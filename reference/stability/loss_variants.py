"""Detached multipliers for the final eight-arm native PPO study.

These functions never change advantages, masks, PPO anchors, or denominators.
They import torch only when called so Ray's worker setup stays stdlib-only.
"""
from __future__ import annotations

ARMS = ("baseline", "full_tis", "negative_tis", "abs_gap_mask",
        "negative_down_mask", "repeat_negative_down_mask", "negative_mass", "sync_fresh")
THRESHOLD = 0.5
TIS_CAP = 2.0
NEGATIVE_LENGTH_CAP = 2048


def _aligned(old, q, advantages, response_mask, repeat_mask=None):
    import torch
    shape = old.shape
    if old.ndim != 2 or any(t.shape != shape for t in (q, advantages, response_mask)):
        raise ValueError("Require aligned padded [responses, tokens] tensors")
    if repeat_mask is not None and repeat_mask.shape != shape:
        raise ValueError("Repeat mask must have exactly the response shape")
    valid = response_mask.bool()
    if any(not torch.isfinite(t[valid]).all() for t in (old, q, advantages)):
        raise ValueError("Nonfinite probability or advantage on a valid token")
    return valid


def make_loss_weights(arm, old_log_probs, rollout_log_probs, advantages, response_mask,
                      repeat_mask=None, supplied_weights=None):
    """Return detached FP32 multipliers; invalid/padded positions are always one.

    All thresholds use the frozen per-round p0, NOT the updating actor logp.
    Method 6 must be computed on original global optimizer minibatches by the
    driver and transported in supplied_weights; local rank/microbatch matching
    would implement a different objective and is deliberately disallowed.
    """
    import torch
    if arm not in ARMS:
        raise ValueError("Unknown study arm: " + str(arm))
    valid = _aligned(old_log_probs, rollout_log_probs, advantages, response_mask, repeat_mask)
    with torch.no_grad():
        # TIS follows the frozen FP32 arithmetic. Hard gates use FP64 operands
        # before subtraction, matching the completed frozen701 ablation: a
        # true difference just beyond +/-0.5 must not round back onto the edge.
        s = old_log_probs.detach().float() - rollout_log_probs.detach().float()
        hard_s = (old_log_probs.detach().double() - rollout_log_probs.detach().double()
                  if arm in ('abs_gap_mask', 'negative_down_mask', 'repeat_negative_down_mask') else None)
        negative = advantages.detach() < 0
        one = torch.ones_like(s)
        if arm in ("baseline", "sync_fresh"):
            weights = one
        elif arm in ("full_tis", "negative_tis"):
            tis = torch.exp(s.clamp(-20.0, 20.0)).clamp(max=TIS_CAP)
            weights = tis if arm == "full_tis" else torch.where(negative, tis, one)
        elif arm == "abs_gap_mask":
            weights = (~(hard_s.abs() > THRESHOLD)).float()
        elif arm in ("negative_down_mask", "repeat_negative_down_mask"):
            excluded = negative & (hard_s < -THRESHOLD)
            if arm == "repeat_negative_down_mask":
                if repeat_mask is None:
                    raise ValueError("Method 5 requires the exact-union-strict-noisy repeat mask")
                excluded &= repeat_mask.bool()
            weights = (~excluded).float()
        else:
            if supplied_weights is None or supplied_weights.shape != s.shape:
                raise ValueError("Method 6 requires aligned global128-matched weights")
            weights = supplied_weights.detach().to(device=s.device, dtype=torch.float32).clone()
            if (weights[valid] <= 0).any():
                raise ValueError("Negative-mass weights must be strictly positive")
            if not torch.equal(weights[valid & ~negative], one[valid & ~negative]):
                raise ValueError("Method 6 changed positive/zero-advantage weights")
            # A is sequence-constant in GRPO; so must the supplied row weight be.
            for i in range(weights.shape[0]):
                selected = weights[i][valid[i]]
                if selected.numel() and not torch.equal(selected, selected[0].expand_as(selected)):
                    raise ValueError("Method 6 weight is not constant within a response")
        weights = torch.where(valid, weights, one).detach()
        if not torch.isfinite(weights[valid]).all() or (weights[valid] < 0).any():
            raise ValueError("Invalid detached loss multiplier")
        return weights


def original_global128_groups(rows=512):
    """Native eight-rank layout: 64 rows/rank, four 16-row local minibatches."""
    if rows != 512:
        raise ValueError("Expected original balanced512 actor batch")
    return [[64 * rank + 16 * inner + j for rank in range(8) for j in range(16)]
            for inner in range(4)]


def negative_mass_weights(advantages, response_mask, groups=None, length_cap=NEGATIVE_LENGTH_CAP):
    """Match original negative |A|*T mass separately in each global minibatch.

    Returns (response-shaped FP32 weights, auditable group records). The default
    groups are the original global128 minibatches; tests may supply a smaller
    partition. Reordering must precede this function, never follow it.
    """
    import torch
    if advantages.ndim != 2 or advantages.shape != response_mask.shape:
        raise ValueError("Require aligned response advantages and mask")
    valid = response_mask.bool()
    lengths = valid.sum(-1)
    if not (lengths > 0).all() or not torch.isfinite(advantages[valid]).all():
        raise ValueError("Every response must contain finite valid advantages")
    first = valid.long().argmax(-1)
    a = advantages.detach().gather(1, first[:, None]).squeeze(1).double()
    if not torch.where(valid, advantages == a[:, None], True).all():
        raise ValueError("Method 6 requires sequence-constant original advantages")
    if groups is None:
        groups = original_global128_groups(advantages.shape[0])
    flattened = [int(i) for group in groups for i in group]
    if sorted(flattened) != list(range(advantages.shape[0])):
        raise ValueError("Minibatch groups must cover each original row exactly once")
    if length_cap <= 0:
        raise ValueError("Length cap must be positive")
    with torch.no_grad():
        row_weights = torch.ones_like(a, dtype=torch.float32)
        records = []
        for inner, rows in enumerate(groups, 1):
            index = torch.tensor(rows, dtype=torch.long, device=a.device)
            t, aa = lengths[index].double(), a[index]
            neg = aa < 0
            u = (float(length_cap) / t).clamp(max=1.0)
            mass = (-aa[neg] * t[neg]).sum()
            denominator = (-aa[neg] * t[neg] * u[neg]).sum()
            lam = mass / denominator if denominator > 0 else mass.new_tensor(1.0)
            w = torch.where(neg, lam * u, torch.ones_like(t)).float()
            row_weights[index] = w
            realized_mass = (-aa[neg] * t[neg] * w[neg].double()).sum()
            relative_error = float((realized_mass - mass).abs() / mass) if mass > 0 else 0.0
            if relative_error > 1e-6:
                raise RuntimeError("FP32 transported weights failed negative-mass matching")
            records.append({"inner": inner, "rows": list(rows), "global_rows": len(rows),
                            "D_physical": int(t.sum()), "negative_rows": int(neg.sum()),
                            "lambda": float(lam), "length_cap": float(length_cap),
                            "negative_mass_original": float(mass),
                            "negative_mass_weighted": float(realized_mass),
                            "relative_mass_error": relative_error,
                            "negative_weight_min": float(w[neg].min()) if neg.any() else None,
                            "negative_weight_max": float(w[neg].max()) if neg.any() else None})
        weights = row_weights[:, None].expand_as(advantages).clone()
        return torch.where(valid, weights, torch.ones_like(weights)).detach(), records


def _selections(mask, advantages, repeat_mask=None):
    basic = (("all", mask), ("positive", mask & (advantages > 0)),
             ("negative", mask & (advantages < 0)), ("zero", mask & (advantages == 0)))
    if repeat_mask is None:
        return basic
    if repeat_mask.shape != mask.shape:
        raise ValueError("Repeat mask must be aligned for stratified statistics")
    repeat = repeat_mask.bool()
    selections = list(basic) + [("repeat", mask & repeat), ("nonrepeat", mask & ~repeat)]
    for sign, selected in basic[1:]:
        selections.extend(((sign + "_repeat", selected & repeat),
                           (sign + "_nonrepeat", selected & ~repeat)))
    return tuple(selections)


def discrepancy_sums(logp, reference, advantages, response_mask, repeat_mask=None):
    """Additive moments; aggregate across microbatches/ranks before dividing.

    abs_gt_0_5 conditional mean is sum_abs_gt_0_5 / abs_gt_0_5_tokens,
    or zero when empty. RMS is sqrt(sum_squared / tokens), L2 sqrt(sum_squared).
    Directional tails distinguish p<q from p>q, with the same strict threshold.
    """
    import torch
    valid = response_mask.bool()
    with torch.no_grad():
        d = logp.detach().double() - reference.detach().double()
        if not torch.isfinite(d[valid]).all():
            raise ValueError("Nonfinite discrepancy on actual training tokens")
        out = {}
        for name, selected in _selections(valid, advantages, repeat_mask):
            x = d[selected]
            absolute = x.abs()
            tail, down, up = absolute > THRESHOLD, x < -THRESHOLD, x > THRESHOLD
            keys = ("tokens", "sum_signed", "sum_abs", "sum_squared", "abs_gt_0_5_tokens",
                    "sum_abs_gt_0_5", "down_lt_minus_0_5_tokens", "sum_abs_down_lt_minus_0_5",
                    "up_gt_0_5_tokens", "sum_abs_up_gt_0_5", "sum_squared_gt_0_5")
            values = torch.stack((x.new_tensor(x.numel()), x.sum(), absolute.sum(), x.square().sum(),
                                  tail.sum().double(), absolute[tail].sum(), down.sum().double(),
                                  absolute[down].sum(), up.sum().double(), absolute[up].sum(),
                                  x[tail].square().sum())).cpu().tolist()
            out[name] = dict(zip(keys, values))
            for key in ("tokens", "abs_gt_0_5_tokens", "down_lt_minus_0_5_tokens", "up_gt_0_5_tokens"):
                out[name][key] = int(out[name][key])
        return out


def likelihood_sums(logp, advantages, response_mask, repeat_mask=None):
    import math
    import torch
    with torch.no_grad():
        p = logp.detach().double()
        out = {}
        for name, selected in _selections(response_mask.bool(), advantages, repeat_mask):
            x = p[selected]
            if not torch.isfinite(x).all() or (x > 1e-4).any():
                raise ValueError("Invalid chosen-token log probability")
            keys = ("tokens", "sum_nll", "sum_nll_squared", "q_ge_0_9_tokens",
                    "q_ge_0_99_tokens", "q_ge_0_999_tokens", "q_ge_0_9999_tokens")
            values = torch.stack((x.new_tensor(x.numel()), -x.sum(), x.square().sum(),
                                  (x >= math.log(.9)).sum().double(), (x >= math.log(.99)).sum().double(),
                                  (x >= math.log(.999)).sum().double(), (x >= math.log(.9999)).sum().double())).cpu().tolist()
            out[name] = dict(zip(keys, values))
            for k in keys:
                if k.endswith("tokens"):
                    out[name][k] = int(out[name][k])
        return out


def multiplier_sums(weights, advantages, response_mask, repeat_mask):
    import torch
    with torch.no_grad():
        out = {}
        valid = response_mask.bool()
        tis_floor = torch.exp(torch.tensor(-20., dtype=torch.float32, device=weights.device)).double()
        for name, selected in _selections(valid, advantages):
            w, a = weights[selected].double(), advantages[selected].double().abs()
            keys = ("tokens", "weight_sum", "weight_squared_sum", "changed_tokens", "zero_weight_tokens",
                    "increased_tokens", "reduced_tokens", "cap2_tokens", "tis_floor_weight_tokens", "repeat_tokens",
                    "anchor_abs_A_mass_unweighted", "anchor_abs_A_mass_weighted")
            values = torch.stack((w.new_tensor(w.numel()), w.sum(), w.square().sum(),
                                  (w != 1).sum().double(), (w == 0).sum().double(),
                                  (w > 1).sum().double(), (w < 1).sum().double(),
                                  (w == TIS_CAP).sum().double(), (w == tis_floor).sum().double(),
                                  repeat_mask.bool()[selected].sum().double(),
                                  a.sum(), (a * w).sum())).cpu().tolist()
            out[name] = dict(zip(keys, values))
            for k in keys:
                if k.endswith("tokens"):
                    out[name][k] = int(out[name][k])
        return out


def coefficient_sums(coefficient, advantages, response_mask, weights, dp_size):
    """Actual one-backward d(loss)/d(logp), divided by native DP factor.

    This is direct token coefficient mass, not a decomposition of parameter
    gradients (whose token Jacobians and vector alignment also matter).
    """
    import torch
    with torch.no_grad():
        c = coefficient.detach().double() / dp_size
        if not torch.isfinite(c[response_mask.bool()]).all():
            raise ValueError("Nonfinite actual logp derivative")
        out = {}
        for name, selected in _selections(response_mask.bool(), advantages):
            x, w = c[selected], weights[selected]
            keys = ("tokens", "active_tokens", "zero_coefficient_tokens", "sum_signed", "sum_abs",
                    "sum_squared", "nonzero_at_zero_weight_tokens")
            values = torch.stack((x.new_tensor(x.numel()), (x != 0).sum().double(),
                                  (x == 0).sum().double(), x.sum(), x.abs().sum(), x.square().sum(),
                                  ((x != 0) & (w == 0)).sum().double())).cpu().tolist()
            out[name] = dict(zip(keys, values))
            for k in keys:
                if k.endswith("tokens"):
                    out[name][k] = int(out[name][k])
        return out


def merge_sums(target, incoming):
    """In-place recursive sum of numeric sufficient statistics, never means."""
    for key, value in incoming.items():
        if isinstance(value, dict):
            merge_sums(target.setdefault(key, {}), value)
        else:
            target[key] = target.get(key, 0) + value
    return target
