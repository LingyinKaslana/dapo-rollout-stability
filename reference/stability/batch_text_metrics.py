"""CPU-only text metrics and frozen exact OR strict-noisy target masks.

The detector reads original token IDs and user-question text only. Probabilities,
advantages and rewards are not inputs. Workers use spawn, hide CUDA before
loading a local tokenizer and persist across rounds. No model/API calls occur.
Mask coordinates are the ORIGINAL padded response target coordinates.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import time
import numpy as np

from repetition_mask import build_repetition_masks
from noisy_repetition_mask import build_noisy_repetition_masks
from repeat_targets_v2 import decode, normalize
from strict_rules import classify_obvious

VERSION = "final-eight-arm-text-metrics-v1"
PROMPT_TEMPLATE = "qwen-single-user-assistant-prefix-v1"
_TOKENIZER = None
_POOL_CREATOR_PID = None
_IDENTITY_KEYS = ("uid", "qid", "question_id", "response_uid", "round", "step",
                  "replicate", "request_order", "source_population")
_FLAGS = (
    "strict_repetition", "normalized_text_cycle128_any",
    "normalized_text_cycle128_alphabetic", "normalized_text_cycle128_numeric",
    "normalized_text_cycle128_other", "exact_confirmed_repeat_any",
    "noisy_repeat_primary_any", "noisy_repeat_strict_any", "repeat_strict_union_any",
    "gibberish_candidate", "response_hit_token_cap", "first_word_alright",
    "first_word_to", "has_think_tag_prefix")


def sha_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha_ids(ids):
    """SHA256 over contiguous little-endian signed-int64 original ID bytes."""
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes(order="C")).hexdigest()


def parse_original_prompt(prompt_ids, tokenizer, expected_user_text=None):
    text = decode(tokenizer, prompt_ids)
    match = re.fullmatch(
        r"<\|im_start\|>user\n(.*?)<\|im_end\|>\n<\|im_start\|>assistant\s*", text, re.S)
    if match is None:
        raise ValueError("Original prompt does not match the frozen single-user Qwen template")
    user = match.group(1)
    if expected_user_text is not None and user != expected_user_text:
        raise ValueError("Recovered original user field differs from supplied user text")
    return text, user


def gibberish_candidates(text):
    """Same mechanical cues as the previous evaluator; not a semantic verdict."""
    out = []
    for name, pattern in [("replacement_character", "\ufffd"),
                          ("control_character", r"[\x00-\x08\x0b\x0c\x0e-\x1f]"),
                          ("upper_letter_run24", r"\b[A-Z]{24,}\b"),
                          ("symbol_cluster32", r"[^\w\s]{32,}")]:
        for match in list(re.finditer(pattern, text))[:8]:
            out.append({"rule": name, "start_char": match.start(),
                        "end_char": match.end()})
    return out


def _process_one(row, tokenizer):
    ids = list(map(int, row["response_ids"]))
    prompt_ids = list(map(int, row["prompt_ids"]))
    if not ids or any(i < 0 for i in ids + prompt_ids):
        raise ValueError("Responses must contain at least one valid nonnegative original token")
    prompt, user = parse_original_prompt(prompt_ids, tokenizer, row.get("expected_user_text"))
    text = decode(tokenizer, ids)
    response_sha = sha_text(text)
    prompt_sha = sha_ids(prompt_ids)
    checks = {"response_text_sha256": response_sha,
              "response_token_ids_sha256": sha_ids(ids),
              "prompt_token_ids_sha256": prompt_sha,
              "prompt_text_sha256": sha_text(prompt),
              "user_text_sha256": sha_text(user)}
    for key, actual in checks.items():
        expected = row.get("expected_" + key)
        if expected is not None and expected != actual:
            raise ValueError(f"Original input identity mismatch: {key}")
    if "expected_response_text" in row and row["expected_response_text"] != text:
        raise ValueError("Original response decoded text identity mismatch")
    exact = build_repetition_masks(ids, tokenizer, prompt_text=user)
    noisy = build_noisy_repetition_masks(ids, tokenizer, prompt_text=user)
    union = exact["confirmed_mask"] | noisy["strict_mask"]
    if union.shape != (len(ids),) or union.dtype != np.bool_:
        raise AssertionError("Original-target mask shape/dtype changed")
    strict = classify_obvious(text, user)
    # All128-character cycle candidates are monitored, including numeric cycles
    # that intentionally do not pass the conservative online mask thresholds.
    cycles = exact["episodes"]
    categories = {c["category"] for c in cycles}
    garble = gibberish_candidates(text)
    first = re.match(r"\s*([A-Za-z]+)", text)
    first_word = first.group(1) if first else None
    token_cap = int(row.get("token_cap", 8192))
    if token_cap < 1 or len(ids) > token_cap:
        raise ValueError("Invalid declared generation token cap")
    result = {k: row[k] for k in _IDENTITY_KEYS if k in row}
    result.update(
        row_index=int(row.get("row_index", 0)), tokens=len(ids),
        **checks, token_id_sha_encoding="little-endian-int64-C-order",
        decoded_text_sha_encoding="UTF-8, skip_special_tokens=False, cleanup=False",
        prompt_template=PROMPT_TEMPLATE, prompt_template_verified=True,
        normalized_characters=len(normalize(text)),
        strict_repetition=bool(strict["obvious"]), strict_types=strict["types"],
        normalized_text_cycle128_any=bool(cycles),
        normalized_text_cycle128_alphabetic="alphabetic" in categories,
        normalized_text_cycle128_numeric="numeric" in categories,
        normalized_text_cycle128_other="other" in categories,
        exact_confirmed_repeat_any=bool(exact["confirmed_mask"].any()),
        exact_confirmed_repeat_tokens=int(exact["confirmed_mask"].sum()),
        noisy_repeat_primary_any=bool(noisy["primary_mask"].any()),
        noisy_repeat_primary_tokens=int(noisy["primary_mask"].sum()),
        noisy_repeat_strict_any=bool(noisy["strict_mask"].any()),
        noisy_repeat_strict_tokens=int(noisy["strict_mask"].sum()),
        repeat_strict_union_any=bool(union.any()), repeat_strict_union_tokens=int(union.sum()),
        gibberish_candidate=bool(garble), gibberish_candidate_rules=sorted({x["rule"] for x in garble}),
        response_hit_token_cap=len(ids) >= token_cap,
        ended_tokenizer_eos=ids[-1] in _id_list(getattr(tokenizer, "eos_token_id", None)),
        first_generated_token_id=ids[0], first_generated_token_decoded=decode(tokenizer, ids[:1]),
        first_generated_word=first_word,
        first_word_alright=bool(first_word and first_word.lower() == "alright"),
        first_word_to=bool(first_word and first_word.lower() == "to"),
        has_think_tag_prefix=text.lstrip().startswith("<think>"),
        exact_alignment=exact["audit"]["decoder_alignment"],
        noisy_alignment=noisy["audit"]["decoder_alignment"],
        invalid_exact_target_boundaries=exact["audit"]["unstable_or_nonexact_visible_token_boundaries"],
        noisy_excluded_boundary_counts={k: v for k, v in noisy["audit"]["checks"].items() if "excluded" in k},
        exact_protection_counts=dict(Counter(reason for c in cycles for reason in c["protected_reasons"])),
        noisy_protection_counts=noisy["audit"]["protected_reasons"],
        detector_uses_probabilities_advantages_rewards=False)
    return result, union


def _id_list(value):
    return [] if value is None else list(value) if isinstance(value, (list, tuple)) else [int(value)]


def _initialize(tokenizer_path):
    global _TOKENIZER
    # Executed only in a spawned dedicated CPU process. Never mutate the GPU
    # driver's CUDA visibility, tokenizer state, model state or random generator.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=False,
                                               local_files_only=True)
    import sys
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        raise RuntimeError("CPU text worker unexpectedly initialized CUDA")


def _process_task(row):
    if _TOKENIZER is None:
        raise RuntimeError("CPU text worker was not initialized")
    return _process_one(row, _TOKENIZER)


def aggregate_rows(rows):
    n = len(rows)
    lengths = np.asarray([r["tokens"] for r in rows], dtype=np.int64)
    counts = {k: sum(bool(r[k]) for r in rows) for k in _FLAGS}
    token_keys = ["exact_confirmed_repeat_tokens", "noisy_repeat_primary_tokens",
                  "noisy_repeat_strict_tokens", "repeat_strict_union_tokens"]
    total = int(lengths.sum())
    return {"responses": n, "tokens": total,
            "mean_length": float(lengths.mean()) if n else None,
            "median_length": float(np.median(lengths)) if n else None,
            "length_quantiles": {str(q): float(np.quantile(lengths, q)) if n else None
                                 for q in [.1, .25, .5, .75, .9, .95, .99]},
            "counts": counts, "fractions": {k: v / n if n else None for k, v in counts.items()},
            "token_counts": {k: sum(r[k] for r in rows) for k in token_keys},
            "token_fractions": {k: sum(r[k] for r in rows) / total if total else None for k in token_keys},
            "first_word_counts_case_sensitive": dict(Counter(r["first_generated_word"] or "<non-word>" for r in rows)),
            "probability_of_first_token_measured": False,
            "semantic_gibberish_verdict_measured": False}


class TextMetricsPool:
    """One instance in the driver, reused for all rounds and all populations.

    process_rows takes ORIGINAL unpadded IDs and optional expected SHA fields.
    process_batch takes padded CPU arrays and original response/prompt masks.
    Its returned repeat_mask_strict_union has exactly the response array shape.
    These methods block until the requested population is fully assessed; CPU
    workers run in parallel. The caller can place non-loss diagnostics on its
    own thread/queue while retaining loss5's mask dependency.
    """
    def __init__(self, tokenizer_path, workers=16, chunksize=1):
        self.tokenizer_path = str(tokenizer_path)
        self.workers = int(workers)
        self.chunksize = int(chunksize)
        if not 1 <= self.workers <= 64 or self.chunksize < 1:
            raise ValueError("Invalid CPU worker count or chunk size")
        self.creator_pid = os.getpid()
        self.pool = ProcessPoolExecutor(max_workers=self.workers,
            mp_context=multiprocessing.get_context("spawn"), initializer=_initialize,
            initargs=(self.tokenizer_path,))

    def process_rows(self, rows, padded_length=None):
        if os.getpid() != self.creator_pid:
            raise RuntimeError("Do not share a driver pool across forked/Ray workers")
        started = time.monotonic()
        tasks = []
        for index, original in enumerate(rows):
            row = dict(original)
            row["row_index"] = index
            tasks.append(row)
        outputs = list(self.pool.map(_process_task, tasks, chunksize=self.chunksize))
        metrics = [x[0] for x in outputs]
        max_len = max((len(x[1]) for x in outputs), default=0)
        width = max_len if padded_length is None else int(padded_length)
        if width < max_len:
            raise ValueError("Declared padded response width shorter than a valid response")
        masks = np.zeros((len(outputs), width), dtype=np.bool_)
        for index, (row, mask) in enumerate(outputs):
            if row["row_index"] != index:
                raise AssertionError("CPU result row ordering changed")
            masks[index, :len(mask)] = mask
        return {"version": VERSION, "rows": metrics, "aggregate": aggregate_rows(metrics),
                "repeat_mask_strict_union": masks,
                "mask_sha256": hashlib.sha256(masks.tobytes(order="C")).hexdigest(),
                "cpu_workers": self.workers, "elapsed_seconds": time.monotonic() - started,
                "GPU_calls": 0, "API_calls": 0,
                "classification": "Mechanical repetition/garble candidates; not semantic diagnoses"}

    def process_batch(self, responses, response_mask, prompts, prompt_mask, row_metadata=None):
        responses, response_mask, prompts, prompt_mask = map(np.asarray,
            (responses, response_mask, prompts, prompt_mask))
        if responses.ndim != 2 or prompts.ndim != 2 or responses.shape != response_mask.shape or prompts.shape != prompt_mask.shape:
            raise ValueError("Expected original two-dimensional token arrays and matching masks")
        n, width = responses.shape
        if prompts.shape[0] != n or row_metadata is not None and len(row_metadata) != n:
            raise ValueError("Mismatched batch rows")
        if not np.isin(response_mask, [0, 1]).all() or not np.isin(prompt_mask, [0, 1]).all():
            raise ValueError("Masks must contain only bool/0/1")
        tasks = []
        for i in range(n):
            mask = response_mask[i].astype(bool)
            length = int(mask.sum())
            if not mask[:length].all() or mask[length:].any():
                raise ValueError("Response mask is not a contiguous valid prefix")
            row = dict(row_metadata[i]) if row_metadata is not None else {}
            if "response_ids" in row or "prompt_ids" in row:
                raise ValueError("Metadata must not override original token IDs")
            row.update(response_ids=responses[i, :length].tolist(),
                       prompt_ids=prompts[i, prompt_mask[i].astype(bool)].tolist())
            tasks.append(row)
        result = self.process_rows(tasks, padded_length=width)
        if np.any(result["repeat_mask_strict_union"] & ~response_mask.astype(bool)):
            raise AssertionError("A masked target lies in padding")
        return result

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def probability_summary(logp, logq, valid_mask, threshold=.5):
    """Separate CPU diagnostic: observed-action log probabilities, never a mask input.

    Tail means are zero when no token exceeds the STRICT threshold. RMS is
    length-normalized; L2 deliberately is not. log_PPL is mean(-logp), NOT d.
    Finite valid values are required; invalid values fail rather than disappear.
    """
    p, q, valid = np.asarray(logp), np.asarray(logq), np.asarray(valid_mask, dtype=bool)
    if p.shape != q.shape or p.shape != valid.shape:
        raise ValueError("Probability diagnostic shapes differ")
    p, q = p[valid].astype(np.float64), q[valid].astype(np.float64)
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError("Nonfinite valid token log probability")
    s = p - q
    d = np.abs(s)
    n = len(s)
    tail, down, up = d > threshold, s < -threshold, s > threshold
    mean = lambda a: float(np.mean(a)) if len(a) else 0.
    out = {"tokens": n, "threshold": threshold, "mean_signed_gap": mean(s),
           "mean_absolute_gap": mean(d), "absolute_gap_L2": float(np.linalg.norm(d)),
           "absolute_gap_RMS": float(np.sqrt(mean(d * d))),
           "absolute_gap_max": float(d.max()) if n else 0.,
           "log_PPL_p": mean(-p), "log_PPL_q": mean(-q),
           "absolute_gap_threshold_count": int(tail.sum()),
           "absolute_gap_threshold_fraction": mean(tail),
           "absolute_gap_threshold_conditional_mean_or_zero": mean(d[tail]),
           "absolute_gap_threshold_excess_mean_or_zero": mean(d[tail] - threshold),
           "p_down_count": int(down.sum()), "p_up_count": int(up.sum()),
           "p_down_fraction": mean(down), "p_up_fraction": mean(up),
           "p_down_absolute_gap_mean_or_zero": mean(d[down]),
           "p_up_absolute_gap_mean_or_zero": mean(d[up]),
           "p_down_q_probability_mean_or_zero": mean(np.exp(q[down])),
           "p_up_q_probability_mean_or_zero": mean(np.exp(q[up])),
           "empty_population": n == 0}
    for prob in (.9, .99, .999):
        out[f"p_ge_{prob}_fraction"] = mean(p >= np.log(prob))
        out[f"q_ge_{prob}_fraction"] = mean(q >= np.log(prob))
    return out
