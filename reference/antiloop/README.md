# Confirmed-loop unlikelihood reference

These two modules contain the detector and loss used in our Qwen3-8B-Base APRIL experiments. They are reference components, not a complete trainer or a demonstrated solution to RL instability. The original algorithms are copied unchanged; `source_manifest.json` records their source hashes.

## Requirements and CPU tests

Python 3.10+, NumPy and PyTorch are required. The detector accepts a tokenizer exposing `get_vocab()`, `decode()` and special-token IDs; using a real model additionally requires its compatible tokenizer. The packed-loss helper requires PyTorch's jagged nested tensors. Our CPU check used PyTorch 2.12; the original runtime used a different GPU build, so this check does not establish GPU/FSDP equivalence.

```sh
python -m unittest discover -s tests -v
```

All test inputs are synthetic. Passing these tests verifies selected implementation properties; their values are not experimental evidence about a trained model.

## Included components

- `repeat_targets_v2.py`: finds already-confirmed exact text cycles using character-wise NFKC normalization and whitespace removal. It selects at most 64 subsequent token positions from the first eligible episode. Alphabetic cycles require at least 128 preceding normalized characters and 16 cycles. Numeric/symbol cycles additionally require 512 preceding characters and 256 visible response tokens; whitespace does not count toward the token threshold. Heuristics protect explicitly requested long mathematical expansions, numbers and layouts. A short-answer request, such as “last two digits,” takes precedence over a large integer appearing in the question.
- `ul_loss.py`: computes the stable family loss `softplus(logsumexp(bad logits) - logsumexp(other logits))` in FP32. The bad family includes token aliases continuing the same confirmed character cycle. This avoids subtracting a rounded probability from one. It also provides exact packed query/target alignment, a graph-connected zero loss for empty masks, and gradient-budget arithmetic.

The detector needs the original question text for its mathematical protections. Do not pass a decoded chat template as a substitute without checking identity. EOS, special tokens and whitespace-only pieces are excluded from the bad family.

The server-specific online bank, collector hooks, checkpoint restoration and optimizer wrapper are deliberately not packaged. They rely on native APRIL/FSDP contracts and require separate verification. In particular, these modules alone do not implement distributed scaling, preserve optimizer state, or resume an experiment.

## Actual 600→620 protocol

The short experiment ran one UL arm for 20 RL rounds / 80 original Adam updates from a full step-600 checkpoint. Main RL rewards, grouped advantages, PPO clipping, token denominator and ordinary decoding were unchanged. A separate auxiliary backward used eight global trajectories, at most 64 targets each, and a fixed denominator of 512. Missing slots used real current training contexts with zero auxiliary masks.

The online bank observed completed training groups before mixed-group filtering, including all-wrong groups. It excluded the full 1,621-question development set, scanned a deterministic hash-selected maximum of 128 candidate responses per round, kept up to 512 trajectories for ten observation rounds, and filled 32 slots per round with question round-robin selection. It did not select by reward, discrepancy or current model probability.

The auxiliary coefficient was detached and bounded by both `lambda_max = 0.01` and `0.05 * ||g_RL|| / ||g_UL||`, with no floor. The original global clip of one and one Adam update followed. This bounds the auxiliary gradient component; it does not bound Adam parameter changes by 5%. The RL-only norm must be reported separately from the combined norm.

## Observed outcome and limits

The short run completed. On the same 64-question, eight-sample native evaluation panel:

| Step | Accuracy | Text-cycle candidates | Mean response tokens |
| --- | ---: | ---: | ---: |
| 600 | 51.1719% | 0 / 512 | 2433.45 |
| 610 | 49.6094% | 43 / 512 | 2133.43 |
| 620 | 44.5313% | 117 / 512 | 3043.12 |

These observations do **not** demonstrate successful prevention. There was no matched new baseline arm, and twenty rounds cannot establish later stability or rebound. Mechanical cycle candidates are not automatic semantic judgments. Zero flags in the initial finite panel do not mean the starting model was repetition-free.

The auxiliary loss was nonzero in 48 of 80 updates, with 303 trajectory uses from 76 distinct responses / 73 questions and 9,032 target-token uses. Actual selected categories were 302 alphabetic uses and one symbol/other use; no numerical-cycle trajectory was selected in this run. Therefore the generic numerical branch passed implementation tests but was not exercised by real numerical-loop supervision here.

A separate earlier study started at step 720 and used a **fixed E-only bank**. It must not be described as this generic online-bank experiment. At step 740, all arms had zero legacy strict repetition flags, while numeric cycles remained; “strict repetition = 0” did not mean all repetition disappeared.

The [published evidence](../../data/investigation/repetition_interventions_evidence.json) records archived artifact identifiers, hashes, populations and exclusions. Keep those actual measurements separate from the synthetic CPU tests. No API key, model weight, generated training corpus or server address is included in this package.
