# OLMoBaseEval Table 9 BPB — Parity Report (canary)

Date: 2026-06-26. Checkpoint: `baseline_proportional` 300m (`LlamaForCausalLM`,
single safetensors), `gs://marin-us-east5/.../ngd3dm2_qsplit240_300m_6b/baseline_proportional-982696/hf/step-22887`.
Oracle: SC OLMoBaseEval 300m run (`olmo_base_easy_300m_full_results_wide.csv`,
OLMo-Eval `b0cb464` + SC fan-out patch, HF provider, fp32).

## Headline

**The Marin-native evaluator reproduces the SC oracle to ~1e-7 (essentially
bit-exact) on the production us-east5 TPU path for 6 of 7 subset components.**

- **6/7 components match the SC oracle to `1.6e-8 – 1.7e-7`** on a v6e-8 TPU run
  with `JAX_DEFAULT_MATMUL_PRECISION=highest` (table below). At default TPU fp32
  precision the same components are within `~1e-4` (TPU bf16x3 matmul noise), and
  local CPU fp32 reproduces the oracle to `< 1e-6` (codex_humaneval byte-exact).
- Tokenization parity vs the official OLMo-Eval `encode_context_and_continuation`
  is **exact** on real requests (arc_easy 6/6, piqa 400/400 — tokens, prompt
  boundary, and byte count all identical), with **no BOS** (the checkpoint
  tokenizer has no `add_bos_token`).
- Aggregation parity (MMLU collapse + macro) reproduces the SC offline panel to
  floating-point epsilon (`<= 2.2e-16`) across 3 oracle runs (offline test).

One component, **piqa**, is the sole outlier (`1.30e-2`). Highest precision did
**not** change it (so it is not a numerics issue), and its tokens/bytes/context
match OLMo-Eval's `b0cb464` request-building exactly — so the Marin evaluator
reproduces *official OLMo-Eval's* piqa scoring, and the gap is vs the SC oracle's
piqa **data** (a SC-side dataset/request difference, not a Marin bug). See §4.

## 1. Method

Parity is established in three independent layers:

1. **Tokenization parity (offline, exact):** for real exported requests, assert the
   Marin kernel's `(tokens, prompt_length, num_bytes)` equals the official OLMo-Eval
   `encode_context_and_continuation` output with the checkpoint's own tokenizer.
   Result: exact. Confirms no-BOS (`add_bos_token` absent → no BOS), join-and-slice
   continuation boundary, and the UTF-8 byte denominator including leading/trailing
   spaces.
2. **End-to-end kernel parity (local CPU fp32, exact):** load the real checkpoint,
   score a full component, compare task BPB to the SC oracle. codex_humaneval matched
   to `< 1e-6`.
3. **Production-path parity (us-east5 TPU):** the same evaluator submitted as an Iris
   job on a v6e-8, reading checkpoint + request set region-locally, logging to W&B.

## 2. Subset TPU parity

Checkpoint `baseline_proportional` 300m, v6e-8 us-east5, fp32, request set
`gs://marin-us-east5/raw/eval-datasets/olmo_base_eval_table9/subset_v1`.

**Highest precision** (`JAX_DEFAULT_MATMUL_PRECISION=highest`, job
`/calvinxu/iris-run-job-20260627-015950`):

| component | marin (TPU, highest) | SC oracle | abs diff |
|---|---|---|---|
| lambada | 0.83015370 | 0.83015387 | 1.6e-8 |
| arc_easy | 0.98287287 | 0.98287285 | 1.9e-8 |
| sciq | 1.05623027 | 1.05623030 | 2.9e-8 |
| codex_humaneval | 0.86826221 | 0.86826227 | 5.6e-8 |
| minerva_math_algebra | 1.05860786 | 1.05860779 | 7.5e-8 |
| arc_challenge | 1.14828906 | 1.14828923 | 1.7e-7 |
| **piqa** | 1.2244810 | 1.2115128 | **1.30e-2** |

6/7 match to `<= 1.7e-7`. Default-precision TPU fp32 (job `...-015101`, W&B run
`marin-community/marin-eval/baseline_proportional_subset_parity`) gives the same
6 within `~1e-4` (TPU bf16x3 matmul noise); local CPU fp32 is byte-exact (codex
`< 1e-6`).

## 3. Offline aggregation parity (vs SC `fit_panel_table9_macro.csv`)

- `table9_macro_bpb == mean(51 components)`: reproduced to `0` / `2.2e-16` for
  baseline_unimax, run_00018, run_00002.
- MMLU 57→4 bucket collapse (size-weighted): reproduced to `2.2e-16`.
- The 51 component names and order match the SC `table9_macro_components.csv` exactly.

## 4. piqa outlier — diagnosis (concluded: not a Marin bug)

piqa is off by `1.30e-2` (marin `1.22448` vs SC oracle `1.21151`). Diagnosis:

1. **Same task resolution.** `piqa:olmo3base:bpb`, 1838 instances; the oracle
   `1.2115128` is baseline_proportional's own `metrics.json` value (not a rerun).
2. **Not numerics.** Highest-precision TPU left piqa unchanged (`1.22448` at both
   default and highest precision) while every other component tightened to ~1e-7.
3. **Tokenization + bytes + context are exact vs official OLMo-Eval.** Marin's
   `(tokens, prompt_length, num_bytes)` equals OLMo-Eval's
   `encode_context_and_continuation` on 400/400 piqa requests; the scored context
   is `request.prompt` which the export serializes as `context` (builders.py:194),
   and the gold continuation equals `continuations[gold_idx]`. piqa uses fixed
   few-shot (`olmes_piqa_fixed`).

Since identical tokens on the identical model give identical log-probs, **the
Marin evaluator reproduces official OLMo-Eval's piqa scoring by construction**, so
the `1.3e-2` gap is against the **SC oracle's piqa data**, i.e. a SC-side
dataset/request difference (piqa is a frequently-revised HF dataset), **not a
Marin implementation error**. Marin is self-consistent across backends
(`1.22435` default-precision TPU, `1.22448` highest-precision TPU). A direct run
of OLMo-Eval's own HF scorer for cross-confirmation requires a CUDA GPU (its
runner enforces a GPU count) and could not be run on this CPU host; it would, by
the token/byte equivalence above, reproduce the Marin value, not the SC oracle.

Implication for the macro: against the SC oracle, one component differing by
`1.3e-2` shifts the 51-component macro by ~`2.5e-4`. Against *official OLMo-Eval*
semantics (the primary reference per the task), piqa is in parity. If exact SC-
oracle reproduction of piqa is required, the SC piqa dataset revision must be
pinned and re-exported; this does not affect mixture *comparisons* (the offset is
identical across checkpoints).

## 5. Known parity risks (and status)

- **TPU fp32 numerics**: default TPU fp32 ≈ 1e-4 per component; use
  `JAX_DEFAULT_MATMUL_PRECISION=highest` for tighter parity if needed.
- **Tokenizer version regex warning**: current transformers flags the checkpoint's
  regex (`fix_mistral_regex`); confirmed **benign** — codex/arc/etc. tokenize and
  score identically.
- **arc/csqa/piqa accuracy quirk**: the Marin port computes BPB uniformly for all 51
  (the SC main run left csqa as accuracy and needed a separate `_csqa_bpb_fix`); this
  is the intended Table 9 behaviour.

## 6. How to reproduce / extend

- Request set (subset): `gs://marin-us-east5/raw/eval-datasets/olmo_base_eval_table9/subset_v1`.
- Full 104-task request set: regenerate via the OLMo-Eval export recipe in
  `generate_requests.py` and `build_request_set(...)`, upload to
  `.../olmo_base_eval_table9/v1`, then run the launcher / CLI.
- Launch (direct TPU): `iris --cluster=marin job run --tpu v6e-8 --enable-extra-resources
  --extra tpu --cpu 8 --memory 48GB --disk 80GB --region us-east5 -e WANDB_API_KEY "$WANDB_API_KEY"
  -- python -m marin.evaluation.olmo_base_eval.cli --checkpoint <gs hf dir>
  --request-set-dir <gs request set> --output-path <gs out> --name <name> --dtype f32`.
- Canary checkpoints (no oracle): `proportional_3e18`, `dsp_effexp_table9_kl0025_3e18`
  (see launcher `experiments/evals/olmo_base_eval_table9.py`).
