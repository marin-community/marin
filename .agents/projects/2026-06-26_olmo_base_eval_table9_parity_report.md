# OLMoBaseEval Table 9 BPB — Parity Report (canary)

Date: 2026-06-26. Parity checkpoint: `baseline_proportional` 300m
(`LlamaForCausalLM`, single safetensors),
`gs://marin-us-east5/.../ngd3dm2_qsplit240_300m_6b/baseline_proportional-982696/hf/step-22887`.
Oracle: SC OLMoBaseEval 300m run (`olmo_base_easy_300m_full_results_wide.csv`,
OLMo-Eval `b0cb464` + SC fan-out patch, HF provider, fp32).

## Headline — full 51-component parity

Running the Marin-native evaluator on the parity checkpoint on a us-east5 v6e-8
(fp32, `JAX_DEFAULT_MATMUL_PRECISION=highest`), over the full 104-task / 88,592-
instance request set, reproduces the SC oracle:

- **Macro: marin `1.086283` vs oracle `1.086283`, abs diff `1.9e-8`** (target `<= 2e-4`).
- **All 51/51 components within `1e-3`** — max abs diff `2.36e-6`, mean `1.5e-7`
  (target: components mostly `<= 1e-3`).
- Tokenization matches the official OLMo-Eval `encode_context_and_continuation`
  exactly (no BOS — the checkpoint tokenizer has no `add_bos_token`).
- Aggregation (MMLU 57→4 collapse + unweighted-mean macro) reproduces the SC
  offline panel to `<= 2.2e-16` (offline test, 3 oracle runs).

Both the macro and the per-component targets are met by ~4 orders of magnitude.

## Method

Three independent layers, all confirmed:

1. **Tokenization parity (offline, exact)** vs OLMo-Eval's own tokenizer function
   on real requests — `(tokens, prompt_length, num_bytes)` identical.
2. **End-to-end kernel parity (local CPU fp32, byte-exact)** — codex_humaneval
   matched the oracle to `< 1e-6` early in development.
3. **Production-path parity (us-east5 TPU)** — the same evaluator as an Iris job,
   reading checkpoint + request set region-locally, logging to W&B. Default TPU
   fp32 lands components within `~1e-4` (bf16x3-pass matmul noise);
   `highest` precision tightens them to `~1e-7` (used for the headline above).

## The one real bug found (gold selection)

The first full-104-task run was far off on multiple-choice tasks whose gold answer
is not the first choice (winogrande `2.55` vs `1.37`, plus coqa, socialiqa,
basic_skills, hellaswag — all ~2x). Root cause: the request bridge scored the
singular `request.continuation`, which is only `continuations[0]`. That is correct
for single-continuation PPL tasks (arc, squad, sciq) and for MC tasks whose gold is
choice 0 (csqa, mmlu) — which is why the 7-task smoke subset looked clean — but
wrong when the gold is a later choice. **This is also why piqa looked off in the
subset; it was this bug, not an SC-side data difference (an earlier, incorrect
hypothesis).**

Fix: select `continuations[gold_idx]` for multi-choice records (and
`continuation_prompts[gold_idx]` as the context for per-choice-prompt tasks like
winogrande), keeping `continuations[0]` for the single-continuation case (the BPB
metric's single-output path). After the fix all 51 components match (above).

## Canary checkpoints (no SC oracle)

Run end-to-end on the two delphi 3e18 mixtures (both `Qwen3ForCausalLM`,
step-3006) to validate the pipeline + W&B writeback and to compare the mixtures:

| mixture | table9 macro BPB |
|---|---|
| `proportional_3e18` | 1.198731 |
| `dsp_effexp_table9_kl0025_3e18` | 1.143605 |
| **delta (optimized − proportional)** | **−0.055126** |

These have no on-disk SC oracle; the run demonstrates the full Iris→TPU→W&B path
and produces the headline `olmo_base_easy/table9_macro_bpb` for each mixture. The
Table-9-optimized DSP mixture is `0.055` BPB lower than proportional — the
expected direction (it was optimized for this macro), and the kind of comparison
this evaluator exists to make.

The canaries are Qwen3, not Llama; loading them surfaced a second issue —
`HFCheckpointConverter.from_hf` probes every registered config and constructs each
one's converter, and the gemma configs try to load the gated `google/gemma-2b`
reference, so a Qwen3 checkpoint (registered after gemma) 403s. Fixed by deriving
the config class from `config.json`'s `model_type` directly; verified on both the
Llama parity checkpoint and the Qwen3 canaries.

## Provenance & artifacts

- Request set: `gs://marin-us-east5/raw/eval-datasets/olmo_base_eval_table9/v2`
  (104 tasks, 88,592 instances; OLMo-Eval `b0cb464` + SC patch).
- W&B: `marin-community/marin-eval`, group `olmo_base_eval_table9`; each run logs
  `olmo_base_easy/table9/<component>/bpb`, `olmo_base_easy/table9_macro_bpb`, and
  SC-compatible `olmo_base_eval/easy_bpb/<task>/bpb` keys, plus checkpoint URI,
  request-set version, and OLMo-Eval SHA as provenance.

## Notes / known limitations

- **Sharded safetensors**: all three checkpoints are single `model.safetensors`,
  so Levanter's loader handles the sharded layout (it is the standard big-model
  path) but these specific 300m checkpoints do not exercise it here.
- **TPU fp32 numerics**: default TPU fp32 ≈ `1e-4` per component; the evaluator
  forces `jax_default_matmul_precision="highest"` so all launch paths reach `~1e-7`.
- **arc/csqa/piqa accuracy quirk**: the Marin port computes BPB uniformly for all
  51 (the SC main run left csqa as accuracy and needed a separate `*_csqa_bpb_fix`);
  this is the intended Table 9 behaviour.
- **Offline request export**: the OLMo-Eval mock runner deadlocks above ~20 tasks,
  so the request set is exported in small per-suite groups (a one-time build step).
