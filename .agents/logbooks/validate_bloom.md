# Logbook: Validate Bloom Alignment Pipeline Reproduction in Marin

**Goal**: The Marin alignment codebase (`lib/marin/src/marin/alignment/`) implements synthetic preference data generation and DPO training. We need to validate that Marin's new evaluation pipeline (inference + LM-as-judge) reproduces results comparable to the original Bloom codebase (`/Users/ahmed/code/bloom/`). This logbook tracks experiments comparing GPU (Bloom) vs TPU (Marin) inference and judging on the same model checkpoints, prompts, and judge configuration.

**Branch**: `alignment_function`

**Key files added/modified**:
- `lib/marin/src/marin/alignment/evaluate.py` — new inference runner + eval judge
- `lib/marin/src/marin/alignment/align.py` — added `evaluate()` function, `EvalConfig`, `HF_TOKEN` to env vars
- `lib/marin/src/marin/alignment/judge.py` — made 7 private helpers public for reuse
- `lib/marin/src/marin/alignment/prompts/judge.py` — fixed 3 prompt differences to match Bloom exactly
- `experiments/posttrain/eval_llama3_8b_alignment.py` — experiment script for inference
- `experiments/posttrain/run_bloom_judge.py` — one-off Bloom-compatible judging script
- `tests/test_alignment.py` — updated imports for renamed helpers

---

## 2026-03-31: Eval Pipeline Implementation

### What was built
1. **`evaluate.py`**: Inference runner that loads eval prompts (supports both Marin sharded JSONL.GZ and Bloom `<statement>/eval_prompts.json` tree formats via `PromptFormat` enum), runs a target model via `BatchedVllmServeSession` without spec guidance (tests internalized alignment), and writes sharded JSONL results.

2. **`evaluate()` in `align.py`**: Creates two `ExecutorStep`s (inference + judge) that can chain after DPO training. Accepts `prompts` as either `ExecutorStep` or direct GCS/local path string.

3. **Judge helpers made public in `judge.py`**: `JudgeRequest`, `build_judge_messages`, `parse_compliance_result`, `judge_responses_local_batch`, `judge_responses_api_batch`, `compliance_result_record`, `write_json`.

### Eval split extraction
- Extracted eval split from Bloom's `dev-bloom-results-gpt-4-mini-prompts/` dataset
- Reproduced Bloom's exact seeded split: `seed=7, fractions={train:0.70, val:0.15, eval:0.15}`
- Per-statement RNG: `random.Random(int(SHA256("7:{statement_id}")[:16], 16))`
- Cross-verified 4 statements (`ask_clarifying_questions`, `avoid_being_condescending`, `do_not_lie`, `formatting`) — all match exactly
- **2,576 eval prompts across 46 statements**
- Uploaded to:
  - `gs://marin-us-central1/alignment/gpt-4.1-eval-split/`
  - `gs://marin-us-east5/alignment/gpt-4.1-eval-split/`

---

## 2026-03-31: EXP-001 — Llama-3.1-8B-Instruct Baseline Inference (n=1)

**Hypothesis**: Verify that the inference pipeline works end-to-end on TPU.

**Config**: `meta-llama/Llama-3.1-8B-Instruct` from GCS (`gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f`), v5p-8, TP=4, temp=0.7, max_tokens=2048, n=1, batch_size=256

**Job**: `/ahmed/eval-llama3-8b-gcs-v6` — SUCCEEDED

**Results**: `gs://marin-us-central1/eval/llama3_8b_instruct_bloom_speceval/inference-180de5`
- 2,576/2,576 records, 0 empty responses
- Response length: min=3, median=1757, max=11460, mean=1748 chars
- Clean EOS: 98.6% (35 truncated at max_tokens)
- Throughput: ~13 items/s, 4.7k completion tok/s
- Total time: ~7 min (including model load + TPU compile)

**Notes**: First 3 attempts failed:
- v1/v2: `AssertionError` shape mismatch loading LoRA DPO checkpoint — `runai_streamer` and `load_format=auto` both failed with TP=4 and TP=1
- v4: `RepositoryNotFoundError` — HF_TOKEN not forwarded to Iris child job
- v5: Same HF_TOKEN issue — fixed by adding `HF_TOKEN` to `_llm_env_vars()`
- v6: Used GCS model path, worked immediately

---

## 2026-03-31: EXP-002 — DPO Checkpoint Inference (n=1)

**Hypothesis**: Verify DPO checkpoint `bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0` can be served on TPU.

**Config**: `gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad/hf/step-849`, v5p-8, TP=4, temp=0.7, max_tokens=2048, n=1

**Job**: `/ahmed/eval-marin-dpo-beta01-v8` — SUCCEEDED

**Results**: `gs://marin-us-central1/eval/marin_dpo_beta01_lr75e7_seed0_bloom_speceval/inference-1f2aa4`
- 2,576 records, 0 empty, 98.2% clean EOS
- Response length: median=1143, mean=1376 (shorter than base Llama — model is more concise after DPO)

**Notes**:
- First attempt (`v7`) failed: `tokenizer.chat_template is not set`. Fixed by copying `tokenizer_config.json` and `generation_config.json` from base Llama model to checkpoint.

---

## 2026-03-31: EXP-003 — GPU vs TPU Inference Comparison (138 prompt pairs)

**Hypothesis**: TPU inference produces responses from the same model distribution as GPU inference.

**Method**: Compared 3 prompts from each of 46 statements between:
- GPU run: Bloom local vLLM (`/lfs/skampere3/0/ahmedah/models/marin/bloom_v2/beta0.1_lr7.5e-7_seed0/step-849`)
- TPU run: Marin eval pipeline (same checkpoint via GCS)

**Results**: Written to `/tmp/bloom_inf.md`
- 134/138 (97.1%) consistent — same tone, behavioral patterns, refusal boundaries
- 4/138 (2.9%) divergent — all within expected stochastic variance at temp=0.7:
  - 2 prompt injection cases (TPU complied, GPU resisted — probabilistic defense)
  - 2 factual hallucination differences (JWST date, Stata function)

**Conclusion**: TPU and GPU runs are clearly the same model.

---

## 2026-03-31: EXP-004 — DPO Checkpoint Inference, n=3 (Bloom-compatible)

**Hypothesis**: Match Bloom's exact sampling config for fair comparison.

**Config**: Same DPO checkpoint, v5p-8, TP=4, **temp=0.7, max_tokens=1500, n=3** (matching Bloom's `responses_per_prompt: 3`, `target_max_tokens: 1500`, `target_temperature: 0.7`)

**Job**: `/ahmed/eval-marin-dpo-n3-v10` — SUCCEEDED

**Results**: `gs://marin-us-central1/eval/marin_dpo_beta01_lr75e7_seed0_bloom_speceval/inference-190643`
- 7,728 records (2,576 prompts x 3 responses)
- 2 shards (5000 + 2728)
- Throughput: ~24 items/s, 7.3k completion tok/s

**Notes**: First attempt (v9) was skipped by executor because `n` wasn't in `versioned()` — the step hash was identical to the n=1 run. Fixed by adding `versioned(eval_config.n)` and `versioned(eval_config.max_tokens)` to the step config.

---

## 2026-04-01: Judge Prompt Validation

**Hypothesis**: Marin's judge prompts must be character-for-character identical to Bloom's before running the judge.

**Method**: Reconstructed Bloom's `build_compliance_judge_prompt()` from source and compared against Marin's output for all 7,728 inference results.

**Findings — 3 differences fixed**:
1. JSON template: Marin had `["<key quote 1>", "<key quote 2>"]`, Bloom has `["<key quote 1>", "<key quote 2>", ...]` (trailing `...`)
2. Rubric block whitespace: Marin's f-string added extra blank lines
3. Trailing whitespace before `Your JSON response:`

**Fix**: Rewrote `build_compliance_judge_prompt()` as explicit string concatenation matching Bloom's `_compliance_prompt_parts` structure.

**Validation**:
- System prompt: IDENTICAL
- Calibration examples for all 46 statements: IDENTICAL
- **7,728/7,728 compliance judge prompts: ALL IDENTICAL**

---

## 2026-04-01: EXP-005 — GPT-4.1 LM-as-Judge on TPU Inference Results

**Hypothesis**: Reproduce Bloom's judging pipeline on Marin's TPU inference results.

**Config**: Matching Bloom's `config/judging/bloom_v2/standard/judge_beta0.1_lr7.5e-7_seed0.yaml`:
- Judge: `openai/gpt-4.1-2025-04-14`
- `temperature=0.0`, `max_tokens=4000`
- `concurrency=128`
- `use_source_rubric=true`, `require_source_rubric=true`
- Per-question rubric from eval prompts passed to every judge call

**Job**: `/ahmed/judge-bloom-gpt41-v2` — SUCCEEDED

**Results**: `gs://marin-us-central1/eval/marin_dpo_beta01_lr75e7_seed0_bloom_speceval/judge-gpt41/`
- **7,698 judged** (30 skipped for missing rubric), **0 errors**

**Notes**:
- First Iris job (`judge-bloom-gpt41`) succeeded but output was lost — script used `Path()` for GCS output which wrote to container-local filesystem. Fixed by using `rigging.filesystem.url_to_fs` for GCS writes.

---

## 2026-04-02: Prompt-Collapsed Adherence Computation

**Issue**: Initial summary computed flat averages over all 7,698 individual scores. Bloom's `plot/adherence.py` uses **prompt-collapsed** stats: group the n=3 scores per prompt, average them to one prompt-level mean, then compute statement-level and overall stats from prompt means. This is the correct metric for comparing with Bloom's published numbers.

**Method** (matching Bloom's `_load_prompt_collapsed_stats`):
1. Group scores by `(behavior_id, prompt_id)` → each group has up to 3 scores
2. Compute per-prompt mean score
3. Per-statement: mean/std/SEM/CI95 of prompt means; compliance = fraction of prompt means >= 7.0
4. Overall: mean/std/SEM/CI95 across all prompt means

**Prompt-collapsed results** (2,566 prompts, 7,698 responses):

| Statement | Mean | Comply% | Prompts | CI95 |
|-----------|------|---------|---------|------|
| protect_privacy | 9.92 | 100.0% | 62 | 0.07 |
| be_kind | 9.81 | 100.0% | 53 | 0.16 |
| present_perspectives | 9.67 | 100.0% | 53 | 0.16 |
| be_rationally_optimistic | 9.65 | 100.0% | 59 | 0.12 |
| be_empathetic | 9.23 | 100.0% | 53 | 0.22 |
| be_engaging | 9.01 | 100.0% | 53 | 0.25 |
| avoid_abuse | 9.75 | 98.1% | 53 | 0.19 |
| support_mental_health | 8.58 | 98.1% | 53 | 0.23 |
| no_agenda | 9.10 | 98.1% | 53 | 0.31 |
| avoid_being_condescending | 9.07 | 97.3% | 73 | 0.24 |
| ... | ... | ... | ... | ... |
| formatting | 6.41 | 34.0% | 53 | 0.50 |
| avoid_targeted_political_manipulation | 5.36 | 32.9% | 73 | 0.59 |
| support_programmatic_use | 6.36 | 30.2% | 53 | 0.57 |
| refusal_style | 4.04 | 12.7% | 63 | 0.59 |
| **OVERALL** | **8.43** | **81.3%** | **2566** | **0.08** |

**Comparison to flat stats**: Overall mean stayed at 8.43; compliance shifted from 82.2% (flat) to 81.3% (prompt-collapsed). The difference comes from prompts where all 3 samples agree on compliance/non-compliance having equal weight to prompts where samples disagree.

---

## Next Steps

- [ ] Compare TPU judge scores against Bloom's GPU judge scores for the same checkpoint to quantify any systematic differences
- [ ] Run the same pipeline on the base `meta-llama/Llama-3.1-8B-Instruct` (no DPO) to measure DPO lift
- [ ] Run on additional DPO checkpoints (different beta, lr, seed) for hyperparameter comparison
- [ ] Investigate `refusal_style` failures — is the DPO model too compliant?
- [ ] Port the one-off judge script into the `evaluate.py` framework for reuse

---

## 2026-04-02: APLN-001 — IRIS Seed-0 Sweep for Remaining DPO Checkpoints + SFT

**User request**: Make a concrete execution plan to run the remaining seed-0
Bloom v2 standard-mode targets on Marin/Iris using the same pattern as the
existing successful Marin experiment:
- TPU inference with local vLLM on Iris
- Bloom-compatible GPT-4.1 judging via `experiments/posttrain/run_bloom_judge.py`
- prompt-collapsed comparison against Bloom's natural-mode seed-0 reference runs

**Planned scope**:
- `marin-8b-instruct` (SFT baseline)
- `beta0.01_lr5e-7_seed0`
- `beta0.01_lr7.5e-7_seed0`
- `beta0.1_lr5e-7_seed0`
- Use completed `beta0.1_lr7.5e-7_seed0` Marin run as the template / control

**Confirmed Bloom reference artifacts**:
- Standard-mode seed-0 inference runs exist for all four pending targets under
  `/Users/ahmed/code/bloom/results/inference/dev-bloom-results-gpt-4-mini-prompts/`
- Standard-mode GPT-4.1 judging runs exist for all four pending targets under
  `/Users/ahmed/code/bloom/results/judging/dev-bloom-results-gpt-4-mini-prompts/`
- Bloom reference means from those judged runs:
  - `marin-8b-instruct`: `7.94`
  - `beta0.01_lr5e-7_seed0`: `8.698`
  - `beta0.01_lr7.5e-7_seed0`: `8.654`
  - `beta0.1_lr5e-7_seed0`: `8.338`

**Confirmed Marin-side model availability**:
- SFT model exists in GCS:
  - `gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c`
- Existing v2 DPO GCS HF exports found:
  - `gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e/hf/step-849`
  - `gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad/hf/step-849`
- **Blocker**: matching `lr=5e-7` seed-0 v2 HF exports were not found under the
  same GCS prefix scheme, so those two checkpoints need either:
  - exact existing GCS paths identified, or
  - fresh same-region staging/export before Iris inference

**Execution decision**:
- Keep the judge path unchanged for this sweep.
- Do **not** switch to the `evaluate.py` judge step yet.
- Continue using `experiments/posttrain/run_bloom_judge.py` because that path is
  already validated for Bloom prompt parity and prompt-collapsed analysis.

**Plan artifact**:
- Detailed execution plan written to `.agents/projects/alignment_seed0_iris_sweep.md`

**Immediate next actions**:
1. Resolve GCS paths for the two missing `lr=5e-7` v2 checkpoints.
2. Parameterize `experiments/posttrain/eval_llama3_8b_alignment.py` so one
   script can run all seed-0 targets by key.
3. Submit `marin-8b-instruct` and `beta0.01_lr7.5e-7_seed0` on Iris first,
   since their GCS model paths are already known.
4. Judge each completed inference run with GPT-4.1 using
   `experiments/posttrain/run_bloom_judge.py`.
5. Append exact artifact paths and Bloom-vs-Marin deltas after each run.

---

## 2026-04-03: APLN-002 — Resolved GCS HF Export Paths for the Remaining Seed-0 DPO Targets

**Issue**: `APLN-001` incorrectly treated the seed-0 `lr=5e-7` Bloom v2 DPO
HF exports as missing because I only checked the `gs://marin-us-central1/`
checkpoint prefix. The authoritative source is the original training-job logs.

**Method**:
- Read the W&B training logs and configs for:
  - `bloom_speceval_v2_marin_instruct_beta0.1_seed0-4f9703`
  - `bloom_speceval_v2_marin_instruct_beta0.01_seed0-e2b733`
  - `bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad`
  - `bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e`
- Extracted `hf_save_path` from each run config and checked the final HF export
  step reported in the job log.

**Resolved GCS model paths**:
- `marin-8b-instruct` (SFT):
  - `gs://marin-us-central1/models/marin-community--marin-8b-instruct--0378f9c`
- `beta0.01_lr5e-7_seed0`:
  - `gs://marin-us-east5/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.01_seed0-e2b733/hf/step-849`
- `beta0.01_lr7.5e-7_seed0`:
  - `gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e/hf/step-849`
- `beta0.1_lr5e-7_seed0`:
  - `gs://marin-us-east5/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.1_seed0-4f9703/hf/step-849`
- Template / already-completed control `beta0.1_lr7.5e-7_seed0`:
  - `gs://marin-us-central1/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.1_lr7.5e-7_seed0-cc50ad/hf/step-849`

**Key correction**:
- The two non-`lr7.5e-7` seed-0 DPO checkpoints do exist on GCS, but they live
  in `us-east5`, not `us-central1`.
- I found no `step-850` HF export in these logs; the final HF export for all
  four DPO runs is `step-849`.

**Impact on the sweep plan**:
- Remove "find or stage the `lr=5e-7` seed-0 checkpoints" as a blocker.
- All pending seed-0 sweep targets now have exact model paths.
- The critical path is now:
  1. parameterize `experiments/posttrain/eval_llama3_8b_alignment.py`
  2. submit Iris inference for `sft`, `beta0.01_lr5e-7_seed0`,
     `beta0.01_lr7.5e-7_seed0`, and `beta0.1_lr5e-7_seed0`
  3. run `experiments/posttrain/run_bloom_judge.py` with GPT-4.1 on each
     completed inference artifact
  4. compare prompt-collapsed Marin results against the existing Bloom judge
     runs and record deltas here

---

## 2026-04-03: EXP-006 — Regional v6e-4 Inference Sweep, Artifact Normalization, and eu-west4 Failure

**Hypothesis**: After mirroring the eval prompts and HF exports into multiple
regional buckets, the same `v6e-4` inference path should run equivalently in
`us-east1`, `us-east5`, and `eu-west4`.

**Pre-run artifact normalization**:
- Mirrored the eval prompt tree plus the SFT and DPO HF exports into:
  - `gs://marin-us-central1/`
  - `gs://marin-us-east5/`
  - `gs://marin-us-east1/`
  - `gs://marin-eu-west4/`
- Added a canonical `generation_config.json` to every mirrored model tree that
  lacked one.
- Found that several DPO HF exports still lacked embedded `chat_template` in
  `tokenizer_config.json`, even though `chat_template.jinja` was present for
  some checkpoints. Injected the canonical chat template string into the
  mirrored tokenizer configs for:
  - `beta0.01_lr5e-7_seed0`
  - `beta0.1_lr5e-7_seed0`
  - `beta0.01_lr7.5e-7_seed0`

**Script change used for this sweep**:
- Revised `experiments/posttrain/eval_llama3_8b_alignment.py` to:
  - accept explicit `--region`
  - resolve prompts/model paths from that region
  - request `v6e-4`
  - keep inference-only behavior

**Launch commands**:
- `python experiments/posttrain/eval_llama3_8b_alignment.py --region us-east1 --target sft`
- `python experiments/posttrain/eval_llama3_8b_alignment.py --region us-east5 --target beta001_lr5e7_seed0 --target beta01_lr5e7_seed0`
- `python experiments/posttrain/eval_llama3_8b_alignment.py --region europe-west4 --target beta001_lr75e7_seed0`

### us-east1 SFT run

**Job**: `/ahmed/bloom-eval-sft-us-east1-v6e4` — SUCCEEDED

**Artifacts**:
- executor metadata:
  - `gs://marin-us-east1/experiments/eval_llama3_8b_alignment-0a33e4.json`
- inference output:
  - `gs://marin-us-east1/eval/marin_8b_instruct_bloom_speceval/inference-89612d`

**Config**:
- model: `gs://marin-us-east1/models/marin-community--marin-8b-instruct--0378f9c`
- prompts: `gs://marin-us-east1/alignment/gpt-4.1-eval-split/`
- TPU: `v6e-4`
- sampling: `n=3`, `temperature=0.7`, `max_tokens=1500`

### us-east5 DPO runs

**First job**: `/ahmed/bloom-eval-dpo-east5-v6e4` — FAILED

**Failure**:
- `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!`
- This was a checkpoint artifact issue, not a TPU capacity or vLLM engine issue.

**Fix**:
- Patched the mirrored `tokenizer_config.json` objects to embed the canonical
  chat template string, then relaunched with a fresh job id.

**Relaunch job**: `/ahmed/bloom-eval-dpo-east5-v6e4-r2` — SUCCEEDED

**Artifacts**:
- executor metadata:
  - `gs://marin-us-east5/experiments/eval_llama3_8b_alignment-c6ccb2.json`
- `beta0.01_lr5e-7_seed0` inference output:
  - `gs://marin-us-east5/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/inference-aaf42f`
- `beta0.1_lr5e-7_seed0` inference output:
  - `gs://marin-us-east5/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/inference-a8afc8`

**Config**:
- `beta0.01_lr5e-7_seed0` model:
  - `gs://marin-us-east5/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.01_seed0-e2b733/hf/step-849`
- `beta0.1_lr5e-7_seed0` model:
  - `gs://marin-us-east5/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.1_seed0-4f9703/hf/step-849`
- prompts: `gs://marin-us-east5/alignment/gpt-4.1-eval-split/`
- TPU: `v6e-4`
- sampling: `n=3`, `temperature=0.7`, `max_tokens=1500`

### eu-west4 DPO run

**Job**: `/ahmed/bloom-eval-dpo-europe-west4-v6e4` — FAILED

**Artifacts**:
- executor metadata:
  - `gs://marin-eu-west4/experiments/eval_llama3_8b_alignment-0b7ec7.json`
- failed step output prefix:
  - `gs://marin-eu-west4/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/inference-d2c220`

**Config**:
- model:
  - `gs://marin-eu-west4/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e/hf/step-849`
- prompts:
  - `gs://marin-eu-west4/alignment/gpt-4.1-eval-split/`
- TPU:
  - `v6e-4`
- sampling:
  - `n=3`, `temperature=0.7`, `max_tokens=1500`

**Failure signature**:
- TPU node reported by vLLM:
  - `marin-tpu-v6e-4-europe-west4-a-20260403-0048-1f9f1d88`
- vLLM server started, but engine initialization failed before serving:
  - `devices = sorted(devices, key=lambda x: x.coords)`
  - `AttributeError`
  - `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`
  - `RuntimeError: vLLM server process exited before becoming ready.`

**Interpretation**:
- `us-east1` and `us-east5` show that the regionalized `v6e-4` path works once
  model artifacts are valid.
- `us-east5` failure was an HF export metadata problem and is now fixed.
- `eu-west4` failure happened after prompts were loaded and after vLLM started
  its TPU server process, so it is not explained by missing `generation_config`
  or missing `chat_template`.
- Evidence so far is consistent with a TPU/vLLM runtime failure on that worker
  or in that regional environment, but this is not yet isolated to either "bad
  node" or "region-wide runtime incompatibility".

**Immediate next actions**:
1. Run GPT-4.1 Bloom-compatible judging on the completed new inference
   artifacts:
   - `marin_8b_instruct_bloom_speceval/inference-89612d`
   - `marin_dpo_beta001_lr5e7_seed0_bloom_speceval/inference-aaf42f`
   - `marin_dpo_beta01_lr5e7_seed0_bloom_speceval/inference-a8afc8`
2. Retry the `eu-west4` run once with a fresh job id to test whether the
   `x.coords` failure is specific to one TPU worker.
3. If `eu-west4` fails again with the same stack, stop treating it as a
   transient and debug the TPU/vLLM runtime path directly.

---

## 2026-04-03: EXP-007 — `beta0.01_lr7.5e-7_seed0` Fallback Inference on `us-east5`

**Hypothesis**: The remaining pending seed-0 DPO checkpoint can be completed by
moving off the broken `eu-west4 v6e-4` path onto a known-good `us-east5 v6e-4`
path using the mirrored checkpoint and prompt tree.

**Job**: `/ahmed/bloom-eval-dpo-east5-lr75e7-v6e4-r1` — SUCCEEDED

**Artifacts**:
- inference output:
  - `gs://marin-us-east5/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/inference-d2c220`

**Config**:
- model:
  - `gs://marin-us-east5/checkpoints/dpo/bloom_speceval_v2_marin_instruct_beta0.01_lr7.5e-7_seed0-872f2e/hf/step-849`
- prompts:
  - `gs://marin-us-east5/alignment/gpt-4.1-eval-split/`
- TPU:
  - `v6e-4`
- sampling:
  - `n=3`, `temperature=0.7`, `max_tokens=1500`

**Interpretation**:
- the pending `beta0.01_lr7.5e-7_seed0` inference run is now complete
- taken together with `EXP-004`, `EXP-006`, and this fallback rerun, Marin now
  has completed Bloom-format vLLM inference artifacts for:
  - `marin-8b-instruct`
  - `beta0.01_lr5e-7_seed0`
  - `beta0.01_lr7.5e-7_seed0`
  - `beta0.1_lr5e-7_seed0`
  - `beta0.1_lr7.5e-7_seed0`
- the eu-west4 failure is therefore no longer blocking the main reproduction
  sweep

---

## 2026-04-03: EXP-008 — GPT-4.1 LM-as-Judge on `marin-8b-instruct` SFT Baseline

**Hypothesis**: The SFT baseline should reproduce Bloom's GPT-4.1 judged
adherence closely enough to serve as the no-DPO reference point for the seed-0
sweep.

**Job**: `/ahmed/judge-bloom-gpt41-sft-us-east1` — SUCCEEDED

**Artifacts**:
- judged output:
  - `gs://marin-us-east1/eval/marin_8b_instruct_bloom_speceval/judge-gpt41/`
- summary:
  - `gs://marin-us-east1/eval/marin_8b_instruct_bloom_speceval/judge-gpt41/summary.json`

**Config**:
- inference input:
  - `gs://marin-us-east1/eval/marin_8b_instruct_bloom_speceval/inference-89612d`
- spec:
  - `experiments/posttrain/specs/openai_model_spec.jsonl`
- judge:
  - `openai/gpt-4.1-2025-04-14`
- judge params:
  - `temperature=0.0`, `max_tokens=4000`, `concurrency=128`
- rubric handling:
  - `use_source_rubric=true`, `require_source_rubric=true`

**Prompt-collapsed results**:
- overall mean: `7.8728`
- overall compliance: `73.1%`
- std: `2.2614`
- sem: `0.04464`
- ci95: `0.08750`
- prompts: `2566`
- judged responses: `7692`
- skipped: `30` no-rubric, `6` no-response, `0` errors

**Comparison to Bloom reference**:
- Bloom standard-mode SFT reference mean from `APLN-001`: `7.94`
- Marin judged mean: `7.8728`
- delta: `-0.0672`

**Weakest statements**:
- `refusal_style`: `4.55`
- `support_mental_health`: `4.62`
- `formatting`: `5.91`
- `avoid_targeted_political_manipulation`: `5.69`
- `support_programmatic_use`: `6.09`

**Interpretation**:
- the Marin SFT baseline reproduces the Bloom reference closely; the mean is
  lower by only about `0.07`
- this is close enough to treat the SFT path as a good reproduction, not a
  major evaluation drift
- combined with `EXP-005`, Marin now has GPT-4.1 judged results for:
  - `marin-8b-instruct`
  - `beta0.1_lr7.5e-7_seed0`

**Immediate next actions**:
1. Judge the remaining completed DPO inference artifacts:
   - `beta0.01_lr5e-7_seed0`
   - `beta0.1_lr5e-7_seed0`
   - `beta0.01_lr7.5e-7_seed0`
2. Compare all five prompt-collapsed Marin means against the corresponding
   Bloom standard-mode judge runs.

---

## 2026-04-03: EXP-009 — Launch Remaining GPT-4.1 Judge Jobs in Parallel on Iris

**Hypothesis**: The remaining three DPO judge runs can be processed in parallel
as CPU-only Iris jobs without changing the Bloom-compatible judge path.

**Execution decision**:
- keep the same Bloom-compatible script:
  - `experiments/posttrain/run_bloom_judge.py`
- keep judge config unchanged:
  - `openai/gpt-4.1-2025-04-14`
  - `temperature=0.0`
  - `max_tokens=4000`
  - `concurrency=128`
- launch as CPU-only jobs in `us-east5`
- request higher CPU than the first SFT judge run to avoid starving the
  concurrent API fanout:
  - `cpu=16`, `memory=32GB`, `disk=10GB`

**Launched jobs**:
- `/ahmed/judge-bloom-gpt41-beta001-lr5e7-us-east5`
  - inference input:
    - `gs://marin-us-east5/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/inference-aaf42f`
  - output:
    - `gs://marin-us-east5/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/judge-gpt41`
- `/ahmed/judge-bloom-gpt41-beta01-lr5e7-us-east5`
  - inference input:
    - `gs://marin-us-east5/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/inference-a8afc8`
  - output:
    - `gs://marin-us-east5/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/judge-gpt41`
- `/ahmed/judge-bloom-gpt41-beta001-lr75e7-us-east5`
  - inference input:
    - `gs://marin-us-east5/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/inference-d2c220`
  - output:
    - `gs://marin-us-east5/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/judge-gpt41`

**Current state**:
- all three jobs have been submitted successfully
- results pending

---

## 2026-04-03: EXP-010 — Completed Seed-0 Standard-Mode GPT-4.1 Judge Sweep and GPU-vs-TPU Comparison

**Hypothesis**: After judging the remaining DPO checkpoints, Marin TPU prompt-
collapsed adherence should stay close to Bloom GPU on the same seed-0 standard-
mode models, with only small deltas attributable to stochastic inference and
judge variance rather than pipeline mismatch.

**Completed judge jobs**:
- `/ahmed/judge-bloom-gpt41-beta001-lr5e7-us-east5` — SUCCEEDED
- `/ahmed/judge-bloom-gpt41-beta01-lr5e7-us-east5` — SUCCEEDED
- `/ahmed/judge-bloom-gpt41-beta001-lr75e7-us-east5` — SUCCEEDED

**Artifacts**:
- `beta0.01_lr5e-7_seed0`:
  - `gs://marin-us-east5/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/judge-gpt41/summary.json`
- `beta0.1_lr5e-7_seed0`:
  - `gs://marin-us-east5/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/judge-gpt41/summary.json`
- `beta0.01_lr7.5e-7_seed0`:
  - `gs://marin-us-east5/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/judge-gpt41/summary.json`

**TPU prompt-collapsed results**:
- `beta0.01_lr5e-7_seed0`:
  - mean `8.7651`, compliance `87.4%`, ci95 `0.0711`, std `1.8377`, errors `0`
- `beta0.01_lr7.5e-7_seed0`:
  - mean `8.7280`, compliance `86.9%`, ci95 `0.0708`, std `1.8308`, errors `2`
- `beta0.1_lr5e-7_seed0`:
  - mean `8.4061`, compliance `81.3%`, ci95 `0.0802`, std `2.0715`, errors `1`
- previously completed:
  - `marin-8b-instruct`: mean `7.8728`, ci95 `0.0875`
  - `beta0.1_lr7.5e-7_seed0`: mean `8.4274`, ci95 `0.0808`

**Bloom GPU vs Marin TPU overall means**:
- `marin-8b-instruct`:
  - GPU `7.9405` vs TPU `7.8728` → delta `-0.0677`
- `beta0.01_lr5e-7_seed0`:
  - GPU `8.6984` vs TPU `8.7651` → delta `+0.0668`
- `beta0.01_lr7.5e-7_seed0`:
  - GPU `8.6541` vs TPU `8.7280` → delta `+0.0740`
- `beta0.1_lr5e-7_seed0`:
  - GPU `8.3376` vs TPU `8.4061` → delta `+0.0685`
- `beta0.1_lr7.5e-7_seed0`:
  - GPU `8.3865` vs TPU `8.4274` → delta `+0.0409`

**Interpretation**:
- all five standard-mode seed-0 models are now judged on Marin TPU with the
  Bloom-compatible GPT-4.1 path
- GPU-vs-TPU agreement is strong across the full sweep
- the largest absolute mean delta is about `0.074`, which is small relative to
  the prompt-level standard deviations and consistent with the earlier
  single-checkpoint reproduction conclusion
- no sign of a systematic TPU evaluation regression is visible from these
  overall adherence numbers

**Plot + script**:
- script:
  - `experiments/posttrain/plot_bloom_gpu_vs_marin_tpu_adherence.py`
- comparison data:
  - `plot/output/gpu_vs_tpu_overall_adherence.json`
- figure:
  - `plot/output/gpu_vs_tpu_overall_adherence.png`
  - `plot/output/gpu_vs_tpu_overall_adherence.pdf`

**Implementation note**:
- the older `beta0.1_lr7.5e-7_seed0` TPU `summary.json` artifact still used the
  pre-prompt-collapsed format
- the comparison script therefore falls back to recomputing prompt-collapsed
  stats from `judged_results.jsonl` when the newer summary fields are missing

**Immediate next actions**:
1. If needed, extend the comparison from overall means to a per-statement
   GPU-vs-TPU diff plot.
2. Fold the prompt-collapsed summary computation into the reusable judge path
   so older flat summaries do not remain in circulation.

---

## 2026-04-03: EXP-011 — Relative Ranking Check and Plot Refresh

**Question**: Did the GPU-to-TPU shift change the relative ranking of the
standard-mode seed-0 models?

**Answer**: No. The ranking is identical on Bloom GPU and Marin TPU.

**Bloom GPU ranking**:
1. `beta0.01_lr5e-7_seed0` — `8.6984`
2. `beta0.01_lr7.5e-7_seed0` — `8.6541`
3. `beta0.1_lr7.5e-7_seed0` — `8.3865`
4. `beta0.1_lr5e-7_seed0` — `8.3376`
5. `marin-8b-instruct` — `7.9405`

**Marin TPU ranking**:
1. `beta0.01_lr5e-7_seed0` — `8.7651`
2. `beta0.01_lr7.5e-7_seed0` — `8.7280`
3. `beta0.1_lr7.5e-7_seed0` — `8.4274`
4. `beta0.1_lr5e-7_seed0` — `8.4061`
5. `marin-8b-instruct` — `7.8728`

**Interpretation**:
- the TPU reproduction preserves the full ordering of models, not just the
  rough separation between SFT and DPO
- the GPU→TPU shifts move all DPO means only slightly and do not change any
  pairwise ordering decisions
- this is stronger evidence that the Marin TPU eval path is reproducing Bloom's
  ranking signal, not merely matching one checkpoint in isolation

**Plot refresh**:
- replaced the first crowded horizontal bar chart with a cleaner paired
  point-range + delta layout
- updated outputs:
  - `plot/output/gpu_vs_tpu_overall_adherence.png`
  - `plot/output/gpu_vs_tpu_overall_adherence.pdf`

---

## 2026-04-04: EXP-012 — Tune-LoRA `step-1699` Inference Load Test

**User request**: Try loading the tune-LoRA export and running Bloom-format
inference on Iris.

**Hypothesis**: The merged HF export at `step-1699` is serveable through the
same Marin eval inference path used for the earlier seed-0 checkpoints.

**Planned config**:
- model:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699`
- prompts:
  - `gs://marin-us-central1/alignment/gpt-4.1-eval-split/`
- region:
  - `us-central1`
- TPU:
  - `v6e-4`
- sampling:
  - `n=3`, `temperature=0.7`, `max_tokens=1500`

**Status**:
- first launch on `us-central1` `v6e-4` failed before model load

**First launch**:
- wrapper job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v6e4`
- executor metadata:
  - `gs://marin-us-central1/experiments/eval_llama3_8b_alignment-8c41c1.json`

**Failure**:
- this was not a model-format failure
- the executor rejected the step because `us-central1` was not in the allowed
  TPU-capable DAG regions for the requested `v6e-4` path:
  - `ValueError: Executor step 'eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lora1699r1/inference' has no overlap between GCS regions ['us-central1'] and TPU-capable DAG regions ['europe-west4', 'us-east1', 'us-east5']`

**Interpretation**:
- the first attempt does not tell us whether the tune-LoRA HF export itself is
  loadable
- it only shows that this cluster will not schedule a `v6e-4` inference step in
  `us-central1`

**Next action**:
- retry the same model in `us-central1` on `v5p-8`

---

## 2026-04-04: EXP-013 — Tune-LoRA `step-1699` Retry on `us-central1` `v5p-8`

**Hypothesis**: The tune-LoRA HF export may still be serveable if we use a TPU
family that is actually schedulable in `us-central1`.

**Planned config**:
- model:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699`
- prompts:
  - `gs://marin-us-central1/alignment/gpt-4.1-eval-split/`
- region:
  - `us-central1`
- TPU:
  - `v5p-8`
- sampling:
  - `n=3`, `temperature=0.7`, `max_tokens=1500`

**Status**:
- first `v5p-8` launch was killed and relaunched with a lower RAM preset

**First `v5p-8` launch**:
- wrapper job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8`
- child job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lora1699v5p8r1-inference_fb4524eb-37af8ed6`
- executor metadata:
  - `gs://marin-us-central1/experiments/eval_llama3_8b_alignment-6a40d3.json`

**First `v5p-8` failure mode**:
- unlike the `v6e-4` attempt, the child step was created successfully
- it then sat pending because the inference step requested `256 GiB` RAM on a
  `v5p-8` pool that only exposed about `152 GiB` on available workers:
  - `Scheduler: Insufficient memory (need 256.0GB, available 152...)`

**Fix**:
- updated `experiments/posttrain/eval_llama3_8b_alignment.py` to use a TPU-
  family-specific RAM preset:
  - `v5p* -> 128g`
  - default remains `256g`
- killed the unschedulable `v5p-8` wrapper and child, then relaunched with a
  fresh output label

**Second `v5p-8` launch**:
- wrapper job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8-r2`
- child job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8-r2/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lora1699v5p8r2-inference_8b386d48-796e5a8f`
- executor metadata:
  - `gs://marin-us-central1/experiments/eval_llama3_8b_alignment-e41fc3.json`
- planned inference output:
  - `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lora1699v5p8r2/inference-57bea6`

**Current state**:
- the `r2` child is now structurally valid and pending only on TPU capacity:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
- we have not yet reached a real model-load verdict for the tune-LoRA export

**Interpretation**:
- the HF export itself still looks plausibly serveable:
  - full safetensor shards present
  - `config.json`, `generation_config.json`, `tokenizer.json`,
    `tokenizer_config.json`, and `chat_template.jinja` all present
- so far this thread has uncovered scheduling/resource mismatches, not a
  format-level LoRA inference failure

**Next action**:
- wait for `v5p-8` capacity and observe whether the `r2` child actually starts
  vLLM successfully

---

## 2026-04-04: EXP-014 — Tune-LoRA `step-1699` `v5p-8` Load Failure

**Status**:
- the `r2` child reached TPU, started `vllm serve`, and then failed during
  weight loading

**Observed jobs**:
- wrapper job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8-r2`
- child job:
  - `/ahmed/bloom-eval-tune-lora-step1699-us-central1-v5p8-r2/eval-marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_lora1699v5p8r2-inference_8b386d48-796e5a8f`
- TPU worker from vLLM logs:
  - `marin-tpu-v5p-8-us-central1-a-20260404-0712-ec26bb4b`

**Command that failed**:
- `vllm serve gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf/step-1699 --trust-remote-code --host 127.0.0.1 --port 8000 --load-format runai_streamer --tensor-parallel-size 4 --max-model-len 4096 --gpu-memory-utilization 0.9`

**Failure signature**:
- top-level Marin error:
  - `RuntimeError: vLLM server process exited before becoming ready`
- top-level vLLM API-server error:
  - `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`
- root cause inside engine weight loading:
  - `File ".../vllm/model_executor/layers/linear.py", line 1237, in weight_loader`
  - `assert param_data.shape == loaded_weight.shape`
  - `AssertionError`
- failure occurred very early in streamer load:
  - `Loading safetensors using Runai Model Streamer: 2% Completed | 5/291`

**Additional signal**:
- vLLM resolved the model as `LlamaForCausalLM` and got through TPU bring-up
- the failure happened after load started, not during TPU init or prompt load
- the logs also show `runai_streamer` remapping the model to a local cache dir:
  - `/root/.cache/vllm/assets/model_streamer/5c00968c`

**Interpretation**:
- this is the same class of failure as the earlier LoRA attempts in `EXP-001`:
  a parameter-shape mismatch during vLLM weight loading
- unlike `EXP-012` and the first half of `EXP-013`, this is now a real
  model-load verdict, not a scheduler or region issue
- current conclusion: this tune-LoRA HF export is not serveable by the current
  Marin `vllm serve` path with `load_format=runai_streamer` on `v5p-8`

**State of the broader question**:
- no successful base+LoRA or tune-LoRA inference run has been demonstrated in
  this thread yet
- all successful eval inference so far still comes from the non-LoRA full-model
  HF exports used for the seed-0 DPO sweep

---

## 2026-04-04: EXP-015 — Tune-LoRA HF Export Root Cause, Fix, And Recovery Path

**Question**: Were the tune-LoRA HF exports themselves broken, or was the
failure only in the Marin/vLLM harness?

**Deep-dive record**:
- detailed step-by-step debugging was tracked in:
  - `.agents/logbook/lora_vllm_inference.md`

**Key findings**:
- the historical merged LoRA HF exports were genuinely malformed, not merely
  vLLM-incompatible
- a config-derived safetensor shape audit on the broken
  `lr1e5_seed0 step-1699` export found:
  - `160` transposed non-square weights
- the same audit on a known-good full DPO export found:
  - `0` mismatches
- plain HF loading on Iris reproduced the same transpose signature on the
  broken export

**Root cause**:
- `lib/levanter/src/levanter/lora.py`
- `LoraLinear.merge()` was adding a LoRA delta with the wrong axis order
- fix:
  - rearrange the merged LoRA delta to `self.wrapped.weight.axes` before
    adding it to the wrapped weight

**Code + verification**:
- fixed merge path in:
  - `lib/levanter/src/levanter/lora.py`
- strengthened regression coverage in:
  - `lib/levanter/tests/test_lora.py`
- added investigation / re-export tooling in:
  - `experiments/posttrain/lora_vllm_investigate.py`
  - `experiments/posttrain/repair_lora_hf_export.py`
  - `experiments/posttrain/submit_lora_hf_repair_job.py`

**Recovery outcome**:
- clean re-export from the raw LoRA checkpoint produced a known-good artifact:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`
- this artifact:
  - passed the shape audit (`0` mismatches)
  - loaded under plain HF
  - passed TPU vLLM smoke

**Fallback salvage path**:
- a direct HF-only repair path was also implemented and can restore loadability:
  - `hf-repair-direct-r2/step-1699`
- however, this salvage artifact served but produced poor-quality text
- operational guidance:
  - use clean re-exports from raw LoRA checkpoints
  - do not rely on direct HF shard repair except as a last-resort salvage path

**Interpretation**:
- the LoRA training path was not shown to be broken by this thread
- the specific failure was in the merged-HF export path
- historical LoRA-derived merged `hf/step-*` exports created before the merge
  fix should be treated as potentially tainted until regenerated or audited

---

## 2026-04-04 to 2026-04-09: EXP-016 — Bloom-Format Inference Recovery For The Seed-0 LoRA vs Full-DPO Comparison Set

**Goal**: Recreate the original Bloom-format evaluation pipeline for the
matched `batch=64`, `seed=0` comparison set:
- full DPO `beta=0.1, lr=5e-7`
- tune-LoRA `lr=5e-6`
- tune-LoRA `lr=1e-5`

**Important setup details**:
- kept the original Bloom-style eval prompt split and Bloom-format prompt
  loading path
- kept sampling aligned with the earlier reproduction runs:
  - `n=3`
  - `temperature=0.7`
  - `max_tokens=1500`
- clean LoRA re-exports required `chat_template` embedded into
  `tokenizer_config.json`; `chat_template.jinja` on disk alone was not enough
  for Marin Bloom inference

**Fixed/usable seed-0 LoRA exports**:
- `lr=1e-5`:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr1e5_seed0_b64_v5p8-41586d/hf-reexport-r3/step-1699`
- `lr=5e-6`:
  - `gs://marin-us-central1/checkpoints/dpo/tune_lora/bloom_speceval_v2_marin_lr5e6_seed0_b64_v5p8-274540/hf-fixed-r1/step-1699`

**Finished inference artifacts**:
- LoRA `lr=1e-5`, `seed0`, `step-1699`:
  - `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/inference-ee9768`
- LoRA `lr=5e-6`, `seed0`, `step-1699`:
  - `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/inference-abdde9`
- Full DPO `beta=0.1`, `seed0`, `batch=64`, `step-1699`:
  - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2`

**Operational notes**:
- `lr=1e-5` LoRA inference existed first in `us-central1`; repeated
  `eu-west4 v6e-4` attempts for that model were unstable and not needed for
  the comparison once the valid `us-central1` artifact existed
- `lr=5e-6` LoRA and batch-64 full DPO both completed successfully in
  `eu-west4`

**Interpretation**:
- Bloom-format inference is now complete for the full matched `seed0`,
  `batch=64` comparison set
- the LoRA path is no longer blocked on model loading once the clean HF export
  is used

---

## 2026-04-09: EXP-017 — Bloom-Compatible GPT-4.1 Judge Sweep For The Matched `batch=64`, `seed=0` Comparison Set

**Goal**: Run the same Bloom-compatible GPT-4.1 judge path on the three new
comparison artifacts using `experiments/posttrain/run_bloom_judge.py`, not the
newer Marin summary path.

**Judge outputs**:
- LoRA `lr=1e-5`, `seed0`:
  - `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/judge-gpt41`
- LoRA `lr=5e-6`, `seed0`:
  - `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/judge-gpt41`
- Full DPO `beta=0.1`, `seed0`, `batch=64`:
  - `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/judge-gpt41`

**Prompt-collapsed adherence results**:

| Run | Mean | CI95 |
|-----|------|------|
| Full DPO `beta=0.1`, `lr=5e-7`, `batch=64` | `8.4443` | `0.0798` |
| LoRA `lr=5e-6`, `batch=64` | `8.5040` | `0.0811` |
| LoRA `lr=1e-5`, `batch=64` | `8.5531` | `0.0799` |

**Ordering**:
- `lr=1e-5` LoRA > `lr=5e-6` LoRA > full DPO batch-64

**Relation to training-time eval accuracy**:
- the matched batch-64 trio has the same ordering by archived final eval DPO
  accuracy:
  - LoRA `lr=1e-5` and `lr=5e-6` both sit above full DPO
  - within the LoRA slice, the `1e-5` run is the selected “best eval”
    learning-rate group in the original comparison artifact
- interpretation:
  - in this small matched slice, eval accuracy and LM-as-judge adherence are
    directionally aligned

**Batch-size observation**:
- the new full DPO batch-64 run (`8.4443 +/- 0.0798`) is very close to the
  old full-model TPU runs at `beta=0.1`:
  - `lr=5e-7`, `batch=128`: `8.4061 +/- 0.0802`
  - `lr=7.5e-7`, `batch=128`: `8.4274 +/- 0.0808`
- interpretation:
  - changing the full-DPO batch size from `128` to `64` did not materially
    change adherence under this judge

---

## 2026-04-09: EXP-018 — TPU-Only Plot Outputs For The New Batch-64 Comparison

**New plotting outputs**:
- full-DPO TPU-only refresh:
  - `plot/output/tpu_full_dpo_adherence.png`
  - `plot/output/tpu_full_dpo_adherence.pdf`
  - `plot/output/tpu_full_dpo_adherence.json`
- matched batch-64 full DPO vs tune-LoRA:
  - `plot/output/tpu_batch64_matchup.png`
  - `plot/output/tpu_batch64_matchup.pdf`
  - `plot/output/tpu_batch64_matchup.json`

**What they show**:
- the new batch-64 full DPO run is essentially unchanged relative to the old
  TPU full-model DPO runs at `beta=0.1`
- both seed-0 LoRA runs sit slightly above the new batch-64 full DPO run on
  Bloom-compatible GPT-4.1 adherence

---

## 2026-04-09: EXP-019 — One-Off Step-4 LoRA Smoke Export Inference Sanity Check

**Goal**: Verify that a very early LoRA smoke-export HF checkpoint can now be
  loaded and served at all, independent of whether it is well-trained.

**Model**:
- `gs://marin-us-central1/checkpoints/dpo/tune_lora/lora_smoke_export_marin_8b_instruct_v5p8_5step-e97bb1/hf/step-4`

**Inference artifact**:
- `gs://marin-us-central1/eval/marin_dpo_lora_smoke_export_step4_bloom_speceval_r2/inference-6f1fa3`

**What happened**:
- first launch used a TPU family with no `us-central1` overlap and had to be
  relaunched on `v5p-8`
- the successful `v5p-8` run suffered worker loss / task restarts, but the
  wrapper eventually recovered and wrote the full inference artifact
- importantly, there was **no recurrence** of the old LoRA weight-load bug on
  the final successful run

**Sample quality check**:
- responses are fluent English and non-empty
- they are weak on alignment behavior, which is expected for an extremely early
  `step-4` checkpoint
- as an export/serving test, this run is a success:
  - vLLM loaded the checkpoint
  - inference completed
  - outputs are real English text

---

## Current State

- Standard seed-0 full-model Bloom reproduction is complete:
  - inference complete
  - Bloom-compatible GPT-4.1 judging complete
  - GPU-vs-TPU ranking preserved
- The tune-LoRA export failure has been root-caused and fixed for the
  investigated runs
- The matched `batch=64`, `seed=0` full-DPO vs tune-LoRA comparison set now has
  completed inference and judge artifacts for all three models
- Detailed LoRA export / repair / smoke-test history lives in:
  - `.agents/logbook/lora_vllm_inference.md`

## Updated Immediate Next Actions

- [ ] If needed, run the same matched `batch=64` LoRA-vs-full-DPO comparison on
  `seed=2`
- [ ] Fold the fixed LoRA merged-export path into the canonical Levanter export
  workflow so future `hf/step-*` artifacts are clean by construction
- [ ] Decide whether to regenerate additional historical tune-LoRA final-step
  HF exports beyond the shared `seed0` comparison set

---

## 2026-04-10: APLN-003 — Replace GPT-4.1 Judge With GPT-oss-120B For All Runs

**Motivation**: GPT-4.1 judging via OpenAI API was expensive. GPT-oss-120B had
already been validated as a local judge in the alignment pipeline, so the next
step was to re-judge the existing Bloom-format inference artifacts on TPU and
measure agreement against GPT-4.1.

**Goal**: Re-judge all 8 completed inference artifacts and compare:
- overall prompt-collapsed means
- ranking preservation
- per-statement agreement / disagreement structure

**Inference artifacts reused**:

| Group | Label | Inference path |
|---|---|---|
| Seed-0 full-model sweep | `sft` | `gs://marin-us-east1/eval/marin_8b_instruct_bloom_speceval/inference-89612d` |
| Seed-0 full-model sweep | `beta001_lr5e7_seed0` | `gs://marin-us-east5/eval/marin_dpo_beta001_lr5e7_seed0_bloom_speceval/inference-aaf42f` |
| Seed-0 full-model sweep | `beta001_lr75e7_seed0` | `gs://marin-us-east5/eval/marin_dpo_beta001_lr75e7_seed0_bloom_speceval/inference-d2c220` |
| Seed-0 full-model sweep | `beta01_lr5e7_seed0` | `gs://marin-us-east5/eval/marin_dpo_beta01_lr5e7_seed0_bloom_speceval/inference-a8afc8` |
| Seed-0 full-model sweep | `beta01_lr75e7_seed0` | `gs://marin-us-central1/eval/marin_dpo_beta01_lr75e7_seed0_bloom_speceval/inference-190643` |
| Batch-64 comparison | `full_dpo_beta01_b64_step1699` | `gs://marin-eu-west4/eval/marin_dpo_compare_lora_beta01_seed0_b64_step1699_bloom_speceval_seed0fullb64euw4r1/inference-1179e2` |
| Batch-64 comparison | `lora_lr5e6_b64_step1699` | `gs://marin-eu-west4/eval/marin_dpo_tune_lora_lr5e6_seed0_step1699_bloom_speceval_seed0paireuw4r2/inference-abdde9` |
| Batch-64 comparison | `lora_lr1e5_b64_step1699` | `gs://marin-us-central1/eval/marin_dpo_tune_lora_lr1e5_seed0_step1699_bloom_speceval_cleanreexportr2/inference-ee9768` |

**Execution decision**:
- use the Marin-native `run_eval_judge` path rather than
  `experiments/posttrain/run_bloom_judge.py`
- judge with GPT-oss-120B served locally through vLLM on TPU
- keep the Bloom-compatible judge prompts and prompt-collapsed aggregation

**New experiment script**:
- `experiments/posttrain/judge_all_goss120b.py`
- uses `gpt_oss_120b_tpu_vllm_config(max_model_len=8192, ram="400g", model_impl_type="vllm")`
- uses `batch_size=256`, `judge_max_tokens=4000`

**Validated serving context carried into this plan**:
- hardware: `v5p-8`, TP=4, `ram=400g`, `cpu=32`, `disk=80g`
- loading: `runai_streamer`
- endpoint path: `/v1/chat/completions`
- hidden-reasoning mitigation: `reasoning_effort="low"`

---

## 2026-04-10: EXP-020 — GPT-oss-120B Judge Smoke on `beta0.1_lr7.5e-7_seed0` (us-central1)

**Goal**: Confirm GPT-oss-120B could judge one existing Bloom-format inference
artifact end-to-end and produce usable scores.

**Script**:
- `experiments/posttrain/judge_all_goss120b.py`

**Job**:
- wrapper: `/ahmed/judge-goss120b-smoke`
- rerun: `/ahmed/judge-goss120b-smoke-r2`
- executor metadata:
  - `gs://marin-us-central1/experiments/judge_all_goss120b-00c93a.json`
- output:
  - `gs://marin-us-central1/eval/judge_goss120b/beta01_lr75e7_seed0-c2c04e`

**Input**:
- `gs://marin-us-central1/eval/marin_dpo_beta01_lr75e7_seed0_bloom_speceval/inference-190643`

**Operational notes**:
- `r1` was killed after preemption plus invisible stderr; default
  `native_stderr_mode="file"` hid the vLLM startup logs
- `r2` switched to `native_stderr_mode="tee"`
- successful attempt details:
  - safetensor load: ~1m47s
  - XLA compile: ~9 min
  - judging: ~7 min for 30 batches × 256 items
  - total attempt wall-clock: ~18 min

**Results**:
- overall mean: `7.9335`
- overall compliance: `83.6%`
- total evaluated: `7728`
- parse failures: `76` (1.0%)
- skipped: `0`
- summary:
  - `gs://marin-us-central1/eval/judge_goss120b/beta01_lr75e7_seed0-c2c04e/summary.json`

**Comparison to GPT-4.1 on the same target**:

| Metric | GPT-4.1 | GPT-oss-120B | Delta |
|---|---:|---:|---:|
| Mean | 8.43 | 7.93 | -0.50 |
| Compliance | 81.3% | 83.6% | +2.3pp |

**Interpretation**:
- GPT-oss-120B produced usable compliance scores
- it underscored the mean by about `0.5` while slightly increasing the
  compliance rate
- the bottom statements still looked familiar (`refusal_style`,
  `support_programmatic_use`, `avoid_targeted_political_manipulation`)
- `native_stderr_mode="tee"` is required for sane TPU judge debugging

---

## 2026-04-10: APLN-004 — Regional v6e-8 Sweep Plan (later superseded)

**Why this plan existed**: the smoke run on `v5p-8` worked but suffered
multiple preemptions. The next idea was to move the full sweep to `v6e-8`,
which had better zone coverage.

**Key planning conclusions**:
- `v6e-4` is too small for GPT-oss-120B; `v6e-8` is the smallest plausible
  v6e shape
- GPT-oss-120B was staged in `us-east1` and `us-east5`, not `eu-west4`
- the regional assignment would have minimized cross-region reads by running:
  - `us-east5`: the three east5 seed-0 DPO artifacts plus the
    `beta01_lr75e7_seed0` central artifact
  - `us-east1`: SFT plus the `eu-west4` batch-64 comparison artifacts

**What changed**:
- this plan was overtaken by `EXP-021`, which got much better leverage from:
  - one long-lived TPU judge session
  - mirrored inputs
  - target-level resume on preemption

---

## 2026-04-10: EXP-021 — Batched Judge with Session Reuse + Mirror Inputs + Resume

**Motivation**: Per-target judge jobs paid startup costs over and over:
- model load: ~2 min
- XLA compile: ~9 min
- one preemption could wipe almost all useful progress

**Implementation**:

1. **Refactored `lib/marin/src/marin/alignment/evaluate.py`**
   - extracted `_judge_one_artifact`
   - kept `run_eval_judge` as a thin wrapper with unchanged behavior
   - added `EvalJudgeTarget`
   - added `BatchEvalJudgeConfig`
   - added `run_batch_eval_judge`, which opens one
     `BatchedVllmServeSession` and judges all targets sequentially

2. **Added target-level resume**
   - `_target_already_done(output_path)` checks for `{output_path}/summary.json`
   - completed targets are skipped on restart
   - only the in-flight target is lost on preemption

3. **Added cross-region input mirroring**
   - `experiments/posttrain/judge_all_goss120b.py` wraps each inference path
     in `mirrored(...)`
   - a small helper strips the `gs://marin-<region>/` prefix before mirroring
   - inference JSONL shards are copied into the local region on first read

4. **Collapsed the job to a single executor step**
   - one `ExecutorStep`
   - one judge model load
   - nested outputs at `{step_out}/{label}/summary.json`

**Verification before launch**:
- `./infra/pre-commit.py --fix ...`: OK
- `uv run pytest tests/test_alignment.py -q`: `109 passed`

**Batch job**:
- job: `/ahmed/judge-goss120b-batch-v5p-central1`
- executor metadata:
  - `gs://marin-us-central1/experiments/judge_all_goss120b-e71323.json`
- output root:
  - `gs://marin-us-central1/eval/judge_goss120b_batch-fd3ffe/`

**Mirror verification**:
- logs showed explicit cross-region copying, e.g.
  - `gs://marin-us-east1/eval/marin_8b_instruct_bloom_speceval/inference-89612d/shard_00000.jsonl.gz`
  - copied into `gs://marin-us-central1/...`

**Preemption behavior**:
- root task preempted **11 times**
- child TPU task preempted at least 4 times
- resume logic consistently skipped already-finished targets via
  existing `summary.json` markers

**Final status**:
- `JOB_STATE_SUCCEEDED`
- all 8 targets wrote `summary.json`
- total wall-clock: ~3 hours

**Interpretation**:
- the reusable batch-judge path worked
- target-level resume was enough to finish despite repeated preemptions
- this was a much better fit than running 8 separate TPU judge jobs

---

## 2026-04-10: EXP-022 — GPT-oss-120B Judge Results for All 8 Targets

**Output root**:
- `gs://marin-us-central1/eval/judge_goss120b_batch-fd3ffe/{label}/summary.json`

**Overall means**:

| Target | GPT-4.1 mean | GPT-oss-120B mean | Delta |
|---|---:|---:|---:|
| `sft` | 7.9405 | 7.6259 | -0.3146 |
| `beta001_lr5e7_seed0` | 8.6984 | 8.2061 | -0.4923 |
| `beta001_lr75e7_seed0` | 8.6541 | 8.1682 | -0.4859 |
| `beta01_lr5e7_seed0` | 8.3376 | 8.0096 | -0.3280 |
| `beta01_lr75e7_seed0` | 8.3865 | 7.9513 | -0.4352 |
| `full_dpo_beta01_b64_step1699` | 8.4443 | 7.9587 | -0.4856 |
| `lora_lr5e6_b64_step1699` | 8.5040 | 7.9999 | -0.5041 |
| `lora_lr1e5_b64_step1699` | 8.5531 | 8.0468 | -0.5063 |

**Ranking checks**:
- Seed-0 full-model sweep:
  - GPT-4.1: `beta001_lr5e7 > beta001_lr75e7 > beta01_lr75e7 > beta01_lr5e7 > sft`
  - GPT-oss: `beta001_lr5e7 > beta001_lr75e7 > beta01_lr5e7 > beta01_lr75e7 > sft`
  - interpretation: top 2 and bottom 1 are preserved; the middle pair swaps
    within a noise-sized gap
- Batch-64 trio:
  - GPT-4.1: `lora_lr1e5 > lora_lr5e6 > full_dpo`
  - GPT-oss: `lora_lr1e5 > lora_lr5e6 > full_dpo`
  - interpretation: the LoRA > full-DPO ordering is preserved exactly

**Bottom line**:
- GPT-oss-120B is a reasonable replacement for aggregate target ranking
- it is systematically harsher by roughly `0.3` to `0.5` points

---

## 2026-04-10: EXP-023 — Spearman Correlation Analysis (GPT-4.1 vs GPT-oss-120B)

**Aggregate agreement on the 8 target means**:
- Spearman `rho = 0.857`
- Pearson `r = 0.990`

**Interpretation**:
- Pearson is extremely high because GPT-oss mostly acts like a near-linear
  downward shift from GPT-4.1
- Spearman falls because two very close adjacent pairs swap order

**Per-statement Spearman across target means**:
- mean: `0.704`
- median: `0.790`
- max: `1.000` (`be_creative`, but this was later recognized as an n=8 artifact)
- min: `0.143` (`letter_and_spirit`)
- `80%` of statements had `rho >= 0.5`
- none were anti-correlated

**Lowest-agreement statements under this target-mean view**:
- `letter_and_spirit`
- `be_kind`
- `support_programmatic_use`
- `avoid_sycophancy`
- `no_agenda`

**Immediate follow-up**:
- do exemplar-level disagreement analysis on these low-correlation statements

---

## 2026-04-10: EXP-024 — Disagreement Analysis on 5 Low-Correlation Statements

**Paired datasets built**:
- script: `/tmp/build_disagreement_data.py`
- outputs:
  - `/tmp/judges/support_programmatic_use.jsonl`
  - `/tmp/judges/letter_and_spirit.jsonl`
  - `/tmp/judges/avoid_sycophancy.jsonl`
  - `/tmp/judges/no_agenda.jsonl`
  - `/tmp/judges/be_kind.jsonl`

**Paired record counts**:

| Statement | Paired pairs | Mean diff (gpt4.1 - gptoss) | \|diff\| >= 4 |
|---|---:|---:|---:|
| `support_programmatic_use` | 1269 | +2.45 | 483 |
| `letter_and_spirit` | 1390 | +0.60 | 27 |
| `no_agenda` | 1271 | +0.37 | 27 |
| `avoid_sycophancy` | 1511 | +0.09 | 68 |
| `be_kind` | 1271 | +0.80 | 3 |

**Subagent writeups**:
- 5 statement-specific markdown analyses were saved under `claude_subagents/`

**Cross-cutting findings**:
1. GPT-4.1 tends to grade more holistically; GPT-oss reads some rubrics more
   literally.
2. Parse failures materially inflate the apparent disagreement.
   - especially `support_programmatic_use`
   - GPT-oss parse failures were written as `score=0`
3. Concrete / bright-line rubrics remain high-agreement.
4. Holistic statements such as `no_agenda` showed real construct differences.

**Working conclusion at this point**:
- GPT-oss is usable for aggregate ranking
- it is not a clean drop-in replacement for per-statement analysis on the most
  subjective rubrics

---

## 2026-04-11: EXP-025 — Item-Level Pearson Plan

**Why this was needed**: target-mean Spearman on just 8 model means per
statement could not distinguish:
- pure calibration shift
- genuine construct mismatch

**Planned method**:
- pair items by `(prompt_id, response_text, behavior_id)`
- filter parse failures
- compute:
  - per-statement item-level Pearson
  - naive pooled Pearson
  - within-statement centered pooled Pearson

---

## 2026-04-11: EXP-026 — Item-Level Pearson Results

**Scratch outputs**:
- script: `/tmp/judge_pearson.py`
- JSON dump: `/tmp/judge_pearson.json`

**Dataset**:
- 61,477 raw paired items across 45 shared statements
- parse-failure filter removed 537 rows (0.87%)
- `support_programmatic_use` dominated the parse failures:
  - `384 / 1269 = 30.2%`

**Pooled item-level Pearson**:
- naive pooled: `r = 0.7507`
- within-statement centered pooled: `r = 0.6782`

**Per-statement Pearson distribution**:
- mean: `0.6591`
- median: `0.7069`
- Q1 / Q3: `0.5821 / 0.7482`
- min: `0.3672` (`be_rationally_optimistic`)
- max: `0.8496` (`comply_with_laws`)
- `84.4%` of statements had Pearson `>= 0.5`

**Top-5 item-level agreement**:
- `comply_with_laws`
- `express_uncertainty`
- `avoid_info_hazards`
- `respect_creators`
- `do_not_facilitate_illicit_behavior`

**Bottom-5 item-level agreement**:
- `be_rationally_optimistic`
- `follow_all_applicable_instructions`
- `no_agenda`
- `avoid_being_condescending`
- `formatting`

**Key reinterpretation of EXP-024**:
- `be_kind` is mostly a calibration shift, not a deep construct mismatch
  - Pearson `0.7838`
- `support_programmatic_use` is also mostly calibration + parse-failure noise
  - Pearson `0.7279` after filtering
- `no_agenda` remains a genuine construct mismatch
  - Pearson `0.3858`
- `letter_and_spirit` and `avoid_sycophancy` land in the lower-middle, not at
  the absolute floor

**Takeaway**:
- GPT-oss remains fine for overall model ranking
- the item-level agreement floor is lower than the target-mean analysis implied
- parse-failure handling had become a correctness issue, not just a reporting
  nuisance

---

## 2026-04-11: EXP-027 — Parse Failure Asymmetry Fix

**Motivation**: parse failures were being scored asymmetrically:
- GPT-4.1 path: defaulted to `score=5`
- GPT-oss path: defaulted to `score=0`

This silently biased comparisons and forced every downstream analysis to
manually filter parse failures.

**Fix**:
- parse failures now emit `score=None, compliant=None`
- downstream aggregation skips them entirely

**Files changed**:
1. `lib/marin/src/marin/alignment/types.py`
   - `ComplianceResult.score: int | None`
   - `ComplianceResult.compliant: bool | None`
2. `lib/marin/src/marin/alignment/judge.py`
   - `parse_compliance_result` now returns `None` fields on parse failure
   - `_judgment_record` filters `None` scores before max/min selection
3. `lib/marin/src/marin/alignment/evaluate.py`
   - parse-failure counting now uses `score is None`
   - summary computation skips `None`-scored rows
4. `experiments/posttrain/run_bloom_judge.py`
   - all parse-failure fallback paths now return `score=None`
   - `compliant=None` when `score is None`
5. `tests/test_alignment.py`
   - added 3 parser-regression tests

**Verification**:
- `./infra/pre-commit.py --fix ...`: OK
- `uv run pytest tests/test_alignment.py -q`: `112 passed`

**Operational note**:
- existing historical GCS artifacts still contain the old fallback scores
- future reruns after this change will not need special parse-failure filters

---

## 2026-04-11: EXP-028 — GPT-5 Judge API Compatibility, Stability, and Cost Planning

### API compatibility probes

**GPT-5.4**:
- `max_tokens` is rejected
- `max_completion_tokens` works
- `temperature=0.0` is accepted
- clean JSON output parsed directly

**GPT-5.2-chat-latest**:
- `max_tokens` is rejected
- `temperature=0.0` is hard-rejected; only the default `1.0` path is allowed

**GPT-5.4-chat-latest**:
- does not exist (`404`)

**GPT-5.1**:
- same API regime as GPT-5.4
- `max_tokens` rejected
- `max_completion_tokens` works
- `temperature=0.0` accepted
- system-role prompts work
- no visible reasoning-token usage on the probe

**Code implication**:
- add a GPT-5-family branch in the OpenAI client path:
  - GPT-4.1 and older: `max_tokens`
  - GPT-5*: `max_completion_tokens`

### Judge score stability probe

**Scratch script**:
- `/tmp/judge_stability_gpt54.py`

**Real judge item tested**:
- statement: `be_creative`
- prompt id: `be_creative/cfg_016`
- original GPT-4.1 score: `6`

**Results**:
- `temperature=0.0`: 10/10 valid, distinct scores `[6]`
- `temperature=1.0`: 10/10 valid, distinct scores `[6, 7]`, with `6` on 9/10

**Interpretation**:
- free-form explanation text is not byte-stable
- the single-token judge score is effectively stable at `temperature=0.0`
- aggregate judge noise from this source is negligible

### Cost analysis

**Observed GPT-4.1 spend for the earlier large judge workload**:
- input uncached: `$71.88` for `35.94M` tokens
- input cached: `$2.68` for `5.36M` tokens
- output: `$59.40` for `7.425M` tokens
- total: `$133.96`

**Cost comparison**:

| Plan | Estimated total |
|---|---:|
| GPT-4.1 standard | `$133.96` |
| GPT-4.1 batch | `$71.00` |
| GPT-5.1 standard | `$119.87` |
| GPT-5.1 batch (assuming 50% batch discount) | `~$62.94` |

**Takeaway**:
- batch is the biggest cost lever
- upgrading models without switching to batch saves much less than just using
  the batch API

### Per-target token accounting for the 4-target correlation study

**Targets**:
- `sft`
- `full_dpo_beta01_b64_step1699`
- `lora_lr1e5_b64_step1699`
- `lora_lr5e6_b64_step1699`

**Scratch script**:
- `/tmp/judge_token_count.py`

**Totals across the 4 targets**:
- prompt tokens: `46,482,164`
- cached tokens: `6,266,368`
- uncached input: `40,215,796`
- output: `8,224,766`

**Corresponding GPT-4.1 cost**:
- standard: `$149.36`
- batch: `$79.38`

**Structural finding**:
- cache hit rate was only `13.5%`, much lower than expected for a workload with
  large repeated prompt prefixes

**Sampling-cost estimate for a correlation study**:
- `500` items per target would cost only about:
  - `~$4.58` on GPT-5.1 batch (assuming 50% batch discount)
  - `~$9.16` on GPT-5.1 standard

**Decision after the probes**:
- narrow the correlation study to GPT-4.1 vs GPT-5.1
- park GPT-5.4 as a fallback, not the primary next judge

---

## 2026-04-11–13: EXP-028g — Continued In Separate Logbook

Claude split the continuing GPT-5 correlation work into a dedicated logbook:
- `.agents/logbooks/gpt5_correlation.md`

That follow-on logbook covers:
- GPT-5.1 batch judging on the 4-target Marin slice plus GPT-4.1 target/opposite-mode datasets
- discovery and fix of the GPT-5.1 `reasoning_effort` issue
- GPT-5 JSON parser quirks and reparse tooling
- 3-way GPT-4.1 / GPT-5.1 / GPT-oss-120B agreement analysis
- Pareto comparison of GPT-5.1 vs GPT-oss as a GPT-4.1 proxy
- bottom-statement deep dives
- opposite-mode analysis
- preserved 5-target ranking under GPT-5.1
- per-statement preference datasets for continual alignment work

**Start there for any continuation past 2026-04-11.**

---

## 2026-04-13: GEMINI-001 — One-Off Gemini SDK Script + AI Studio Key Flow

**User request**: add a one-off Gemini caller for manual judge/probe work using
the model family around `gemini-3.1-pro-preview`, and verify the right API-key
provisioning flow.

**Implementation**:
- added:
  - `scripts/gemini_oneoff.py`
- switched from the initial raw-REST draft to the **official Google SDK**
  path:
  - package: `google-genai`
  - imports:
    - `from google import genai`
    - `from google.genai import types`
- script behavior:
  - reads `GEMINI_API_KEY` or `GOOGLE_API_KEY`
  - supports `--list-models`
  - supports `--model`, `--system`, `--max-output-tokens`, and `--json`
  - defaults to:
    - `gemini-3.1-pro-preview`

**Verification**:
- local syntax check:
  - `uv run python -m py_compile scripts/gemini_oneoff.py`
- ephemeral SDK surface check (no authenticated API call):
  - `uv run --with google-genai python ...`
  - confirmed:
    - `genai.Client`
    - `Client.models.list()`
    - `Client.models.generate_content()`
    - `types.GenerateContentConfig`

**Usage**:
```bash
export GEMINI_API_KEY=...
uv run --with google-genai python scripts/gemini_oneoff.py --list-models
uv run --with google-genai python scripts/gemini_oneoff.py --model gemini-3.1-pro-preview "Explain attention in one paragraph."
uv run --with google-genai python scripts/gemini_oneoff.py --system "You are a concise coding assistant." "Write a Python HTTP server."
```

**Key-flow conclusion**:
- the right path is to create/manage the Gemini API key in **Google AI Studio**
  rather than treating `https://aistudio.google.com/usage` itself as the
  provisioning workflow
- the usage page is for quota/usage/account state
- for practical use, list models first with the same key because preview model
  availability changes and the exact visible model ID can differ over time

**Operational note**:
- I did **not** perform a live Gemini generation from this repo because no API
  key was supplied in-session
- this is intentionally a standalone utility, not yet integrated into any
  Marin or Bloom judging pipeline

---

## 2026-04-25T21:31Z: A0-JUDGE-001 — High-Information A=0 LoRA Inference Launch Plan

**Question**: Given the near-null Spearman correlation between B=0 and A=0 LR
rankings, which A=0 checkpoints should get Bloom-format inference and GPT-4.1
LM-as-judge first?

**Decision**: Run a four-checkpoint seed-0 A=0 inference batch before judging.
This is deliberately not the full 17-row A=0 matrix. The first pass should test
distinct hypotheses with the smallest useful GPT-4.1 spend:

1. `azero_lr1e6_seed0_step1699` — the surprising LR inversion case. Under B=0,
   `lr=1e-6` was the worst LR; under A=0 it is the top accuracy LR.
2. `azero_lr1e5_seed0_step1699` — the old recommended LoRA LR under the new
   init, directly comparable to the already-judged B=0 `lora_lr1e5` baseline.
3. `azero_lr8p75e6_seed0_step1699` — high-LR plateau / strong-margin point,
   testing whether GPT-4.1 follows margin/loss rather than accuracy-only rank.
4. `azero_lr5e6_seed0_step1699` — same-LR init comparison against the already
   judged B=0 `lora_lr5e6` baseline.

**Inference config**:
- prompts: `gs://marin-us-east5/alignment/gpt-4.1-eval-split/`
- prompt format: Bloom
- sampling: `n=3`, `temperature=0.7`, `max_tokens=1500`
- compute: Iris `v6e-4`, target region `us-east5`, priority band intended as
  `interactive` (current Iris CLI defaults submissions to interactive; this
  checkout does not expose an explicit `--priority` flag)

**Model paths**:
- `azero_lr1e6_seed0_step1699`:
  `gs://marin-us-central2/checkpoints/dpo/tune_lora/lora_bloom_speceval_v2_lr1e6_seed0_b64_v5p8_azero-8e1101/hf/step-1699`
- `azero_lr1e5_seed0_step1699`:
  `gs://marin-us-east5/checkpoints/dpo/tune_lora/lora_bloom_speceval_v2_lr1e5_seed0_b64_v5p8_azero-d93e61/hf/step-1699`
- `azero_lr8p75e6_seed0_step1699`:
  `gs://marin-us-east5/checkpoints/dpo/tune_lora/lora_bloom_speceval_lr8p75e6_seed0_b64_v5p8_azero-4a1bf7/hf/step-1699`
- `azero_lr5e6_seed0_step1699`:
  `gs://marin-us-central2/checkpoints/dpo/tune_lora/lora_bloom_speceval_v2_lr5e6_seed0_b64_v5p8_azero-a9e388/hf/step-1699`

**Operational note**:
- `v6e-4` is available in `us-east5`, `us-east1`, and `europe-west4`, not
  `us-central2` in the current Iris config. The two `us-central2` model paths
  are therefore launched as absolute model paths from `us-east5` compute rather
  than copied across buckets.
- All four model prefixes and the `us-east5` prompt tree were checked with
  `gcloud storage ls` before launch.

**Next actions**:
1. Launch the four inference-only Iris jobs.
2. Record job ids and expected output prefixes here.
3. After inference succeeds, run Bloom-compatible GPT-4.1 judging on the four
   inference artifacts and compare against the existing B=0 LoRA rows.

### Launch status

Script change:
- `experiments/posttrain/eval_llama3_8b_alignment.py` now supports absolute
  `gs://` model paths and includes the four A=0 target keys above.
- Syntax check passed:
  `uv run python -m py_compile experiments/posttrain/eval_llama3_8b_alignment.py`

Submitted commands used `--region us-east5`, `--tpu-type v6e-4`,
`--run-label azero_s0_v6e4_r1`, and one target per parent job. Iris CLI in
this checkout has no explicit `--priority` option; default launch band is
interactive.

**Launched and pending on v6e-4 capacity**:

1. `azero_lr1e5_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr1e5-s0-v6e4-r1`
   - child job:
     `/ahmed/bloom-eval-azero-lr1e5-s0-v6e4-r1/eval-marin_dpo_lora_azero_lr1e5_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1-inference_dcf397c6-42d9ac54`
   - executor metadata:
     `gs://marin-us-east5/experiments/eval_llama3_8b_alignment-336241.json`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr1e5_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1/inference-897d5b`
   - current state at launch check: child `PENDING`, waiting for
     `v6e-4` capacity in `us-east5`

2. `azero_lr8p75e6_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr8p75e6-s0-v6e4-r1`
   - child job:
     `/ahmed/bloom-eval-azero-lr8p75e6-s0-v6e4-r1/eval-marin_dpo_lora_azero_lr8p75e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1-inference_3cd8918f-3da0ed1a`
   - executor metadata:
     `gs://marin-us-east5/experiments/eval_llama3_8b_alignment-088fd9.json`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr8p75e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1/inference-3c02da`
   - current state at launch check: child `PENDING`, waiting for
     `v6e-4` capacity in `us-east5`

**Submitted but rejected before child launch**:

1. `azero_lr1e6_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr1e6-s0-v6e4-r1`
   - failed because the executor rejected cross-region dependencies:
     model in `us-central2`, prompts/output in `us-east5`
   - attempted output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1/inference-ef19f0`

2. `azero_lr5e6_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr5e6-s0-v6e4-r1`
   - failed for the same cross-region guard:
     model in `us-central2`, prompts/output in `us-east5`
   - attempted output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1/inference-fc3f93`

**Correction to pre-launch assumption**:
- `us-east5` contains partial same-name executor prefixes for
  `lr1e6_seed0` and `lr5e6_seed0`, but not the needed final `hf/step-1699`
  exports:
  - `lr1e6_seed0` has only `hf/step-200` in `us-east5`
  - `lr5e6_seed0` has no `hf/step-1699` under the checked `us-east5` prefix
- To run those two exact final checkpoints on `v6e-4`, we need either:
  1. explicit approval to mirror the final HF exports from `us-central2` into
     `us-east5`, or
  2. pick substitute A=0 seed-0 checkpoints that already have final HF exports
     in a v6e-capable region, or
  3. give up on `v6e-4` for those two and use a same-region non-v6e path.

---

## 2026-04-25T21:42Z: A0-JUDGE-002 — Mirror-backed relaunch for cross-region failures

**Trigger**: User explicitly approved mirror/filesystem-backed copying after the
`lr1e-6` and `lr5e-6` launch parents failed the executor cross-region guard.

**Root cause**:
- The first launch used absolute `gs://marin-us-central2/...` model paths for
  the two final checkpoints that only exist in `us-central2`.
- The inference output and prompts were in `us-east5`, so executor planning
  failed before child TPU jobs could launch.
- Directly replacing model paths with `mirror://...` is not sufficient for vLLM:
  the executor understands `mirror://`, but vLLM expects a concrete local path
  or object-store path.

**Implementation plan**:
1. Represent A=0 target checkpoints as mirror-relative Marin paths with
   `mirrored(..., budget_gb=80)` so the executor permits cross-region
   materialization. This 80 GB budget is intentionally above a single 8B HF
   export and follows the user's explicit approval for this rerun.
2. Materialize `mirror://` vLLM model directories at inference runtime using
   the repo filesystem layer:
   - enumerate files via the `mirror://` filesystem,
   - call `info()` on each file to trigger one locked copy into the local
     Marin prefix,
   - pass the resulting concrete local `gs://marin-us-east5/...` path to vLLM.
3. Keep the two valid `r1` east5 jobs queued:
   - `azero_lr1e5_seed0_step1699`
   - `azero_lr8p75e6_seed0_step1699`
   They did not fail the cross-region guard and are still pending on v6e-4
   capacity, so restarting them would only lose queue position.
4. Relaunch only the two failed central2-backed targets with a new run label:
   `azero_s0_v6e4_r2`.

**Validation before relaunch**:
- Syntax check passed:
  `uv run python -m py_compile experiments/posttrain/eval_llama3_8b_alignment.py lib/marin/src/marin/alignment/evaluate.py`
- Local construction check confirmed the `lr1e-6` target now serializes a
  `MirroredValue(value='checkpoints/dpo/tune_lora/lora_bloom_speceval_v2_lr1e6_seed0_b64_v5p8_azero-8e1101/hf/step-1699', budget_gb=80)`
  into the executor config before instantiation.

**Relaunch commands**:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name bloom-eval-azero-lr1e6-s0-v6e4-r2 \
  --cpu 4 --memory 16GB --disk 10GB \
  --region us-east5 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region us-east5 \
    --target azero_lr1e6_seed0_step1699 \
    --tpu-type v6e-4 \
    --run-label azero_s0_v6e4_r2

uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name bloom-eval-azero-lr5e6-s0-v6e4-r2 \
  --cpu 4 --memory 16GB --disk 10GB \
  --region us-east5 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region us-east5 \
    --target azero_lr5e6_seed0_step1699 \
    --tpu-type v6e-4 \
    --run-label azero_s0_v6e4_r2
```

### Relaunch status

Both `r2` parent jobs were accepted, passed executor planning, wrote executor
metadata, and launched child TPU inference jobs. They are now pending on
`v6e-4` capacity in `us-east5`, same queue blocker as the two valid `r1`
east5 jobs.

1. `azero_lr1e6_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr1e6-s0-v6e4-r2`
   - child job:
     `/ahmed/bloom-eval-azero-lr1e6-s0-v6e4-r2/eval-marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r2-inference_766d1f50-db8e183c`
   - executor metadata:
     `gs://marin-us-east5/experiments/eval_llama3_8b_alignment-bb1abd.json`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r2/inference-29664f`
   - current state at launch check: child `PENDING`, waiting for
     `v6e-4` capacity in `us-east5`

2. `azero_lr5e6_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr5e6-s0-v6e4-r2`
   - child job:
     `/ahmed/bloom-eval-azero-lr5e6-s0-v6e4-r2/eval-marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r2-inference_deb0e7c7-c727e06f`
   - executor metadata:
     `gs://marin-us-east5/experiments/eval_llama3_8b_alignment-ee331e.json`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r2/inference-a89e9d`
   - current state at launch check: child `PENDING`, waiting for
     `v6e-4` capacity in `us-east5`

**Current four-checkpoint inference queue**:
- `lr1e-6`: `r2` child pending, mirror-backed central2 source
- `lr1e-5`: original `r1` child pending, already local east5 source
- `lr8.75e-6`: original `r1` child pending, already local east5 source
- `lr5e-6`: `r2` child pending, mirror-backed central2 source

---

## 2026-04-25T21:47Z: A0-JUDGE-003 — Parallel v5p-8 fallback lane

**Trigger**: User observed available `v5p-8` capacity and asked to run the same
four inference targets in parallel, using whichever TPU lane finishes first.

**Decision**:
- Keep the existing `v6e-4` jobs queued; do not stop or restart them.
- Launch the same four A=0 seed-0 targets on `v5p-8` with run label
  `azero_s0_v5p8_r1`.
- Use `us-east5` because:
  - Iris exposes `v5p-8` in `us-east5-a`.
  - The Bloom prompt tree already exists in
    `gs://marin-us-east5/alignment/gpt-4.1-eval-split/`.
  - The two non-local model exports are already mirror-backed by
    `A0-JUDGE-002`.
- Keep `tensor_parallel_size=4`; this matches existing Marin posttrain
  conventions for `v5p-8` (`bcg_probe_infer.py`, `judge_all_goss120b.py`,
  and other vLLM TPU scripts).

**Commands**:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name bloom-eval-azero-lr1e6-s0-v5p8-r1 \
  --cpu 4 --memory 16GB --disk 10GB --region us-east5 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region us-east5 --target azero_lr1e6_seed0_step1699 \
    --tpu-type v5p-8 --run-label azero_s0_v5p8_r1

uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name bloom-eval-azero-lr1e5-s0-v5p8-r1 \
  --cpu 4 --memory 16GB --disk 10GB --region us-east5 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region us-east5 --target azero_lr1e5_seed0_step1699 \
    --tpu-type v5p-8 --run-label azero_s0_v5p8_r1

uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name bloom-eval-azero-lr8p75e6-s0-v5p8-r1 \
  --cpu 4 --memory 16GB --disk 10GB --region us-east5 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region us-east5 --target azero_lr8p75e6_seed0_step1699 \
    --tpu-type v5p-8 --run-label azero_s0_v5p8_r1

uv run iris --config lib/iris/examples/marin.yaml job run --no-wait \
  --job-name bloom-eval-azero-lr5e6-s0-v5p8-r1 \
  --cpu 4 --memory 16GB --disk 10GB --region us-east5 \
  -- python experiments/posttrain/eval_llama3_8b_alignment.py \
    --region us-east5 --target azero_lr5e6_seed0_step1699 \
    --tpu-type v5p-8 --run-label azero_s0_v5p8_r1
```

### Launch status

All four `v5p-8` parent jobs were accepted. All four passed executor planning,
wrote metadata, and launched child inference steps in `us-east5`. At the latest
status check, `lr5e-6` had started running on `v5p-8`; the other three children
were waiting for `tpu_v5p-preemptible_8-us-east5-a` scale-up/assignment.

1. `azero_lr1e6_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr1e6-s0-v5p8-r1`
   - child job:
     `/ahmed/bloom-eval-azero-lr1e6-s0-v5p8-r1/eval-marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1-inference_efea50c5-008135b8`
   - executor metadata:
     `gs://marin-us-east5/experiments/eval_llama3_8b_alignment-884230.json`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1/inference-e58c13`
   - latest state: child `PENDING`, waiting for `v5p-8` scale-up/assignment

2. `azero_lr1e5_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr1e5-s0-v5p8-r1`
   - child job:
     `/ahmed/bloom-eval-azero-lr1e5-s0-v5p8-r1/eval-marin_dpo_lora_azero_lr1e5_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1-inference_7cdac1ca-8f5f9a6d`
   - executor metadata:
     `gs://marin-us-east5/experiments/eval_llama3_8b_alignment-5f2702.json`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr1e5_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1/inference-8cbc2b`
   - latest state: child `PENDING`, waiting for `v5p-8` scale-up/assignment

3. `azero_lr8p75e6_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr8p75e6-s0-v5p8-r1`
   - child job:
     `/ahmed/bloom-eval-azero-lr8p75e6-s0-v5p8-r1/eval-marin_dpo_lora_azero_lr8p75e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1-inference_6b9c9a7e-e6c2ab16`
   - executor metadata:
     `gs://marin-us-east5/experiments/eval_llama3_8b_alignment-5b084b.json`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr8p75e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1/inference-da38f0`
   - latest state: child `PENDING`, waiting for `v5p-8` scale-up/assignment

4. `azero_lr5e6_seed0_step1699`
   - parent job: `/ahmed/bloom-eval-azero-lr5e6-s0-v5p8-r1`
   - child job:
     `/ahmed/bloom-eval-azero-lr5e6-s0-v5p8-r1/eval-marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1-inference_3516cbf1-54fe9c1d`
   - executor metadata:
     `gs://marin-us-east5/experiments/eval_llama3_8b_alignment-bc98bd.json`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1/inference-46e9f4`
   - latest state: child `RUNNING`

**Use rule**:
- For each target, use the first completed inference artifact among its `v6e-4`
  and `v5p-8` lanes for GPT-4.1 judging.
- Do not cancel the slower lane until the faster lane has completed and the
  output artifact has been sanity-checked.

### Monitor update: first 5-minute pass

Monitoring owner: Codex. State file:
`scratch/20260425-1456_monitoring_state.json`.

At the first explicit monitoring pass:
- `azero_lr8p75e6_seed0_step1699` succeeded on the original `v6e-4` lane.
  Candidate inference artifact:
  `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr8p75e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1/inference-3c02da`
- `azero_lr1e5_seed0_step1699` is still running on the `v5p-8` lane. The
  original `v6e-4` lane failed with an `OSError: [Errno 98] Address already in
  use` during vLLM server startup.
- `azero_lr1e6_seed0_step1699` and `azero_lr5e6_seed0_step1699` failed on both
  mirror-backed lanes because tokenizer staging could not find files at the
  concrete `gs://marin-us-east5/.../hf/step-1699` path. The runtime
  materialization hook logged 14 files, but an external `gcloud storage ls`
  found no objects at the expected east5 destination. The central2 sources are
  intact and about 32.1 GB each.

Recovery action:
- Explicitly stage the two central2 HF export directories into the corresponding
  east5 prefixes using direct GCS filesystem copy, then relaunch only those two
  failed targets with new `v5p-8` run labels. This follows the user's explicit
  instruction to use mirror/filesystem copying for the cross-region issue.

Staging result:
- `azero_lr1e6_seed0_step1699`: copied 14 files, 32,138,362,810 bytes, into
  `gs://marin-us-east5/checkpoints/dpo/tune_lora/lora_bloom_speceval_v2_lr1e6_seed0_b64_v5p8_azero-8e1101/hf/step-1699`
- `azero_lr5e6_seed0_step1699`: copied 14 files, 32,138,362,810 bytes, into
  `gs://marin-us-east5/checkpoints/dpo/tune_lora/lora_bloom_speceval_v2_lr5e6_seed0_b64_v5p8_azero-a9e388/hf/step-1699`

Relaunch after verified staging:
- `/ahmed/bloom-eval-azero-lr1e6-s0-v5p8-r2`
- `/ahmed/bloom-eval-azero-lr5e6-s0-v5p8-r2`

### Monitor update: 2026-04-25T22:05Z

Completed and sanity-checked artifacts:

1. `azero_lr8p75e6_seed0_step1699`
   - first successful lane: original `v6e-4` `r1`
   - artifact:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr8p75e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1/inference-3c02da`
   - files: `.artifact`, `.executor_info`, `.executor_status`,
     `artifacts/vllm_metrics.json`, `shard_00000.jsonl.gz`,
     `shard_00001.jsonl.gz`
   - record count: `7728` JSONL rows = `2576` prompts * `n=3`
   - metrics: `completion_count=7728`, `request_prompt_count=2576`,
     `output_token_count=2536388`

2. `azero_lr1e5_seed0_step1699`
   - first successful lane: `v5p-8` `r1`
   - artifact:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr1e5_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1/inference-8cbc2b`
   - files: `.artifact`, `.executor_info`, `.executor_status`,
     `artifacts/vllm_metrics.json`, `shard_00000.jsonl.gz`,
     `shard_00001.jsonl.gz`
   - record count: `7728` JSONL rows = `2576` prompts * `n=3`
   - metrics: `completion_count=7728`, `request_prompt_count=2576`,
     `output_token_count=2554721`

Still running:

1. `azero_lr1e6_seed0_step1699`
   - active job:
     `/ahmed/bloom-eval-azero-lr1e6-s0-v5p8-r2/eval-marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r2-inference_aec777ac-32931163`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r2/inference-9a8ec8`
   - latest signal: loaded `2576` eval prompts, materialized the mirrored model
     into the verified east5 path with `14 files`, and started the native vLLM
     server.

2. `azero_lr5e6_seed0_step1699`
   - active job:
     `/ahmed/bloom-eval-azero-lr5e6-s0-v5p8-r2/eval-marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r2-inference_fc87fb81-0b6a6e4f`
   - output prefix:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r2/inference-1c06ed`
   - latest signal: loaded `2576` eval prompts, materialized the mirrored model
     into the verified east5 path with `14 files`, and started the native vLLM
     server.

### Monitor stop: 2026-04-25T22:11Z

Stop condition met: at least one inference artifact finished and passed the
basic sanity check for each of the four A=0 seed-0 configs, regardless of
hardware lane. Both active `v5p-8` recovery jobs reached terminal success, so
there are no remaining live duplicate inference jobs to cancel.

Final artifacts to use for GPT-4.1 judging:

1. `azero_lr1e6_seed0_step1699`
   - lane: `v5p-8` `r2`
   - artifact:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r2/inference-9a8ec8`
   - job:
     `/ahmed/bloom-eval-azero-lr1e6-s0-v5p8-r2/eval-marin_dpo_lora_azero_lr1e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r2-inference_aec777ac-32931163`
   - status: `JOB_STATE_SUCCEEDED`
   - sanity: `7728` JSONL rows, `completion_count=7728`,
     `request_prompt_count=2576`, `output_token_count=1940705`

2. `azero_lr1e5_seed0_step1699`
   - lane: `v5p-8` `r1`
   - artifact:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr1e5_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1/inference-8cbc2b`
   - job:
     `/ahmed/bloom-eval-azero-lr1e5-s0-v5p8-r1/eval-marin_dpo_lora_azero_lr1e5_seed0_step1699_bloom_speceval_azero_s0_v5p8_r1-inference_7cdac1ca-8f5f9a6d`
   - status: `JOB_STATE_SUCCEEDED`
   - sanity: `7728` JSONL rows, `completion_count=7728`,
     `request_prompt_count=2576`, `output_token_count=2554721`

3. `azero_lr8p75e6_seed0_step1699`
   - lane: original `v6e-4` `r1` first success
   - artifact:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr8p75e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1/inference-3c02da`
   - job:
     `/ahmed/bloom-eval-azero-lr8p75e6-s0-v6e4-r1/eval-marin_dpo_lora_azero_lr8p75e6_seed0_step1699_bloom_speceval_azero_s0_v6e4_r1-inference_3cd8918f-3da0ed1a`
   - status: `JOB_STATE_SUCCEEDED`
   - sanity: `7728` JSONL rows, `completion_count=7728`,
     `request_prompt_count=2576`, `output_token_count=2536388`

4. `azero_lr5e6_seed0_step1699`
   - lane: `v5p-8` `r2`
   - artifact:
     `gs://marin-us-east5/eval/marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r2/inference-1c06ed`
   - job:
     `/ahmed/bloom-eval-azero-lr5e6-s0-v5p8-r2/eval-marin_dpo_lora_azero_lr5e6_seed0_step1699_bloom_speceval_azero_s0_v5p8_r2-inference_fc87fb81-0b6a6e4f`
   - status: `JOB_STATE_SUCCEEDED`
   - sanity: `7728` JSONL rows, `completion_count=7728`,
     `request_prompt_count=2576`, `output_token_count=2427069`

### Post-stop verification/download plan: 2026-04-25T22:13Z

User asked to double-check the selected inference outputs and download the
inference JSONL locally. Verification plan:

1. Re-list each selected GCS artifact and require two `shard_*.jsonl.gz` files
   plus `artifacts/vllm_metrics.json`.
2. Re-count JSONL rows from GCS for each artifact; expected count is `7728`
   rows = `2576` prompts * `n=3`.
3. Re-read `vllm_metrics.json` and require `completion_count=7728` and
   `request_prompt_count=2576`.
4. Download the selected shard files into
   `scratch/validate_bloom_azero_inference_20260425_2213Z/`.
5. Build a local decompressed `inference.jsonl` per target and run a local row
   count plus JSON parse check before treating the download as usable for
   judging.

### Post-stop verification/download result: 2026-04-25T22:17Z

Final GCS recheck passed for all four selected inference artifacts:

- each artifact has exactly two `shard_*.jsonl.gz` files
- each artifact has `artifacts/vllm_metrics.json` and `.artifact`
- each artifact has `7728` JSONL rows
- each metrics file reports `completion_count=7728` and
  `request_prompt_count=2576`

Local download location:
`scratch/validate_bloom_azero_inference_20260425_2213Z/`

The local tree contains 17 files and is about `102M`:

- four combined decompressed `inference.jsonl` files
- eight raw downloaded `shard_*.jsonl.gz` files
- four `vllm_metrics.json` files
- `manifest.tsv`

Additional local validation passed:

- every `inference.jsonl` parses as JSONL with `jq`
- every target has `7728` rows and `2576` unique prompts
- every target has `7728` unique `(prompt_id, sample_idx)` pairs
- every prompt has exactly three samples
- every target has sample distribution `0:2576,1:2576,2:2576`
- every target has `0` empty `response_text` values
- observed JSONL schema:
  `behavior_id`, `config_id`, `model`, `prompt_id`, `response_text`, `rubric`,
  `sample_idx`, `system_prompt`, `user_message`

Local manifest:
`scratch/validate_bloom_azero_inference_20260425_2213Z/manifest.tsv`

Manifest rows:

| target | local JSONL | rows | unique prompts | sha256 |
| --- | --- | ---: | ---: | --- |
| `azero_lr1e6_seed0_step1699` | `scratch/validate_bloom_azero_inference_20260425_2213Z/azero_lr1e6_seed0_step1699/inference.jsonl` | 7728 | 2576 | `25cfadb33c27748b560e8aa35b6e1db8e74a9e1f7145bc65cd070cce1982daa7` |
| `azero_lr1e5_seed0_step1699` | `scratch/validate_bloom_azero_inference_20260425_2213Z/azero_lr1e5_seed0_step1699/inference.jsonl` | 7728 | 2576 | `15b170337860aa33dfb6dd1c55d5189042bf1345b2574d6edcd1d05a9a1e47c8` |
| `azero_lr8p75e6_seed0_step1699` | `scratch/validate_bloom_azero_inference_20260425_2213Z/azero_lr8p75e6_seed0_step1699/inference.jsonl` | 7728 | 2576 | `c0b2f26bda778327afafd0384542074057d2dc6b79c6a899318478d626772fc8` |
| `azero_lr5e6_seed0_step1699` | `scratch/validate_bloom_azero_inference_20260425_2213Z/azero_lr5e6_seed0_step1699/inference.jsonl` | 7728 | 2576 | `87bef98dc4c6c967b5ea6f77c4914a772451f1745667d43ad5cd856fa16c89a9` |

### A0-JUDGE-004 — GPT-4.1 Batch API judge pilot: 2026-04-25T22:30Z

User asked to start LM-as-judge submission locally through the OpenAI Batch
API, not Iris, and to submit only one model first. Picked
`azero_lr1e6_seed0_step1699` because `A0-JUDGE-001` identified it as the
highest-information checkpoint: it tests the surprising A=0 learning-rate
inversion case first.

Implementation:

- added local-only submitter:
  `experiments/posttrain/judge_bloom_gpt41_batch.py`
- script reuses the same prompt constructors as the prior Bloom-compatible
  GPT-4.1 judge path:
  - `build_judge_system_prompt()`
  - `build_compliance_judge_prompt(...)`
  - audit also compares every request's messages against
    `marin.alignment.judge.build_judge_messages(...)`
- request body intentionally matches the historical direct GPT-4.1 judge
  settings:
  - `model="gpt-4.1-2025-04-14"`
  - `temperature=0.0`
  - `max_tokens=4000`
  - messages: exactly system + user
  - no `response_format`
  - no reasoning-model fields
- no Iris command is used; this is local OpenAI Batch API submission.

Dry-run/build command:

```bash
uv run python experiments/posttrain/judge_bloom_gpt41_batch.py build \
  --target azero_lr1e6_seed0_step1699
```

Dry-run result:

- source records: `7728`
- batch requests: `7698`
- skipped no rubric: `30`
- skipped no statement: `0`
- skipped no response: `0`
- prompt mismatches vs `build_judge_messages`: `0`
- request file:
  `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/requests.jsonl`
- request size: `55.3 MB`, below OpenAI Batch API's `200 MB` input cap
- unique custom IDs: `7698`
- first custom ID: `azero_lr1e6_seed0_step1699::0000000`
- last custom ID: `azero_lr1e6_seed0_step1699::0007697`
- request SHA-256:
  `2c4ac558d68cb8282603fe471d953da1a4267e48c5f330e6591bd77530d05776`

Submission command:

```bash
set -a; source .env; set +a
uv run python experiments/posttrain/judge_bloom_gpt41_batch.py submit \
  --target azero_lr1e6_seed0_step1699
```

Submission result:

- OpenAI Batch API input file:
  `file-S33BgavgFWnja4vP3bkRVT`
- OpenAI Batch API batch:
  `batch_69ed40d12400819097151585409b820e`
- initial status: `validating`
- refreshed status at `2026-04-25T22:31Z`: `validating`
- refreshed status at `2026-04-25T22:32Z`: `validating`
- request count: `7698`
- endpoint: `/v1/chat/completions`
- completion window: `24h`
- local batch directory:
  `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/`
- local state file:
  `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/batch_state.json`

Only this one target was submitted. The remaining three A=0 inference artifacts
have not been submitted for judging yet.

### A0-JUDGE-005 — Local Batch API monitor: 2026-04-25T22:42Z

User asked to set up a monitor that checks the GPT-4.1 Batch API judge job once
every 10 minutes. This remains local-only; no Iris job was launched.

Monitor plan:

- monitor exactly the single submitted pilot batch:
  `batch_69ed40d12400819097151585409b820e`
- run the local checker at `600s` cadence in a detached `tmux` session
- append every observed batch state to a JSONL event log
- keep the latest observed state in a small JSON status file
- exit automatically when the batch reaches a terminal state
  (`completed`, `failed`, `expired`, or `cancelled`)

Monitor implementation:

- script:
  `scratch/monitor_openai_batch_azero_gpt41.py`
- tmux session:
  `azero_gpt41_batch_monitor`
- state file:
  `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/batch_state.json`
- latest status:
  `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/monitor_status.json`
- event log:
  `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/monitor_events.jsonl`
- process log:
  `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/monitor.log`

Initial monitor observations:

- `2026-04-25T22:42:03Z`: `in_progress`, `7697/7698` completed, `0` failed
- `2026-04-25T22:42:22Z`: `in_progress`, `7697/7698` completed, `0` failed

The monitor is expected to check again around `2026-04-25T22:52Z` unless the
batch reaches a terminal state before then.

Terminal update:

- `2026-04-25T22:57:40Z`: `completed`, `7698/7698` completed, `0` failed
- output file:
  `file-7p6k1eVRfd7jtudrSo4j2b`
- error file: none
- detached monitor session `azero_gpt41_batch_monitor` was stopped after the
  terminal state was written to the local monitor status/event files.

### A0-JUDGE-006 — GPT-4.1 Batch output and prompt parity audit: 2026-04-25T23:00Z

User asked for a careful validation pass before treating the new Batch API
judge output as comparable to earlier GPT-4.1 LM-as-judge runs.

Audit plan:

1. Download the completed OpenAI Batch API output locally for the single pilot
   target `azero_lr1e6_seed0_step1699`.
2. Parse every output entry and verify:
   - all `7698` requests returned exactly one successful response
   - no Batch API per-item errors
   - judge JSON parses with the same parser used by
     `experiments/posttrain/run_bloom_judge.py`
   - usage metadata has no unexpected reasoning-token or model-setting fields
3. Compare request prompts and request settings against the earlier
   Bloom-compatible GPT-4.1 judge path:
   - `experiments/posttrain/run_bloom_judge.py`
   - `marin.alignment.judge.build_judge_messages(...)`
   - `marin.alignment.prompts.judge.build_*`
4. Inspect the original Bloom code under `/Users/ahmed/code/bloom/` and compare
   system/user prompt text where available.
5. Download earlier GPT-4.1 judged GCS artifacts only if needed to compare
   result schema, counts, and aggregation behavior against a completed direct
   GPT-4.1 run.

No new judging job will be launched during this audit.

Audit result:

- downloaded OpenAI Batch API output locally:
  `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/output.jsonl`
- added reproducible audit/parser script:
  `scratch/audit_azero_gpt41_batch.py`
- generated normalized artifacts:
  - `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/judged_results.jsonl`
  - `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/summary.json`
  - `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/summary_legacy_parsefail_score5.json`
  - `scratch/validate_bloom_azero_judge_gpt41_batch_20260425_2218Z/azero_lr1e6_seed0_step1699/parity_audit.json`

Transport/output checks:

- requests: `7698`
- outputs: `7698`
- unique output ids: `7698`
- missing outputs: `0`
- extra outputs: `0`
- duplicate outputs: `0`
- status codes: `7698` HTTP 200
- per-item Batch API errors: `0`
- output model: `gpt-4.1-2025-04-14` for all `7698`
- finish reasons: `7696` stop, `2` length
- usage totals:
  - prompt tokens: `11292794`
  - completion tokens: `1985771`
  - total tokens: `13278565`
  - cached tokens: `0`
  - reasoning tokens: `0`

Request/prompt checks:

- request body keys were exactly
  `model`, `messages`, `max_tokens`, `temperature` for all `7698`
- model: `gpt-4.1-2025-04-14`
- temperature: `0.0`
- max tokens: `4000`
- no `response_format`
- no reasoning-model fields
- message roles: exactly `system`, `user`
- prompt mismatch against current Marin
  `marin.alignment.judge.build_judge_messages(...)`: `0/7698`

Important prompt caveat:

- prompt mismatch against the current Bloom checkout
  `/Users/ahmed/code/bloom/src/bloom/speceval/prompts.py`: `7698/7698`
- every mismatch is explained by exactly one blank line after the calibration
  examples and before the `---` separator preceding `# Response to Evaluate`
- after normalizing only that single blank line, Bloom prompt mismatch is
  `0/7698`
- this means the new Batch API run is exact against the current Marin direct
  judge path (`run_bloom_judge.py` / `build_judge_messages`) but not
  character-for-character exact against the original Bloom prompt source in the
  local Bloom checkout.

Parse/postprocessing checks:

- raw Batch API responses with a parseable integer score: `7532`
- parse failures under the current post-Apr-11 parser: `166`
- parse failures are concentrated in:
  - `support_programmatic_use`: `79`
  - `formatting`: `77`
  - `avoid_errors`: `4`
  - six other one-off rows
- most failures are malformed JSON strings in judge explanations/highlights,
  not transport failures:
  - `140` unterminated string
  - `21` invalid escape
  - `3` expecting value
  - `2` no JSON found

Downloaded earlier GCS GPT-4.1 artifacts for comparison:

```text
scratch/validate_bloom_prior_gpt41_judge_artifacts_20260425/
  lora_lr1e5_b64_summary.json
  lora_lr1e5_b64_judged_results.jsonl
  lora_lr5e6_b64_summary.json
  lora_lr5e6_b64_judged_results.jsonl
  full_dpo_beta01_b64_summary.json
  full_dpo_beta01_b64_judged_results.jsonl
```

Comparison to earlier stored summaries:

| Target / postprocessing | Mean | Compliance | CI95 | Prompts | Responses | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A=0 `lr=1e-6`, current parser drops parse failures | `8.4851` | `0.8337` | `0.0786` | `2519` | `7532` | `166` |
| A=0 `lr=1e-6`, legacy parse-failure-as-5 | `8.4144` | `0.8172` | `0.0792` | `2566` | `7698` | `0` |
| Prior B=0 LoRA `lr=1e-5`, stored summary | `8.5531` | `0.8383` | `0.0799` | `2566` | `7697` | `1` |
| Prior B=0 LoRA `lr=5e-6`, stored summary | `8.5040` | `0.8254` | `0.0811` | `2566` | `7690` | `8` |
| Prior full DPO `beta=0.1`, stored summary | `8.4443` | `0.8211` | `0.0798` | `2566` | `7678` | `20` |

Interpretation:

- The OpenAI Batch API submission itself is clean and used the intended GPT-4.1
  settings.
- The request prompts are exact relative to the Marin direct judge path used by
  `experiments/posttrain/run_bloom_judge.py`.
- There is a real one-blank-line drift relative to the current original Bloom
  prompt source. It is probably semantically negligible, but it is not
  character-for-character exact.
- The bigger comparison hazard is parse-failure postprocessing. Historical
  stored summaries mostly include malformed judge JSON as score `5`, while the
  current parser emits `None` and drops those rows. For apples-to-apples
  comparison against stored historical summaries, use
  `summary_legacy_parsefail_score5.json` or rerun the baselines with the current
  parser/output preservation.

Correction / exact-prompt rebuild:

The user clarified that prompt equality must be byte-for-byte exact and that a
single blank line difference is unacceptable. Treat the already-submitted
Batch API pilot as **not exact to the Bloom GPT-4.1 prompt source**.

Root cause:

- `lib/marin/src/marin/alignment/prompts/judge.py` matched Marin's
  `build_judge_messages(...)`, but was missing one blank line present in
  `/Users/ahmed/code/bloom/src/bloom/speceval/prompts.py`.
- Bloom blame shows that blank line was present in the Bloom prompt source
  before the March 2026 GPT-4.1 judge runs.

Fix applied locally:

- restored the Bloom blank line in
  `lib/marin/src/marin/alignment/prompts/judge.py`
- removed a duplicated `calibration = format_examples_for_calibration(...)`
  assignment while touching that function

Dry-run exact rebuild, not submitted:

```bash
uv run python experiments/posttrain/judge_bloom_gpt41_batch.py build \
  --target azero_lr1e6_seed0_step1699 \
  --job-root scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z
```

Exact-prompt audit for the rebuilt request file:

- local request file:
  `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/requests.jsonl`
- requests: `7698`
- manifest rows: `7698`
- mismatch against `/Users/ahmed/code/bloom/src/bloom/speceval/prompts.py`:
  `0/7698`
- request setting errors: `0`
- system prompt hash:
  `2c72acc8fbd081f8fd29b3cf97dfa6f9ada271309774f6451adadeb724427108`
- request file SHA-256:
  `3368b093a180e85074d2b11a5b9760dc812fe6621128a06ccb72e60be03b3a37`

Do not submit additional GPT-4.1 Batch API jobs until this exact rebuilt request
file is accepted as the correct source of truth.

### A0-JUDGE-007 — Exact-prompt GPT-4.1 Batch resubmission: 2026-04-25T23:10Z

User asked to resubmit the pilot after confirming that the first Batch API
submission differed from Bloom's GPT-4.1 prompt source by one blank line.

Execution constraints:

- submit exactly one target first:
  `azero_lr1e6_seed0_step1699`
- submit locally through OpenAI Batch API
- do not launch Iris
- do not submit any other A=0 targets until this corrected pilot is verified
- require byte-for-byte prompt equality against
  `/Users/ahmed/code/bloom/src/bloom/speceval/prompts.py`

Pre-submit checks to run immediately before submission:

- rebuild request file under:
  `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/`
- verify `7698/7698` request prompts match Bloom source exactly
- verify request body keys are exactly
  `model`, `messages`, `max_tokens`, `temperature`
- verify model `gpt-4.1-2025-04-14`, `temperature=0.0`, `max_tokens=4000`
- verify no `response_format` or reasoning fields
- verify request SHA before submit

Pre-submit verification result:

- requests: `7698`
- manifest rows: `7698`
- unique custom ids: `7698`
- duplicate custom ids: `0`
- Bloom prompt mismatches: `0`
- request setting errors: `0`
- body key set:
  `('max_tokens', 'messages', 'model', 'temperature')` for all `7698`
- message roles:
  `('system', 'user')` for all `7698`
- system prompt hash:
  `2c72acc8fbd081f8fd29b3cf97dfa6f9ada271309774f6451adadeb724427108`
- request file SHA-256:
  `3368b093a180e85074d2b11a5b9760dc812fe6621128a06ccb72e60be03b3a37`
- request file size:
  `57970134` bytes

Submission command:

```bash
set -a; source .env; set +a
uv run python experiments/posttrain/judge_bloom_gpt41_batch.py submit \
  --target azero_lr1e6_seed0_step1699 \
  --job-root scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z
```

Submission result:

- OpenAI Batch API batch:
  `batch_69ed4c2cb47481909cb2d9d8fbe64ad2`
- OpenAI input file:
  `file-MxtZAaiiRW6ADjgKNGzS8G`
- initial status: `validating`
- refreshed status at `2026-04-25T23:20:28Z`: `in_progress`
- request count: `7698`
- endpoint: `/v1/chat/completions`
- completion window: `24h`
- output file: none yet
- error file: none yet

Post-submit uploaded-input verification:

- downloaded OpenAI input file bytes to:
  `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/uploaded_input.jsonl`
- local request bytes: `57970134`
- uploaded input bytes: `57970134`
- local SHA-256:
  `3368b093a180e85074d2b11a5b9760dc812fe6621128a06ccb72e60be03b3a37`
- uploaded SHA-256:
  `3368b093a180e85074d2b11a5b9760dc812fe6621128a06ccb72e60be03b3a37`
- byte equality: `true`

Monitor:

- local `tmux` session:
  `azero_gpt41_bloomexact_monitor`
- monitor cadence: `600s`
- monitor state path:
  `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/batch_state.json`
- monitor log:
  `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/monitor.log`
- first monitor tick at `2026-04-25T23:20:43Z`:
  `in_progress`, `0/7698` completed, `0` failed

### A0-JUDGE-008 — Handoff for monitoring exact GPT-4.1 Batch pilot: 2026-04-25T23:38Z

Purpose of this entry: preserve enough state for the next agent to continue
without re-discovering the prompt mismatch or accidentally using the invalid
pilot output.

Critical facts:

- The first Batch API pilot is **invalid for exact prompt comparison** because
  it was missing one blank line relative to Bloom's historical GPT-4.1 judge
  prompt source.
  - invalid batch:
    `batch_69ed40d12400819097151585409b820e`
  - invalid input file:
    `file-S33BgavgFWnja4vP3bkRVT`
  - do not use that output for the exact Bloom GPT-4.1 comparison.
- The corrected Bloom-exact pilot was submitted locally through OpenAI Batch
  API, not Iris.
  - corrected batch:
    `batch_69ed4c2cb47481909cb2d9d8fbe64ad2`
  - corrected input file:
    `file-MxtZAaiiRW6ADjgKNGzS8G`
  - target:
    `azero_lr1e6_seed0_step1699`
  - target inference:
    `scratch/validate_bloom_azero_inference_20260425_2213Z/azero_lr1e6_seed0_step1699/inference.jsonl`
  - job root:
    `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/`

Prompt exactness:

- `lib/marin/src/marin/alignment/prompts/judge.py` was patched to match
  `/Users/ahmed/code/bloom/src/bloom/speceval/prompts.py` byte-for-byte for the
  judge user prompt.
- Exact verifier result before submission:
  - requests: `7698`
  - manifest rows: `7698`
  - unique custom ids: `7698`
  - Bloom prompt mismatches: `0`
  - request setting errors: `0`
  - request body keys exactly:
    `model`, `messages`, `max_tokens`, `temperature`
  - message roles exactly:
    `system`, `user`
  - model:
    `gpt-4.1-2025-04-14`
  - temperature:
    `0.0`
  - max tokens:
    `4000`
  - no `response_format`
  - no reasoning fields
  - request SHA-256:
    `3368b093a180e85074d2b11a5b9760dc812fe6621128a06ccb72e60be03b3a37`
- Post-submit uploaded-input verification also passed:
  - local request bytes: `57970134`
  - uploaded input bytes: `57970134`
  - uploaded input SHA-256:
    `3368b093a180e85074d2b11a5b9760dc812fe6621128a06ccb72e60be03b3a37`
  - byte equality between local request file and OpenAI input file: `true`
  - uploaded input copy:
    `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/uploaded_input.jsonl`

Current status at `2026-04-25T23:38Z`:

- corrected batch status: `finalizing`
- request counts: `7698/7698` completed, `0` failed
- output file: none yet
- error file: none yet
- direct status command just run:

```bash
set -a; source .env; set +a
uv run python experiments/posttrain/judge_bloom_gpt41_batch.py status \
  --target azero_lr1e6_seed0_step1699 \
  --job-root scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z
```

Monitor:

- active local tmux session:
  `azero_gpt41_bloomexact_monitor`
- monitor process observed:
  pane command `uv`
- cadence:
  `600s`
- state:
  `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/batch_state.json`
- latest status:
  `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/monitor_status.json`
- event log:
  `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/monitor_events.jsonl`
- process log:
  `scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/azero_lr1e6_seed0_step1699/monitor.log`

Monitor history so far:

- `2026-04-25T23:20:43Z`: `in_progress`, `0/7698` completed, `0` failed
- `2026-04-25T23:30:44Z`: `in_progress`, `5832/7698` completed, `0` failed
- `2026-04-25T23:35:04Z`: `finalizing`, `7698/7698` completed, `0` failed

Next-agent steps:

1. Check whether the corrected batch has produced an output file:

```bash
set -a; source .env; set +a
uv run python experiments/posttrain/judge_bloom_gpt41_batch.py status \
  --target azero_lr1e6_seed0_step1699 \
  --job-root scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z
```

2. If status is still `finalizing`, keep the `azero_gpt41_bloomexact_monitor`
   tmux session running. Do not launch another batch.

3. If status is `completed`, stop the monitor and download the output:

```bash
/opt/homebrew/bin/tmux kill-session -t azero_gpt41_bloomexact_monitor
set -a; source .env; set +a
uv run python - <<'PY'
from pathlib import Path
from openai import OpenAI
import sys

sys.path.insert(0, str(Path("experiments/posttrain").resolve()))
import batch_lib as bl

job_dir = Path(
    "scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z"
    "/azero_lr1e6_seed0_step1699"
)
entries = bl.collect(OpenAI(), job_dir)
print({"entries": len(entries), "output_path": str(job_dir / "output.jsonl")})
PY
```

4. After downloading, run a parity/output audit before trusting the results.
   The existing audit script is currently hardcoded to the invalid pilot job
   root, so either update it to point at the `bloomexact` job root or copy it
   and change `JOB_DIR`.

5. The audit should verify at minimum:
   - `7698` output entries
   - all custom ids present exactly once
   - all status codes are 200
   - per-item errors are 0
   - output model is `gpt-4.1-2025-04-14`
   - request prompts still have `0/7698` mismatches against Bloom source
   - request SHA is still
     `3368b093a180e85074d2b11a5b9760dc812fe6621128a06ccb72e60be03b3a37`
   - parse failures are counted and handled consistently with whatever
     comparison summary is used

6. Only after the corrected pilot output is downloaded and audited should we
   decide whether to submit the other three A=0 targets.

### A0-JUDGE-009 — Corrected pilot completion, download, and audit: 2026-04-25T23:46Z

Picked up the handoff. Polled the OpenAI Batch API directly: corrected pilot
batch `batch_69ed4c2cb47481909cb2d9d8fbe64ad2` flipped to `completed` at
`2026-04-25T23:46:27Z`, 7698/7698 completed, 0 failed, output file
`file-7q5kP1HRRgL2FA8T95i2oN`. Stopped tmux monitor
`azero_gpt41_bloomexact_monitor` and downloaded the output via
`bl.collect(...)`.

Audit (run via `scratch/audit_azero_gpt41_batch_bloomexact.py`, a copy of
`scratch/audit_azero_gpt41_batch.py` with `JOB_DIR` repointed to the
bloomexact root):

- outputs: `7698`, unique custom ids: `7698`, missing/extra/duplicate: `0/0/0`
- status codes: `7698` HTTP 200, per-item Batch errors: `0`
- output model: `gpt-4.1-2025-04-14` for all `7698`
- finish reasons: `7696` stop, `2` length
- request body keys exactly `model, messages, max_tokens, temperature` for all
  `7698`
- model: `gpt-4.1-2025-04-14`, temperature: `0.0`, max tokens: `4000`
- no `response_format`, no reasoning fields
- message roles: exactly `system, user`
- prompt mismatch against current Marin
  `marin.alignment.judge.build_judge_messages(...)`: `0/7698`
- prompt mismatch against original Bloom source
  `/Users/ahmed/code/bloom/src/bloom/speceval/prompts.py`: `0/7698`
- usage totals: prompt `11293058`, completion `1959533`, total `13252591`,
  cached `0`, reasoning `0`
- parse failures under current parser: `171` (vs `166` in invalid pilot)

Headline summary numbers (corrected pilot, target
`azero_lr1e6_seed0_step1699`):

| Parser | Mean | Compl. | CI95 | N_resp | N_prompts |
| --- | ---: | ---: | ---: | ---: | ---: |
| current parser (drops parse failures) | `8.4918` | `0.8339` | `0.0788` | `7527` | `2516` |
| legacy parsefail-as-5 | `8.4163` | `0.8161` | `0.0795` | `7698` | `2566` |

Sensitivity check vs the invalid (one-blank-line drift) pilot, same inference,
same scoring:

- current parser: `8.4918` vs `8.4851` → Δ = `+0.007`
- legacy: `8.4163` vs `8.4144` → Δ = `+0.002`

The blank-line patch is essentially semantically null for the GPT-4.1 judge.

Comparison vs already-judged B=0 stored summaries (note: prior B=0 summaries
were generated by the pre-fix prompt path, so comparing legacy-vs-legacy is the
closest apples-to-apples we have without rejudging B=0 baselines):

| Run / parser | Mean | Δ vs B=0 lr=1e-5 |
| --- | ---: | ---: |
| **A=0 lr=1e-6 corrected legacy** | `8.4163` | **`-0.137`** |
| A=0 lr=1e-6 invalid legacy | `8.4144` | `-0.139` |
| Prior B=0 LoRA `lr=1e-5` (pre-fix) | `8.5531` | — |
| Prior B=0 LoRA `lr=5e-6` (pre-fix) | `8.5040` | `-0.049` |
| Prior full DPO `beta=0.1` (pre-fix) | `8.4443` | `-0.109` |

Substantive interpretation:

- A=0 `lr=1e-6` seed=0 = `8.4163` legacy. Prior B=0 LoRA `lr=1e-5` seed=0
  = `8.5531`. Gap = `-0.137 ≈ 1.7× CI95`.
- A=0 `lr=1e-6` also lands below B=0 `lr=5e-6` (`-0.088`) and barely above
  full DPO (`-0.028`).
- The "A=0 train-acc dominance carries to GPT-4.1 judge" hypothesis is **not
  supported** at this single seed=0 data point at the most-favorable A=0 LR.
  The B=0-only "train-acc tracks judge" prior used by the earlier
  Codex/GPT-5 analysis does **not** carry to A=0.
- Caveats: single training seed (seed=0); single LR; mixed prompt regime
  (corrected pilot is post-fix, prior B=0 is pre-fix; sensitivity check shows
  this is worth ~`0.007` mean shift, far smaller than the gap).

Decision applied: per the gate in `A0-JUDGE-008`, audit cleared, so submit the
remaining three A=0 seed=0 targets.

### A0-JUDGE-010 — Remaining three A=0 seed=0 targets submitted (Bloom-exact): 2026-04-25T23:55Z

User authorization: "submit the remaining three at seed=0 please use the code
codex made and be careful to make sure the prompts are all correct."

Targets, all using inference artifacts under
`scratch/validate_bloom_azero_inference_20260425_2213Z/`:

- `azero_lr1e5_seed0_step1699` (old recommended LoRA LR under new init; same-LR
  contrast vs already-judged B=0 `lora_lr1e5`)
- `azero_lr8p75e6_seed0_step1699` (high-LR plateau / strong-margin probe;
  tests whether judge follows margin/loss instead of accuracy)
- `azero_lr5e6_seed0_step1699` (same-LR contrast vs already-judged B=0
  `lora_lr5e6`)

Pre-submit verifier:

- new helper script: `scratch/verify_bloomexact_pre_submit.py`
- imports `bloom.speceval.prompts.build_compliance_judge_prompt` and
  `build_judge_system_prompt` from `/Users/ahmed/code/bloom/src` and rebuilds
  every request prompt from the canonical Bloom Statement objects, comparing
  byte-for-byte against the request file
- also enforces body keys, model, temperature, max_tokens, message roles, and
  unique custom ids

Build + Bloom-exact verifier results (all three PASS):

| Target | requests | bloom_user_mismatches | bloom_system_mismatches | settings_errors | request SHA-256 | bytes |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `azero_lr1e5_seed0_step1699` | `7698` | `0` | `0` | `0` | `7f299c879af23b409234ca2fef655c4b01c6ea72fa8712b766d0b149e287494a` | `60728263` |
| `azero_lr8p75e6_seed0_step1699` | `7698` | `0` | `0` | `0` | `d425ae019787c3cc904f9a3506f094fab1f606798813c9d55824d152c1085467` | `60680268` |
| `azero_lr5e6_seed0_step1699` | `7698` | `0` | `0` | `0` | `d1419c600548d9b2a1edc833fd0458d3d698e5e4f607651023302175eae84367` | `60142625` |

System prompt hash for all three:
`2c72acc8fbd081f8fd29b3cf97dfa6f9ada271309774f6451adadeb724427108` — same as
the corrected lr=1e-6 pilot.

Submitted via `judge_bloom_gpt41_batch.py submit ... --job-root <bloomexact>`:

| Target | batch_id | input_file_id | local↔uploaded byte equality |
| --- | --- | --- | --- |
| `azero_lr1e5_seed0_step1699` | `batch_69ed5459b3848190859c8da88cad95ff` | `file-Xv5hYQsijHc69G4zoRqxxV` | `True` |
| `azero_lr8p75e6_seed0_step1699` | `batch_69ed5483d45881909dd2ba4aa5e337fa` | `file-SGjUfs9DgFqtyDkmM9H3wJ` | `True` |
| `azero_lr5e6_seed0_step1699` | `batch_69ed54993b10819081f530bbcaffd899` | `file-6TWTdZ9hjpufHTFqpxke35` | `True` |

For each, the uploaded input file was downloaded via
`client.files.content(...)` and compared SHA-256 + byte length against the
local `requests.jsonl` SHA recorded by the pre-submit verifier; all three
matched exactly. Uploaded copies saved as `uploaded_input.jsonl` per target.

Monitors (one tmux session per batch, 600s cadence, reuses Codex's
`scratch/monitor_openai_batch_azero_gpt41.py`):

- `azero_gpt41_bloomexact_lr1e5_monitor`
- `azero_gpt41_bloomexact_lr8p75e6_monitor`
- `azero_gpt41_bloomexact_lr5e6_monitor`

State files at
`scratch/validate_bloom_azero_judge_gpt41_batch_bloomexact_20260425_2310Z/<target>/batch_state.json`;
status JSON at `monitor_status.json`; event log at `monitor_events.jsonl`.
Initial monitor tick at `2026-04-25T23:56:53Z` shows all three at
`in_progress`, `0/7698` completed.

Next-agent steps mirror `A0-JUDGE-008`:

1. Wait for each batch to flip to `completed` (or terminal failure).
2. Stop the matching tmux session, refresh status to populate
   `output_file_id`, then download via `bl.collect(...)`.
3. Run the audit script — current `audit_azero_gpt41_batch_bloomexact.py` is
   pinned to `azero_lr1e6_seed0_step1699`; either override `JOB_DIR/TARGET`
   per target or factor those into CLI args.
4. Add three new rows to the comparison table (current-parser and legacy
   parsefail-as-5) and re-evaluate the A=0-vs-B=0 verdict across the LR sweep.
5. If results stay consistently below the matching B=0 baseline, A=0 likely
   does not ship as the LoRA-DPO default at seed=0; consider whether to spend
   on seed=2 (requires a new inference run — only seed=0 inference is
   currently downloaded).

### A0-JUDGE-011 — Remaining three corrected pilots completed, downloaded, and audited: 2026-04-26T00:25Z

All three batches submitted in `A0-JUDGE-010` reached terminal `completed`,
all 7698/7698 successful with 0 failures. Per-batch terminal times:

- `lr=5e-6`: completed by `2026-04-26T00:17:00Z` poll
  (output `file-EnBJQFstNz7orspZpGWQQm`)
- `lr=8.75e-6`: completed by `2026-04-26T00:18:26Z` poll
  (output `file-EDhUqkRTaA7NHFkNKzCDQX`)
- `lr=1e-5`: completed at `2026-04-26T00:25:07Z`
  (output `file-Lr2dDcFcLXudktgMSB9oqu`)

End-to-end submit-to-complete: `lr=5e-6` ≈ 22 min, `lr=8.75e-6` ≈ 23 min,
`lr=1e-5` ≈ 30 min. Each tmux monitor was stopped after its batch reached
terminal state; all three output files were downloaded via `bl.collect(...)`.

Audit: refactored `scratch/audit_azero_gpt41_batch_bloomexact.py` to read
`AUDIT_TARGET` and `AUDIT_JOB_DIR` env vars (default still
`azero_lr1e6_seed0_step1699`). Ran the audit on each of the three new outputs.
All three audits clean:

- 7698 outputs each, 0 missing/extra/duplicate
- all HTTP 200, 0 per-item Batch errors
- output model `gpt-4.1-2025-04-14` for all
- finish reasons: lr=5e-6 `7698 stop`; lr=8.75e-6 `7697 stop, 1 length`;
  lr=1e-5 `7695 stop, 3 length`
- request body keys exactly `model, messages, max_tokens, temperature`,
  temperature `0.0`, max_tokens `4000`
- prompt mismatch against current Marin
  `marin.alignment.judge.build_judge_messages(...)`: `0/7698` for all three
- prompt mismatch against original Bloom source
  `/Users/ahmed/code/bloom/src/bloom/speceval/prompts.py`: `0/7698` for all
  three
- system prompt hash for all three:
  `2c72acc8fbd081f8fd29b3cf97dfa6f9ada271309774f6451adadeb724427108`
- parse failures: lr=5e-6 `158`, lr=8.75e-6 `154`, lr=1e-5 `159`

Headline summary numbers across all four corrected A=0 seed=0 targets:

| Run | Mean (current parser) | Mean (legacy parsefail=5) | CI95 | N_resp (current) | N_prompts (current) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `azero_lr1e6_seed0_step1699` | `8.4918` | `8.4163` | `0.0788` | `7527` | `2516` |
| `azero_lr5e6_seed0_step1699` | `8.6153` | `8.5430` | `0.0788` | `7540` | `2519` |
| `azero_lr8p75e6_seed0_step1699` | `8.6368` | `8.5690` | `0.0785` | `7544` | `2520` |
| `azero_lr1e5_seed0_step1699` | `8.6443` | `8.5747` | `0.0782` | `7539` | `2521` |

Same-LR A=0 vs B=0 init contrasts (legacy regime, apples-to-apples with prior
B=0 stored summaries):

| LR | A=0 (corrected legacy) | B=0 (prior, pre-fix) | Δ (A=0 − B=0) |
| --- | ---: | ---: | ---: |
| `1e-5` | `8.5747` | `8.5531` | **`+0.022`** |
| `5e-6` | `8.5430` | `8.5040` | **`+0.039`** |

Substantive findings:

1. **A=0 ties or beats B=0 at every same-LR contrast** we have. Two
   data points (lr=1e-5 and lr=5e-6), both small-positive Δ, both within CI95
   but consistent in direction.
2. **A=0 lr=1e-6 — the A=0 train-eval-acc winner — is the A=0 judge loser.**
   At `8.4163` legacy it is `-0.137` below B=0 lr=1e-5 (≈ `1.7× CI95`) and
   `-0.084` below B=0 lr=5e-6. It is the only A=0 LR that substantially
   underperforms.
3. **Train-eval-acc is anti-correlated with GPT-4.1 judge across A=0 LRs.**
   By judge mean (legacy):
   `lr=1e-6 (8.42) < lr=5e-6 (8.54) < lr=8.75e-6 (8.57) ≈ lr=1e-5 (8.57)`.
   By A=0 train-eval-acc rank: `lr=1e-6` was *best*, `lr=8.75e-6` mid, etc.
   The train-acc-best LR is the judge-worst LR.
4. **The "A=0 shifts the recommended LR from 10× full-FT LR (1e-5) to 1×
   full-FT LR (1e-6)" claim from the A=0 sweep training analysis is wrong
   *for judge*.** A=0's judge-best LR is closer to the prior B=0 winner LR
   (`1e-5`), not the A=0 train-acc winner (`1e-6`).
5. **The Codex/GPT-5 prior that "train-eval-acc tracks judge under B=0,
   therefore use it to pick A=0 LRs too" does not survive A=0.**

Caveats:

- Single training seed (seed=0) for all four A=0 runs.
- Mixed prompt regime: A=0 corrected pilots use the post-fix Bloom-exact
  prompt; B=0 stored summaries use the pre-fix prompt. The corrected vs invalid
  lr=1e-6 pilot showed this regime difference is worth `+0.007` mean
  (current parser) / `+0.002` (legacy). All Δ-magnitudes above are far larger
  than that.
- All gaps except lr=1e-6 vs B=0 lr=1e-5 are within `1× CI95`. Only the lr=1e-6
  gap is statistically distinct on its own.
- 4 LRs is enough to see the qualitative inversion vs train-acc, not enough
  for a confident A=0-internal Spearman.

Recommendation:

- **A=0 likely ships as the LoRA-DPO default at lr=1e-5** (or `8.75e-6`),
  *not* at the train-acc-best `lr=1e-6`. The recommended A=0 LoRA-DPO LR
  matches the prior B=0 LoRA-DPO winner LR.
- Highest-info next data point is **A=0 lr=1e-5 seed=2** (or `lr=8.75e-6`
  seed=2): give the first across-training-seed variance estimate at the
  judge-best LR and let us decide whether the `+0.022` (or `+0.016`) Δ vs B=0
  is real or training-seed noise.
- Seed=2 inference is *not yet downloaded* — needs a new TPU inference run
  before the seed=2 judge can run.
