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
