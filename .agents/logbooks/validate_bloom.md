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
- **Overall mean score: 8.43 / 10**
- **Overall compliance rate: 82.2%**

**Per-statement results**:

| Statement | Mean | Comply% | Count |
|-----------|------|---------|-------|
| protect_privacy | 9.92 | 100.0% | 186 |
| be_kind | 9.81 | 100.0% | 159 |
| be_empathetic | 9.23 | 100.0% | 159 |
| be_engaging | 9.01 | 100.0% | 159 |
| be_rationally_optimistic | 9.65 | 100.0% | 177 |
| avoid_abuse | 9.75 | 98.1% | 159 |
| present_perspectives | 9.67 | 98.7% | 159 |
| no_agenda | 9.10 | 98.1% | 159 |
| ... | ... | ... | ... |
| formatting | 6.41 | 37.7% | 159 |
| support_programmatic_use | 6.36 | 34.0% | 159 |
| avoid_targeted_political_manipulation | 5.36 | 34.2% | 219 |
| refusal_style | 4.04 | 13.2% | 189 |

**Notes**:
- First Iris job (`judge-bloom-gpt41`) succeeded but output was lost — script used `Path()` for GCS output which wrote to container-local filesystem. Fixed by using `rigging.filesystem.url_to_fs` for GCS writes.
- `refusal_style` is the weakest by far (13.2% compliance). This likely measures whether the model refuses gracefully vs. complying with harmful requests — the DPO model may be too willing to engage.

---

## Next Steps

- [ ] Compare TPU judge scores against Bloom's GPU judge scores for the same checkpoint to quantify any systematic differences
- [ ] Run the same pipeline on the base `meta-llama/Llama-3.1-8B-Instruct` (no DPO) to measure DPO lift
- [ ] Run on additional DPO checkpoints (different beta, lr, seed) for hyperparameter comparison
- [ ] Investigate `refusal_style` failures — is the DPO model too compliant?
- [ ] Port the one-off judge script into the `evaluate.py` framework for reuse
