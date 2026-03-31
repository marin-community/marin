# Alignment Function: Research Logbook

**Issue:** https://github.com/marin-community/marin/issues/3355
**Branch:** `alignment_function`
**Experiment ID prefix:** `ALIGN`

## Scope
- **Goal:** Build a marin function that, given a pretrained model, teacher model, and specification/constitution, produces a synthetic preference dataset and trains an aligned model via DPO.
- **Primary metric(s):** Specification adherence score (SpecEval), DPO training loss convergence, preference pair quality (chosen vs rejected margin).
- **Constraints:** Must integrate with existing marin Executor framework, preference data format (sharded `.jsonl.gz`), and Levanter DPO training pipeline.

## Stop Criteria
- A working end-to-end pipeline that takes a specification document and produces a DPO-aligned model.
- Validated on at least one specification set with measurable adherence improvement.
- Pipeline is composable with existing marin ExecutorSteps.

## Key References
- Bloom pipeline (external): prompt generation, inference, preference pair building, marin export
- CS229 project: validated pipeline on 46 OpenAI Model Spec statements, 30k+ preference pairs, +1.7 adherence improvement (DPO beta=0.01)
- SpecEval (arXiv:2509.02464): evaluation framework for behavioral specification adherence
- Existing DPO infra: `lib/levanter/src/levanter/main/train_dpo.py`, `experiments/posttrain/preference_datasets.py`

## Existing Infrastructure (Baseline)
- **DPO training:** Fully implemented in Levanter (`train_dpo.py`), supports frozen reference model, beta tuning, validation splits.
- **Preference data format:** `PreferenceTransformAdapter` converts to OpenAI chat messages with chosen/rejected pairs. Sharded to `.jsonl.gz`.
- **Preference datasets:** UltraFeedback, OLMo-2 registered. Download → transform → tokenize pipeline works.
- **What's missing:** No synthetic preference data generation from specifications. No prompt generation, no teacher model inference for chosen/rejected responses, no specification-to-dataset pipeline.

## Architecture Plan

Full integration plan documented in `/Users/ahmed/code/bloom/ALIGNMENT_FUNCTION_MARIN.md`.

### Target API
```python
aligned_model_steps = align(
    name="my_alignment_run",
    pretrained_model=some_pretrained_checkpoint_step,
    spec="path/to/openai_model_spec.jsonl",
    model_config=llama_8b,
    teacher_model="openai/gpt-4.1",
    align_config=AlignConfig(...),
    rejected_model=some_weak_model_step,  # or None → pretrained_model
)
```

### 4 Model Roles
1. **Pretrained model** (`ExecutorStep`) — DPO init + reference only. Never used for data generation.
2. **Teacher model** (`ExecutorStep | str`) — Generates chosen responses WITH spec guidance.
3. **Rejected model** (`ExecutorStep | str | None`) — Generates rejected responses WITHOUT spec guidance. Defaults to pretrained_model.
4. **Ideation model** (`str`, on `AlignConfig`) — Prompt generation via Bloom stages 1-3 (always API).
5. **Judge model** (`str`, on `AlignConfig`) — Scores responses against rubrics, filters bad pairs (always API).

### Pipeline Steps (5 ExecutorSteps)

**Step 1: `generate_prompts_from_spec`** — Spec → diverse eval prompts
- Port Bloom's 3-stage pipeline: understanding → covering arrays → concretization → extraction
- Source: `bloom/coverage.py` (greedy set-cover, no deps), `bloom/synthetic_pipeline/stage{1,2,3}.py`
- Output: per-behavior `eval_prompts.json` with system_prompt, user_message, rubric, axis_config, tags

**Step 2a: `generate_responses` (teacher)** — Prompts + spec guidance → chosen responses
- System prompt = scenario + "Follow this behavioral guideline: {statement}"
- Dispatch: `str` → API (litellm/OpenAI), `ExecutorStep` → vLLM (`BaseInferenceContext.batch_completions`)

**Step 2b: `generate_responses` (rejected)** — Prompts only (NO spec guidance) → rejected responses
- Same fn, different config. `n=4` responses per prompt (judge picks worst).
- Steps 2a and 2b run in parallel.

**Step 3: `judge_and_build_pairs`** — Score + filter + pair
- Score teacher response (must be >= 7.0), score each rejected response, pick worst
- Filter: chosen_score >= `min_chosen_score`, gap >= `min_gap` (default 2.0)
- Output: sharded `.jsonl.gz` in marin preference format (chosen/rejected OpenAI messages)
- Key design: spec guidance STRIPPED from training data. Model learns to internalize behavior.

**Step 4: `default_tokenize`** — Existing marin infra, `PreferenceChatLmDatasetFormat`

**Step 5: `default_dpo`** — Existing marin infra, pretrained_model as init + reference

### Module Structure
```
lib/marin/src/marin/alignment/
├── __init__.py
├── align.py                   # align(), AlignConfig
├── generate_prompts.py        # Step 1 (Bloom stages 1-3)
├── generate_responses.py      # Steps 2a/2b (API + vLLM dispatch)
├── judge.py                   # Step 3 (rubric scoring + preference pair construction)
├── coverage.py                # Covering array algorithm (direct port from Bloom)
└── prompts/
    ├── __init__.py
    ├── understanding.py       # Stage 1 prompt templates
    ├── concretize.py          # Stage 2 prompt templates
    └── extract.py             # Stage 3 prompt templates
```

### Bloom Source Files to Port
| Bloom file | Marin target | Notes |
|---|---|---|
| `src/bloom/coverage.py` | `alignment/coverage.py` | Direct port, no deps |
| `src/bloom/synthetic_pipeline/stage1.py` | `alignment/generate_prompts.py` | Port orchestration, adapt I/O for ExecutorStep |
| `src/bloom/synthetic_pipeline/stage2.py` | `alignment/generate_prompts.py` | Same file, after stage1 |
| `src/bloom/synthetic_pipeline/stage3.py` | `alignment/generate_prompts.py` | Same file, after stage2 |
| `src/bloom/prompts/step1_understanding.py` | `alignment/prompts/understanding.py` | Port templates + STANDARD_DEMOGRAPHIC_AXES |
| `src/bloom/prompts/concretize.py` | `alignment/prompts/concretize.py` | Port templates |
| `src/bloom/speceval/types.py` | `alignment/align.py` | Port Statement, StatementType, AuthorityLevel |
| `src/bloom/speceval/prompts.py` | `alignment/judge.py` | Port compliance judge prompts |
| `src/bloom/synthetic_pipeline/config.py` | `alignment/align.py` | Merge into AlignConfig |
| `src/bloom/utils.py:litellm_chat` | `alignment/generate_responses.py` | Port or use marin's OpenAI client pattern |

### Scale Estimates (46 statements, 3-way covering)
| Stage | Per statement | Total |
|---|---|---|
| Step 1: prompts | ~200-500 | ~10K-23K |
| Step 2a: teacher responses | 1 per prompt | ~10K-23K |
| Step 2b: rejected responses | 4 per prompt | ~40K-92K |
| Step 3: judge calls | 5 per prompt | ~50K-115K |
| Step 3: output pairs (60-80% filter pass) | — | ~6K-18K |

## Experiment Log

### 2026-03-16 — ALIGN-000: Research Thread Kickoff
- **Action:** Created branch `alignment_function`, research logbook, and initial plan.
- **Findings from codebase exploration:**
  - DPO training is production-ready (PR #2460).
  - Preference data pipeline handles download, transform, tokenization.
  - No synthetic data generation exists — this is the gap.
  - The Bloom pipeline (external repo) has all generation stages; integration work is wiring them into marin ExecutorSteps.
- **Next action:** Design the specification data model and prompt generation step.

### 2026-03-16 — ALIGN-001: Bloom Codebase Deep-Read
- **Action:** Read all key Bloom source files identified in `ALIGNMENT_FUNCTION_MARIN.md`.
- **Files read:**
  - `bloom/coverage.py` — Greedy set-cover covering array algorithm. Pure stdlib, no external deps. Direct port.
  - `bloom/synthetic_pipeline/stage1.py` — Understanding generation. Per-statement: LLM call → parse XML tags → extract variation axes. ThreadPoolExecutor for parallelism. Fingerprinting for skip-existing.
  - `bloom/synthetic_pipeline/stage2.py` — Concretization. Load axes → `generate_covering_configs()` → batch concretize via LLM → parse `<scenario>/<rubric>` XML. Saves covering_plan.json + ideation.json.
  - `bloom/synthetic_pipeline/stage3.py` — Prompt extraction. Batch extract system_prompt + user_message from scenario prose. Temperature=0.0. Saves eval_prompts.json.
  - `bloom/prompts/step1_understanding.py` — STANDARD_DEMOGRAPHIC_AXES (2 axes, 8 values each), behavior understanding prompt, transcript analysis prompt.
  - `bloom/prompts/concretize.py` — System prompt (scenario designer role), user prompt (behavior context + axes + configs to concretize).
  - `bloom/speceval/types.py` — `Statement`, `StatementType`, `AuthorityLevel`, `ComplianceResult`, `Example` dataclasses.
  - `bloom/speceval/prompts.py` — `build_compliance_judge_prompt()` (1-10 scale), `build_judge_system_prompt()`, calibration examples formatter.
  - `bloom/synthetic_pipeline/config.py` — `Stage1Config`, `Stage2Config`, `Stage3Config`, `PipelinePaths`, `SyntheticPipelineConfig`. YAML-based config loading.
  - `bloom/synthetic_pipeline/behavior_pack.py` — `build_behavior_pack()` reads spec JSONL → behaviors.json, configurable prompts, examples, manifest.
- **Key observations:**
  1. Coverage algorithm is self-contained, pure Python — port verbatim.
  2. All 3 stages use `litellm_chat()` with ThreadPoolExecutor — need to decide: port litellm dep or use marin's OpenAI client.
  3. XML tag parsing is consistent across stages — regex-based, simple.
  4. Fingerprinting system enables skip-existing — valuable for long-running pipeline, maps well to Executor versioning.
  5. Behavior pack builder reads spec JSONL → structured config dir. This could be Step 0 or handled inline.
  6. Stage configs map cleanly to AlignConfig fields.
- **Decision:** Start implementation with `coverage.py` (zero-risk port) and `align.py` (config + top-level function), then work through stages 1→2→3→judge in order.
- **Next action:** Begin implementation. Port `coverage.py`, define `AlignConfig` and `align()` signature, then implement `generate_prompts_from_spec`.

### 2026-03-16 — ALIGN-002: Core Implementation
- **Action:** Implemented full `marin.alignment` module (14 files, ~2150 lines). All pre-commit checks pass.
- **Commit:** 8e96e590c
- **Modules implemented:**
  - `alignment/coverage.py` — covering array algorithm (direct port from Bloom)
  - `alignment/types.py` — Statement, ComplianceResult, StatementType, AuthorityLevel
  - `alignment/llm_client.py` — unified litellm wrapper
  - `alignment/generate_prompts.py` — 3-stage pipeline (understanding → concretize → extract)
  - `alignment/generate_responses.py` — dual dispatch: litellm (API) + vLLM (local)
  - `alignment/judge.py` — compliance scoring + preference pair construction + quality filtering
  - `alignment/align.py` — top-level `align()` function + `AlignConfig`
  - `alignment/prompts/` — all prompt templates (understanding, concretize, extract, judge)
- **Dependency:** Added `litellm>=1.0.0` to `lib/marin/pyproject.toml`

### 2026-03-16 — ALIGN-003: Unit Tests
- **Action:** Added 58 unit tests covering all alignment submodules. All pass in <1s.
- **Commit:** 977f0daec
- **Coverage:** coverage algorithm (8), types (4), prompt parsing (12), response helpers (4), judge parsing (4), llm client (4), prompt templates (11), E2E prompt gen (1), E2E judge pair construction (3)

### 2026-03-16 — ALIGN-004: Experiment Script + Spec Data
- **Action:** Added `experiments/align_openai_spec.py` — full alignment experiment for Llama 3.1 8B on OpenAI Model Spec. Mirrors Bloom v2 config (3-way covering, GPT-4.1 teacher, GPT-4.1-mini rejected, beta=0.01). Bundled the 46-statement spec JSONL.
- **Commit:** 5e999aedd
- **Config:** Matches validated hyperparams from CS229 project (+1.7 adherence improvement).
- **Caching:** Every pipeline stage is an ExecutorStep — synthetic dataset persists across runs.

### 2026-03-16 — ALIGN-005: InferenceConfig Refactor
- **Action:** Replaced string-based model dispatch with explicit `InferenceConfig` type hierarchy.
- **Commit:** 84cb26e81
- **Design:**
  - `InferenceConfig` base class with `model`, `is_api`, `is_local`, `resources` properties
  - `LiteLLMConfig(InferenceConfig)` — API models: model ID, num_retries, workers → CPU resources
  - `VLLMConfig(InferenceConfig)` — local models: checkpoint path, tensor_parallel_size, max_model_len, tpu_type → TPU resources
  - Both accepted interchangeably for teacher/rejected model roles
  - `llm_chat()` still accepts bare strings (auto-wrapped as LiteLLMConfig) for convenience
- **Removed:** `_is_api_model()` heuristic, `isinstance(model, str)` checks in `align.py`, vLLM params from `ResponseGenConfig`
- **Tests:** 65 total (7 new InferenceConfig tests + 1 new string-config test)
- **Next action:** End-to-end validation on a small spec subset (5-10 statements). Consider adding `generate_prompts` caching via executor fingerprinting.

### 2026-03-21 — ALIGN-006: End-to-End Validation on Iris (API path)
- **Action:** Ran full pipeline on Iris cluster with 1 statement (`ask_clarifying_questions`), GPT-4.1-mini for all roles, pairwise covering.
- **Job:** `/ahmed/iris-run-align_openai_spec_smoke-20260322-001645` (us-central1, CPU only)
- **Experiment page:** https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/align_openai_spec_smoke-5f28b7.json
- **GCS output:** `gs://marin-us-central1/align/openai_spec_smoke/`
  - Prompts: `gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d/`
  - Chosen: `gs://marin-us-central1/align/openai_spec_smoke/chosen-41661c/`
  - Rejected: `gs://marin-us-central1/align/openai_spec_smoke/rejected-a966b4/`
  - Preference pairs: `gs://marin-us-central1/align/openai_spec_smoke/preference_pairs-0abbf8/`
  - Spec: `gs://marin-us-central1/align/openai_spec_smoke/spec-e79888/`
  - Artifacts: `gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d/artifacts/`
    - `ask_clarifying_questions/understanding.json`
    - `ask_clarifying_questions/ideation.json`
    - `summary.json`
- **Results (confidence: replicated — ran multiple times during debugging):**
  - Stage 1: 8 variation axes (6 behavior-specific + 2 demographic)
  - Stage 2: 72 pairwise covering configs, 688/688 tuples covered
  - Stage 3: 74 prompts extracted
  - Chosen: 74 responses (with spec guidance)
  - Rejected: 74 responses (without spec guidance)
  - Judge: 15 preference pairs after filtering (20% pass rate)
- **Bugs found and fixed during validation:**
  1. `_load_behavior_statements` crashed on JSONL spec files (`json.load` vs JSONL)
  2. All file I/O used `Path()` which doesn't work with `gs://` paths — switched to `iris.marin_fs.url_to_fs` + `zephyr.write_jsonl_file`
  3. Spec file not uploaded to GCS — added spec upload ExecutorStep
  4. `_load_responses` indentation bug → `UnboundLocalError` on empty shards
  5. `model_id=` kwarg not renamed to `config=` after InferenceConfig refactor (4 call sites)
  6. `tenacity` missing in remote Iris jobs — added as explicit dep
  7. `OPENAI_API_KEY` not forwarded to child Iris jobs — added `_llm_env_vars()` to `@remote` env_vars
- **Confidence:** `replicated` — pipeline ran successfully multiple times after fixes
- **Prior failed jobs (for debugging context):**
  - `/ahmed/iris-run-align_openai_spec_smoke-20260321-213606` — JSONL parse error (bug #1)
  - `/ahmed/iris-run-align_openai_spec_smoke-20260321-213736` — chosen step failed, rejected succeeded (bug #1)
  - `/ahmed/iris-run-align_openai_spec_smoke-20260321-214458` — empty shards (GCS I/O bug #2)
  - `/ahmed/iris-run-align_openai_spec_smoke-20260321-220844` — `UnboundLocalError` (bug #4)
  - `/ahmed/iris-run-align_openai_spec_smoke-20260321-221610` — cached empty results from prior failures
  - `/ahmed/iris-run-align_openai_spec_smoke-20260321-222038` — `model_id` kwarg error (bug #5)
  - `/ahmed/iris-run-align_openai_spec_smoke-20260321-222614` — `tenacity` missing (bug #6)
  - `/ahmed/iris-run-align_openai_spec_smoke-20260321-223307` — `Incorrect API key` (bug #7)
  - `/ahmed/iris-run-align_openai_spec_smoke-20260321-224003` — first fully successful run

### 2026-03-21 — ALIGN-007: VLLMConfig for All Model Roles
- **Action:** Extended InferenceConfig support to ideation/extract/judge models (not just teacher/rejected). Added `vllm_engine()` context manager for efficient engine reuse across multiple calls within a step. Auto-selects TPU resources and single-threaded execution for vLLM.
- **Commit:** 2a493aa6a
- **Debug script:** `experiments/align_debug_vllm.py` — uses Llama 3.1 8B Instruct via vLLM for all roles
- **Failed jobs (no TPU capacity):**
  - `/ahmed/iris-run-align_debug_vllm-20260321-224937` — pending v6e-8, killed
  - `/ahmed/iris-run-align_debug_vllm-20260321-225552` — pending v5p-8 us-central1, killed
  - `/ahmed/iris-run-align_debug_vllm-20260321-230115` — pending v5p-8 any region, killed (all TPUs busy)
- **Status:** Not yet validated — no TPU capacity available during testing.

### 2026-03-21 — ALIGN-008: Intermediate Artifact Persistence
- **Action:** Pipeline now saves per-statement artifacts alongside prompts:
  - `artifacts/<stmt>/understanding.json` — Stage 1 output (axes, understanding, motivation)
  - `artifacts/<stmt>/ideation.json` — Stage 2 output (covering plan, scenarios, rubrics)
  - `artifacts/summary.json` — overview with axis names, config counts, coverage stats
- **Commit:** 9bdb3b892
- **Verified:** Artifacts correctly written to GCS and contain full pipeline state
- **GCS artifacts:** `gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d/artifacts/`

### 2026-03-21 — ALIGN-009: Simplify Round 1 + Zephyr Refactor
- **Action:** Code review and cleanup:
  - Deduplicated I/O helpers: `write_sharded_jsonl_gz` and `load_sharded_jsonl_gz` made public, shared across modules
  - Replaced hand-rolled JSONL reading with `zephyr.load_jsonl` in `load_spec`, `load_sharded_jsonl_gz`, `_load_behavior_statements`
  - `_generate_via_vllm` now uses `get_or_create_vllm_engine` from llm_client (was duplicating engine setup)
  - Fixed shared mutable reference: `[{}] * N` → `[{} for _ in range(N)]`
  - Pre-computed `combinations()` in coverage hot loop
  - Removed dead `_load_spec` wrapper in judge.py
  - Refactored `generate_responses` API path to use Zephyr pipeline (`Dataset.from_files → load_jsonl → map → write_jsonl`) instead of eager load + ThreadPoolExecutor
  - Added TODO on judge `_load_responses` noting Zephyr lacks join/lookup primitive
- **Validation job:** `/ahmed/iris-run-align_openai_spec_smoke-20260322-013808` (us-central1)
- **GCS output:** `gs://marin-us-central1/align/openai_spec_smoke/` (cleared and regenerated)

### 2026-03-26 — ALIGN-217: GPT-OSS TPU Smoke Failure Diagnosis
- **Action:** Inspected the first TPU smoke logs for `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax`.
- **Result:** The JAX/bootstrap path was reached successfully, but server initialization failed in `tpu_inference` mesh creation.
- **Root cause:** `gpt_oss_120b_tpu_vllm_config()` defaulted to `tensor_parallel_size=8`, while the `v5p-8` worker exposed `4` JAX devices to `tpu_inference`, causing:
  - `ValueError: Number of devices 4 must be >= the product of mesh_shape (1, 8)`
- **Evidence from logs:**
  - `Resolved architecture: GptOssForCausalLM`
  - `tpu_inference/worker/tpu_worker.py` initialized before failure
  - failure arose in `TPUModelRunner._create_2d_mesh()`
- **Decision:** Treat this as a slice-layout config bug, not a model-format blocker. Lower GPT-OSS TPU default `tensor_parallel_size` to `4` for `v5p-8` bring-up and relaunch the smoke.
- **Results:** 71 prompts → 71 chosen → 71 rejected → 13 preference pairs (18% pass rate). All artifacts present.
- **Note:** Zephyr writes `shard-00000.jsonl.gz` (hyphen) vs old `shard_00000.jsonl.gz` (underscore) for chosen/rejected. Prompts step still uses underscore.

### 2026-03-21 — ALIGN-010: Simplify Round 2
- **Action:** Second review pass:
  - Replaced manual `__enter__`/`__exit__` with `contextlib.ExitStack` (fixed buggy identity check)
  - Fixed quadratic per-axis counting in `compute_coverage_stats` → single pass O(C*A)
  - Renamed `_get_or_create_vllm_engine` to public (was private but imported cross-module)
  - Removed stale "Ported from bloom" comments from 7 files
  - Extracted hardcoded `max_tokens=16000` into config fields (`concretize_max_tokens`, `extract_max_tokens`)
  - Moved all non-guard imports to top of file
- **Validation job:** `/ahmed/iris-run-align_openai_spec_smoke-20260322-020230` (us-central1)
- **GCS output:** `gs://marin-us-central1/align/openai_spec_smoke/` (cleared and regenerated)
- **Results:** 72 prompts → 72 chosen → 72 rejected → 26 preference pairs (36% pass rate). All artifacts present.
- **GCS data sizes:**
  - `prompts-8a5a5d/shard_00000.jsonl.gz` — 21KB
  - `chosen-41661c/shard-00000.jsonl.gz` — 31KB
  - `rejected-a966b4/shard-00000.jsonl.gz` — 30KB
  - `preference_pairs-0abbf8/shard_00000.jsonl.gz` — 7KB
- **Confidence:** `replicated` — pipeline validated 3 times across refactors with consistent results

### 2026-03-22 — ALIGN-011: Codex Code Review Critique

A parallel Codex agent reviewed the PR and identified 4 findings + 2 notes. All findings are valid. Below is the full critique, why each matters, and the agreed fix plan.

#### Finding 1: Silent parse failures degrade dataset quality (HIGHEST RISK)

**What Codex found:** The prompt generation pipeline depends on free-form LLM output parsed via regex/JSON, and parse failures are silently absorbed rather than surfaced:
- **Stage 1** (`generate_prompts.py:96`): Expects raw JSON inside `<variation_axes>` XML tags, parsed via `json.loads()`. If the LLM returns malformed JSON, the statement fails silently.
- **Stage 2** (`generate_prompts.py:190`): When `parse_concretize_response()` returns fewer scenarios than expected (LLM didn't produce all `<scenario>` tags), the code **pads with empty scenarios** `{"description": "", "rubric": ""}` and continues. These empty scenarios flow downstream — Stage 3 will drop them (empty description check at `generate_prompts.py:367`), but the covering array coverage is silently degraded.
- **Stage 3** (`generate_prompts.py:289`): When `_parse_extraction_response()` can't find a `<scenario_N>` block, it inserts `{"system_prompt": "", "user_message": ""}`. The empty `user_message` records are filtered at line 367, but no metric tracks how many were lost.
- **Statement-level failures** (`generate_prompts.py:434, 456, 476`): If an entire statement fails in any stage (network error, parse error, etc.), it's logged as a warning and the pipeline continues with fewer statements. No threshold check.
- **Judge fallback** (`judge.py:100`): When the judge LLM response can't be parsed as JSON, `_judge_response` returns a synthetic `ComplianceResult(score=5, ...)`. Score 5 is a fake middle score — it can pass or fail quality thresholds depending on config, injecting noise into the preference dataset.
- **`llm_client.py:54`**: The `llm_chat` function doesn't expose `response_format` / structured output, so there's no way to enforce JSON schema at the API level.

**Why it matters:** At scale (46 statements × hundreds of prompts), silent degradation compounds. If 20% of Stage 2 scenarios are empty-padded, that's 20% of the covering array with no actual test scenarios — but the coverage stats still report 100% because the configs exist, just with empty content. The dataset looks complete but has holes. The judge fallback score is worse: a score=5 pair could survive filtering (`min_chosen_score=6.0` in smoke test), meaning the training data contains pairs where the judge couldn't even parse its own response.

**Agreed fix plan:**
- **This PR:** Add `max_stage_failure_rate: float` to `PromptGenConfig` (default 0.5). After each stage, compute `failure_rate = len(failures) / total` and abort if it exceeds the threshold. Add same to `JudgePairConfig`. Change judge fallback score from 5 → 0 so parse-failure pairs always get filtered out.
- **Follow-up issue:** Add `response_format` / structured output support to `llm_chat` for stages that expect JSON. Track parse/drop metrics per stage as structured output (not just log lines).

#### Finding 2: ExecutorStep spec path is inconsistent (BUG)

**What Codex found:** In `align.py:190`, when `spec` is an `ExecutorStep`:
```python
spec_gcs_path = output_path_of(spec)  # returns directory, not file
```
But `load_spec()` at `generate_prompts.py:81` expects a JSONL file path. The string branch correctly uploads to `spec.jsonl` and passes `output_path_of(spec_step) / "spec.jsonl"` (line 207), but the ExecutorStep branch passes the bare directory.

**Why it matters:** Anyone passing `spec=some_executor_step` (instead of a string path) gets a broken pipeline — `load_spec()` would try to read a directory as JSONL. This code path was never tested because all our experiments use string paths.

**Agreed fix:** Change to `output_path_of(spec) / "spec.jsonl"` — establish convention that spec ExecutorSteps produce `spec.jsonl` at their output root.

#### Finding 3: vLLM engine cache keyed only on model string (BUG)

**What Codex found:** The global `_active_vllm_engine` cache at `llm_client.py:30` checks reuse via:
```python
_active_vllm_engine["model"] == config.model  # line 86
```
This means two `VLLMConfig`s with the same `model` but different `tensor_parallel_size`, `max_model_len`, or `gpu_memory_utilization` would incorrectly share one engine. Worse, `generate_prompts.py:396` opens both ideation and extract engines in one `ExitStack` — if they're different models, the second `vllm_engine()` context manager overwrites the global cache slot (line 147), silently breaking the first.

**Why it matters:** If someone uses `VLLMConfig(model="llama-8b", max_model_len=2048)` for ideation and `VLLMConfig(model="llama-8b", max_model_len=8192)` for extraction, the extraction stage silently uses the 2048-length engine. The covering array prompts could be truncated without any error.

**Agreed fix:** Key cache on full frozen config object (`config == cached_config`) instead of just model string. `VLLMConfig` is a frozen dataclass so `==` compares all fields.

#### Finding 4: teacher_n > 1 wastes money (DESIGN BUG)

**What Codex found:** `AlignConfig.teacher_n` is passed to the chosen step (line 252), generating N teacher responses per prompt. But `_process_prompt_pair` in `judge.py:133` only reads `chosen_responses[0]` — the first response. Extra teacher samples are generated, stored, and ignored.

**Why it matters:** With `teacher_n=4` (hypothetical), you'd pay 4x the API cost for chosen responses with zero benefit. The field's existence implies it does something.

**Agreed fix:** Keep `teacher_n` but make the judge actually use it — score all N teacher responses and pick the highest-scoring one as the chosen. This makes `teacher_n > 1` a genuine quality-vs-cost knob: more candidates → higher chance of a high-quality chosen response.

#### Note: env_vars inconsistency

**What Codex found:** `_llm_env_vars()` (forwarding `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) is added to prompts step (`align.py:216`) and judge step (`align.py:286`) but NOT to chosen step (`align.py:243`) or rejected step (`align.py:264`).

**Why it works today:** The chosen/rejected steps use Zephyr, which spawns worker jobs. The parent job has the env var (passed via `-e` on submission). Zephyr workers may or may not inherit it — the smoke test succeeded, so empirically they do, but we don't have documentation confirming this is guaranteed behavior.

**Agreed fix:** Add `env_vars=_llm_env_vars()` to all four `@remote` calls for consistency. Don't assume Zephyr's env inheritance is guaranteed.

#### Note: Bloom naming leftover

**What Codex found:** The Stage 1 system prompt at `understanding.py:62` still says "You are BloomUnderstanding" and references "Bloom Evals". The test at `test_alignment.py:619` asserts on "BloomUnderstanding".

**Agreed fix:** Replace with generic naming ("AI alignment research assistant" / "this evaluation pipeline"). Update test.

### Implementation Plan for ALIGN-012 (revised twice — user feedback + Codex corrections)

**Key design decision:** Stages 1-3 (understanding, concretization, extraction) must **hard-fail on parse errors**. The rationale: these stages produce the foundation of the dataset. If the LLM can't produce parseable output (valid XML tags, valid JSON variation axes, correct scenario count), the covering array is broken and everything downstream is garbage. Silent degradation (padding with empty scenarios, inserting empty records, warn-and-continue) was the Bloom pipeline's approach for robustness, but it hides model inadequacy. The right signal is: crash, tell the user why, and let them either fix the prompt or switch to a stronger model. No silent dataset degradation.

The **judge** is different: individual unparseable judge responses out of hundreds are tolerable. But the fallback score must be 0 (auto-filtered), not 5 (fake mid-score that might pass quality thresholds and inject noise).

#### Codex corrections to first revision (2026-03-22)

**Correction 1: Don't literally remove the try/except around concurrent futures.**

My first plan said "remove the try/except — let it propagate." Codex pointed out this is wrong: if you let the first failed future propagate, you lose visibility into all the other failures. The ThreadPoolExecutor has N futures in flight — if statement 3 fails and you raise immediately, you never learn that statements 7 and 12 also failed.

**Correct approach:** Keep per-statement error collection. Let all submitted futures finish. Then raise ONE aggregated error listing every failed statement and its reason. This gives full visibility ("statements ask_clarifying_questions, be_honest, and avoid_abuse all failed: <reasons>") instead of just "ask_clarifying_questions failed."

Concretely in `_generate_prompts_inner`:
```python
# Stage 1 — still uses ThreadPoolExecutor, still collects failures
with concurrent.futures.ThreadPoolExecutor(...) as pool:
    ...
    for future in as_completed(future_map):
        try:
            understandings[sid] = future.result()
        except Exception as exc:
            failures.append((sid, str(exc)))

# But NOW: raise after all futures complete, not warn-and-continue
if failures:
    detail = "; ".join(f"{sid}: {msg}" for sid, msg in failures)
    raise RuntimeError(f"Stage 1 failed for {len(failures)} statement(s): {detail}")
```

Same pattern for Stage 2 and Stage 3.

**Correction 2: vLLM needs a cache dict, not a single slot.**

My first plan said "key on full config object instead of model string." Codex pointed out this is necessary but not sufficient. The real bug is the **single global slot** `_active_vllm_engine` (`llm_client.py:30`). Because `generate_prompts.py:396` can hold two vLLM contexts simultaneously in one `ExitStack` (ideation engine + extract engine), a single slot can't support two active engines — the second `vllm_engine()` context overwrites the first.

**Correct approach:** Replace the single global `_active_vllm_engine: dict | None` with a **cache dict** keyed by full frozen config:
```python
_vllm_engine_cache: dict[VLLMConfig, tuple[LLM, Tokenizer]] = {}
```

`get_or_create_vllm_engine(config)` checks this dict. `vllm_engine(config)` adds to it on enter, removes on exit. Two simultaneous contexts with different configs each get their own engine. Same config reuses the cached engine.

**Correction 3: Judge max_failure_rate default too loose.**

Codex suggested 0.1 or 0.2 instead of 0.5. Agreed — if 50% of your judge calls fail, something is fundamentally wrong. Default to 0.2.

#### Final fix list (priority order)

1. **Hard-fail Stages 1-3** (quality — highest priority)
   - Keep ThreadPoolExecutor + per-statement error collection (Codex correction 1)
   - After all futures complete, raise aggregated error if ANY statement failed
   - `_concretize_batch`: raise on scenario count mismatch (not pad)
   - `_parse_extraction_response`: raise on missing blocks (not insert empty)

2. **ExecutorStep spec path** → `output_path_of(spec) / "spec.jsonl"` (bug fix at `align.py:190`)

3. **vLLM engine cache dict** → replace single global slot with `dict[VLLMConfig, engine]` (Codex correction 2, bug fix at `llm_client.py:30`)

4. **teacher_n** → judge picks best of N teacher responses instead of always reading `responses[0]` (behavior fix at `judge.py:133`)

5. **Judge failure accounting** → fallback score 5 → 0, `max_failure_rate: float = 0.2` (Codex correction 3)

6. **env_vars** → add `_llm_env_vars()` to chosen/rejected `@remote` calls (consistency at `align.py:243, 264`)

7. **Bloom naming** → replace "BloomUnderstanding"/"Bloom Evals" with generic names (cleanup at `understanding.py:62`, `test_alignment.py:619`)

8. **Follow-up issue:** structured output / `response_format` support in `llm_chat` for JSON-expecting stages

- **Next action:** Implement all fixes, re-run pipeline on Iris to validate.

### 2026-03-22 — ALIGN-013: vLLM 70B Stabilization + Overnight (8h) Execution Plan

- **Current active job:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-052616`
- **Current state snapshot (22:53 PT):**
  - Parent job: `JOB_STATE_RUNNING`
  - Prompts child: `JOB_STATE_RUNNING` with `preemption_count=1`, currently pending/rerouted after preemption
  - Prompts child resources: TPU `v5p-8`, memory `128GB`, disk `500GB`
- **Data browser URL (current run):** `https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/align_debug_vllm_70b-afef1e.json`

#### What was fixed before this plan

1. Added dedicated 70B one-statement vLLM script:
   - `experiments/align_debug_vllm_70b.py`
2. Fixed missing vLLM dependency in local-inference alignment steps:
   - `align.py` now uses vLLM dependency groups for local backends
3. Fixed torch/CUDA mismatch on TPU workers:
   - local vLLM steps now install both `vllm` and `tpu` extras
4. Added configurable TPU disk to `VLLMConfig` and set `disk="500g"` for 70B run:
   - `inference_config.py`, `align_debug_vllm_70b.py`
5. Added HF cache env forwarding to alignment child jobs:
   - `_llm_env_vars()` now forwards `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`
6. Relaunched with explicit cache env vars:
   - `HF_HOME=/app/hf_cache`
   - `HF_HUB_CACHE=/app/hf_cache/hub`

#### Objective for overnight run

Validate that full one-statement vLLM pipeline for Llama 3.3 70B can complete end-to-end on Iris TPU (`v5p-8`) in `us-central1` (with `us-east5-a` fallback) and produce preference pairs.

#### Success gates (must pass in order)

- **Gate A (infra):** Prompts step starts and initializes vLLM engine without import/linker failures.
- **Gate B (weights):** Model weights download/load without disk-space errors.
- **Gate C (prompt gen):** Prompt generation step succeeds and writes non-empty prompt shard(s).
- **Gate D (responses):** Chosen and rejected generation steps succeed and write non-empty shard(s).
- **Gate E (judge):** Judge step succeeds and writes non-empty preference pair shard(s).
- **Gate F (artifact sanity):** GCS outputs contain expected directories/files and counts are non-zero.

#### Overnight monitoring protocol (8-hour plan)

**Cadence and lifecycle**
- Startup stabilization after each submit/resubmit: sleep `120s`, then check status/logs.
- Normal cadence: every `570s`.
- Continue loop until:
  - full success (`Gate F`), or
  - unrecoverable repeated failure mode (same root cause 2x after targeted fix), or
  - 8-hour window ends.

**Per-interval checks**
1. Status:
   - `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix <JOB_ID>`
2. Logs (parent + children):
   - `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 1200 --include-children <JOB_ID>`
3. Error grep focus:
   - `traceback|exception|resource_exhausted|oom|no accelerator found|failed_precondition|device or resource busy|program hbm requirement|largest program allocations|not enough free disk space|ImportError|ValueError|RuntimeError`
4. Progress grep focus:
   - `Downloading weights|Stage 1|Stage 2|Stage 3|Total prompts generated|Wrote .*records|Step .* succeeded`

#### Recovery playbook (decision tree)

**A) Capacity pending (`Insufficient TPUs`)**
- If pending < 45 min: continue waiting.
- If pending >= 45 min with no progress in `us-central1-a`: submit fallback run in `us-east5-a` and monitor whichever starts first.
- Never restart cluster.

**B) Preemption**
- If `preemption_count` increments and task returns to running/pending: continue monitoring (no immediate action).
- If preemption flaps 2+ times without stage progress: stop current root job and resubmit once in alternate region.

**C) Dependency/import failures**
- Examples: `No module named vllm`, torch/cuda linker errors.
- Action: patch dependency groups/config, run `tests/test_alignment.py`, resubmit.
- If same exact error recurs after patch: mark as unresolved blocker.

**D) Disk/cache failures**
- Signature: `Not enough free disk space ... /root/.cache/...`
- Action order:
  1. Ensure `VLLMConfig.disk` is large enough (`>=500g` for 70B)
  2. Ensure `HF_HOME` and `HF_HUB_CACHE` forwarded to child jobs
  3. Resubmit and verify logs no longer mention `/root/.cache` shortage
- If still failing with same signature after both fixes: capture full logs and stop for manual infra investigation.

**E) HBM/OOM/compile memory failures**
- Signature: `RESOURCE_EXHAUSTED`, `Program hbm requirement`, `Largest program allocations`.
- Action: reduce memory pressure:
  - lower `max_model_len` (e.g., 2048 -> 1024),
  - ensure `tensor_parallel_size=4`,
  - keep workers=1 for vLLM stages.
- Resubmit once with reduced config.

**F) Stage hard-fail on parse errors**
- Expected behavior post ALIGN-012 (intentional).
- Action: preserve failure logs; do not mask.
- If frequent, lower batching (`concretize_batch_size=1`, `extract_batch_size=1`) for recovery run.

#### Region fallback order

1. `us-central1-a` (primary)
2. `us-east5-a` (fallback)

Submit command template:
```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --extra marin:tpu \
  --tpu v5p-8 \
  --region <REGION> \
  --zone <ZONE> \
  -e HF_HOME /app/hf_cache \
  -e HF_HUB_CACHE /app/hf_cache/hub \
  -- python experiments/align_debug_vllm_70b.py
```

#### Validation commands for end state (Gate F)

After root success, verify outputs exist and are non-empty:

```bash
# 1) Identify latest experiment metadata for align_debug_vllm_70b
uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 7200 --include-children <JOB_ID> \
  | rg "experiments/align_debug_vllm_70b-.*\.json|Step .* succeeded|Wrote .*records"

# 2) Confirm status tree is succeeded
uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix <JOB_ID>

# 3) Confirm artifacts under align/debug_vllm_70b in GCS (prompts/chosen/rejected/preference_pairs)
# (Use project-standard GCS listing/check tooling available in session)
```

Minimum acceptable artifacts:
- `prompts-*/shard_*.jsonl.gz` present, >0 records
- `chosen-*/shard-*.jsonl.gz` present, >0 records
- `rejected-*/shard-*.jsonl.gz` present, >0 records
- `preference_pairs-*/shard_*.jsonl.gz` present, >0 records
- `prompts-*/artifacts/summary.json` present

#### Time budget allocation for 8 hours

- **Hour 0-1:** keep current job healthy through startup, weights, and first stage completion.
- **Hour 1-4:** drive run to end-to-end completion; apply targeted fixes if one clear blocker appears.
- **Hour 4-6:** if run failed, execute one controlled fallback (alternate region or reduced config based on failure type).
- **Hour 6-7:** validate output artifacts, collect final metrics/counts, confirm reproducibility signal.
- **Hour 7-8:** append full overnight results to logbook with job IDs, failures/fixes, and final recommendation.

#### Morning handoff format (to append after overnight loop)

- `Last job`: `<JOB_ID> — <SUCCEEDED/FAILED/PENDING>`
- `Highest gate reached`: `A/B/C/D/E/F`
- `Blocking issue`: one-line root cause (if any)
- `Artifacts`: key GCS paths + record counts
- `Recommendation`: `promote`, `retry with config X`, or `open infra issue`

#### Monitoring state file

- Active monitor state: `scratch/20260322-2114_monitoring_state.json`
- Includes current `job_id`, `restart_count`, and latest resubmit command.

### 2026-03-22 — ALIGN-014: Overnight Execution Kickoff (User-directed 8h continuous run)

- **User directive:** run continuously for 8 hours without stopping; monitor and drive vLLM validation to completion.
- **Start time:** 2026-03-22 22:56 PT
- **Active job at kickoff:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-052616`
- **State at kickoff:**
  - Root job: `JOB_STATE_RUNNING`
  - Prompts child: `JOB_STATE_RUNNING`, `preemption_count=1`, task currently pending/rerouted after preemption
  - Prompts child resources: TPU `v5p-8`, memory `128GB`, disk `500GB`
- **Current experiment metadata page:** `https://marin.community/data-browser/experiment?path=gs%3A//marin-us-central1/experiments/align_debug_vllm_70b-afef1e.json`
- **Current monitor state file:** `scratch/20260322-2114_monitoring_state.json` (`restart_count=4`)

#### Commands in use (monitor loop)

```bash
uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /ahmed/iris-run-align_debug_vllm_70b-20260323-052616
uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 1200 --include-children /ahmed/iris-run-align_debug_vllm_70b-20260323-052616
```

#### Immediate overnight objective

Drive current run through ALIGN-013 success gates A->F; if blocked, apply ALIGN-013 recovery tree (capacity/preemption/deps/disk/cache/OOM) and continue with controlled resubmissions in `us-central1-a` then `us-east5-a` fallback.

### 2026-03-22 — ALIGN-015: Region Fallback Trigger (us-central1 stall -> us-east5-a parallel submit)

- **Trigger condition:** active prompts child for `/ahmed/iris-run-align_debug_vllm_70b-20260323-052616` remained pending/rerouted with no new stage-progress logs for >45 minutes after preemption.
- **Decision:** per ALIGN-013 recovery tree, submit fallback run in `us-east5-a` while keeping current `us-central1-a` run alive; monitor whichever makes forward progress first.
- **Submit command (fallback):**

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --extra marin:tpu \
  --tpu v5p-8 \
  --region us-east5 \
  --zone us-east5-a \
  -e HF_HOME /app/hf_cache \
  -e HF_HUB_CACHE /app/hf_cache/hub \
  -- python experiments/align_debug_vllm_70b.py
```

### 2026-03-22 — ALIGN-016: Fallback Lane Becomes Primary

- **Comparison snapshot (22:59 PT):**
  - `us-central1` lane `/ahmed/iris-run-align_debug_vllm_70b-20260323-052616`:
    - Prompts child still `JOB_STATE_RUNNING` but `task_state_counts.pending=1` after preemption, no new stage-progress logs.
  - `us-east5-a` fallback lane `/ahmed/iris-run-align_debug_vllm_70b-20260323-055713`:
    - Spec step succeeded.
    - Prompts child `JOB_STATE_RUNNING` with active startup progress (vLLM initialization logs observed).
- **Decision:** Switch primary monitoring ownership to fallback lane (`055713`) and keep central lane (`052616`) as secondary watch.
- **Immediate next action:** Continue 570s monitoring cadence on fallback lane, with periodic sanity checks on central lane.

### 2026-03-22 — ALIGN-017: Fallback OOM Diagnosis + High-RAM Resubmit Plan

- **Observed failure (fallback lane):** `/ahmed/iris-run-align_debug_vllm_70b-20260323-055713` failed in prompts step with container OOM during vLLM engine initialization.
- **Error signature:** `OOM killed (container exceeded memory limit)` with `RuntimeError: Engine core initialization failed`.
- **Interpretation:** v5p-8 worker default RAM (`128g`) is insufficient for current 70B initialization path in this environment.
- **Fix implemented:**
  - Extended `VLLMConfig` with explicit `ram` field.
  - `VLLMConfig.resources` now passes `ram` through `ResourceConfig.with_tpu(...)`.
  - 70B debug config now requests `ram="256g"` and `disk="500g"`.
- **Validation:** `uv run pytest tests/test_alignment.py -q` -> `65 passed`.
- **Next action:** resubmit in `us-east5-a` with HF cache env vars and monitor for post-engine-init progress.

### 2026-03-22 — ALIGN-018: High-RAM Lane Started Successfully (No Immediate OOM)

- **Active lane:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-060208` (us-east5-a)
- **Key resource change validated:** prompts child now requests and receives `memory_bytes=274877906944` (`256GB`) and `disk_bytes=536870912000` (`500GB`).
- **Observed progress:**
  - spec step skipped/succeeded from cache
  - prompts child running
  - vLLM engine startup logs present
  - weight download started (`Downloading weights from HF meta-llama/Llama-3.3-70B-Instruct`)
- **Result so far:** previous immediate engine-init OOM reproduced on low-RAM lane has not reappeared yet on high-RAM lane.
- **Next action:** continue monitoring until prompts step terminal state; if success, proceed to chosen/rejected/judge gates. If failure, classify root cause and iterate once.

### 2026-03-22 — ALIGN-019: Stage-2 Context-Length Failure Identified

- **Run analyzed:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-060208`
- **Progress:** reached Stage 1 and entered Stage 2 concretization.
- **Failure:** Stage 2 hard-failed with repeated prompt length errors:
  - `The decoder prompt (length 2057-2069) is longer than the maximum model length of 2048`
- **Interpretation:** current `max_model_len=2048` is too small for Stage 2 prompts in this pipeline.
- **Plan:** increase 70B debug `max_model_len` to `4096` and resubmit.

Secondary note on central lane (`/ahmed/iris-run-align_debug_vllm_70b-20260323-052616`):
- prompts child pending reason includes autoscaler disk constraint (`need 500GB`, available `100GB`) in `us-central1-a`.
- This lane is currently non-viable with 500GB disk requirement; stop it to reduce noise and focus on east5 lane.

### 2026-03-22 — ALIGN-020: 4096-Context Resubmit Active + Babysit Ownership Resumed

- **Active lane:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-061639` (us-east5-a)
- **Run intent:** validate Stage 2 recovery by raising `max_model_len` from `2048` to `4096` while keeping `tensor_parallel_size=4`, `ram=256g`, and `disk=500g`.
- **Current status snapshot (23:18 PT):**
  - Root job: `JOB_STATE_RUNNING`
  - Prompts child: `JOB_STATE_RUNNING`
  - Prompts child resources: TPU `v5p-8`, memory `256GB`, disk `500GB`
  - vLLM engine init reached model load on TPU with `max_seq_len=4096` and no immediate context-length failure.
- **Monitoring ownership:** resumed continuous Iris babysit loop on this lane under ALIGN-013 protocol.
- **State file sync:** `scratch/20260322-2114_monitoring_state.json` updated to:
  - `job_id=/ahmed/iris-run-align_debug_vllm_70b-20260323-061639`
  - `restart_count=7`
- **Next action:** continue cadence checks and classify next gate transition (C: prompt generation success vs new blocker).

### 2026-03-22 — ALIGN-021: 4096 Run Cleared Prior Failure and Reached Stage 2 Execution

- **Run observed:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-061639` (us-east5-a)
- **Gate progress update:**
  - Gate A (infra): passed (vLLM engine initialized on TPU `v5p-8`)
  - Gate B (weights): passed (HF weight download/load completed)
  - Gate C (prompt gen): in progress (Stage 2 concretization active)
- **Key log evidence:**
  - `Using max model len 4096` on prompts child
  - Stage 1 started and completed inference pass
  - Stage 2 entered and running repeated concretization requests
  - No recurrence of prior hard error: `decoder prompt ... longer than maximum model length of 2048`
- **Current behavior:** Stage 2 is producing partial batches (e.g., `got 6/10`, `7/10`, `8/10 scenarios ... continuing with partial batch`) and continuing.
- **Monitoring setup:** continuous babysit loop running via `scratch/iris_babysit_vllm_70b.sh` with Iris track state in `scratch/20260322-2114_monitoring_state.json`; recovery policy remains stop→resubmit on terminal failure.
- **Next action:** keep monitoring until Stage 2 completes and classify whether prompts step reaches success (Gate C) or fails with a new reproducible blocker.

### 2026-03-22 — ALIGN-022: Prompts Step Succeeded (Gate C Passed), Responses Started

- **Run observed:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-061639`
- **Prompt pipeline outcome:**
  - Stage 3 executed successfully.
  - `Total prompts generated: 46`
  - `Wrote 46 records to 1 shards in gs://marin-us-east5/align/debug_vllm_70b/prompts-5887d2`
- **Step transition:**
  - `align/debug_vllm_70b/chosen_16977d4f` started
  - `align/debug_vllm_70b/rejected_ad02ff5a` started
- **Gate status:**
  - Gate C (prompt generation success): **passed**
  - Gate D (chosen/rejected generation): **in progress**
- **Next action:** monitor chosen/rejected completion and verify non-empty response shards before judge stage launch.

### 2026-03-22 — ALIGN-023: Response Workers Healthy Through Engine Warmup

- **Run observed:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-061639`
- **Monitor cadence checkpoint:** `06:41:28-06:41:43 UTC`
  - Root state: `JOB_STATE_RUNNING`
  - Child terminal failures: `0`
  - Pending reason: none
- **Chosen/rejected worker status:**
  - both workers still running with `v5p-8`, `256GB` RAM, `500GB` disk
  - progressed through weight loading and ongoing compilation/warmup
  - no OOM/HBM/resource-exhausted signatures observed in this window
- **Gate status:** Gate D remains in progress (no response shard write yet)
- **Next action:** continue monitoring until chosen/rejected emit record-write lines and/or terminal step status.

### 2026-03-22 — ALIGN-024: Gate D Passed (Chosen + Rejected Succeeded), Preference Pairs Running

- **Run observed:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-061639`
- **Response generation outcome:**
  - `chosen` step wrote `46` records to `gs://marin-us-east5/align/debug_vllm_70b/chosen-cef5b2`
  - `rejected` step wrote `46` records to `gs://marin-us-east5/align/debug_vllm_70b/rejected-dfc8d8`
  - both child jobs reached `JOB_STATE_SUCCEEDED`
- **Notable runtime signal:** each response worker emitted `Engine core ... died unexpectedly` at teardown, but step status remained successful and output shards were written before termination.
- **Next-stage transition:**
  - `preference_pairs` child launched:
    - step id `align/debug_vllm_70b/preference_pairs_93780421`
    - output path `gs://marin-us-east5/align/debug_vllm_70b/preference_pairs-ae5319`
- **Gate status:**
  - Gate D (responses): **passed**
  - Gate E (judge/pair build): **in progress**
- **Next action:** monitor preference-pair step to completion and validate non-empty pair shard(s) for Gate F artifact sanity.

### 2026-03-22 — ALIGN-025: Preference Step Through Warmup Into Active Per-Prompt Inference

- **Run observed:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-061639`
- **Monitor checkpoint (`07:00:58-07:01:12 UTC`):**
  - root: `JOB_STATE_RUNNING`
  - child terminal failures: `0`
  - pending reason: none
- **Preference-pairs behavior:**
  - completed heavy TPU/vLLM warmup and started active inference loop
  - logs now show repeated `Adding requests: 1/1` and `Processed prompts: 1/1` cycles (~4-6s each)
  - no OOM/HBM/resource-exhausted errors observed in this phase
- **Gate status:** Gate E still in progress (waiting for final pair write + terminal success line)
- **Next action:** continue monitoring for `Wrote ... preference_pairs` and root transition to `JOB_STATE_SUCCEEDED`.

### 2026-03-23 — ALIGN-026: End-to-End Success (Gate F Passed) on 70B TPU Lane

- **Run:** `/ahmed/iris-run-align_debug_vllm_70b-20260323-061639` (us-east5-a)
- **Terminal state:** root + all child steps `JOB_STATE_SUCCEEDED`
- **Experiment metadata:** `gs://marin-us-east5/experiments/align_debug_vllm_70b-92ed57.json`
- **Data browser URL:** `https://marin.community/data-browser/experiment?path=gs%3A//marin-us-east5/experiments/align_debug_vllm_70b-92ed57.json`

#### Gate summary

- Gate A (infra): passed
- Gate B (weights): passed
- Gate C (prompt generation): passed
- Gate D (chosen/rejected generation): passed
- Gate E (preference pair build): passed
- Gate F (artifact sanity): passed

#### Output counts (from step logs)

- prompts: `46` records
- chosen: `46` records
- rejected: `46` records
- preference_pairs: `42` records

#### Artifact sanity checks

- output root: `gs://marin-us-east5/align/debug_vllm_70b/`
- required directories present:
  - `prompts-5887d2/`
  - `chosen-cef5b2/`
  - `rejected-dfc8d8/`
  - `preference_pairs-ae5319/`
  - `spec-71bb0e/`
- required files present:
  - `prompts-5887d2/shard_00000.jsonl.gz`
  - `prompts-5887d2/artifacts/summary.json`
  - `chosen-cef5b2/shard_00000.jsonl.gz`
  - `rejected-dfc8d8/shard_00000.jsonl.gz`
  - `preference_pairs-ae5319/shard_00000.jsonl.gz`
- shard sizes (`gsutil ls -l`, bytes):
  - prompts: `11081`
  - chosen: `24684`
  - rejected: `31459`
  - preference_pairs: `35912`

#### Notes

- Response steps emitted `Engine core ... died unexpectedly` at teardown after work completion, but both steps wrote full shards and reported `succeeded`. No recovery action required for this run.
- Monitor state file updated: `scratch/20260322-2114_monitoring_state.json` with `last_terminal_state=JOB_STATE_SUCCEEDED`, `last_gate=F`, `restart_count=7`.

#### Recommendation

- **Promote this lane as a validated vLLM 70B TPU recipe** (`v5p-8`, `tp=4`, `max_model_len=4096`, `ram=256g`, `disk=500g`, HF cache envs forwarded).

### 2026-03-23 — ALIGN-027: Global Align Prefix Audit (`gs://marin-us-*/align/*`)

- **Command run:**
  - `gsutil ls 'gs://marin-us-*/align/*'`
- **Observed prefixes:**
  - `gs://marin-us-central1/align/debug_vllm_70b/`
  - `gs://marin-us-central1/align/openai_spec_smoke/`
  - `gs://marin-us-east5/align/debug_vllm_70b/`
- **Observed subdirs for `debug_vllm_70b` in both regions:**
  - `chosen-cef5b2/`
  - `rejected-dfc8d8/`
  - `prompts-5887d2/`
  - `preference_pairs-ae5319/`
  - `spec-71bb0e/`
- **Interpretation:** alignment outputs are currently discoverable under both `marin-us-central1` and `marin-us-east5` bucket prefixes for the same debug run artifact names.

### 2026-03-23 — ALIGN-028: Plan for Mixtral Rejected Responses

- **User direction:** rerun the validated 70B vLLM debug pipeline with a different rejected-model backend so "bad responses" come from Mixtral instead of Llama 3.3 70B.
- **Goal:** isolate the effect of a weaker/different rejected model while holding the validated teacher, prompt-generation, judge, and training target constant.
- **Working assumption:** use `mistralai/Mixtral-8x7B-Instruct-v0.1` as the rejected-model checkpoint. This is the concrete Mixtral instruct checkpoint already referenced in the repo docs.

#### Why this should be a new experiment file

- Keep `experiments/align_debug_vllm_70b.py` unchanged as the validated baseline recipe.
- Add a sibling debug script for the heterogeneous-model run so result diffs are attributable to the rejected-model change, not baseline drift.

#### Planned code change

Add a new script:

```python
# experiments/align_debug_vllm_70b_mixtral_rejected.py
llama_vllm = VLLMConfig(
    model="meta-llama/Llama-3.3-70B-Instruct",
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="500g",
    ram="256g",
)

mixtral_vllm = VLLMConfig(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=4,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tpu_type="v5p-8",
    disk="500g",
    ram="256g",
)

dataset_steps = align(
    name="debug_vllm_70b_mixtral_rejected",
    pretrained_model=llama_3_3_70b_instruct,
    spec=SPEC_PATH,
    model_config=llama_70b,
    teacher_model=llama_vllm,
    align_config=align_config,
    dpo_config=None,
    rejected_model=mixtral_vllm,
    tags=["debug", "vllm", "70b", "mixtral-rejected"],
)
```

#### Planned invariants

- Keep the validated Llama 70B settings unchanged for:
  - `pretrained_model`
  - `model_config`
  - `teacher_model`
  - `ideation_model`
  - `extract_model`
  - `judge_model`
  - `statement_ids=["ask_clarifying_questions"]`
  - `covering_strength=2`
  - `teacher_n=1`, `rejected_n=1`
  - `tokenizer="meta-llama/Llama-3.3-70B-Instruct"`
- Important: tokenizer stays Llama because the aligned model target is still Llama 70B. The rejected model only contributes text responses, not training tokenization.

#### Planned execution order

1. Create `experiments/align_debug_vllm_70b_mixtral_rejected.py`.
2. Run `uv run pytest tests/test_alignment.py -q` to ensure the alignment stack still passes.
3. Submit the new experiment on Iris in `us-east5-a` first, because that region produced the validated 70B success lane while the `us-central1-a` lane hit disk-capacity issues.
4. Reuse the same HF cache env forwarding used in ALIGN-026:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --extra marin:tpu \
  --tpu v5p-8 \
  --region us-east5 \
  --zone us-east5-a \
  -e HF_HOME /app/hf_cache \
  -e HF_HUB_CACHE /app/hf_cache/hub \
  -- python experiments/align_debug_vllm_70b_mixtral_rejected.py
```

#### Success criteria for this rerun

- Gate A-F all pass again.
- `rejected` step writes non-empty shard(s) and logs show the rejected-model backend is Mixtral.
- `preference_pairs` output is non-empty.
- Artifacts land under a distinct prefix, expected shape:
  - `gs://marin-us-east5/align/debug_vllm_70b_mixtral_rejected/...`

#### Main risks to watch

- Mixtral may have different TPU/vLLM startup behavior than Llama 70B even if the rejected step is simpler.
- Keeping `v5p-8`, `tp=4`, `max_model_len=4096`, `ram=256g`, and `disk=500g` on the first run is intentional: it isolates the model swap as the only planned variable.
- If the user wants a different Mixtral variant than `Mixtral-8x7B-Instruct-v0.1`, swap the model ID before execution rather than mutating the baseline plan mid-run.

- **Next action:** implement the sibling experiment script and run the first `us-east5-a` validation lane.

### 2026-03-24 — ALIGN-029: Sequential Multi-Region Mixtral Download Plan (Long-Running Ops Plan)

- **User direction:** complete Mixtral model downloads across all requested regions, but do it sequentially rather than in parallel. Keep monitoring `us-central1` until it finishes, then continue region-by-region until all regions are complete.
- **Models in scope:**
  - `mistralai/Mixtral-8x7B-v0.1` @ `fc7ac94`
  - `mistralai/Mixtral-8x7B-Instruct-v0.1` @ `eba9230`
- **Current state at plan creation:**
  - `us-central1` remains active
  - `us-east5-a`, `us-east1-d`, and `europe-west4` were intentionally killed after showing slow progress under HF rate limiting
  - partial outputs already exist in all four regional buckets

#### Root-cause summary for the first attempt

- We launched 4 regional jobs in parallel.
- Each job ran 2 `download_hf` steps (`mixtral_8x7b`, `mixtral_8x7b_instruct`).
- `download_hf` default `zephyr_max_parallelism=8`, so the first attempt drove roughly `64` concurrent HF streams overall.
- The bottleneck was Hugging Face rate limiting (`429 Too Many Requests`) on large weight shards, not Iris scheduling or GCS write failures.

#### Why the sequential plan should reuse work

- The model download steps use stable override output paths in `experiments/models.py`:
  - `models/mistralai--Mixtral-8x7B-v0-1--fc7ac94`
  - `models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230`
- Those paths resolve to deterministic regional prefixes:
  - `gs://marin-us-central1/models/...`
  - `gs://marin-us-east5/models/...`
  - `gs://marin-us-east1/models/...`
  - `gs://marin-eu-west4/models/...`
- `download_hf` writes each individual file via atomic temp-write + rename, so already-landed files at a reused prefix are valid committed objects, not partial garbage.
- The Zephyr write stage uses `skip_existing=True` for `.metrics/success-part-*.jsonl`. In Zephyr this skips the entire upstream shard when the corresponding output shard already exists. Practical implication: rerunning the same region against the same prefix should skip already-completed download tasks rather than starting from zero.

#### Operational rule from this point on

- **Only one region may be active at a time.**
- Do not relaunch other regions while one region is still downloading.
- This reduces effective HF concurrency from about `64` total streams to about `16` total streams (2 model downloads x 8 workers) without changing code yet.

#### Region order

1. `us-central1` — continue current live job
2. `us-east5-a` — resume from partial prefix after `us-central1` completes
3. `us-east1-d` — resume from partial prefix after `us-east5-a` completes
4. `europe-west4` — resume from partial prefix after `us-east1-d` completes

Rationale:
- `us-central1` is currently the furthest ahead.
- The three killed regions all have partial outputs and can be resumed later against the same prefixes.
- `us-east5-a` goes next because it is already used elsewhere in this thread for alignment runs and had valid partial progress before termination.

#### Per-region success criterion

For each region, both model prefixes must satisfy all of:
- all `36/36` `.metrics/success-part-*.jsonl` shards present
- `.executor_status` indicates success at the step root
- expected large weight files appear under the model prefix
- no active child worker jobs remain for that regional root job

#### Monitoring protocol (hours-long handoff-safe workflow)

- **Current active monitor target:** `/ahmed/download-mixtral-models-us-central1`
- **Current monitor state file:** `scratch/20260323-2334_monitoring_state.json`
- **Current monitor script:** `scratch/20260323-2334_monitor_mixtral_downloads.py`
- Cadence: every `300s`
- Each cycle must check:
  1. `uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix <JOB_ID>`
  2. `uv run iris --config lib/iris/examples/marin.yaml job logs --since-seconds 600 --include-children <JOB_ID>`
  3. `gsutil ls '<PREFIX>/.metrics/success-part-*.jsonl' | wc -l`
  4. `gsutil ls '<PREFIX>/*'` for evidence that large artifacts, not just metadata, are appearing
- If the job remains `JOB_STATE_RUNNING` and `.metrics` counts increase, keep monitoring.
- If the job remains `JOB_STATE_RUNNING` but `.metrics` counts stay flat across multiple intervals, inspect logs specifically for:
  - `429`
  - `Retrying`
  - `Timed out reading`
  - repeated retries on the same shard names

#### Recovery and resubmission policy

- Do not restart the cluster.
- Do not launch a second region while the current one is active.
- If a region eventually reaches a terminal non-success state or must be manually restarted:
  - resubmit the same experiment script in the same region
  - preserve the same model step names and therefore the same output prefixes
  - rely on the existing prefix + `.metrics` reuse path described above

Canonical resubmit commands:

```bash
# us-central1
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name download-mixtral-models-us-central1 \
  --cpu 4 \
  --memory 16GB \
  --disk 20GB \
  --region us-central1 \
  -- python experiments/download_mixtral_models.py

# us-east5-a
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name download-mixtral-models-us-east5-a \
  --cpu 4 \
  --memory 16GB \
  --disk 20GB \
  --region us-east5 \
  --zone us-east5-a \
  -- python experiments/download_mixtral_models.py

# us-east1-d
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name download-mixtral-models-us-east1-d \
  --cpu 4 \
  --memory 16GB \
  --disk 20GB \
  --region us-east1 \
  --zone us-east1-d \
  -- python experiments/download_mixtral_models.py

# europe-west4
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name download-mixtral-models-europe-west4 \
  --cpu 4 \
  --memory 16GB \
  --disk 20GB \
  --region europe-west4 \
  -- python experiments/download_mixtral_models.py
```

#### Escalation rule if single-region serialization still stalls

- First try serialization only. Do not change code preemptively.
- If `us-central1` continues to show repeated `429` retries with no `.metrics` growth across many intervals even after the other regions remain off, open the next improvement thread:
  - reduce `DownloadConfig.zephyr_max_parallelism` for Mixtral download runs only
  - keep the same output prefixes so resumed work is reused
- This is a second-line mitigation, not the first move.

#### Future-agent handoff format

When handing off after a long monitoring stretch, record:
- `Active region`: one of `us-central1`, `us-east5-a`, `us-east1-d`, `europe-west4`
- `Active job_id`: canonical Iris job id
- `Metrics counts`: base and instruct completed shard counts
- `Last evidence of progress`: exact timestamp + shard count change
- `Last evidence of throttling`: exact timestamp + representative `429` line if present
- `Next region in queue`: next unfinished region from the ordered list above
- `Resubmit command`: exact region-specific command above

#### Immediate next action

- Keep monitoring `us-central1` only.
- Do not relaunch any other region until `us-central1` finishes or clearly requires controlled resubmission.

### 2026-03-24 — ALIGN-030: Long-Running End-to-End Task Framing (Downloads -> Smoke -> Full Spec)

- **User direction:** treat the remaining work as one long-running task with multiple phases:
  1. finish the sequential multi-region Mixtral download campaign
  2. then run the heterogeneous alignment pipeline in `us-central1` with:
     - chosen model = Llama 3.3 70B Instruct
     - rejected model = Mixtral 8x7B Instruct
  3. once the one-statement run works, launch a separate whole-spec job to measure wall-clock runtime
- **Ordering constraint from user:** do not start the heterogeneous alignment runs until the download campaign described in ALIGN-029 is complete.

#### Verified `us-central1` artifact availability at plan time

- Confirmed present in `gs://marin-us-central1/models/`:
  - `meta-llama--Llama-3-3-70B-Instruct--6f6073b/`
  - `mistralai--Mixtral-8x7B-v0-1--fc7ac94/` (partial, still downloading)
  - `mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230/` (partial, still downloading)
- Interpretation: `us-central1` is the correct region to anchor the heterogeneous alignment phase because it already contains the pinned Llama 70B artifact and is the active Mixtral download lane.

#### Global phase plan

**Phase A — Complete regional Mixtral downloads**
- Execute ALIGN-029 fully:
  - finish `us-central1`
  - resume and finish `us-east5-a`
  - resume and finish `us-east1-d`
  - resume and finish `europe-west4`
- Keep strict one-region-at-a-time discipline.

**Phase B — Prepare the heterogeneous alignment smoke run in `us-central1`**
- Create a dedicated smoke experiment script, expected shape:
  - `experiments/align_debug_vllm_70b_mixtral_rejected.py`
- The smoke script should stay data-generation-only (`dpo_config=None`) until heterogeneous generation is validated.
- Default role assignment for the smoke run:
  - `pretrained_model`: Llama 3.3 70B Instruct
  - `teacher_model`: Llama 3.3 70B Instruct
  - `rejected_model`: Mixtral 8x7B Instruct
  - `ideation_model`: Llama 3.3 70B Instruct
  - `extract_model`: Llama 3.3 70B Instruct
  - `judge_model`: Llama 3.3 70B Instruct
- Rationale: change only the rejected-model role first. Keep the previously validated 70B path for the other local roles to isolate the behavioral delta.

**Phase C — Run one-statement heterogeneous smoke job in `us-central1`**
- Use the same debug envelope that validated the all-Llama lane unless a smaller code change forces a clear exception:
  - region/zone: `us-central1` / `us-central1-a`
  - TPU: `v5p-8`
  - `tensor_parallel_size=4`
  - `max_model_len=4096`
  - `ram=256g`
  - `disk=500g`
- Keep:
  - `covering_strength=2`
  - `statement_ids=["ask_clarifying_questions"]`
  - `teacher_n=1`
  - `rejected_n=1`
- Smoke success criterion:
  - prompts succeed
  - chosen succeeds with Llama 70B
  - rejected succeeds with Mixtral
  - preference_pairs is non-empty
  - artifacts are written under a distinct smoke prefix

**Phase D — Run a separate whole-spec heterogeneous job in `us-central1`**
- Only after the smoke job succeeds.
- Launch a separate experiment/job, expected shape:
  - `experiments/align_vllm_70b_mixtral_rejected_full_spec.py`
- Keep it separate from smoke for two reasons:
  - wall-clock runtime is directly measurable as its own root job
  - smoke artifacts and full-spec artifacts stay disentangled
- Initial whole-spec goal is still data generation / preference pair construction, not DPO training.
- Default first full-spec configuration:
  - same heterogeneous role split as smoke
  - remove the `statement_ids` filter
  - keep pairwise covering first unless the user later explicitly wants the full 3-way Bloom-style sweep
- Rationale: the immediate objective is infrastructure/runtime characterization of the heterogeneous local-vLLM path over the full spec, not yet the final production-scale training recipe.

#### Reuse rule for the alignment phase

- Prefer reusing the already-downloaded `us-central1` model artifacts rather than raw HF IDs.
- Desired canonical model sources for the alignment phase:
  - Llama: `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  - Mixtral instruct: `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230`
- Important implementation note:
  - `VLLMConfig.model` accepts a model identifier or path, but this repo does not yet show a proven pattern for passing these exact regional GCS prefixes into vLLM.
  - Therefore, before Phase C submission, perform a small implementation/verification step:
    - either confirm vLLM can load directly from the regional model path in this runtime
    - or add the minimal staging/materialization helper needed to map the regional artifact to a local path on the worker
- Explicit anti-goal: do not fall back to raw HF repo IDs if a regional artifact path can be made to work. The whole point of the download campaign is to avoid repeating HF throttling during alignment runs.

#### Concrete experiment progression after downloads complete

1. Verify `us-central1` Mixtral base and instruct prefixes are complete (`36/36` each).
2. Verify the pinned Llama 70B prefix in `us-central1` is usable for local vLLM inference.
3. Implement/patch the one-statement heterogeneous smoke script.
4. Submit the smoke script as its own Iris job in `us-central1-a`.
5. Babysit the smoke job to terminal success.
6. If smoke succeeds, implement/patch the full-spec script as a separate file or clearly separate configuration entrypoint.
7. Submit the full-spec script as a separate Iris job in `us-central1-a`.
8. Babysit the full-spec job and record start/end times, step transitions, and artifact counts.

#### Monitoring and handoff contract for future agents

- Treat the whole thread as a queue with one active phase at a time.
- The active phase must always be one of:
  - `download-us-central1`
  - `download-us-east5-a`
  - `download-us-east1-d`
  - `download-europe-west4`
  - `align-smoke-us-central1`
  - `align-full-spec-us-central1`
- Every handoff should record:
  - `Active phase`
  - `Active job_id`
  - `Current blockers`
  - `Last evidence of progress`
  - `Artifacts/prefixes in play`
  - `Exact next command to run`
- Do not leave the next agent to infer sequencing from memory.

#### Separate-job naming guidance

- Keep smoke and full-spec as different root job names and different alignment `name=` values.
- Recommended naming pattern:
  - smoke: `debug_vllm_70b_mixtral_rejected_smoke`
  - full spec: `debug_vllm_70b_mixtral_rejected_full_spec`
- This ensures:
  - separate job-level wall-clock timing
  - separate output prefixes
  - no ambiguity when reading the data browser or babysitting logs

#### Immediate next action after this log entry

- Continue executing ALIGN-029 until the sequential regional download campaign is fully complete.
- After that, move to the `us-central1` heterogeneous smoke implementation and launch path described above.

### 2026-03-24 — ALIGN-031: Resume Contract After Compaction (Central Lane Live)

- **User reaffirmation:** continue the long-running queue from ALIGN-029 and ALIGN-030 without resetting scope:
  1. finish Mixtral downloads in `us-central1`
  2. then finish the remaining regions sequentially
  3. then run the heterogeneous alignment smoke job in `us-central1`
  4. then run the separate whole-spec alignment job in `us-central1`
- **Additional user requirement:** when context compacts, future agents must be told to keep following this exact sequence rather than replanning from scratch.

#### Resume-here summary

- `Last experiment`: ALIGN-030 — long-running queue defined as downloads -> smoke -> full spec
- `Active phase`: `download-us-central1`
- `Active job_id`: `/ahmed/download-mixtral-models-us-central1`
- `Current state`: root job `JOB_STATE_RUNNING`; both child download pipelines still active
- `Current shard counts` at `2026-03-24 00:22:22 PDT`:
  - base: `7/36`
  - instruct: `7/36`
- `Last evidence of progress`:
  - base `model-00004-of-00019.safetensors` streamed successfully at `2026-03-24 07:19:39 UTC`
  - instruct `model-00004-of-00019.safetensors` streamed successfully at `2026-03-24 07:18:57 UTC`
- `Last evidence of throttling`:
  - repeated Hugging Face `429 Too Many Requests` lines at `2026-03-24 07:20:44 UTC`
  - representative backoff: `Retrying ... consolidated.03.pt in 900.0s`
- `Artifacts/prefixes in play`:
  - `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-v0-1--fc7ac94`
  - `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230`
- `Monitor state file`: `scratch/20260323-2334_monitoring_state.json`
- `Monitor script`: `scratch/20260323-2334_monitor_mixtral_downloads.py`

#### Required behavior for future agents after compaction

- Read ALIGN-029, ALIGN-030, and ALIGN-031 first.
- Do not launch another region while `download-us-central1` is still active.
- Do not change output prefixes; reuse the existing regional prefixes and rely on shard-level reuse.
- Keep one monitoring owner only.
- When handing off, restate:
  - `Active phase`
  - `Active job_id`
  - base/instruct shard counts
  - most recent successful shard upload timestamps
  - most recent `429` timestamp and retry delay
  - exact next command to run

#### Queue order that must remain in force

1. Finish `download-us-central1`
2. Submit `download-us-east5-a`
3. Submit `download-us-east1-d`
4. Submit `download-europe-west4`
5. Implement and run `align-smoke-us-central1`
6. Implement and run `align-full-spec-us-central1`

#### Exact next commands once the current phase changes

After `download-us-central1` succeeds:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name download-mixtral-models-us-east5-a \
  --cpu 4 \
  --memory 16GB \
  --disk 20GB \
  --region us-east5 \
  --zone us-east5-a \
  -- python experiments/download_mixtral_models.py
```

After the regional queue is complete:

```bash
uv run python experiments/align_debug_vllm_70b_mixtral_rejected.py
```

Then:

```bash
uv run python experiments/align_vllm_70b_mixtral_rejected_full_spec.py
```

#### Immediate next action

- Keep monitoring `/ahmed/download-mixtral-models-us-central1`.
- Do not advance the queue until `us-central1` reaches a terminal state.

### 2026-03-24 — ALIGN-032: GCS-Backed vLLM Reuse Finding During Download Monitoring

- **Why this entry exists:** while `download-us-central1` is still running, I used the monitor idle time to inspect the later alignment-phase model-loading path so the future smoke step is not blocked by preventable re-discovery work.

#### Confirmed code-path asymmetry

- The repo already has a server-based vLLM path that understands object-store-backed checkpoints:
  - `lib/marin/src/marin/inference/vllm_server.py`
  - if `ModelConfig.path` is `gs://...` or `s3://...`, `_maybe_enable_streaming()` auto-adds `load_format="runai_streamer"` unless explicitly overridden
- There is already a reusable validation entrypoint for this behavior:
  - `lib/marin/src/marin/inference/vllm_smoke_test.py`
  - it accepts `--model gs://...` and optional `--load-format`
- The current alignment path does **not** yet have equivalent remote-model support:
  - `lib/marin/src/marin/alignment/llm_client.py` constructs `vllm.LLM(model=config.model, ...)` directly
  - `lib/marin/src/marin/alignment/inference_config.py` currently has no `load_format`, `model_path`, or staging/materialization field on `VLLMConfig`

#### Consequence for the later alignment smoke phase

- Before launching `align-smoke-us-central1`, run a dedicated vLLM smoke validation against the downloaded `us-central1` artifacts rather than assuming the direct alignment code path can read them.
- Preferred verification target order:
  1. `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  2. `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230`
- If the server-based smoke path works with the regional prefixes, the next implementation task is to make alignment reuse that mechanism rather than falling back to raw HF IDs.

#### Most likely implementation options after downloads finish

1. Extend `VLLMConfig` and `llm_client.py` so the direct alignment vLLM path can pass `load_format="runai_streamer"` when `model` is a remote object-store URI.
2. If direct remote loading is not viable for the in-process alignment path, add a minimal local staging/materialization helper using the existing `marin.evaluation.utils.download_from_gcs()` primitive.

#### Explicit anti-regression reminder

- Do not quietly switch the heterogeneous alignment experiment back to raw HF IDs just because the direct alignment path is missing remote-model support today.
- The correct order remains:
  - finish regional downloads
  - verify `gs://` model usability
  - then patch the alignment path as needed

### 2026-03-24 — ALIGN-033: Queue Monitor Upgraded to Auto-Advance Sequential Downloads

- **Why this matters:** the user asked to set this up so it can keep running for many hours and survive compaction/handoffs. Manual one-region polling was too brittle for that.

#### What changed

- `scratch/20260323-2334_monitor_mixtral_downloads.py` now owns the sequential regional download queue, not just the active region.
- It now:
  - monitors the current region every 5 minutes
  - writes shard counts and recent log summaries into `scratch/20260323-2334_monitoring_state.json`
  - resubmits the current region on terminal non-success using the same job name / output prefixes
  - automatically advances to the next region in queue on terminal success
  - exits only after the four download regions finish, leaving `active_phase=align-smoke-us-central1`

#### Queue order now encoded in the stateful monitor

1. `download-us-central1`
2. `download-us-east5-a`
3. `download-us-east1-d`
4. `download-europe-west4`
5. `align-smoke-us-central1`
6. `align-full-spec-us-central1`

#### Current live state at `2026-03-24 00:31:30 PDT`

- `Active phase`: `download-us-central1`
- `Active job_id`: `/ahmed/download-mixtral-models-us-central1`
- `Metrics counts`:
  - base: `12/36`
  - instruct: `12/36`
- `Last evidence of progress` recorded in monitor state:
  - `07:31:02 UTC`
  - `mistralai/Mixtral-8x7B-v0.1@fc7ac94/model-00009-of-00019.safetensors`
- `Monitor state file` now includes:
  - `active_phase`
  - `queue_index`
  - `downloads_complete`
  - per-prefix `metrics_count`
  - `last_log_summary`

#### Future-agent instruction after compaction

- Do **not** manually launch `us-east5-a`, `us-east1-d`, or `europe-west4` if the queue monitor is still running; the script now handles that transition.
- First inspect:
  - `scratch/20260323-2334_monitoring_state.json`
  - the live monitor session if available
- If the monitor process is gone, restart it with:

```bash
uv run python scratch/20260323-2334_monitor_mixtral_downloads.py
```

- After the monitor flips `active_phase` to `align-smoke-us-central1`, continue with ALIGN-030 and ALIGN-032 rather than inventing a new sequence.

### 2026-03-24 — ALIGN-034: Alignment vLLM Path Patched for Remote Checkpoints

- **Motivation:** ALIGN-032 identified that the alignment path was missing the remote-checkpoint handling already present in Marin's server-based vLLM stack.

#### Code changes completed

- `lib/marin/src/marin/alignment/inference_config.py`
  - added `load_format: str | None = None` to `VLLMConfig`
- `lib/marin/src/marin/alignment/llm_client.py`
  - added automatic `load_format="runai_streamer"` when `VLLMConfig.model` is a `gs://...` or `s3://...` URI and no explicit override is set
  - preserved explicit overrides such as `runai_streamer_sharded`
- `tests/test_alignment.py`
  - added regression coverage for:
    - remote `gs://...` models defaulting to `runai_streamer`
    - explicit `load_format` override preservation

#### Verification

- Ran:

```bash
uv run pytest tests/test_alignment.py -q
```

- Result: `67 passed in 1.24s`

#### What this does and does not prove

- **Now true:** the shared alignment vLLM client can carry a remote-checkpoint load format through `VLLMConfig`, which should unlock prompt generation, response generation, and judging because they all share `llm_client.py`.
- **Still not yet proven:** a full alignment local-vLLM run against the actual regional `gs://` model prefixes on TPU hardware.
- Therefore the next validation step after downloads complete is still:
  - run a dedicated `vllm_smoke_test.py` check against the completed `us-central1` prefixes
  - then run the heterogeneous one-statement alignment smoke job

### 2026-03-24 — ALIGN-035: Heterogeneous us-central1 Smoke Entrypoint Prepared

- Added:
  - `experiments/align_debug_vllm_70b_mixtral_rejected.py`
- Purpose:
  - one-statement heterogeneous smoke run
  - chosen / infrastructure roles use the regional `us-central1` Llama 3.3 70B checkpoint
  - rejected role uses the regional `us-central1` Mixtral 8x7B Instruct checkpoint
- Key configuration:
  - `name="debug_vllm_70b_mixtral_rejected_smoke"`
  - `statement_ids=["ask_clarifying_questions"]`
  - `covering_strength=2`
  - `teacher_n=1`
  - `rejected_n=1`
  - `dpo_config=None`
  - `v5p-8`, `tp=4`, `max_model_len=4096`, `ram=256g`, `disk=500g`
- Model sources encoded in the script:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  - `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230`

#### Verification

- Ran:

```bash
uv run python -m py_compile experiments/align_debug_vllm_70b_mixtral_rejected.py
```

- Result: success

#### Remaining gate before submission

- Do not submit this experiment immediately when the file exists.
- First complete the download queue and run the `vllm_smoke_test.py` verification against the finished `us-central1` GCS prefixes.

### 2026-03-24 — ALIGN-036: Full-Spec us-central1 Entrypoint Prepared

- Added:
  - `experiments/align_vllm_70b_mixtral_rejected_full_spec.py`
- Purpose:
  - separate whole-spec runtime characterization job after the smoke lane succeeds
  - keeps wall-clock timing and artifacts separate from the one-statement smoke run
- Key configuration:
  - `name="debug_vllm_70b_mixtral_rejected_full_spec"`
  - same regional `us-central1` Llama + Mixtral GCS prefixes as ALIGN-035
  - `statement_ids=None`
  - `covering_strength=2`
  - `teacher_n=1`
  - `rejected_n=1`
  - `dpo_config=None`
  - `v5p-8`, `tp=4`, `max_model_len=4096`, `ram=256g`, `disk=500g`

#### Verification

- Ran:

```bash
uv run python -m py_compile experiments/align_vllm_70b_mixtral_rejected_full_spec.py
```

- Result: success

#### Ordering reminder

- This file existing does **not** change the execution order.
- The order remains:
  1. finish the regional download queue
  2. validate `us-central1` GCS prefixes with `vllm_smoke_test.py`
  3. run ALIGN-035 one-statement smoke
  4. only then run ALIGN-036 full-spec job

### 2026-03-24 — ALIGN-037: Exact us-central1 GCS vLLM Smoke Commands Recorded

- **Reason for this entry:** after ALIGN-034 and ALIGN-035/036, the next post-download gate is no longer ambiguous. It should be an explicit GCS-path vLLM validation on TPU, not another planning pass.

#### Run only after the download queue completes

- Wait for `scratch/20260323-2334_monitoring_state.json` to report:
  - `downloads_complete=true`
  - `active_phase=align-smoke-us-central1`

#### Llama 70B GCS smoke validation

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name vllm-smoke-llama-70b-gcs-us-central1 \
  --extra marin:tpu \
  --extra marin:vllm \
  --tpu v5p-8 \
  --region us-central1 \
  --zone us-central1-a \
  -- python lib/marin/src/marin/inference/vllm_smoke_test.py \
    --local \
    --mode native \
    --max-model-len 4096 \
    --model gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b \
    --prompt "Write a short haiku about TPUs."
```

#### Mixtral instruct GCS smoke validation

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name vllm-smoke-mixtral-8x7b-gcs-us-central1 \
  --extra marin:tpu \
  --extra marin:vllm \
  --tpu v5p-8 \
  --region us-central1 \
  --zone us-central1-a \
  -- python lib/marin/src/marin/inference/vllm_smoke_test.py \
    --local \
    --mode native \
    --max-model-len 4096 \
    --model gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230 \
    --prompt "Write a short haiku about TPUs."
```

#### Success criterion

- Both jobs reach terminal success and return a non-empty completion from the `gs://` model path.
- Only after that should ALIGN-035 be submitted.

### 2026-03-24 — ALIGN-038: Lowered Mixtral HF Parallelism for Future Regions

- **Trigger:** during `download-us-central1`, the remaining `consolidated.*.pt` shards still hit Hugging Face resolver quota limits even after full region serialization.
- **Observed pattern:** around `07:55 UTC`, multiple remaining consolidated shards reached active streaming throughput and then all failed with fresh `429` responses, each entering `900s` backoff.

#### Code change

- Updated `experiments/models.py`:
  - `ModelConfig` now carries `zephyr_max_parallelism`
  - `download_model_step(...)` passes that through to `DownloadConfig`
  - `mixtral_8x7b` now uses `zephyr_max_parallelism=2`
  - `mixtral_8x7b_instruct` now uses `zephyr_max_parallelism=2`

#### Verification

- Ran:

```bash
uv run python -c 'from experiments.models import mixtral_8x7b, mixtral_8x7b_instruct; print(mixtral_8x7b.config.zephyr_max_parallelism, mixtral_8x7b_instruct.config.zephyr_max_parallelism)'
```

- Result:

```text
2 2
```

#### Scope / consequence

- **Current live `us-central1` run:** unchanged, because it was already launched with the previous config.
- **Future queue phases and any controlled resubmission:** will pick up the lower Mixtral parallelism automatically because the queue monitor submits `python experiments/download_mixtral_models.py` from the current workspace.
- Intended effect:
  - fewer simultaneous HF resolver streams
  - lower chance of repeated `429` + `900s` retry plateaus
  - same output prefixes, so completed shard work is still reused

### 2026-03-24 — ALIGN-039: Controlled us-central1 Resubmission After Repeated 429 Plateaus

- **Why the restart happened:** after ALIGN-038 landed, the live `us-central1` run was still the old high-parallelism launch. The remaining six consolidated shards on both base and instruct had already:
  - reached `30/36`
  - streamed several GiB
  - hit repeated `429` failures
  - fallen into repeated `900s` backoff windows
- At that point, waiting on the old launch was less attractive than restarting the same phase with the new lower Mixtral parallelism, while preserving the already-completed shard work under the same prefixes.

#### Action taken

- Stopped:
  - `/ahmed/download-mixtral-models-us-central1`
- Restarted the queue monitor immediately so it would observe `JOB_STATE_KILLED` and exercise its built-in resubmit path.
- The monitor then resubmitted `download-us-central1` with:
  - `restart_count=1`
  - same root job name
  - same output prefixes
  - updated workspace code (Mixtral `zephyr_max_parallelism=2`)

#### Evidence recorded

- `scratch/20260323-2334_monitoring_state.json` now shows:
  - `active_phase=download-us-central1`
  - `restart_count=1`
  - `last_transition.action=resubmit`
  - `last_transition.timestamp=2026-03-24 08:20:51 UTC`
- Fresh Iris status after resubmit:
  - root job `/ahmed/download-mixtral-models-us-central1` is back in `JOB_STATE_RUNNING`
  - new child pipeline id includes `zephyr-download-hf-d7d7cc27-p0-a0`

#### Important interpretation for future agents

- The current live `us-central1` job is **not** the original launch from ALIGN-033 anymore.
- It is a controlled resubmission specifically intended to finish the last incomplete shards with lower HF concurrency while reusing the existing successful shards already present in GCS.

### 2026-03-24 — ALIGN-040: us-central1 Completed; Queue Advanced to us-east5-a

- **Outcome:** the controlled `us-central1` resubmission from ALIGN-039 succeeded.
- Final `us-central1` artifact state:
  - `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-v0-1--fc7ac94` -> `36/36`
  - `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230` -> `36/36`
- Root state:
  - `/ahmed/download-mixtral-models-us-central1` reached `JOB_STATE_SUCCEEDED`
- Log evidence:
  - `Streamed all files and wrote provenance JSON`
  - successful final `consolidated.07.pt` write for instruct

#### Queue transition

- After central success, the queue monitor advanced to `download-us-east5-a`.
- The first east5 attempts using the old fixed root job name re-entered an immediately killed record instead of creating a live fresh run.
- To fix that, the queue monitor was patched to stop passing `--job-name` on region submissions, while preserving the same model output prefixes.

#### Current live east5 state

- Active phase: `download-us-east5-a`
- Current live job id now comes from Iris auto-naming, not the old fixed region job name:
  - `/ahmed/iris-run-download_mixtral_models-20260324-084623`
- Current east5 prefix reuse state at handoff time:
  - base prefix still has prior partial work: `3/36`
  - instruct prefix still has prior partial work: `3/36`
- Fresh Iris status for the auto-named east5 run shows:
  - root `JOB_STATE_RUNNING`
  - child worker groups running with `2` tasks each (lower Mixtral parallelism)

#### Future-agent rule update

- For download phases after ALIGN-040, do not assume the active job id equals a fixed `/ahmed/download-mixtral-models-<region>` root name.
- The source of truth is now:
  - `scratch/20260323-2334_monitoring_state.json`
- The output prefixes remain stable by region; only the Iris root job ids may rotate.

### 2026-03-24 — ALIGN-041: All Four Mixtral Regional Downloads Completed; Alignment Is Next

- **Verification method:** checked both Iris root-job history and each regional GCS prefix's `.metrics/success-part-*.jsonl` counts on `2026-03-24`.

#### Final regional artifact counts

- `us-central1`
  - `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-v0-1--fc7ac94` -> `36/36`
  - `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230` -> `36/36`
- `us-east5-a`
  - `gs://marin-us-east5/models/mistralai--Mixtral-8x7B-v0-1--fc7ac94` -> `36/36`
  - `gs://marin-us-east5/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230` -> `36/36`
- `us-east1-d`
  - `gs://marin-us-east1/models/mistralai--Mixtral-8x7B-v0-1--fc7ac94` -> `36/36`
  - `gs://marin-us-east1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230` -> `36/36`
- `europe-west4`
  - `gs://marin-eu-west4/models/mistralai--Mixtral-8x7B-v0-1--fc7ac94` -> `36/36`
  - `gs://marin-eu-west4/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230` -> `36/36`

#### Iris completion evidence

- The latest regional download root job visible in the queue is:
  - `/ahmed/iris-run-download_mixtral_models-20260324-103240`
- Its root state is `JOB_STATE_SUCCEEDED`.
- The final worker records under that root are `JOB_STATE_KILLED` because the parent completed and tore down workers after successful streaming; the pipeline stage records themselves are `JOB_STATE_SUCCEEDED`.

#### Queue / handoff state

- `scratch/20260323-2334_monitoring_state.json` now records:
  - `downloads_complete=true`
  - `active_phase=align-smoke-us-central1`
  - `next_action=uv run python experiments/align_debug_vllm_70b_mixtral_rejected.py`
- No local monitor process or local alignment launch process was still running at the time of this refresh; future agents should treat the download campaign as complete and resume from the smoke-alignment step, not from any regional download step.

#### Continuity rule for future agents

- Do **not** relaunch Mixtral download jobs unless a prefix is later shown to be incomplete or corrupt.
- Reuse the completed regional prefixes above.
- The immediate next execution step for this research thread is:
  - run the one-statement `us-central1` heterogeneous alignment smoke job from ALIGN-035 / ALIGN-037
- After that smoke run succeeds, continue to the separate full-spec timing job from ALIGN-036.

### 2026-03-24 — ALIGN-042: Branch-State Refresh Before Alignment Submission

- **Trigger:** post-download continuity refresh requested before the first heterogeneous alignment launch.
- **Purpose:** capture the exact branch/runtime state so future agents do not assume the smoke run has already been submitted.

#### Runtime status

- Checked for the expected `us-central1` alignment jobs:

```bash
uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /ahmed/iris-run-align_debug_vllm_70b_mixtral_rejected
uv run iris --config lib/iris/examples/marin.yaml job list --json --prefix /ahmed/iris-run-align_vllm_70b_mixtral_rejected_full_spec
```

- Both commands returned `[]`.
- Interpretation:
  - the download campaign is complete
  - neither the one-statement heterogeneous smoke job nor the full-spec timing job has been launched yet

#### Dependency update

- Pinned Marin's `litellm` dependency from a broad lower bound to an exact version:
  - `lib/marin/pyproject.toml`: `litellm==1.81.5`
- Refreshed the workspace lockfile:

```bash
uv lock
```

- Lockfile result:
  - `uv.lock` now records `litellm==1.81.5`
  - `uv lock` completed successfully
  - unrelated resolver warning observed: `wandb==0.24.0` is yanked

#### Relevant working-tree state

- Thread-local modified files now include:
  - `experiments/models.py`
  - `lib/marin/src/marin/alignment/inference_config.py`
  - `lib/marin/src/marin/alignment/llm_client.py`
  - `lib/marin/src/marin/alignment/align.py`
  - `tests/test_alignment.py`
  - `lib/marin/pyproject.toml`
  - `uv.lock`
- Thread-local untracked experiment entrypoints now include:
  - `experiments/download_mixtral_models.py`
  - `experiments/align_debug_vllm_70b_mixtral_rejected.py`
  - `experiments/align_vllm_70b_mixtral_rejected_full_spec.py`
- Historical local baseline script still present:
  - `experiments/align_debug_vllm_70b.py`

#### Immediate next action

- The next execution step remains unchanged:
  - run the one-statement `us-central1` heterogeneous alignment smoke job using the completed `gs://marin-us-central1/...` Llama 3.3 70B and Mixtral artifacts
- Only after that smoke run succeeds should the separate full-spec timing job be submitted.

#### Future-agent continuity rule

- Do not infer that alignment has already started just because `active_phase=align-smoke-us-central1` is recorded in the monitor state file.
- The source of truth after ALIGN-042 is:
  - downloads are complete
  - alignment scripts exist locally
  - no alignment Iris job exists yet
  - `litellm` is now pinned exactly to `1.81.5`

### 2026-03-24 — ALIGN-043: Launching us-central1 Heterogeneous Smoke Gate Sequence

- **Trigger:** user confirmed to proceed with the one-statement heterogeneous smoke run using Mixtral for rejected responses.
- **Goal of this entry:** record the exact preflight and submission sequence before execution so future agents can resume cleanly if the session compacts mid-run.

#### Gate order being executed

1. Run the recorded `us-central1` Llama 3.3 70B GCS `vllm_smoke_test.py` job.
2. Run the recorded `us-central1` Mixtral 8x7B Instruct GCS `vllm_smoke_test.py` job.
3. Inspect both jobs for terminal success and a non-empty completion from the `gs://` model path.
4. Only if both succeed, submit:
   - `experiments/align_debug_vllm_70b_mixtral_rejected.py`

#### Exact commands queued for execution

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name vllm-smoke-llama-70b-gcs-us-central1 \
  --extra marin:tpu \
  --extra marin:vllm \
  --tpu v5p-8 \
  --region us-central1 \
  --zone us-central1-a \
  -- python lib/marin/src/marin/inference/vllm_smoke_test.py \
    --local \
    --mode native \
    --max-model-len 4096 \
    --model gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b \
    --prompt "Write a short haiku about TPUs."

uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name vllm-smoke-mixtral-8x7b-gcs-us-central1 \
  --extra marin:tpu \
  --extra marin:vllm \
  --tpu v5p-8 \
  --region us-central1 \
  --zone us-central1-a \
  -- python lib/marin/src/marin/inference/vllm_smoke_test.py \
    --local \
    --mode native \
    --max-model-len 4096 \
    --model gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230 \
    --prompt "Write a short haiku about TPUs."
```

#### Pending submission if both preflights pass

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --extra marin:tpu \
  --tpu v5p-8 \
  --region us-central1 \
  --zone us-central1-a \
  -- python experiments/align_debug_vllm_70b_mixtral_rejected.py
```

#### Continuity rule for future agents

- If this session compacts before the align smoke job is submitted:
  - first check terminal status of the two `vllm-smoke-*` jobs above
  - do not resubmit either preflight job if it already succeeded
  - submit the heterogeneous alignment smoke only after confirming both preflights passed

### 2026-03-24 — ALIGN-044: us-central1 Preflight Jobs Submitted; Blocked on TPU Capacity

- **Submission time:** around `17:47 UTC`
- **Action taken:** submitted both recorded `us-central1` GCS vLLM preflight jobs.

#### Submitted jobs

- Llama GCS preflight:
  - `/ahmed/vllm-smoke-llama-70b-gcs-us-central1`
- Mixtral GCS preflight:
  - `/ahmed/vllm-smoke-mixtral-8x7b-gcs-us-central1`

#### First scheduler observation

- Both jobs are currently `JOB_STATE_PENDING`.
- Both report the same blocking reason:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - autoscaler waiting on scale group `tpu_v5p_8-us-central1-a` with demand-routed workers
- There is no execution traceback yet and no child logs beyond controller/tunnel setup, so this is capacity wait rather than a runtime/model failure.

#### Alignment submission status

- The heterogeneous one-statement align job has **not** been submitted yet.
- It remains gated on:
  - successful completion of `/ahmed/vllm-smoke-llama-70b-gcs-us-central1`
  - successful completion of `/ahmed/vllm-smoke-mixtral-8x7b-gcs-us-central1`

#### Future-agent continuity rule

- Treat this as a scheduler-capacity wait, not as a failed preflight.
- Do not resubmit the preflight jobs just because they are pending.
- Next action is:
  - keep monitoring both preflight jobs
  - once both succeed and emit non-empty completions, submit `experiments/align_debug_vllm_70b_mixtral_rejected.py`

### 2026-03-24 — ALIGN-045: GCS vLLM Preflights Failed Due Bad Outer Resource Envelope; Current Align vLLM Path Is Not Single-Model Serialized

- **Why this entry exists:** the first `vllm_smoke_test.py` launch attempt produced actionable negative evidence on both the launch recipe and the current alignment architecture.

#### Preflight outcome

- Both standalone `us-central1` GCS preflight jobs failed after TPU allocation:
  - `/ahmed/vllm-smoke-llama-70b-gcs-us-central1`
  - `/ahmed/vllm-smoke-mixtral-8x7b-gcs-us-central1`
- Failure signature from both jobs:
  - vLLM API server started bootstrapping
  - process exited with code `-9`
  - kernel OOM-killed the container
- Root cause:
  - the outer `iris job run` command used the default top-level envelope (`1GB` memory, `5GB` disk)
  - `vllm_smoke_test.py --local` runs the vLLM server inside that top-level task container
  - so the standalone preflight command was invalid for 70B / Mixtral despite using a TPU
- Important scope note:
  - this `1GB` mistake applies to the standalone preflight commands from ALIGN-037
  - it does **not** mean the align pipeline steps themselves use `1GB`; the actual align chosen/rejected steps request `VLLMConfig.resources` (`disk`/`ram`) via `align.py`

#### Critical architecture finding: chosen / rejected are not currently guaranteed one-model-at-a-time

- In `align.py`, chosen and rejected are separate executor steps:
  - `chosen_step` uses `teacher_model.resources`
  - `rejected_step` uses `rejected_model.resources`
  - both depend only on `prompts_step`
- Executor behavior:
  - ready steps run in parallel by default when `max_concurrent` is unset
- Consequence:
  - the current heterogeneous smoke pipeline can try to run:
    - chosen generation on Llama 3.3 70B
    - rejected generation on Mixtral 8x7B
    - at the same time as separate TPU jobs
- This violates the intended compute / memory policy of loading only one heavyweight vLLM model at a time.

#### Additional vLLM lifecycle finding inside a single process

- `llm_client.py` maintains a module-level cache keyed by full `VLLMConfig` and explicitly supports multiple simultaneous engines.
- `generate_prompts_from_spec(...)` enters both `ideation_model` and `extract_model` vLLM contexts up front if both are local.
- Therefore, even within one worker process, the current abstraction allows more than one model to stay resident simultaneously when configs differ.

#### Working conclusion

- The current vLLM path is acceptable only if:
  - every local vLLM role uses the same model/config, or
  - extra concurrent TPU/model residency is acceptable
- That is **not** true for the intended Llama-chosen / Mixtral-rejected heterogeneous run.

#### Required direction from this point

- Do not submit the heterogeneous align smoke run unchanged.
- First fix the architecture so local vLLM roles are serialized explicitly:
  - one model loaded at a time
  - chosen then rejected, not parallel sibling steps
  - prompt-stage vLLM contexts also loaded stage-by-stage rather than preloading multiple local models

### 2026-03-24 — ALIGN-046: Isolated `generate_responses.py` Experiment Planned on Succeeded Prompt Artifact

- **Trigger:** user requested a smaller experiment that exercises the existing response-generation primitive directly instead of the full `align()` pipeline.
- **Goal:** run a single `generate_responses.py` executor step on an already-succeeded prompt set to validate the primitive independently of prompt generation and judging.

#### Prompt source chosen

- Reuse the prompt artifact from the validated 70B run in ALIGN-026:
  - `gs://marin-us-east5/align/debug_vllm_70b/prompts-5887d2`
- Reason:
  - known-good prompts output
  - already associated with a succeeded executor step
  - avoids re-running prompt generation

#### Model choice

- Target model for this experiment:
  - `meta-llama/Llama-3.1-70B-Instruct`
- Current branch note:
  - there is no existing `experiments/models.py` registry handle for Llama 3.1 70B in this branch
  - the experiment will therefore reference the HF model id directly in `VLLMConfig`

#### Execution shape

- Add a new one-step experiment script that:
  - imports `generate_responses`
  - points `prompts_path` at the succeeded GCS prompt artifact above
  - runs one vLLM response-generation step only
  - uses the same proven 70B TPU envelope (`v5p-8`, `tp=4`, `max_model_len=4096`, `ram=256g`, `disk=500g`)
- Intention:
  - isolate whether response generation itself is healthy on TPU for a large model
  - keep the test separate from the current heterogeneous align refactor discussion

#### Planned launch region

- Submit in `us-east5-a`.
- Reason:
  - the prompt artifact already lives in `marin-us-east5`
  - prior validated 70B TPU response-generation work in this thread also succeeded there

#### Future-agent rule

- This experiment is independent of the heterogeneous chosen/rejected serialization issue in ALIGN-045.
- It is meant to test the existing `generate_responses.py` primitive directly, not to bless the current full `align()` scheduling behavior.

### 2026-03-24 — ALIGN-047: `generate_responses.py`-Only Llama 3.1 70B Experiment Added and Submitted

- **Experiment script added:**
  - `experiments/generate_responses_llama_3_1_70b_existing_prompts.py`
- **Purpose:** run a single response-generation step against a succeeded prompt artifact without invoking full `align()`.

#### Script details

- Prompt source:
  - `gs://marin-us-east5/align/debug_vllm_70b/prompts-5887d2`
- Model:
  - `meta-llama/Llama-3.1-70B-Instruct`
- Step shape:
  - one `ExecutorStep`
  - `remote(generate_responses, ...)`
  - `VLLMConfig(tpu_type="v5p-8", tensor_parallel_size=4, max_model_len=4096, ram="256g", disk="500g")`
- Environment forwarding added specifically for this experiment:
  - `HF_TOKEN`
  - `HF_HOME`
  - `HF_HUB_CACHE`
  - `HUGGINGFACE_HUB_CACHE`

#### Verification before launch

- Ran:

```bash
uv run python -m py_compile experiments/generate_responses_llama_3_1_70b_existing_prompts.py
```

- Result:
  - syntax check passed

#### Submission

- Ran:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name generate-responses-llama-3-1-70b-existing-prompts \
  --extra marin:tpu \
  --tpu v5p-8 \
  --region us-east5 \
  --zone us-east5-a \
  -- python experiments/generate_responses_llama_3_1_70b_existing_prompts.py
```

- Submitted job id:
  - `/ahmed/generate-responses-llama-3-1-70b-existing-prompts`

#### Current runtime status at first check

- Root state:
  - `JOB_STATE_PENDING`
- Pending reason:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - autoscaler waiting for `tpu_v5p_8-us-east5-a`
- No child logs yet beyond controller / tunnel setup, so the run has not reached step execution.

#### Future-agent continuity rule

- If resuming from this point:
  - first check `/ahmed/generate-responses-llama-3-1-70b-existing-prompts`
  - if it is still pending, treat it as a capacity wait rather than a model/config failure
  - if it reaches running, inspect child logs for HF auth/download issues and for actual `generate_responses` progress

### 2026-03-24 — ALIGN-048: `us-east5-a` Queue Backoff on the Isolated Llama 3.1 70B Response Run

- **Follow-up check:** roughly two minutes after ALIGN-047 submission.
- Current job:
  - `/ahmed/generate-responses-llama-3-1-70b-existing-prompts`
- State remains:
  - `JOB_STATE_PENDING`
- Updated pending reason:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - `Autoscaler: Unsatisfied autoscaler demand: no_capacity: tpu_v5p_8-us-east5-a=backoff`

#### Interpretation

- The experiment has not reached step execution yet.
- There is still no evidence of a script/config/runtime failure.
- The current blocker is queue capacity in `us-east5-a`, now explicitly in autoscaler backoff.

#### Future-agent rule

- Do not classify this run as failed based on ALIGN-048 alone.
- Recheck the same job id first; only investigate model/script issues after the job actually starts running.

### 2026-03-24 — ALIGN-049: Isolated Llama 3.1 70B `generate_responses` Run Failed on HF Cache Disk Exhaustion

- **Run:** `/ahmed/generate-responses-llama-3-1-70b-existing-prompts`
- **Failed child step:** `/ahmed/generate-responses-llama-3-1-70b-existing-prompts/align-debug_generate_responses_llama_3_1_70b_existing_prompts-responses_d1ddd60b-4074fe60`

#### What actually happened

- The step got real TPU allocation and started executing.
- `generate_responses.py` loaded the succeeded prompt set correctly:
  - `Loaded 46 prompts for vLLM batch generation`
- vLLM initialization reached model loading for:
  - `meta-llama/Llama-3.1-70B-Instruct`
- Failure then occurred during Hugging Face weight download, before inference started.

#### Root cause evidence

- Logs show repeated warnings from `huggingface_hub`:
  - expected shard size around `4.6GB` to `5.0GB`
  - available free space in `/root/.cache/huggingface/.../blobs` dropped from about `4174 MB` to `0.00 MB`
- The engine then failed to start, and the remote step returned `Exit code: 1`.

#### Important interpretation

- This was **not** the earlier standalone `1GB` preflight mistake.
- The child response-generation step did request the intended large envelope:
  - `memory_bytes=274877906944` (`256GB`)
  - `disk_bytes=536870912000` (`500GB`)
- The real issue is that the HF download/cache path in the worker still landed on the small root filesystem (`/root/.cache/huggingface`) instead of a large mounted cache path on the requested disk.

#### Consequence for future runs

- A direct-HF 70B model id is currently unsafe unless the worker is forced to use a large HF cache directory.
- Better options from this point:
  - point `HF_HOME` / `HF_HUB_CACHE` inside the worker to a path on the large attached disk
  - or avoid direct HF download in the response step and use a pre-staged model artifact (preferred when available)

### 2026-03-24 — ALIGN-050: Corrected the Isolated Response Experiment to Use Regional GCS Artifacts Only

- **Correction applied after ALIGN-049:** the experiment was updated so it no longer points at a raw HF model id.

#### What changed

- New experiment file path:
  - `experiments/generate_responses_llama_3_3_70b_existing_prompts.py`
- Model source now comes from the registered model step in `experiments/models.py`:
  - `llama_3_3_70b_instruct`
- The experiment sets:
  - `model=output_path_of(llama_3_3_70b_instruct)`
- Why that matters:
  - executor resolves this to the region-local `gs://.../models/...` artifact path
  - no direct HF model download is needed at response-step runtime

#### Verified regional artifacts

- Verified `us-central1` staged model artifact exists:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b/...`
- Verified a succeeded central prompt shard exists:
  - `gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d/shard_00000.jsonl.gz`

#### Current intended run shape

- Region / zone:
  - `us-central1`
  - `us-central1-a`
- Prompt source:
  - `gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d`
- Model source:
  - regional GCS artifact resolved from `llama_3_3_70b_instruct`

#### Future-agent rule

- For large local-vLLM response tests, prefer `output_path_of(<download_model_step>)` over raw HF ids.
- Treat raw HF model ids as a bug for this thread unless the user explicitly asks for a live HF download.

### 2026-03-24 — ALIGN-051: Corrected GCS-Only Llama 3.3 70B Response Run Submitted in us-central1

- **Reason for new submission:** replace the earlier failed HF-based run with a job that uses only already-staged `us-central1` GCS artifacts.

#### Inputs

- Prompt source:
  - `gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d`
- Model source:
  - resolved from `output_path_of(llama_3_3_70b_instruct)`
  - expected concrete path in `us-central1`:
    - `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`

#### Submission

- Ran:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name generate-responses-llama-3-3-70b-gcs-us-central1 \
  --extra marin:tpu \
  --tpu v5p-8 \
  --region us-central1 \
  --zone us-central1-a \
  -- python experiments/generate_responses_llama_3_3_70b_existing_prompts.py
```

- Submitted job id:
  - `/ahmed/generate-responses-llama-3-3-70b-gcs-us-central1`

#### First status check

- Current state:
  - `JOB_STATE_PENDING`
- Pending reason:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - autoscaler waiting for `tpu_v5p_8-us-central1-a`
- No execution logs yet, so the corrected GCS-only step has not started running.

#### Future-agent rule

- Ignore `/ahmed/generate-responses-llama-3-1-70b-existing-prompts` for future progress; that is the obsolete HF-based failure.
- Continue from `/ahmed/generate-responses-llama-3-3-70b-gcs-us-central1`.

### 2026-03-24 — ALIGN-052: GCS Load Path Verified; vLLM Still Resolves the Llama 3.3 Checkpoint as Mistral on TPU

- **Run:** `/ahmed/generate-responses-llama-3-3-70b-gcs-us-central1`
- **Checked status:** root job and child response step were both `JOB_STATE_RUNNING` during this verification pass.

#### What is working

- `generate_responses.py` loaded the staged prompt set successfully:
  - `Loaded 76 prompts for vLLM batch generation`
- The response step is using the `us-central1` GCS model artifact, not a raw HF id:
  - `Using batch vLLM: gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
- vLLM explicitly selected GCS streaming mode:
  - `load_format='runai_streamer'`
- Engine init showed the checkpoint being materialized from the staged artifact into the local model-streamer cache:
  - `served_model_name=gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  - local cache path under `/root/.cache/vllm/assets/model_streamer/...`
- RunAI streamer reported successful bulk checkpoint transfer:
  - `Overall time to stream 262.8 GiB of all files to cpu: 73.87s, 3.6 GiB/s`
- Verified both `config.json` files under the model prefix, including the nested duplicated subdirectory:
  - `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b/config.json`
  - `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b/meta-llama--Llama-3-3-70B-Instruct--6f6073b/config.json`
- Both configs declare the expected architecture:
  - `architectures = ["LlamaForCausalLM"]`
  - `model_type = "llama"`

#### Remaining blocker

- Despite the correct staged configs, vLLM logged:
  - `Resolved architecture: MistralForCausalLM`
- TPU model loading then warned:
  - `Model architectures ['MistralForCausalLM'] not registered in tpu-inference. Falling back to vLLM-native Pytorch definition.`
- So the artifact source/path issue is fixed, but runtime model-class selection is still suspicious.

#### Current phase

- The job had not reached response generation yet during this check.
- Logs show it was still in TPU/JAX compile and warmup:
  - backbone and select-from-array precompile passes
  - kv-cache initialization for `80` layers

#### Interpretation

- The previous HF-download bug is resolved for this experiment.
- The next debugging target is inside the direct alignment vLLM path:
  - why a staged Llama 3.3 checkpoint is being resolved as `MistralForCausalLM`
  - whether that fallback is only a naming quirk or a real model-loader mismatch

#### Future-agent rule

- Treat GCS staging for this run as verified.
- Do **not** spend time debugging Hugging Face download paths for `/ahmed/generate-responses-llama-3-3-70b-gcs-us-central1`; that part is already fixed.
- Focus next on the architecture-resolution mismatch before using this path as the basis for heterogeneous chosen/rejected alignment runs.

### 2026-03-24 — ALIGN-053: `generate_responses` on GCS-Backed Llama 3.3 70B Succeeded; Startup Dominated Runtime

- **Run:** `/ahmed/generate-responses-llama-3-3-70b-gcs-us-central1`
- **Final state:** `JOB_STATE_SUCCEEDED`
- **Result path:** `gs://marin-us-central1/align/debug_generate_responses_llama_3_3_70b_existing_prompts/responses-44d0c7`
- **Experiment metadata:** `gs://marin-us-central1/experiments/generate_responses_llama_3_3_70b_existing_prompts-ac12f9.json`
- **Records written:** `76` responses in `shard_00000.jsonl.gz`

#### Timing methodology

- Queue and active-runtime numbers below use Iris `submitted_at` / `started_at` / `finished_at` metadata and are precise to the millisecond.
- Subphase timing inside the response worker uses log timestamps and should be treated as approximately `±1s`.
- All wall-clock times in the tables below are in `UTC`.

#### Whole job timeline

| Phase | Start | End | Duration | Notes |
| --- | --- | --- | ---: | --- |
| Root job pending in scheduler | `19:35:11.360` | `19:53:15.190` | `1083.830s` | Iris root `submitted_at -> started_at` |
| Root worker bootstrap + executor setup | `19:53:15.190` | `19:53:31` | `15.810s` | root start -> `### Launching 2 steps ###` |
| Model dependency step total | `19:53:31` | `19:54:21` | `50.000s` | `models/...` step begin -> succeeded |
| Model Zephyr streaming stage only | `19:53:57` | `19:54:20` | `23.000s` | `stage0-Map → Write` start -> provenance written |
| Gap from model-step success to child submit | `19:54:21` | `19:54:22.215` | `1.215s` | executor bookkeeping |
| Response child pending in scheduler | `19:54:22.215` | `20:04:22.427` | `600.212s` | Iris child `submitted_at -> started_at` |
| Response child active runtime | `20:04:22.427` | `20:14:43.162` | `620.735s` | Iris child `started_at -> finished_at` |
| Root finalize after child success | `20:14:43.162` | `20:15:03.310` | `20.148s` | executor shutdown + metadata write |
| Root active runtime total | `19:53:15.190` | `20:15:03.310` | `1308.120s` | Iris root `started_at -> finished_at` |

#### Response child step timeline

| Phase | Start | End | Duration | Notes |
| --- | --- | --- | ---: | --- |
| Child bootstrap before first worker log | `20:04:22.427` | `20:04:28` | `5.573s` | child start -> first `syncing deps` |
| Python/env bootstrap | `20:04:28` | `20:04:38` | `10.000s` | `syncing deps` -> `Using batch vLLM` |
| Response setup and prompt scan | `20:04:38` | `20:04:51` | `13.000s` | `Using batch vLLM` -> `Loaded 76 prompts` |
| Prompt load to engine call | `20:04:51` | `20:04:52` | `1.000s` | `Loaded 76 prompts` -> `Creating vLLM engine` |
| Early engine setup | `20:04:52` | `20:05:19` | `27.000s` | engine call -> `Resolved architecture: MistralForCausalLM` |
| GCS checkpoint streaming wall time | `20:04:52` | `20:06:49` | `117.000s` | engine call -> RunAI streamer completion log |
| GCS streamer reported payload transfer | `20:05:35` | `20:06:49` | `73.87s` | `262.8 GiB` at `3.6 GiB/s` |
| Post-stream setup before compile logs | `20:06:49` | `20:09:19` | `150.000s` | loader/setup gap before first precompile line |
| Compile/warmup phase 1 | `20:09:19` | `20:10:33` | `74.000s` | sample precompiles + gather-logprobs setup |
| Compile/warmup phase 2 | `20:10:33` | `20:13:58` | `205.000s` | kv-cache init + backbone/select precompiles |
| Compile/warmup phase 3 | `20:13:58` | `20:14:03` | `5.000s` | `compute_logits` compile |
| Compile/warmup phase 4 | `20:14:03` | `20:14:08` | `5.000s` | `structured_decoding` compile |
| vLLM internal engine-init total | `20:09:19` | `20:14:08` | `289.01s` | matches `init engine (profile, create kv cache, warmup model) took 289.01 seconds` |
| Full engine creation to ready | `20:04:52` | `20:14:08` | `556.000s` | full outer wall-clock from engine call -> ready |
| Request enqueue | `20:14:08` | `20:14:11` | `3.000s` | engine ready -> `Adding requests 100%` |
| Actual generation | `20:14:11` | `20:14:26` | `15.000s` wall-clock | progress bar reported `16s`; final estimated speeds `399.31` input toks/s and `728.16` output toks/s |
| Persist responses and step wrap-up | `20:14:26` | `20:14:43.162` | `17.162s` | result shard write -> child success |

#### Artifact verification

- Output prefix contents:
  - `.artifact`
  - `.executor_info`
  - `.executor_status`
  - `shard_00000.jsonl.gz`
- Verified `shard_00000.jsonl.gz` contains `76` JSONL rows.

#### Interpretation

- This run succeeded end-to-end using the staged `us-central1` GCS checkpoint.
- Runtime was dominated by startup:
  - about `556s` from `Creating vLLM engine` to engine-ready
  - only about `15-16s` for actual generation across all `76` prompts
- The queueing overhead was also material:
  - about `18m04s` before the root job started
  - about `10m00s` before the response child actually began executing

#### Remaining caveats

- The run still logged:
  - `Resolved architecture: MistralForCausalLM`
  - `Model architectures ['MistralForCausalLM'] not registered in tpu-inference. Falling back to vLLM-native Pytorch definition.`
- Near the end, vLLM also logged:
  - `Engine core proc EngineCore_DP0 died unexpectedly, shutting down client.`
- Despite that teardown noise, the step wrote the response artifact and succeeded.

#### Future-agent rule

- Use this run as the current timing baseline for isolated `generate_responses.py` on staged Llama 3.3 70B in `us-central1`.
- If optimizing throughput for future alignment runs, focus first on reducing startup/compile cost rather than generation speed.

### 2026-03-24 — ALIGN-054: Added Separate `vllm serve` Comparison Experiment on the Same Prompt Artifact

- **Reason for new experiment:** compare Marin's shared `VllmEnvironment` / `vllm serve` path against the direct `llm.generate(...)` path used by `lib/marin/src/marin/alignment/generate_responses.py`.
- **Comparison target:** same prompt set, same staged `us-central1` Llama 3.3 70B checkpoint, same generation settings (`n=1`, `temperature=0.7`, `max_tokens=512`), but a different inference path.

#### Code changes for this experiment

- Added a new standalone experiment script:
  - `experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve.py`
- The script:
  - reuses `gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d`
  - reuses the regional GCS model artifact from `output_path_of(llama_3_3_70b_instruct)`
  - starts `VllmEnvironment(..., mode="native")`
  - sends the prompt set through `/v1/chat/completions`
  - writes normal response shards plus a `timing.json` artifact and a `vllm_server_logs_tail.txt` artifact
- Important timing rule baked into the script:
  - the `timing.json` measurements start only after the remote worker is already running
  - scheduler / TPU allocation wait is intentionally excluded from the experiment's own timing artifact

#### Supporting fix

- Updated `lib/marin/src/marin/inference/vllm_server.py` so `_engine_kwargs_to_cli_args(...)` now forwards:
  - `tensor_parallel_size`
- Why this mattered:
  - without that fix, `VllmEnvironment` would not actually launch `vllm serve` with `--tensor-parallel-size 4`
  - for a 70B model on `v5p-8`, that would make the comparison invalid
- Added regression test:
  - `tests/test_vllm_server.py`

#### Submission

- Submitted with explicit `us-central1` pinning:

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name generate-responses-llama-3-3-70b-vllm-serve-gcs-us-central1 \
  --extra marin:tpu \
  --tpu v5p-8 \
  --region us-central1 \
  --zone us-central1-a \
  -- python experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve.py
```

- Submitted job id:
  - `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-gcs-us-central1`

#### First status check

- Current state:
  - `JOB_STATE_PENDING`
- Pending reason:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - autoscaler backoff on `tpu_v5p_8-us-central1-a`

#### Future-agent rule

- Compare this new run against `ALIGN-053` only using the new script's internal `timing.json` artifact, not Iris scheduler timestamps.
- The intended apples-to-apples comparison is:
  - direct `llm.generate(...)` path from `ALIGN-053`
  - native `vllm serve` path from this run
- Keep the region pinned to `us-central1` for this comparison unless the user explicitly changes that constraint.

### 2026-03-24 — ALIGN-055: `vllm serve` Comparison Run Completed; Roughly Matches Direct `llm.generate(...)`

- **Run:** `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-gcs-us-central1`
- **Final state:** `JOB_STATE_SUCCEEDED`
- **Response artifact:** `gs://marin-us-central1/align/debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve/responses-0635ef`
- **Experiment metadata:** `gs://marin-us-central1/experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve-683c67.json`
- **Rows written:** `76`

#### Internal timing artifact

- Timing source for this run:
  - `gs://marin-us-central1/align/debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve/responses-0635ef/timing.json`
- These timings intentionally exclude Iris scheduler wait and only measure work done inside the remote worker.

#### `vllm serve` worker-side timings

| Metric | Value |
| --- | ---: |
| `prompt_load_seconds` | `0.851s` |
| `vllm_server_start_seconds` | `590.323s` |
| `vllm_request_seconds` | `16.761s` |
| `vllm_only_seconds` | `607.084s` |
| `result_write_seconds` | `0.350s` |
| `total_worker_seconds` | `608.520s` |

#### Side-by-side comparison against `ALIGN-053`

- Broad worker-body comparison:
  - `vllm serve total_worker_seconds = 608.520s`
  - direct `generate_responses.py` child runtime = `620.735s`
  - delta = `-12.215s` (`-1.97%`)
- Tighter comparison focused on the actual vLLM work:
  - `vllm serve vllm_only_seconds = 607.084s`
  - direct `create_engine -> child_finish = 591.162s`
  - delta = `+15.922s` (`+2.69%`)
- Even tighter comparison against direct `create_engine -> write_results`:
  - `vllm serve vllm_only_seconds = 607.084s`
  - direct `create_engine -> write_results = 574.000s`
  - delta = `+33.084s` (`+5.76%`)
- Request/inference-only comparison:
  - `vllm serve vllm_request_seconds = 16.761s`
  - direct generation wall time = `15.000s`
  - delta = `+1.761s` (`+11.74%`)

#### Interpretation

- The run does **not** show a clear `vllm serve` win.
- Best summary:
  - `vllm serve` is in roughly the same range as the direct `llm.generate(...)` path
  - depending on boundary choice, it is either very slightly faster overall or slightly slower for the actual model-serving work
- The strongest conservative conclusion is:
  - `vllm serve` approximately matches the direct path, but does not materially outperform it in this setup

#### Additional observations from `vllm_server_logs_tail.txt`

- The serve path still logged:
  - `Resolved architecture: MistralForCausalLM`
- RunAI streamer was somewhat slower than in `ALIGN-053`:
  - `83.75s` here vs `73.87s` in the direct run
- Internal engine init was somewhat faster than in `ALIGN-053`:
  - `244.88s` here vs `289.01s` in the direct run
- Net result:
  - startup and request overhead largely cancel out any one clear advantage

#### Future-agent rule

- Treat `vllm serve` and direct `llm.generate(...)` as performance peers for this prompt-set scale on staged Llama 3.3 70B in `us-central1`.
- Do not claim a meaningful throughput win for `vllm serve` based on this run alone.

### 2026-03-24 — ALIGN-056: Current `vllm serve` Comparison Uses Per-Prompt Chat Requests; Tighter Batch Comparison Should Use `/v1/completions`

- **Why this note exists:** clarify exactly how the `vllm serve` comparison in `ALIGN-054` / `ALIGN-055` is currently issuing requests, and what a more apples-to-apples batched serve comparison should look like.

#### What the current script does

- `experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve.py`:
  - starts Marin `VllmEnvironment(..., mode="native")`
  - posts directly to the local OpenAI-compatible HTTP server exposed by `vllm serve`
  - uses `/v1/chat/completions`
  - sends **one request per prompt** via a `ThreadPoolExecutor`
- So when discussing the current serve comparison, "HTTP responses" means literal HTTP JSON request/response pairs against the local `vllm serve` endpoint.

#### Why this is not the tightest batching comparison

- The current chat-completions path is valid, but it is not the closest possible match to direct `llm.generate(all_prompts, ...)`.
- It adds:
  - per-request JSON serialization overhead
  - per-request HTTP overhead
  - client thread-pool overhead
- The server can still batch internally as requests arrive, but that is not the same thing as one explicit client-side batch.

#### Better next comparison shape

- For a tighter `vllm serve` benchmark, use `/v1/completions` instead of `/v1/chat/completions`.
- Pre-render the exact prompt strings locally using the same template logic as the direct `generate_responses.py` path.
- Then submit a single batched request whose `prompt` field is a list of prompt strings.
- Rationale:
  - the OpenAI-compatible completions schema accepts `prompt` as a string **or an array of strings**
  - that is much closer to `vllm.LLM.generate(prompts, ...)`

#### Future-agent rule

- Do **not** describe the current `ALIGN-055` serve experiment as a perfectly apples-to-apples client batching comparison.
- If the user asks for a tighter serve-vs-direct benchmark, the next experiment should:
  - keep `us-central1`
  - keep the same staged GCS model path and prompt artifact
  - switch the serve client to one batched `/v1/completions` request over pre-rendered prompt strings

### 2026-03-24 — ALIGN-057: Prepare a Tighter `vllm serve` Benchmark Using One Batched `/v1/completions` Request

- **Goal:** launch a second serve-based comparison that is closer to direct `llm.generate(all_prompt_texts, ...)` than the per-prompt chat-completions experiment in `ALIGN-055`.

#### Planned experiment shape

- New script:
  - `experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched.py`
- Constraints held fixed:
  - region: `us-central1`
  - model artifact: staged Llama 3.3 70B GCS path from `output_path_of(llama_3_3_70b_instruct)`
  - prompt artifact: `gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d`
  - TPU/resources: `v5p-8`, `tp=4`, `max_model_len=4096`, `ram=256g`, `disk=500g`

#### Key implementation difference from `ALIGN-055`

- Instead of:
  - `/v1/chat/completions`
  - one request per prompt
- Use:
  - staged tokenizer files copied locally from the same `gs://...` model artifact
  - local `apply_chat_template(..., add_generation_prompt=True)` to pre-render prompt strings
  - one HTTP POST to `/v1/completions` with `prompt=[...]`

#### Timing rule

- As with `ALIGN-054` / `ALIGN-055`, the experiment's `timing.json` should measure worker-local work only and should exclude Iris scheduler wait.

### 2026-03-24 — ALIGN-058: Batched `/v1/completions` Serve Experiment Added, Syntax-Checked, and Submitted in `us-central1`

- **Script added:**
  - `experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched.py`
- **What it does:**
  - stages the tokenizer/config files locally from the same staged `gs://...` Llama 3.3 70B artifact
  - re-renders the prompt strings with local `apply_chat_template(..., add_generation_prompt=True)`
  - starts Marin `VllmEnvironment(..., mode="native")`
  - sends one OpenAI-compatible batched request to `/v1/completions`
  - writes normal response shards plus:
    - `timing.json`
    - `vllm_server_logs_tail.txt`

#### Local validation

- Syntax-check passed:
  - `uv run python -m py_compile experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched.py`

#### Submission

```bash
uv run iris --config lib/iris/examples/marin.yaml job run \
  --no-wait \
  --job-name generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1 \
  --extra marin:tpu \
  --tpu v5p-8 \
  --region us-central1 \
  --zone us-central1-a \
  -- python experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched.py
```

- Submitted job id:
  - `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1`

#### First status check

- Current state:
  - `JOB_STATE_PENDING`
- Pending reason:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - autoscaler backoff on `tpu_v5p_8-us-central1-a`

#### Future-agent rule

- When this run finishes, compare it against:
  - `ALIGN-053` direct `llm.generate(...)`
  - `ALIGN-055` per-prompt `/v1/chat/completions` serve
- Use only the new run's internal `timing.json` for serve-side timing comparisons.

### 2026-03-24 — ALIGN-059: Killed the Blocked Batched Serve Run; Corrected Child Disk Request from `500g` to `5g`

- **What happened:** the first batched `/v1/completions` serve run never reached vLLM startup because the child TPU step was stuck pending on an oversized disk request.
- CLI/UI diagnosis matched:
  - requested disk: `536,870,912,000` bytes (`500 GB`)
  - available on the placement target: `107,374,182,400` bytes (`100 GB`)
- That was a configuration error in the experiment script, not evidence that `vllm serve` itself needed `500 GB` of local disk.

#### Corrective action

- Terminated:
  - `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1`
  - `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1/align-debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched-responses_4de8b766-338e8351`
- Updated:
  - `experiments/generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched.py`
- Resource change:
  - `disk="500g"` -> `disk="5g"`

#### Rationale

- The successful direct 70B run proved that this path does require substantial RAM and does use local vLLM asset paths.
- But the logs did **not** prove that the entire checkpoint must be fully persisted to local disk.
- The next experiment should therefore test the smallest practical disk request instead of cargo-culting `500g`.

#### Future-agent rule

- If the low-disk rerun fails, inspect the runtime failure mode before increasing disk again.
- Do not assume any future 70B serve run needs `500g` disk unless logs show a concrete local-disk exhaustion signature.

### 2026-03-24 — ALIGN-060: First `disk=5g` Rerun Started Immediately but Reused a Stale Executor Step Key

- **What changed:** lowering the child disk request from `500g` to `5g` removed the TPU-placement bottleneck immediately.
- Evidence:
  - the new root job started effectively instantly instead of waiting on `insufficient_resources: disk`
- **New blocker discovered:** the rerun reused the same executor step name and output path as the killed `ALIGN-058` attempt.
- Result:
  - the executor read the response step status as `RUNNING`
  - it did not launch a fresh child step
  - this would have left the root job waiting on stale executor state

#### Corrective action

- Terminated:
  - `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1-disk5g`
  - `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1-disk5g/align-debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched-responses_4de8b766-19f8b7af`
- Updated the experiment script so the response step now uses a fresh output prefix:
  - `align/debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched_disk5g/responses`

#### Interpretation

- This was not a resource failure.
- It was a reproducibility / executor-state issue caused by reusing the same step key after killing the previous batched run mid-flight.

#### Future-agent rule

- After killing an in-flight executor experiment, do not reuse the same step/output key for the retry unless the previous step status is known to be cleaned up.
- For experimental retries like this one, prefer a new step name / output prefix.

### 2026-03-24 — ALIGN-061: Fresh `disk=5g` Batched Serve Retry Submitted; Disk Constraint Removed, Only TPU Capacity Wait Remains

- **Fresh retry job:**
  - `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1-disk5g-fresh`
- **Why a second retry was needed:** after `ALIGN-060`, the script was updated to use a fresh response-step key/output prefix so the executor would not inherit stale `RUNNING` state from the previously killed attempt.

#### Clean retry status

- Root job:
  - started immediately
  - no disk-related scheduling issue on the lightweight executor process
- Child TPU response step:
  - `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1-disk5g-fresh/align-debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched_disk5g-responses_dad76bea-5d502e34`
  - requests:
    - `v5p-8`
    - `ram=256g`
    - `disk=5g`
- Current child state:
  - `JOB_STATE_PENDING`
- Current pending reason:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - autoscaler backoff on `tpu_v5p_8-us-central1-a`

#### Important interpretation

- The previous `500g` disk over-request is no longer the blocker.
- The current wait is now the expected pure TPU-capacity wait in `us-central1-a`.
- This is the correct low-disk experiment setup for determining whether the serve path can actually run with only `5g` local disk.

#### Future-agent rule

- If this run starts and then fails at runtime, inspect whether the failure is:
  - disk exhaustion
  - tokenizer/config staging failure
  - vLLM asset/temp-file failure
  - something unrelated
- Only increase disk after observing a concrete runtime disk failure signature.

### 2026-03-24 — ALIGN-062: Batched `vllm serve` on `disk=5g` Succeeded; Materially Faster Than Per-Prompt Serve

- **Run:** `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1-disk5g-fresh`
- **Successful child:** `/ahmed/generate-responses-llama-3-3-70b-vllm-serve-batched-gcs-us-central1-disk5g-fresh/align-debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched_disk5g-responses_dad76bea-9308e1f8`
- **Final state:** `JOB_STATE_SUCCEEDED`
- **Response artifact:** `gs://marin-us-central1/align/debug_generate_responses_llama_3_3_70b_existing_prompts_vllm_serve_batched_disk5g/responses-5f5d97`
- **Rows written:** `76`

#### Worker-local timing (`timing.json`)

| Metric | Value |
| --- | ---: |
| `prompt_load_seconds` | `1.085s` |
| `prompt_render_seconds` | `1.554s` |
| `vllm_server_start_seconds` | `445.316s` |
| `vllm_request_seconds` | `16.217s` |
| `vllm_only_seconds` | `461.533s` |
| `result_write_seconds` | `0.434s` |
| `total_worker_seconds` | `464.939s` |

#### Iris timing

| Phase | Value |
| --- | ---: |
| Child pending before TPU allocation | `257.339s` |
| Child runtime after start | `533.328s` |

#### vLLM / Run:ai details from server logs

- Resolved architecture:
  - `MistralForCausalLM`
- Run:ai streaming:
  - `262.8 GiB` in `83.48s` at `3.1 GiB/s`
- CPU buffer line observed:
  - `37.3 GiB`
- Internal engine init:
  - `init engine (profile, create kv cache, warmup model) took 110.51 seconds`
- Request mode:
  - one `POST /v1/completions`

#### Comparison takeaway

- `disk=5g` was sufficient for this successful batched serve run.
- Compared with the earlier per-prompt `/v1/chat/completions` serve run (`ALIGN-055`):
  - `total_worker_seconds`: `464.939s` vs `608.520s`
  - `vllm_only_seconds`: `461.533s` vs `607.084s`
  - both are about `24%` faster in favor of the batched `/v1/completions` path
- The request wall time itself changed only slightly:
  - `16.217s` vs `16.761s`
- So most of the improvement came from faster startup / engine-init behavior in this run, not from the HTTP request stage alone.

#### Future-agent rule

- Treat batched `/v1/completions` as the best-performing `vllm serve` variant tested so far for this prompt-set scale.
- Do not assume the improvement is purely due to client batching; startup variance still matters.

### 2026-03-24 — ALIGN-063: Bottom-Line Comparison Matrix for the Three 70B Response Paths

- **Purpose:** provide one clear, bottom-of-logbook matrix comparing:
  - direct local `llm.generate(...)`
  - `vllm serve` with many per-prompt HTTP chat requests
  - `vllm serve` with one batched `/v1/completions` request
- **Common setup across all three:**
  - prompt count: `76`
  - model artifact: staged `us-central1` Llama 3.3 70B GCS checkpoint
  - TPU: `v5p-8`
  - tensor parallelism: `4`
  - max model len: `4096`

#### Main comparison matrix

| Path | Request shape | Child disk request | Worker-local total | Startup / server ready | Request / generation wall time | Prompt throughput | Generation throughput | Result write | Run:ai stream | Internal engine init | Output rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Direct `llm.generate(...)` | one in-process batch over all prompt texts | `500g` | `620.735s` child runtime | `556.000s` create-engine -> ready | `15.000s` | `399.31 toks/s` final progress estimate | `728.16 toks/s` final progress estimate | `17.162s` | `73.87s` @ `3.6 GiB/s` | `289.01s` | `76` |
| `vllm serve` per-prompt HTTP | `76` concurrent `POST /v1/chat/completions` requests | `500g` | `608.520s` | `590.323s` `vllm_server_start_seconds` | `16.761s` | `613.6 tokens/s` APIServer avg | `941.0 tokens/s` APIServer avg | `0.350s` | `83.75s` @ `3.1 GiB/s` | `244.88s` | `76` |
| `vllm serve` batched HTTP | one `POST /v1/completions` with `prompt=[76 rendered prompts]` | `5g` | `464.939s` | `445.316s` `vllm_server_start_seconds` | `16.217s` | `619.5 tokens/s` APIServer avg | `847.4 tokens/s` APIServer avg | `0.434s` | `83.48s` @ `3.1 GiB/s` | `110.51s` | `76` |

- **Throughput note:** the direct path reports final progress-bar estimates (`est. speed input/output`), while the two `vllm serve` paths report APIServer average throughput lines. They are close enough to be useful, but they are not emitted by the exact same logger.

#### Relative deltas

| Comparison | Worker-local delta | Percent |
| --- | ---: | ---: |
| Batched serve vs per-prompt serve | `-143.581s` | `-23.60%` |
| Batched serve vs direct `llm.generate(...)` | `-155.796s` | `-25.10%` if comparing full child runtime |
| Batched serve `vllm_only` vs direct create-engine -> child-finish | `-129.629s` | `-21.93%` |
| Batched serve request wall time vs per-prompt serve request wall time | `-0.545s` | `-3.25%` |

#### Very clear takeaway

- **Fastest path tested:** batched `vllm serve` via one `/v1/completions` request.
- **Why it wins:** almost all of the gain came from startup / engine-init being lower in that run, not from the HTTP request itself.
- **Important operational result:** the winning batched `vllm serve` run succeeded with only `disk=5g`.

### 2026-03-24 — ALIGN-064: Recommendation for Open-Weight End-to-End Pipeline; Prefer Batched Local Inference

- **Decision:** for local open-weight inference, standardize on **batched** inference, not per-prompt HTTP fanout.
- **Reason for decision:** the tested request windows were nearly identical:
  - per-prompt serve request wall time: `16.761s`
  - batched serve request wall time: `16.217s`
- So per-prompt serving did **not** show a meaningful steady-state generation advantage on this workload.
- When there is no clear performance advantage, defer to **simplicity**:
  - one batch request is simpler than a thread pool issuing one request per prompt
  - one batch request matches the existing direct `llm.generate(all_prompt_texts, ...)` mental model
  - one batch request is the cleaner primitive for offline dataset-generation jobs

#### Stage-by-stage recommendation

- **Chosen / rejected response generation:** batched.
  - Current local implementation is already structured this way in `generate_responses.py` via one `llm.generate(all_prompt_texts, ...)`.
- **Judge step:** move to batched / microbatched local inference.
  - Current implementation is structurally per-response because `_judge_response(...)` calls `llm_chat_single(...)` once per scored response.
  - This is a poor fit for local open-weight models because the judge step eagerly loads all prompt pairs anyway and local vLLM is pinned to `workers=1`.
- **Prompt generation Stage 1 (understanding / axes):** microbatched across statements.
  - This work is independent across statements, but local vLLM currently executes it sequentially with one worker.
- **Prompt generation Stage 2 (concretization):** keep the current batch unit.
  - This stage is already naturally batched because one request covers multiple covering-array configs.
- **Prompt generation Stage 3 (extraction):** keep the current batch unit.
  - This stage is already naturally batched because one request extracts multiple scenarios at once.

#### Higher-priority architectural note

- Before broad open-weight rollout, fix **one-model-at-a-time** execution.
- Current pipeline still has major overlap risks:
  - `align.py` schedules `chosen` and `rejected` as sibling steps
  - `generate_prompts.py` can preload both local engines for ideation and extraction in the same process
- That lifecycle issue matters more than per-prompt vs batched request shape.

#### Exact experiment queue from here

1. **Experiment A — Reusable batched local inference primitive for alignment**
   - **Goal:** add one shared helper for local batched inference using `VllmEnvironment` + one batched `/v1/completions` request.
   - **Scope:** this should become the common primitive for local response generation, local judge microbatches, and local prompt-gen microbatches.
   - **Non-goal:** do **not** keep building separate ad hoc per-prompt serve paths.
   - **Success criteria:**
     - accepts a list of already-rendered prompt strings
     - returns completions grouped by input prompt
     - uses staged `gs://...` model paths only
     - succeeds on `v5p-8` in `us-central1`
     - still works with `disk=5g`

2. **Experiment B — Swap local `generate_responses.py` onto the new batched serve primitive**
   - **Goal:** replace the current direct local `llm.generate(...)` path with the shared batched serve primitive for `VLLMConfig`.
   - **Validation workload:** the same succeeded `76`-prompt artifact already used in `ALIGN-052` through `ALIGN-063`.
   - **Success criteria:**
     - output schema unchanged
     - output row count unchanged (`76`)
     - no Hugging Face download path appears in logs
     - worker-local timing remains in the same ballpark as the current best batched serve run

3. **Experiment C — Add a batched local judge path**
   - **Goal:** stop using one `llm_chat_single(...)` call per judged response when `judge_model` is local.
   - **Implementation shape:** build judge prompts for a microbatch of response candidates, render them up front, then score them through the shared batched serve primitive.
   - **Validation workload:** run on a small fixed chosen/rejected artifact pair first, not the whole spec.
   - **Success criteria:**
     - pair count is stable versus the old judge path on the same artifact
     - parse failure rate does not regress
     - local judge no longer spends nearly all its time in serial per-response calls

4. **Experiment D — Add batched local prompt-generation paths**
   - **Goal:** make local prompt generation use batch-shaped inference instead of repeated `llm_chat_single(...)` calls.
   - **Stage plan:**
     - Stage 1 understanding: microbatch across statements
     - Stage 2 concretization: preserve current request shape, but route through the shared batched local primitive
     - Stage 3 extraction: preserve current request shape, but route through the shared batched local primitive
   - **Validation workload:** one-statement spec first.
   - **Success criteria:**
     - generated prompt schema unchanged
     - same statement coverage and artifact structure
     - no concurrent multi-engine loading inside one prompt-generation worker

5. **Experiment E — Enforce one-model-at-a-time lifecycle in `align()`**
   - **Goal:** make the heterogeneous Llama chosen / Mixtral rejected path safe under TPU memory constraints.
   - **Required changes:**
     - do not allow `chosen` and `rejected` local-model steps to run concurrently
     - do not preload multiple local prompt-generation engines in the same process
   - **Success criteria:**
     - only one local model is active at a time per worker
     - logs show clean teardown before the next model is started

6. **Experiment F — `us-central1` one-statement full open-weight smoke**
   - **Goal:** run the whole pipeline with open-weight models for all model roles after Experiments A-E land.
   - **Target configuration:**
     - ideation: Llama 3.3 70B
     - extraction: Llama 3.3 70B
     - chosen: Llama 3.3 70B
     - rejected: Mixtral 8x7B Instruct
     - judge: Llama 3.3 70B
     - region: `us-central1`
   - **Success criteria:**
     - prompts generated
     - chosen responses generated
     - rejected responses generated
     - preference pairs written
     - no HF download path
     - no simultaneous multi-model load

7. **Experiment G — Separate `us-central1` full-spec runtime characterization**
   - **Goal:** after smoke success, run the whole spec as a separate job to measure wall-clock behavior.
   - **Purpose:** characterize runtime and bottlenecks, not DPO yet.
   - **Success criteria:**
     - full prompt-generation pass completes
     - chosen / rejected / judge complete
     - timing breakdown is recorded by stage
     - outputs are separated from the smoke run

#### Operational rule for future agents

- Do **not** spend more time on the per-prompt `vllm serve` path unless a user explicitly asks for interactive-serving behavior.
- For offline alignment dataset generation, the default local-model direction is now:
  - **batched**
  - **one-model-at-a-time**
  - **`us-central1` staged `gs://...` artifacts only**

### 2026-03-24 — ALIGN-065: Start Experiment A/B; Shared Batched Serve Primitive + `generate_responses.py` Refactor

- **Reason for this entry:** begin the concrete refactor sequence from `ALIGN-064` by first changing only the local response-generation path.
- **Scope for this experiment:**
  - add one reusable alignment-side batched `vllm serve` helper that:
    - stages tokenizer files
    - starts `VllmEnvironment(mode="native")`
    - renders chat prompts locally
    - sends one batched `/v1/completions` request
  - switch `generate_responses.py` local `VLLMConfig` inference onto that helper
  - leave prompt-generation and judge logic unchanged for now
- **Why this boundary:** response generation already has the cleanest batch shape in the current codebase, so it is the lowest-risk place to validate the new primitive before touching judge or prompt generation.

#### Files targeted in this experiment

- `lib/marin/src/marin/alignment/batched_vllm_serve.py`
- `lib/marin/src/marin/alignment/generate_responses.py`
- `tests/test_alignment.py`
- `experiments/generate_responses_llama_3_3_70b_existing_prompts_refactored.py`

#### Verification plan for this experiment

1. Run alignment unit coverage after the refactor:
   - `uv run pytest tests/test_alignment.py -q`
2. Submit a fresh `us-central1` TPU validation job against the known-good `76`-prompt artifact:
   - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name generate-responses-llama-3-3-70b-refactored-us-central1 --extra marin:tpu --tpu v5p-8 --region us-central1 --zone us-central1-a -- python experiments/generate_responses_llama_3_3_70b_existing_prompts_refactored.py`
3. Treat this as successful only if:
   - the job succeeds
   - outputs still contain `76` rows
   - logs show `vllm serve` / `/v1/completions`, not direct `llm.generate(...)`
   - logs show staged `gs://...` loading and no Hugging Face download path
   - `disk=5g` is still sufficient

### 2026-03-24 — ALIGN-066: Experiment A/B Local Refactor Landed; Validation Run Re-Submitted Region-Only and Is Pending TPU Capacity

- **Implementation completed locally:**
  - added `lib/marin/src/marin/alignment/batched_vllm_serve.py`
  - switched `generate_responses.py` local `VLLMConfig` path onto batched `vllm serve`
  - added a focused unit test in `tests/test_alignment.py`
  - added a fresh Iris validation script: `experiments/generate_responses_llama_3_3_70b_existing_prompts_refactored.py`
- **Local verification passed:**
  - `uv run pytest tests/test_alignment.py -q`
  - result: `68 passed`
- **Execution notes:**
  - first submission used `--region us-central1 --zone us-central1-a` under job name `/ahmed/generate-responses-llama-3-3-70b-refactored-us-central1`
  - that root remained `JOB_STATE_PENDING` on TPU capacity, so it was killed before any child step started
  - relaunched as region-only under `/ahmed/generate-responses-llama-3-3-70b-refactored-us-central1-region-only`
- **Current state of the live run:**
  - root job: `JOB_STATE_PENDING`
  - pending reason:
    - `Scheduler: Insufficient TPUs (need 4, available 0)`
    - constraints now include `region` but no explicit `zone`
  - important negative evidence:
    - there is **no disk blocker**
    - requested disk remains `5 GB`
    - so the refactored path is at least preserving the intended low-disk resource envelope
- **What still needs verification once capacity arrives:**
  - child logs must show `vllm serve` startup
  - request path must be `/v1/completions`
  - output row count must remain `76`
  - no Hugging Face download path may appear

### 2026-03-24 — ALIGN-067: Experiment A/B Succeeded; `generate_responses.py` Local Path Now Verified on Batched `vllm serve`

- **Live validation run:** `/ahmed/generate-responses-llama-3-3-70b-refactored-us-central1-region-only`
- **Successful child step:** `/ahmed/generate-responses-llama-3-3-70b-refactored-us-central1-region-only/align-debug_generate_responses_llama_3_3_70b_existing_prompts_refactored-responses_dd0eb71a-27d16f2e`
- **Result:** both root and child are `JOB_STATE_SUCCEEDED`

#### Timing from Iris metadata

- Root pending before start: `83.628s`
- Root active runtime: `598.377s`
- Child pending before start: `15.681s`
- Child active runtime: `518.648s`

#### Output artifact

- `gs://marin-us-central1/align/debug_generate_responses_llama_3_3_70b_existing_prompts_refactored/responses-761166`
- Files present:
  - `.artifact`
  - `.executor_info`
  - `.executor_status`
  - `shard_00000.jsonl.gz`
- Verified row count in `shard_00000.jsonl.gz`: `76`

#### Worker-log evidence

- Step output path resolved to:
  - `gs://marin-us-central1/align/debug_generate_responses_llama_3_3_70b_existing_prompts_refactored/responses-761166`
- Response step log confirms the refactored path ran:
  - `Using batch vLLM: gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  - `Loaded 76 prompts for batched vLLM serve generation`
  - `Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
  - `vLLM environment ready`
  - `Wrote 76 records to 1 shards ...`
- Negative evidence:
  - no `HF` / `huggingface` / `Downloading weights` strings were observed in the worker logs
  - no disk-capacity blocker appeared; the child still ran with `disk=5 GB`

#### Important caveat

- The Iris worker logs did **not** surface an HTTP access-log line showing `POST /v1/completions`.
- So the request endpoint is confirmed by the new helper implementation path (`batched_vllm_serve.py`), while the runtime logs confirm only that the step used batched serve generation plus `vllm serve` startup successfully.

#### Decision after this experiment

- Experiment A is complete.
- Experiment B is complete.
- `generate_responses.py` local `VLLMConfig` inference is now verified on batched `vllm serve`.

#### Next exact experiment

- Proceed to **Experiment C** from `ALIGN-064`:
  - add a batched local judge path
  - validate it first on a small fixed chosen/rejected artifact pair before touching the full pipeline

### 2026-03-24 — ALIGN-068: Start Experiment C; Add Batched Local Judge Path and Validate on Fixed Pair

- **Reason for this entry:** continue the open-weight batched-serve migration after successful response-path validation in `ALIGN-067`.
- **Scope for this experiment:**
  - keep API judge behavior unchanged
  - add a local batched judge path for `VLLMConfig` in `judge.py`
  - route local judging through the shared batched serve helper from Experiment A/B
  - validate on a tiny fixed chosen/rejected artifact pair in `us-central1`

#### Files targeted in this experiment

- `lib/marin/src/marin/alignment/judge.py`
- `lib/marin/src/marin/alignment/align.py`
- `lib/marin/src/marin/alignment/batched_vllm_serve.py`
- `tests/test_alignment.py`
- `experiments/judge_llama_3_3_70b_fixed_pairs_refactored.py`

#### Fixed-pair validation dataset

- Two prompt pairs:
  - `be_helpful/cfg_000`
  - `be_honest/cfg_000`
- Chosen responses are intentionally strong and policy-compliant.
- Rejected responses are intentionally bad:
  - rude refusal for helpfulness
  - fabricated private-bank-balance claim for honesty
- Expected output pair count: `2`

#### Verification plan for this experiment

1. Run alignment unit coverage after the judge refactor:
   - `uv run pytest tests/test_alignment.py -q`
2. Submit a fresh `us-central1` TPU validation job:
   - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name judge-llama-3-3-70b-fixed-pairs-refactored-us-central1 --extra marin:tpu --tpu v5p-8 --region us-central1 -- python experiments/judge_llama_3_3_70b_fixed_pairs_refactored.py`
3. Treat this as successful only if:
   - the job succeeds
   - the preference-pair output row count is exactly `2`
   - logs show local `vllm serve` startup for judge inference
   - logs show batched local judge scoring rather than serial per-response calls
   - no Hugging Face download path appears
   - `disk=5g` is still sufficient

### 2026-03-24 — ALIGN-069: Experiment C Submitted; Currently Pending Only on `us-central1` TPU Capacity

- **Live validation run:** `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1`
- **Current state:** `JOB_STATE_PENDING`
- **Pending reason:**
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - constraints are `device-type`, `device-variant`, `region`, `reservation-job`
  - no explicit zone pin is present in the submission
- **Important evidence already gathered before TPU execution:**
  - local code changes for Experiment C are in place
  - `uv run pytest tests/test_alignment.py -q` passed with `69` tests
  - repo fix/check pass succeeded on the changed files
- **Important negative evidence:**
  - there is no disk-capacity blocker on the root request
  - requested disk remains `5 GB`
- **What future agents should do next:**
  - keep checking this root until it starts
  - once a child judge step appears, inspect logs for:
    - `Starting vLLM native server`
    - `Sending batched vLLM serve request to /v1/completions`
    - `Scoring ... responses via local batched judge`
  - after completion, verify the pair artifact contains exactly `2` rows

### 2026-03-24 — ALIGN-070: First Experiment C Run Failed; Root Cause Was Stringifying `output_path_of(prepare_step)` in the Fixed-Pair Script

- **Failed run:** `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1`
- **Observed step status:**
  - prepare step succeeded
  - judge step failed immediately before actual judging
- **Judge-step failure:**
  - `FileNotFoundError: [Errno 2] No such file or directory: "/app/InputName(step=ExecutorStep(...))/spec/spec.jsonl"`
- **Diagnosis:**
  - the fixed-pair experiment script used Python f-strings around `output_path_of(prepare_step)`
  - that coerced the executor `InputName` object into a literal placeholder string too early
  - the judge worker then tried to open that placeholder text as a local file path under `/app/...`
- **Concrete fix applied locally:**
  - replaced:
    - `f"{output_path_of(prepare_step)}/chosen"`
    - `f"{output_path_of(prepare_step)}/rejected"`
    - `f"{output_path_of(prepare_step)}/spec/spec.jsonl"`
  - with executor-aware path joins:
    - `output_path_of(prepare_step) / "chosen"`
    - `output_path_of(prepare_step) / "rejected"`
    - `output_path_of(prepare_step) / "spec" / "spec.jsonl"`
- **Implication:**
  - this was not a GCS-read failure
  - it was a config-construction bug in the experiment script
- **Next action:**
  - rerun Experiment C with a fresh job name after this path fix

### 2026-03-24 — ALIGN-071: Experiment C Retry Cleared the Path Bug; Current Blocker Is Root-Job Preemption, Not Missing GCS Artifacts

- **Retry run:** `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry1`
- **What the retry proved:**
  - the judge worker successfully loaded the fixed artifacts from GCS after the `InputName` path fix
  - worker logs showed:
    - `Loaded 2 chosen, 2 rejected responses`
    - `Processing 2 prompt pairs via local batched judge`
    - `Starting vLLM environment`
    - `Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
- **Interpretation:**
  - the earlier `FileNotFoundError` was fully explained by the experiment-script path bug
  - there is no evidence of a missing `gs://.../spec/spec.jsonl` artifact
  - this is now past artifact loading and into actual local-judge startup
- **What happened next:**
  - first retry child:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry1/align-debug_local_judge_llama_3_3_70b_fixed_pairs-preference_pairs_2dc64343-5207a060`
    - was killed with `Parent task preempted`
  - the root job itself stayed alive with `preemption_count=1`, restarted, reacquired the step lock, and resubmitted a new child:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry1/align-debug_local_judge_llama_3_3_70b_fixed_pairs-preference_pairs_2dc64343-f2f68796`
  - current state of the replacement child: `JOB_STATE_PENDING` on `us-central1` TPU capacity
- **Operational takeaway for future agents:**
  - do **not** keep debugging GCS/spec-path handling for Experiment C unless a new file-path error appears
  - if this retry loop keeps suffering root preemption, relaunch the experiment with a CPU-only root submission and let only the judge child request TPUs

### 2026-03-24 — ALIGN-072: Relaunched Experiment C with a CPU-Only Root So Only the Judge Child Holds TPUs

- **Why this entry exists:**
  - the previous retry used a TPU-root submission, so the root executor itself was holding `v5p-8` resources
  - that caused unnecessary TPU contention and made root preemption materially more disruptive
- **Action taken:**
  - killed:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry1`
  - relaunched with CPU-only root:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry2-cpu-root`
  - submission shape:
    - `--cpu 4 --memory 16GB --disk 10GB --region us-central1`
    - no TPU attached to the root job
- **Current state:**
  - root is `JOB_STATE_RUNNING` on CPU only
  - judge child:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry2-cpu-root/align-debug_local_judge_llama_3_3_70b_fixed_pairs-preference_pairs_2dc64343-35f79a18`
    - is `JOB_STATE_PENDING` only on `us-central1` TPU capacity
- **Important implication:**
  - this is now the correct operational recipe for Experiment C and similar executor-based validation scripts:
    - CPU-only root
    - TPU requested only by the child step via `VLLMConfig.resources`

### 2026-03-24 — ALIGN-073: Experiment C Succeeded; Batched Local Judge Validated End-to-End

- **Successful run:** `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry2-cpu-root`
- **Successful child:** `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry2-cpu-root/align-debug_local_judge_llama_3_3_70b_fixed_pairs-preference_pairs_2dc64343-35f79a18`
- **Result:** both root and child are `JOB_STATE_SUCCEEDED`

#### Timing from Iris metadata

- Root pending before start: `0.184s`
- Root active runtime: `686.752s`
- Child pending before start: `155.138s`
- Child active runtime: `509.765s`

#### Output artifact

- `gs://marin-us-central1/align/debug_local_judge_llama_3_3_70b_fixed_pairs/preference_pairs-01dd5c`
- Files include:
  - `.artifact`
  - `.executor_info`
  - `.executor_status`
  - `shard_00000.jsonl.gz`
- Verified row count in `shard_00000.jsonl.gz`: `2`

#### Worker-log evidence

- Judge worker loaded the fixed artifacts successfully:
  - `Loaded 2 chosen, 2 rejected responses`
  - `Processing 2 prompt pairs via local batched judge`
- Local serving path was used:
  - `Starting vLLM environment`
  - `Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
  - `vLLM environment ready`
- Batched request path was used:
  - `Scoring 4 responses via local batched judge`
  - `Rendering 4 chat prompts for batched vLLM serve`
  - `Sending batched vLLM serve request to /v1/completions for 4 prompts (n=1)`
- Output and filtering behavior matched expectations:
  - `Built 2 preference pairs (0 filtered, 0 failures)`
  - `Wrote 2 preference pairs to gs://marin-us-central1/align/debug_local_judge_llama_3_3_70b_fixed_pairs/preference_pairs-01dd5c`

#### Conclusion

- Experiment C is now validated.
- The local judge path in `judge.py` successfully runs through batched `vllm serve` on staged GCS model weights and produces the expected fixed-pair artifact.
- The next queued refactor from the plan is prompt-generation migration to the shared batched local serve primitive, followed by the one-model-at-a-time lifecycle work in `align()`.

### 2026-03-24 — ALIGN-074: Judge Refactor Now Persists Full Scored Judgments Before Filtering

- **Reason for this entry:** the previous judge artifact only kept final `chosen` / `rejected` message pairs, which made it impossible to inspect chosen score, rejected score, gap, confidence, explanation, highlights, or raw judge output after the run.
- **Code changes:**
  - rewrote `alignment/judge.py` around two explicit stages:
    - `judge_responses(JudgeConfig)` — writes a full judgments dataset
    - `build_preference_pairs(PreferencePairFilterConfig)` — filters persisted judgments into DPO pairs
  - updated `alignment/align.py` so the main pipeline now has:
    - `align/<name>/judgments`
    - `align/<name>/preference_pairs`
  - kept `judge_and_build_pairs(JudgePairConfig)` only as a convenience wrapper for scripts, but it now writes judgments first and filters second
- **What the new judgments artifact contains per prompt:**
  - prompt metadata:
    - `prompt_id`
    - `behavior_id`
    - `system_prompt`
    - `user_message`
    - `rubric`
    - structured `statement`
  - full scored candidate lists:
    - `chosen_candidates[]`
    - `rejected_candidates[]`
  - each candidate includes:
    - `response_index`
    - `response_text`
    - parsed `judgment`
      - `score`
      - `compliant`
      - `confidence`
      - `explanation`
      - `highlights`
      - `raw_response`
  - selected extrema and filter inputs:
    - `best_chosen`
    - `worst_rejected`
    - `gap`
    - `status`
    - `errors`
- **What the new pair-filter step now logs:**
  - final DPO pairs at the pair output root
  - `artifacts/filter_decisions/` with one record per prompt containing:
    - `best_chosen_score`
    - `worst_rejected_score`
    - `gap`
    - `passed`
    - `reason`
  - `artifacts/filter_summary.json` with aggregate counts by reason
- **Validation:**
  - `make fix` passed
  - `uv run pytest tests/test_alignment.py -q` passed: `68 passed`
  - updated tests now verify:
    - local batched judge writes full scored judgments
    - pair filtering is a separate operation over persisted judgments
    - filter decisions and summary artifacts are written
- **Operational consequence for future agents:**
  - if someone asks “where are the judge scores?”, the correct answer should now be:
    - in the judgments artifact from `align/<name>/judgments`
    - and in the pair step’s `artifacts/filter_decisions/`
  - to get those live on GCS for Experiment C or a full align run, rerun the relevant experiment after this refactor lands

### 2026-03-24 — ALIGN-075: Rerun Experiment C on the Refactored Judge Path to Materialize Scores on GCS

- **Reason for this entry:** the prior successful Experiment C artifact (`preference_pairs-01dd5c`) predates `ALIGN-074` and therefore still lacks persisted judge-score metadata.
- **Planned action now:**
  - rerun `experiments/judge_llama_3_3_70b_fixed_pairs_refactored.py`
  - keep the CPU-only root submission pattern from `ALIGN-072`
  - use a fresh job name so the new run materializes:
    - filtered preference pairs
    - `artifacts/judgments/` from the wrapper
    - `artifacts/filter_decisions/`
    - `artifacts/filter_summary.json`
- **Success criteria:**
  - root and child succeed
  - final pair row count remains `2`
  - the new output contains full scored judgments and filter-decision artifacts

### 2026-03-24 — ALIGN-076: Aborted the `us-central1` Logged Rerun After User Redirected to `us-east5-a`; Launch Is Blocked by Missing Staged Llama 3.3 70B Artifact in `us-east5`

- **Killed run:**
  - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry4-logged-separate`
  - killed child:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-central1-retry4-logged-separate/align-debug_local_judge_llama_3_3_70b_fixed_pairs_logged-judgments_8cb4e877-c6703787`
- **Why the requested move to `us-east5-a` is blocked:**
  - checked:
    - `gs://marin-us-east5/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  - result:
    - path does **not** exist
- **Implication:**
  - launching the same Experiment C job in `us-east5-a` right now would either:
    - trigger a fresh model download path into `us-east5`, or
    - require cross-region checkpoint reads from `us-central1`
  - both are undesirable for this thread; we have been explicitly avoiding Hugging Face downloads and large cross-region transfers
- **Next action needed before `us-east5-a` launch is safe:**
  - stage `meta-llama/Llama-3.3-70B-Instruct` into `gs://marin-us-east5/models/...`
  - only then rerun the logged Experiment C script in `us-east5-a`

### 2026-03-24 — ALIGN-077: User Approved `us-east5-a` DAG Staging; Launch the Logged Experiment C Script There

- **Decision change:** the user explicitly approved using the executor DAG to stage the missing `llama_3_3_70b_instruct` dependency in `us-east5-a`.
- **What this means operationally:**
  - the existing dependency on `output_path_of(llama_3_3_70b_instruct)` is sufficient
  - no manual extra model step needs to be added to `executor_main(...)`
  - the `models/meta-llama--Llama-3-3-70B-Instruct--6f6073b` step will run in `us-east5` because the artifact is not already present there
- **Action to take now:**
  - submit `experiments/judge_llama_3_3_70b_fixed_pairs_refactored.py`
  - CPU-only root
  - pin root submission to `us-east5-a`
  - let the DAG materialize both:
    - `gs://marin-us-east5/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
    - the logged Experiment C outputs in `gs://marin-us-east5/align/...`

### 2026-03-24 — ALIGN-078: `us-east5-a` Experiment C Relaunch Is Healthy; Model Staging Is Actively Progressing at `40/53` Files

- **Active root job:**
  - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-east5a-retry5-stage-model`
- **Current child state:**
  - fixed-artifacts CPU step succeeded:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-east5a-retry5-stage-model/align-debug_local_judge_llama_3_3_70b_fixed_pairs_logged-artifacts_341775e9-8b982a2c`
  - model-staging Zephyr coordinator is still running:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-east5a-retry5-stage-model/zephyr-download-hf-0c401190-p0-a0`
  - Zephyr worker pool is still running with `8` workers:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-east5a-retry5-stage-model/zephyr-download-hf-0c401190-p0-a0/zephyr-download-hf-0c401190-p0-workers`
- **What the current logs show:**
  - executor launched `4` steps from `3` provided steps because it discovered the model dependency through the DAG
  - the model step is targeting:
    - `gs://marin-us-east5/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  - the Hugging Face staging job identified:
    - `53` files
    - `262.87 GB`
- **Direct GCS progress check:**
  - prefix exists:
    - `gs://marin-us-east5/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  - `.metrics/success-part-*.jsonl` currently reports:
    - `40` completed files
  - total expected files from the downloader metadata:
    - `53`
  - sample staged payload already includes:
    - `config.json`
    - `generation_config.json`
    - `model-00001-of-00030.safetensors`
    - `model-00002-of-00030.safetensors`
- **Interpretation:**
  - the run is healthy
  - it is no longer just writing metadata; large weight shards are landing in `us-east5`
  - the logged judgments step has not started yet because it still depends on the model step finishing
- **Immediate next check for future agents:**
  - wait for the model step to finish
  - then look for the new logged judgments child under the same root
  - after that, verify the new `judgments` and `filter_decisions` artifacts in `gs://marin-us-east5/align/debug_local_judge_llama_3_3_70b_fixed_pairs_logged/...`

### 2026-03-24 — ALIGN-079: Experiment C Succeeded End-to-End in `us-east5-a` with Logged Judgments, Filter Decisions, and Final Pairs

- **Successful root job:**
  - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-east5a-retry5-stage-model`
- **Successful child steps:**
  - model staging:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-east5a-retry5-stage-model/zephyr-download-hf-0c401190-p0-a0`
  - logged local judge:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-east5a-retry5-stage-model/align-debug_local_judge_llama_3_3_70b_fixed_pairs_logged-judgments_99a30903-86e02dcc`
  - separate pair filter:
    - `/ahmed/judge-llama-3-3-70b-fixed-pairs-refactored-us-east5a-retry5-stage-model/align-debug_local_judge_llama_3_3_70b_fixed_pairs_logged-preference_pairs_efa79973-fba5b217`
- **Artifacts written:**
  - logged judgments:
    - `gs://marin-us-east5/align/debug_local_judge_llama_3_3_70b_fixed_pairs_logged/judgments-1610db`
  - final pairs:
    - `gs://marin-us-east5/align/debug_local_judge_llama_3_3_70b_fixed_pairs_logged/preference_pairs-37ec92`
- **Verified contents:**
  - `judgments-1610db/shard_00000.jsonl.gz` has `2` rows
  - judgment row keys include:
    - `chosen_candidates`
    - `rejected_candidates`
    - `best_chosen`
    - `worst_rejected`
    - `gap`
    - `status`
    - `errors`
  - `preference_pairs-37ec92/shard_00000.jsonl.gz` has `2` rows
  - pair row keys remain:
    - `chosen`
    - `rejected`
  - pair sidecars are present under:
    - `gs://marin-us-east5/align/debug_local_judge_llama_3_3_70b_fixed_pairs_logged/preference_pairs-37ec92/artifacts/filter_decisions`
    - `gs://marin-us-east5/align/debug_local_judge_llama_3_3_70b_fixed_pairs_logged/preference_pairs-37ec92/artifacts/filter_summary.json`
- **Timing from Iris metadata:**
  - root runtime:
    - `2070s` (`1774413869603 - 1774411790549`)
  - judgments child runtime:
    - `599.953s`
  - preference-pairs child runtime:
    - `25.193s`
- **Operational conclusion:**
  - the logged-judge refactor is now externally validated on GCS, not just by unit tests
  - future agents looking for judge scores should inspect the `judgments-1610db` artifact in `us-east5`
  - Experiment C is complete; the next planned experiment remains prompt-generation migration to shared batched `vllm serve`

### 2026-03-24 — ALIGN-080: Start Experiment D; Refactor Prompt Generation onto Shared Batched `vllm serve` with Same-Model Session Reuse

- **Goal:** move prompt-generation stages 1, 2, and 3 off direct `llm_chat_single(...)`/`vllm_engine(...)` and onto the shared batched serve primitive.
- **Lifecycle rule for the refactor:**
  - never have more than one local model active at once
  - if `ideation_model == extract_model` and both are local `VLLMConfig`, reuse a single `BatchedVllmServeSession` across stages 1, 2, and 3
  - if they differ, run stages 1-2 under the ideation session, tear it down, then open the extraction session for stage 3
- **Implementation scope:**
  - `lib/marin/src/marin/alignment/generate_prompts.py`
  - `lib/marin/src/marin/alignment/align.py`
  - `tests/test_alignment.py`
- **Validation plan before launching Iris:**
  - add focused unit tests for:
    - same-model session reuse across stages 1-3
    - clean session handoff when ideation and extraction configs differ
  - keep the existing API-path prompt-generation test green
- **Dedicated Experiment D script to launch after local validation:**
  - `experiments/generate_prompts_llama_3_3_70b_refactored.py`
  - one statement only:
    - `ask_clarifying_questions`
  - staged checkpoint:
    - `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  - run only prompt generation, not full `align()`
- **Success criteria:**
  - prompt schema unchanged
  - one-statement artifact structure unchanged
  - logs show batched `vllm serve`, not direct `llm.generate(...)`
  - same-model run uses one local serve session across stages 1-3

### 2026-03-24 — ALIGN-081: Experiment D Refactor Landed; Unit Tests Passed; One-Statement `us-central1` Validation Run Submitted

- **Code changes landed:**
  - `lib/marin/src/marin/alignment/generate_prompts.py`
    - local stages 1-3 now route through `BatchedVllmServeSession`
    - stage 1 microbatches statements
    - stages 2 and 3 batch prompt requests through one local serve session
    - same-model local configs reuse one session across stages 1-3
    - different ideation vs extraction local configs now run sequentially with clean session handoff
  - `lib/marin/src/marin/alignment/align.py`
    - prompt-generation step now requests local resources if either ideation or extraction is local
    - added `prompt_batch_size` to `AlignConfig`
    - prompt step now rejects mixed local ideation/extraction resources inside one worker
  - `tests/test_alignment.py`
    - added local prompt-generation tests for:
      - same-model session reuse
      - different-model session switching
  - `experiments/generate_prompts_llama_3_3_70b_refactored.py`
    - dedicated one-statement prompt-generation validation script
- **Local verification:**
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py lib/marin/src/marin/alignment/align.py experiments/generate_prompts_llama_3_3_70b_refactored.py tests/test_alignment.py`
    - passed
  - `uv run pytest tests/test_alignment.py -q`
    - passed: `70 passed`
- **Experiment D validation run submitted:**
  - root:
    - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-central1`
  - current state at submission check:
    - root `JOB_STATE_RUNNING`
    - spec-upload child created and running:
      - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-central1/align-debug_generate_prompts_llama_3_3_70b_refactored-spec_62dc78f2-c904bb8f`
- **Validation target:**
  - one statement:
    - `ask_clarifying_questions`
  - model:
    - `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
- **Immediate next check for future agents:**
  - wait for the prompt-generation child to start
  - verify logs show batched `vllm serve`
  - verify the output prompt artifact and prompt count
  - then proceed to Experiment E if the one-statement run succeeds

### 2026-03-24 — ALIGN-082: First Experiment D Run Failed with `HTTP 400` from `vllm serve`; Root Cause Was an Impossible Local Token Budget

- **Failed run:**
  - root:
    - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-central1`
  - failed child:
    - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-central1/align-debug_generate_prompts_llama_3_3_70b_refactored-prompts_2c1080f8-daa82b37`
- **What happened:**
  - the prompt-generation TPU child successfully:
    - loaded the spec
    - filtered to `ask_clarifying_questions`
    - started `vllm serve`
    - reached `vLLM environment ready`
  - it then failed on the very first Stage 1 batched request:
    - `requests.exceptions.HTTPError: 400 Client Error: Bad Request for url: http://127.0.0.1:8000/v1/completions`
- **Diagnosis:**
  - this was not a region or model-path problem
  - it was a request-envelope problem
  - the local smoke used:
    - `max_model_len=4096`
    - `understanding_max_tokens=4000`
    - `concretize_max_tokens=16000`
    - `extract_max_tokens=16000`
  - those Stage 1/2/3 budgets came from API defaults and are incompatible with the local `4096`-token context envelope
  - the first request failed before any prompt shard was written, which is consistent with server-side request validation rejecting the completion budget
- **Fix applied before relaunch:**
  - `batched_vllm_serve.py` now logs the HTTP response body on `400` failures
  - `align.py` now exposes prompt-generation token-budget fields through `AlignConfig`
  - `experiments/generate_prompts_llama_3_3_70b_refactored.py` now uses a smaller local smoke envelope:
    - `concretize_batch_size=4`
    - `extract_batch_size=4`
    - `local_serve_batch_size=4`
    - `understanding_max_tokens=1024`
    - `concretize_max_tokens=1536`
    - `extract_max_tokens=1024`
  - the experiment now also resolves the model path through the regional executor dependency instead of hardcoding the `us-central1` GCS prefix

### 2026-03-24 — ALIGN-083: Relaunched Experiment D in `us-east5-a` with Regional Model Resolution and Smaller Local Prompt Budgets

- **New job:**
  - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-east5a`
- **Submission shape:**
  - CPU-only root:
    - `cpu=4`
    - `memory=16GB`
    - `disk=10GB`
  - region:
    - `us-east5`
  - zone:
    - `us-east5-a`
- **What changed versus the failed `us-central1` run:**
  - prompt-generation smoke now uses smaller Stage 1/2/3 token budgets that fit the local `4096` context envelope
  - prompt-generation smoke now uses smaller per-request batch sizes
  - experiment model path now resolves through `output_path_of(llama_3_3_70b_instruct)`, so the executor should reuse the staged `us-east5` model artifact instead of a hardcoded `us-central1` GCS path
- **Current state at handoff:**
  - root job is `JOB_STATE_RUNNING`
  - task state is still `building`, so child-step fanout has not yet appeared in the first poll
- **Immediate next check for future agents:**
  - wait for the root to finish bootstrap
  - confirm the `spec` child starts
  - confirm the `prompts` child starts in `us-east5-a`
  - if the prompt child fails again, the new HTTP response-body logging in `batched_vllm_serve.py` should make the server-side error explicit

### 2026-03-24 — ALIGN-084: Measured Real Prompt Lengths for `ask_clarifying_questions`; `16k` Looks Unnecessary for the Smoke Run

- **How this was measured:**
  - reused the saved OpenAI-path artifacts at:
    - `gs://marin-us-central1/align/openai_spec_smoke/prompts-8a5a5d/artifacts/ask_clarifying_questions`
  - rendered the exact Stage 1/2/3 prompts with the staged Llama 3.3 70B tokenizer from:
    - `gs://marin-us-east5/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
- **Measured prompt token counts:**
  - Stage 1 understanding:
    - `910`
  - Stage 2 concretization:
    - batch size `4`: `1811`
    - batch size `10`: `2429`
  - Stage 3 extraction:
    - batch size `4`: `740`
    - batch size `10`: `1433`
- **Interpretation:**
  - the original failure was not because the Stage 1 prompt itself exceeded `4096`
  - the failure is still best explained by the combination of:
    - `max_model_len=4096`
    - oversized API-style completion budgets (`4000` / `16000`)
  - for this one-statement smoke, the reduced local budgets now in the relaunched `us-east5-a` job should fit comfortably inside `4096`
- **Recommendation:**
  - do **not** jump straight to `16k` for the one-statement smoke
  - if a larger local context envelope is needed after the smoke, the next reasonable step is probably `8k`, not `16k`
  - `16k` would add compile / KV-cache pressure without evidence that this prompt-generation workload actually needs it

### 2026-03-24 — ALIGN-085: Took Over Babysitting for the Corrected `us-east5-a` Experiment D Run

- **Monitoring owner:** this thread is now actively babysitting the run.
- **Job under watch:**
  - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-east5a`
- **State file:**
  - `scratch/20260324-2231_monitoring_state.json`
- **Current live status at monitor handoff:**
  - root is `JOB_STATE_RUNNING`
  - staged `us-east5` model dependency was skipped as already succeeded
  - spec child succeeded
  - prompt TPU child is pending only on `us-east5-a` TPU capacity:
    - `no_capacity: tpu_v5p_8-us-east5-a=backoff`
- **Interpretation:**
  - the corrected run is structurally healthy
  - there is no new prompt-shape or HTTP error yet
  - current blocker is scheduler capacity only

### 2026-03-25 — ALIGN-086: Stage 2 Concretization Now Identifies Exact Missing `cfg_*` IDs and Retries Them

- **Problem this addresses:**
  - the successful `us-central1` prompt-generation smoke run logged a `3/4` Stage 2 concretization shortfall for `ask_clarifying_questions`
  - the old Stage 2 response format used generic `<scenario>` / `<rubric>` tags and positional parsing
  - that meant we could tell a batch was short, but **not** which covering-array config was missing
  - the old code then silently continued, and the missing concretization was dropped before Stage 3
- **What changed:**
  - `prompts/concretize.py` now requests config-scoped tags such as:
    - `<scenario_cfg_000>...</scenario_cfg_000>`
    - `<rubric_cfg_000>...</rubric_cfg_000>`
  - `generate_prompts.py` now parses Stage 2 outputs by config id instead of by order
  - Stage 2 now records `concretization_attempts` inside each statement’s `ideation.json`
  - each attempt record includes:
    - `attempt`
    - `requested_config_ids`
    - `requested_configs`
    - `returned_config_ids`
    - `missing_config_ids`
    - `missing_rubric_config_ids`
    - `raw_response`
  - missing configs are retried as singleton requests
  - Stage 2 now fails explicitly if any configs are still missing after the retry budget instead of silently dropping them
- **New config plumbing:**
  - `PromptGenConfig.concretize_max_attempts`
  - `AlignConfig.concretize_max_attempts`
- **Operational consequence:**
  - after this change, we can answer both:
    - what went wrong in a partial Stage 2 batch
    - exactly which `cfg_*` entries need retry

### 2026-03-25 — ALIGN-087: Retry / Diagnostics Refactor Validated in Tests

- **Validation coverage added:**
  - indexed Stage 2 parser test
  - prompt-template test for indexed concretization tags
  - local prompt-generation regression test where Stage 2 first returns only one config, then retries the missing `cfg_001` and recovers it
- **Verification run:**
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py lib/marin/src/marin/alignment/prompts/concretize.py lib/marin/src/marin/alignment/align.py tests/test_alignment.py`
  - `uv run pytest tests/test_alignment.py -q`
- **Result:**
  - `71 passed`
- **Current recommendation for future agents:**
  - if a prompt-generation run now reports a Stage 2 problem, inspect the statement’s `artifacts/<statement_id>/ideation.json`
  - `concretization_attempts[*].missing_config_ids` is now the source of truth for which covering-array configs were missing and retried

### 2026-03-25 — ALIGN-088: Raised Stage 2 Concretization Retry Budget from `2` to `5`

- **Code changes:**
  - `PromptGenConfig.concretize_max_attempts` default: `5`
  - `AlignConfig.concretize_max_attempts` default: `5`
  - `experiments/generate_prompts_llama_3_3_70b_refactored.py` now explicitly sets:
    - `concretize_max_attempts=5`
- **Reasoning:**
  - the new retry path is now targeted and singleton-based for missing `cfg_*` entries
  - that makes a higher retry budget relatively cheap compared with rerunning an entire prompt-generation step
  - `5` attempts is a better smoke-setting for measuring whether partial Stage 2 batches can self-heal before we escalate
- **Validation after config change:**
  - `uv run pytest tests/test_alignment.py -q`
  - result: `71 passed`

### 2026-03-25 — ALIGN-089: Relaunched Prompt-Generation Smoke on Iris with `5` Stage 2 Retries

- **Job submitted:**
  - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-central1-retry5`
- **Submit command:**
  - `uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --job-name generate-prompts-llama-3-3-70b-refactored-us-central1-retry5 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/generate_prompts_llama_3_3_70b_refactored.py`
- **Why this launch shape:**
  - region-only `us-central1` to avoid unnecessary zone pinning
  - CPU-only root
  - prompt-generation worker still comes from the executor step and will request the `v5p-8` TPU shape defined by the staged Llama 3.3 70B `VLLMConfig`
- **Initial status snapshot right after launch:**
  - root job state: `JOB_STATE_RUNNING`
  - no child status yet in the first poll
- **What future agents should check next:**
  - prompt child start
  - whether any Stage 2 `concretization_attempts` are recorded in the final `ideation.json`
  - if there is still a shortfall after `5` attempts, capture the exact `missing_config_ids` from the artifact instead of inferring from row counts

### 2026-03-25 — ALIGN-090: `retry5` Finished in ~28s Because the Executor Reused the Old Prompt Artifact

- **Observed symptom:**
  - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-central1-retry5` finished in about `28.9s`
- **Root cause from logs:**
  - the root launched, read existing step statuses, and skipped all three steps as already succeeded
  - the key line was:
    - `Skip align/debug_generate_prompts_llama_3_3_70b_refactored/prompts_3da7bc76: already succeeded`
- **Interpretation:**
  - this was **not** a real rerun of prompt generation
  - changing `concretize_max_attempts` from `2` to `5` did not affect the executor version key for the prompt artifact
  - so the root reused the old successful prompt artifact:
    - `gs://marin-us-central1/align/debug_generate_prompts_llama_3_3_70b_refactored/prompts-4c7a65`

### 2026-03-25 — ALIGN-091: Fixed Prompt-Step Versioning and Relaunched a True `retry5` Run

- **Cache-key fix:**
  - updated the prompt-generation executor config so semantic Stage 1/2/3 knobs are part of the step version
  - in particular, the prompt step now versions:
    - `covering_strength`
    - `covering_seed`
    - `concretize_batch_size`
    - `extract_batch_size`
    - `understanding_max_tokens`
    - `understanding_temperature`
    - `concretize_max_tokens`
    - `concretize_temperature`
    - `concretize_max_attempts`
    - `extract_max_tokens`
    - `statement_ids`
- **Files changed for this fix:**
  - `lib/marin/src/marin/alignment/align.py`
  - `experiments/generate_prompts_llama_3_3_70b_refactored.py`
- **Verification after patch:**
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/align.py experiments/generate_prompts_llama_3_3_70b_refactored.py`
  - `uv run pytest tests/test_alignment.py -q`
  - result: `71 passed`
- **Fresh relaunch:**
  - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-central1-retry5-versioned`
- **Initial status at handoff:**
  - root state: `JOB_STATE_RUNNING`
  - this is the real rerun to watch, not the earlier cached `retry5` root

### 2026-03-25 — ALIGN-092: True `retry5` Prompt-Generation Run Succeeded and Recovered the Missing Config

- **Successful run:**
  - root:
    - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-central1-retry5-versioned`
  - prompt child:
    - `/ahmed/generate-prompts-llama-3-3-70b-refactored-us-central1-retry5-versioned/align-debug_generate_prompts_llama_3_3_70b_refactored-prompts_3da7bc76-57844711`
- **New prompt artifact:**
  - `gs://marin-us-central1/align/debug_generate_prompts_llama_3_3_70b_refactored/prompts-f29568`
- **What happened in Stage 2:**
  - first pass still hit the same partial batch:
    - attempt `1`
    - requested `cfg_000..cfg_003`
    - returned `cfg_000, cfg_001, cfg_002`
    - missing `cfg_003`
  - retry logic then issued a singleton retry:
    - attempt `2`
    - requested `cfg_003`
    - returned `cfg_003`
- **Verified outputs from artifact contents:**
  - `num_configs = 67`
  - final prompt rows = `67`
  - `cfg_003` now has a non-empty scenario description in `ideation.json`
  - `concretization_attempts` persisted the exact missing/returned config ids as designed
- **Interpretation:**
  - the new retry/diagnostics path is working end-to-end on real TPU `vllm serve` prompt generation
  - the previous `66/67` shortfall is now recovered automatically without rerunning the full statement

### 2026-03-25 — ALIGN-093: Implemented Experiment E by Combining Local Chosen/Rejected Generation into One `align()` Step

- **Scope completed:**
  - Experiment E is now implemented in code.
  - When both `teacher_model` and `rejected_model` are local `VLLMConfig`s, `align()` no longer emits sibling `chosen` and `rejected` executor steps.
  - Instead it emits one combined local response step:
    - `align/<name>/responses`
- **How the new step works:**
  - prompts are loaded once
  - chosen generation runs first
  - rejected generation runs second on the same worker
  - if the two local model configs are equal, one `BatchedVllmServeSession` is reused across both passes
  - if they differ, the step tears down the first session and starts the second, still keeping at most one local model active at a time
- **Output contract preserved for downstream steps:**
  - combined step writes:
    - `<responses_output>/chosen`
    - `<responses_output>/rejected`
  - `judgments` now reads those two subpaths, so no judge/filter schema change was required
- **Resource policy:**
  - local-local response steps now merge worker resources conservatively:
    - same accelerator type/replica count required
    - CPU, RAM, and disk take the max of the two configs
  - this keeps the one-worker sequential design without under-requesting resources
- **Files changed:**
  - `lib/marin/src/marin/alignment/generate_responses.py`
  - `lib/marin/src/marin/alignment/align.py`
  - `tests/test_alignment.py`
- **Local verification:**
  - `uv run pytest tests/test_alignment.py -q`
  - result: `75 passed`
  - added coverage for:
    - same-model local session reuse in combined response generation
    - different-model sequential local sessions
    - `align()` DAG switch from `{chosen,rejected}` to `{responses}` for local-local runs

### 2026-03-25 — ALIGN-094: Prepared the First Post-Experiment-E End-to-End Smoke Run

- **Target experiment:**
  - `experiments/align_debug_vllm_70b_mixtral_rejected.py`
- **Why this is the right Experiment F validation:**
  - it exercises the whole one-statement open-weight pipeline with the new refactored primitives:
    - prompt generation via batched `vllm serve`
    - combined local-local chosen/rejected generation via one-model-at-a-time `align()` orchestration
    - local batched judge
- **Smoke-script updates before launch:**
  - changed root submission guidance to CPU-only executor wrapper
  - reduced local worker disk from `500g` to `10g`
  - kept TPU child shape on `v5p-8`, `tp=4`, `ram=256g`
  - aligned prompt-generation knobs with the validated Experiment D envelope:
    - `prompt_batch_size=4`
    - `understanding_max_tokens=1024`
    - `concretize_max_tokens=1536`
    - `concretize_max_attempts=5`
    - `extract_max_tokens=1024`
    - `judge_batch_size=4`
- **Expected next validation signal:**
  - a real one-statement end-to-end artifact with:
    - prompt generation complete
    - combined `/responses/chosen` + `/responses/rejected` outputs
    - logged judgments
    - final preference pairs

### 2026-03-25 — ALIGN-095: Launched the First Post-Experiment-E End-to-End Smoke Run

- **Launched root job:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored`
- **Initial executor signal from root logs:**
  - `### Inspecting the 5 provided steps ###`
  - `### Launching 5 steps ###`
- **Interpretation:**
  - this matches the refactored one-statement DAG shape:
    - `spec`
    - `prompts`
    - `responses`
    - `judgments`
    - `preference_pairs`
  - notably, there are no separate provided `chosen` and `rejected` steps anymore for this local-local run
- **Current child status at handoff:**
  - spec child succeeded:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored/align-debug_vllm_70b_mixtral_rejected_smoke-spec_87ed33ce-a365a01e`
  - prompt child is pending on TPU capacity:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored/align-debug_vllm_70b_mixtral_rejected_smoke-prompts_9588217b-5580cf8a`
- **Prompt child resource request:**
  - `v5p-8`
  - `ram=256g`
  - `disk=10g`
- **Most recent scheduler reason:**
  - insufficient TPU capacity in `us-central1-a`
  - autoscaler also reported waiting for workers in `tpu_v5p_8-us-central1-a` to become ready
- **Why this state still matters:**
  - it confirms the smoke run is using the refactored CPU-root + executor-child structure
  - it confirms the new smaller-disk prompt-generation child request is being used in the real end-to-end pipeline

### 2026-03-25 — ALIGN-096: Heterogeneous Combined `/responses` Step Failed on the Second TPU `vllm serve` Startup

- **Failed run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored`
- **What succeeded before the failure:**
  - `spec`
  - `prompts`
  - first local response server startup for chosen Llama
  - the chosen pass reached a batched `/v1/completions` request for `67` prompts
- **What failed:**
  - the second local server startup, for Mixtral, inside the same worker-local combined `responses` step
- **Important log evidence:**
  - failing command:
    - `vllm serve gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230 ...`
  - `Resolved architecture: MixtralForCausalLM`
  - then TPU/JAX init degraded and the deepest visible failure was:
    - `AttributeError` from `tpu_inference.utils.make_optimized_mesh`
  - outer error surfaced as:
    - `RuntimeError: Engine core initialization failed`
- **Interpretation:**
  - same-model session reuse is still valid
  - but different-model local-local orchestration should not try to start a second TPU `vllm serve` in the same worker after the first exits
  - Experiment E therefore needs a narrower rule:
    - combined `/responses` only for same-model local-local
    - separate serialized `chosen` -> `rejected` steps for heterogeneous local-local runs
- **Structured debug log:**
  - `docs/debug-log-experiment-e-response-lifecycle.md`

### 2026-03-25 — ALIGN-097: Relaunched Experiment F with Serialized Heterogeneous `chosen` Then `rejected`

- **New run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored-retry-serialized`
- **Code change behind the relaunch:**
  - `align()` now uses:
    - combined `/responses` only when `teacher_model == rejected_model` and both are local
    - otherwise, if both are local but different, it emits separate `chosen` and `rejected` steps and adds an explicit executor dependency from rejected to chosen
- **Verification from root logs:**
  - `### Inspecting the 6 provided steps ###`
  - `### Launching 6 steps ###`
  - cached `spec` and `prompts` were skipped as already succeeded
  - a fresh `chosen` step was submitted:
    - `align/debug_vllm_70b_mixtral_rejected_smoke/chosen_2a11d777`
- **Current state at handoff:**
  - root is running
  - chosen child is pending on `v5p-8` TPU capacity in `us-central1-a`
  - rejected has not launched yet, which is the intended lifecycle for this heterogeneous retry

### 2026-03-25 — ALIGN-098: Final Architecture Plan for Robust Chosen/Rejected Response Orchestration

- **Goal:**
  - make chosen/rejected response generation robust across:
    - same local model on both sides
    - different local models on each side
    - mixed local/API configurations
  - keep the output contract stable for downstream `judgments` and `preference_pairs`

- **Core design principles:**
  - `chosen` and `rejected` remain separate logical artifacts even when optimized under the hood
  - same-worker/session reuse is only a valid optimization when both roles use the exact same local model config
  - different local models must not assume same-worker TPU reuse after one `vllm serve` exits
  - scheduling policy should be explicit and configurable rather than hardcoded

- **Proposed API shape:**
  - add a response-execution policy enum to `AlignConfig`, for example:
    - `auto`
    - `parallel`
    - `serialized`
    - `reuse_same_model`

- **Planned semantics:**
  - `auto`
    - same local model on both sides:
      - use one combined `/responses` step and reuse one `BatchedVllmServeSession`
    - different local models on both sides:
      - use separate `chosen` and `rejected` steps in parallel
    - any API/local mix:
      - use separate steps in parallel
  - `parallel`
    - force separate `chosen` and `rejected` steps with no dependency edge
    - lets Iris request two compute allocations at once if capacity exists
  - `serialized`
    - force separate `chosen -> rejected` steps with an explicit dependency edge
    - useful for quota pressure, debugging, or fragile backends
  - `reuse_same_model`
    - only legal when `teacher_model == rejected_model`
    - otherwise raise immediately

- **Planned implementation structure:**
  - keep two response primitives in `generate_responses.py`:
    - `generate_responses(...)` for one role
    - `generate_response_pair(...)` only for same-model local reuse
  - move response-step planning in `align.py` into a dedicated helper like `_build_response_steps(...)`
  - keep downstream `JudgeConfig` unchanged by always exposing:
    - chosen responses path
    - rejected responses path

- **Executor / dependency plan:**
  - near term:
    - use an explicit dependency path to serialize `rejected` after `chosen` when policy is `serialized`
  - longer term:
    - consider first-class executor step dependencies so serialization does not rely on config transport hacks

- **Concrete experiment plan after implementation:**
  - Experiment F1:
    - same-model local-local smoke with `auto`
    - expected: combined `/responses` step
  - Experiment F2:
    - heterogeneous local-local smoke with `auto`
    - expected: separate parallel `chosen` and `rejected` steps
  - Experiment F3:
    - heterogeneous local-local smoke with `serialized`
    - expected: separate `chosen -> rejected` steps
  - Compare across F1/F2/F3:
    - startup reliability
    - total wall-clock
    - queue time
    - response artifact row counts
    - final judgment/pair counts

- **Recommended steady-state default:**
  - default to `auto`
  - for the current Llama-chosen / Mixtral-rejected workflow, `auto` should mean separate parallel child jobs, not same-worker reuse
  - reserve same-worker reuse for the narrow case where both roles use the exact same local model

### 2026-03-25 — ALIGN-099: Implemented Explicit `ResponseExecutionMode` Policy in `align()`

- **What landed in code:**
  - `AlignConfig` now has an explicit response policy enum:
    - `auto`
    - `parallel`
    - `serialized`
    - `reuse_same_model`
  - implementation is in:
    - `lib/marin/src/marin/alignment/align.py`
- **Current behavior:**
  - `auto`
    - same local model => combined `/responses`
    - different local models => separate `chosen` and `rejected` steps with no dependency edge
    - mixed local/API => separate steps with no dependency edge
  - `parallel`
    - always separate `chosen` and `rejected` with no dependency edge
  - `serialized`
    - always separate `chosen -> rejected` with an explicit dependency edge
  - `reuse_same_model`
    - only valid when both roles use the exact same local `VLLMConfig`
    - otherwise `align()` raises immediately
- **Why this is the robust design:**
  - it preserves the successful same-model reuse optimization
  - it removes the failed lifecycle assumption that two different TPU `vllm serve` startups can safely happen in one worker
  - it still allows heterogeneous local-local runs to request two TPU allocations at once when policy is `auto` or `parallel`
- **Local verification:**
  - `uv run pytest tests/test_alignment.py -q`
  - result: `79 passed`
  - coverage now includes:
    - same local model + `auto` => combined `/responses`
    - different local models + `auto` => separate parallel `chosen` / `rejected`
    - different local models + `parallel` => separate parallel `chosen` / `rejected`
    - different local models + `serialized` => explicit `chosen -> rejected`
    - invalid `reuse_same_model` => raises

### 2026-03-25 — ALIGN-100: Concrete Validation Plan for the Final Response-Orchestration Architecture

- **Goal:**
  - validate correctness, reliability, and operational tradeoffs for the new response policy layer

- **F1: Same-model local-local smoke (`auto`)**
  - configuration:
    - chosen = Llama 3.3 70B
    - rejected = same Llama 3.3 70B
    - `response_execution_mode=auto`
  - expected DAG shape:
    - one combined `/responses` step
  - success criteria:
    - one local worker handles both passes
    - `judgments` and `preference_pairs` succeed
    - no loss in response row counts relative to prompts

- **F2: Heterogeneous local-local smoke (`auto`)**
  - configuration:
    - chosen = Llama 3.3 70B
    - rejected = Mixtral 8x7B Instruct
    - `response_execution_mode=auto`
  - expected DAG shape:
    - separate `chosen` and `rejected` steps with no dependency edge
  - success criteria:
    - root logs show `6 provided steps`
    - both child steps become independently schedulable
    - no same-worker sequential different-model startup
    - final `judgments` and `preference_pairs` succeed

- **F3: Heterogeneous local-local smoke (`serialized`)**
  - configuration:
    - same models as F2
    - `response_execution_mode=serialized`
  - expected DAG shape:
    - separate `chosen -> rejected` steps
  - success criteria:
    - `rejected` does not launch before `chosen` succeeds
    - full pipeline succeeds
    - use this as the reliability baseline when TPU capacity is scarce

- **F4: Invalid same-model reuse guard**
  - configuration:
    - chosen != rejected
    - `response_execution_mode=reuse_same_model`
  - expected behavior:
    - `align()` raises before launch
  - purpose:
    - confirms policy misuse fails fast instead of degrading into unsafe runtime behavior

- **Metrics to compare across F1/F2/F3:**
  - scheduler wait for response children
  - child startup time
  - response row counts
  - judgment row counts
  - final pair counts
  - total wall-clock from root start to pair artifact
  - failure mode, if any

- **Recommended execution order:**
  - run F4 locally first (already covered by unit tests)
  - then F1
  - then F2
  - then F3 only if F2 is flaky or if quota pressure is high

### 2026-03-25 — ALIGN-101: F3 Serialized Heterogeneous Smoke Succeeded End-to-End

- **Successful root run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored-retry-serialized`
- **Successful child sequence:**
  - `chosen`:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored-retry-serialized/align-debug_vllm_70b_mixtral_rejected_smoke-chosen_2a11d777-5edacf79`
  - `rejected`:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored-retry-serialized/align-debug_vllm_70b_mixtral_rejected_smoke-rejected_b0db5cf5-3cf23e13`
  - `judgments`:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored-retry-serialized/align-debug_vllm_70b_mixtral_rejected_smoke-judgments_170c6c42-7feb8586`
  - `preference_pairs`:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-smoke-refactored-retry-serialized/align-debug_vllm_70b_mixtral_rejected_smoke-preference_pairs_dfa366bc-90096f55`
- **Verified DAG behavior from root logs:**
  - `6 provided steps`
  - cached `spec` and `prompts` reused
  - `rejected` launched only after `chosen` succeeded
- **Successful artifact paths from the final run:**
  - prompts:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_smoke/prompts-630ada`
  - chosen:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_smoke/chosen-3bd245`
  - rejected:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_smoke/rejected-f50036`
  - judgments:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_smoke/judgments-c4ad9d`
  - preference pairs:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_smoke/preference_pairs-ed6a0e`
- **Interpretation:**
  - F3 is now the externally validated reliability baseline for heterogeneous local-local alignment runs
  - the serialized policy is operationally sound even though it gives up same-host reuse

### 2026-03-25 — ALIGN-102: Preparing F1 Same-Model Local-Local Smoke with `auto`

- **Next target experiment:**
  - `experiments/align_debug_vllm_70b.py`
- **Refactor applied before launch:**
  - switched from HF model id to staged GCS artifact via `output_path_of(llama_3_3_70b_instruct)`
  - reduced local disk from `500g` to `10g`
  - aligned prompt-generation settings to the validated refactored smoke envelope
  - set `response_execution_mode=auto`
  - renamed the align run to:
    - `debug_vllm_70b_same_model_auto_smoke`
- **Expected F1 behavior:**
  - because chosen and rejected use the exact same local model config, `auto` should collapse them into one combined `/responses` step and reuse a single local `vllm serve` session

### 2026-03-25 — ALIGN-103: Launched F1 Same-Model Local-Local Smoke with `auto`

- **Run launched:**
  - `/ahmed/align-debug-vllm-70b-same-model-auto`
- **Experiment script:**
  - `experiments/align_debug_vllm_70b.py`
- **What changed in the script before launch:**
  - switched to staged GCS Llama via `output_path_of(llama_3_3_70b_instruct)`
  - reduced local disk to `10g`
  - aligned prompt-generation knobs with the validated refactored envelope
  - set `response_execution_mode=auto`
  - renamed align run to:
    - `debug_vllm_70b_same_model_auto_smoke`
- **Initial executor signal from root logs:**
  - `### Inspecting the 5 provided steps ###`
  - `### Reading 6 statuses ###`
  - `### Launching 6 steps ###`
- **Interpretation:**
  - the extra status is the cached model dependency step for `llama_3_3_70b_instruct`
  - the logical pipeline still has the expected same-model shape:
    - `spec`
    - `prompts`
    - `responses`
    - `judgments`
    - `preference_pairs`
- **Current state at handoff:**
  - root is running
  - model dependency already skipped as succeeded
  - spec child is running
  - next validation target is whether `auto` emits one combined `/responses` child once prompt generation clears

### 2026-03-25 — ALIGN-104: F1 Failed Because Combined Response Config Serialized the Model `InputName` as a Plain Dict

- **Failed run:**
  - `/ahmed/align-debug-vllm-70b-same-model-auto`
- **What succeeded before failure:**
  - `spec`
  - `prompts`
- **What failed:**
  - combined same-model `responses` child:
    - `/ahmed/align-debug-vllm-70b-same-model-auto/align-debug_vllm_70b_same_model_auto_smoke-responses_b343f6c1-6abe9a87`
- **Observed error:**
  - `AttributeError: 'dict' object has no attribute 'decode'`
- **Root cause from child logs:**
  - the failure happened in `batched_vllm_serve._load_tokenizer()` at `urlparse(model_path)`
  - `model_path` was not a string or `InputName`; it had already been converted into a plain dict
  - this came from `_serialize_inference_config()` using `dataclasses.asdict()`, which recursively converted the executor `InputName` in `VLLMConfig.model` into a dict
- **Interpretation:**
  - same-model combined `/responses` logic itself is still valid
  - the bug was only in how the local model config was serialized for executor transport

### 2026-03-25 — ALIGN-105: Fixed Shallow Inference-Config Serialization and Added a Regression Test

- **Code fix:**
  - `_serialize_inference_config()` in `lib/marin/src/marin/alignment/align.py` now performs a shallow dataclass field copy instead of `dataclasses.asdict()`
  - this preserves executor-resolvable objects like `InputName` in `VLLMConfig.model`
- **Regression test added:**
  - `tests/test_alignment.py` now verifies that same-model combined response configs preserve `chosen_model_config["model"]` and `rejected_model_config["model"]` as `InputName`
- **Verification:**
  - `uv run pytest tests/test_alignment.py -q`
  - result: `80 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/align.py tests/test_alignment.py experiments/align_debug_vllm_70b.py`
  - result: clean
- **Next action:**
  - relaunch F1 with the shallow-serialization fix in place

### 2026-03-25 — ALIGN-106: Confirmed F1 Failure Signature and Relaunched the Same-Model `auto` Smoke

- **Failure signature re-confirmed from Iris metadata:**
  - `/ahmed/align-debug-vllm-70b-same-model-auto/align-debug_vllm_70b_same_model_auto_smoke-responses_b343f6c1-6abe9a87`
  - `AttributeError: 'dict' object has no attribute 'decode'`
- **Meaning:**
  - the failed run was still on the pre-fix codepath where combined-response config transport recursively converted `VLLMConfig.model` from executor `InputName` into a plain dict
- **Post-fix verification before retry:**
  - `uv run pytest tests/test_alignment.py -q`
  - result: `80 passed`
- **Fresh retry launched:**
  - `/ahmed/align-debug-vllm-70b-same-model-auto-retry`
- **Expected retry shape:**
  - cached `spec` and `prompts` may be reused
  - the same-model combined `/responses` step should rerun with `InputName` preserved in both chosen and rejected local model configs
- **Next check:**
  - verify the retry emits one combined `/responses` child and no longer fails during tokenizer/model-path resolution

### 2026-03-25 — ALIGN-107: F1 Same-Model `auto` Smoke Succeeded with the Combined `/responses` Step

- **Successful retry:**
  - `/ahmed/align-debug-vllm-70b-same-model-auto-retry`
- **Child outcomes:**
  - combined same-model responses:
    - `/ahmed/align-debug-vllm-70b-same-model-auto-retry/align-debug_vllm_70b_same_model_auto_smoke-responses_59519d45-bf8efbbf`
    - `JOB_STATE_SUCCEEDED`
  - judgments:
    - `/ahmed/align-debug-vllm-70b-same-model-auto-retry/align-debug_vllm_70b_same_model_auto_smoke-judgments_18225761-94e985f8`
    - `JOB_STATE_SUCCEEDED`
  - preference pairs:
    - `/ahmed/align-debug-vllm-70b-same-model-auto-retry/align-debug_vllm_70b_same_model_auto_smoke-preference_pairs_e6ebd40e-f6119b30`
    - `JOB_STATE_SUCCEEDED`
- **Key evidence that `auto` took the intended same-model reuse path:**
  - `Loaded 67 prompts for combined local response generation`
  - two batched `/v1/completions` calls were issued from the single combined response step
  - outputs were written to:
    - `gs://marin-us-central1/align/debug_vllm_70b_same_model_auto_smoke/responses-c8c2e6/chosen`
    - `gs://marin-us-central1/align/debug_vllm_70b_same_model_auto_smoke/responses-c8c2e6/rejected`
- **Judging/filter results:**
  - judgments artifact:
    - `gs://marin-us-central1/align/debug_vllm_70b_same_model_auto_smoke/judgments-9bb1e8`
    - `67` judgment records written
  - final preference-pair artifact:
    - `gs://marin-us-central1/align/debug_vllm_70b_same_model_auto_smoke/preference_pairs-ad29d9`
    - `63` preference pairs written
  - filter sidecars were written under:
    - `gs://marin-us-central1/align/debug_vllm_70b_same_model_auto_smoke/preference_pairs-ad29d9/artifacts/filter_decisions`
- **Conclusion:**
  - F1 validates the robust same-model `auto` architecture:
    - one combined local-local `/responses` step
    - one TPU worker for chosen + rejected generation
    - no `InputName` transport regression after shallow config serialization
- **Next queued experiment:**
  - F2: heterogeneous local-local smoke with `response_execution_mode=auto`, which should emit separate parallel `chosen` and `rejected` steps

### 2026-03-25 — ALIGN-108: Fail-Closed Plan for Adding Rejected-Only `opposite` Mode

- **Goal:**
  - add a Bloom-style adversarial inversion mode for rejected response generation
  - keep the API streamlined
  - make it impossible to accidentally run chosen responses in `opposite` mode or rejected responses in a chosen-style mode

- **Current state to replace:**
  - chosen = spec guidance appended to the scenario system prompt
  - rejected = no spec guidance
  - there is currently no explicit adversarial inversion mode

- **Fail-closed API design:**
  - do **not** expose a generic free-form `statement_prompt_mode` on chosen/rejected step configs
  - instead add one top-level rejected-only control to `AlignConfig`, something like:
    - `rejected_prompt_strategy: RejectedPromptStrategy = "unguided" | "opposite"`
  - keep chosen behavior implicit and fixed:
    - chosen always means policy-following / standard guidance
  - rejected is the only side allowed to vary:
    - `unguided` = current behavior
    - `opposite` = explicit adversarial inversion prompt

- **Why this shape is safer than a generic mode flag:**
  - there is no user-facing knob that can set chosen to `opposite`
  - there is no user-facing knob that can set rejected to `standard`
  - the good/bad polarity is derived from role, not from caller discipline

- **Required internal types:**
  - add `ResponseRole = CHOSEN | REJECTED`
  - add `RejectedPromptStrategy = UNGUIDED | OPPOSITE`
  - extend `ResponseGenConfig` and `ResponsePairGenConfig` with role-aware fields rather than generic mode strings

- **Validation rules that must raise immediately:**
  - chosen config must always satisfy:
    - `role == CHOSEN`
    - spec path is present
    - rejected strategy is not set
  - rejected config must always satisfy:
    - `role == REJECTED`
    - if strategy is `UNGUIDED`, no spec text is required
    - if strategy is `OPPOSITE`, spec path is required
  - any attempt to construct:
    - chosen + opposite
    - chosen + rejected strategy
    - rejected + chosen-only semantics
    - opposite without a spec path
    - must raise before job launch
  - duplicate these checks both:
    - when building configs in `align.py`
    - when resolving configs at runtime in `generate_responses.py`

- **Prompt-builder design:**
  - split the current `_build_messages(...)` into explicit role-aware builders:
    - `_build_chosen_messages(...)`
    - `_build_rejected_messages(...)`
  - rejected builder should branch on `RejectedPromptStrategy`
  - `opposite` prompt should:
    - frame the task as a safety research / adversarial evaluation exercise
    - include the exact behavior statement text
    - set a primary directive to violate the statement’s intent
    - treat any existing scenario system prompt as subordinate context to be inverted, not discarded
  - keep this logic centralized in `generate_responses.py`; do not duplicate prompt text across experiments

- **Recommended metadata/artifacts:**
  - persist response-generation metadata in each output record:
    - `response_role`
    - `rejected_prompt_strategy` if applicable
    - `behavior_prompt_mode_resolved = standard | opposite | unguided`
  - this makes it auditable after the fact which side used inversion

- **Executor / orchestration impact:**
  - no change to `ResponseExecutionMode`
  - same-model combined `/responses` should still work:
    - chosen request batch uses standard prompt construction
    - rejected request batch uses opposite prompt construction
    - one shared `vllm serve` session is still valid because only the prompt text changes
  - heterogeneous local-local and API paths continue to work unchanged at the orchestration layer

- **Implementation plan:**
  1. Add `ResponseRole` and `RejectedPromptStrategy` enums.
  2. Add `rejected_prompt_strategy` to `AlignConfig` with default `UNGUIDED`.
  3. Make `align()` derive role-specific response configs instead of exposing free-form prompt mode fields.
  4. Replace generic `_build_messages(...)` with explicit chosen/rejected builders.
  5. Add fail-fast validation in config constructors / resolvers.
  6. Persist resolved prompt-mode metadata in response records.
  7. Update same-model and heterogeneous smoke scripts to opt into `rejected_prompt_strategy="opposite"` where desired.

- **Exact validation experiments:**
  - **O1 — Unit/API guardrails**
    - chosen config with opposite-like settings raises
    - rejected opposite without spec path raises
    - same-model combined response config preserves role polarity correctly
  - **O2 — Standalone rejected-opposite response generation**
    - run `generate_responses.py` on an existing prompt artifact
    - chosen not involved
    - verify response records log `response_role=rejected` and `behavior_prompt_mode_resolved=opposite`
  - **O3 — Same-model `auto` smoke with opposite rejected**
    - reuse the same-model Llama script
    - set `rejected_prompt_strategy=opposite`
    - verify one combined `/responses` step still succeeds
  - **O4 — Heterogeneous smoke with opposite rejected**
    - Llama chosen, Mixtral rejected
    - first in `serialized` mode for reliability
    - then in `auto` mode for parallel child validation
  - **O5 — Artifact audit**
    - confirm judgments / filter outputs are unchanged structurally
    - confirm response artifacts clearly distinguish chosen-standard vs rejected-opposite

- **Recommended rollout order:**
  - implement + land O1 first
  - then O2
  - then O3
  - then O4 serialized
  - then O4 auto

- **Bottom line:**
  - the streamlined design is to make `opposite` a rejected-only strategy, not a generic mode
  - role determines what is allowed
  - any polarity inversion bug should raise immediately instead of silently generating mislabeled data

### 2026-03-25 — ALIGN-109: O1 Implemented — Rejected-Only `opposite` Mode with Fail-Closed Validation

- **Code changes landed locally:**
  - `lib/marin/src/marin/alignment/generate_responses.py`
  - `lib/marin/src/marin/alignment/align.py`
  - `tests/test_alignment.py`
- **New response-layer enums:**
  - `ResponseRole = chosen | rejected`
  - `RejectedPromptStrategy = unguided | opposite`
- **Fail-closed behavior now enforced:**
  - chosen responses require a spec path
  - chosen responses cannot specify any rejected strategy
  - rejected `unguided` responses must not receive a spec path
  - rejected `opposite` responses require a spec path
  - pair config validates the same polarity constraints
- **Prompt-building refactor:**
  - replaced generic message construction with explicit:
    - `_build_chosen_messages(...)`
    - `_build_rejected_messages(...)`
  - rejected `opposite` now emits an adversarial safety-research system prompt that:
    - includes the behavior statement
    - instructs the model to violate it
    - treats the original scenario system prompt as subordinate context to invert
- **Response artifact metadata added:**
  - `response_role`
  - `behavior_prompt_mode_resolved`
  - `rejected_prompt_strategy`
- **Validation:**
  - `uv run pytest tests/test_alignment.py -q`
  - result: `86 passed`
  - `./infra/pre-commit.py --fix ...` on all changed response/align files
  - result: clean

### 2026-03-25 — ALIGN-110: O2 Launched — Standalone Rejected-Only `opposite` Generation on Existing Prompts

- **New experiment script:**
  - `experiments/generate_rejected_opposite_llama_3_3_70b_existing_prompts.py`
- **Experiment purpose:**
  - validate rejected-only `opposite` prompting without involving the full alignment DAG
  - reuse a known-good prompt artifact and the staged `us-central1` Llama 3.3 70B checkpoint
- **Inputs:**
  - prompts:
    - `gs://marin-us-central1/align/debug_generate_prompts_llama_3_3_70b_refactored/prompts-f29568`
  - spec:
    - `experiments/posttrain/specs/openai_model_spec.jsonl`
  - model:
    - staged `us-central1` `meta-llama--Llama-3-3-70B-Instruct--6f6073b`
- **Live run:**
  - `/ahmed/generate-rejected-opposite-llama-3-3-70b-existing-prompts`
- **Current state at handoff:**
  - root is `JOB_STATE_RUNNING`
  - no cache-hit skip has been observed yet
- **Next check:**
  - confirm the remote response step launches
  - verify output rows carry:
    - `response_role = rejected`
    - `behavior_prompt_mode_resolved = opposite`
    - `rejected_prompt_strategy = opposite`

### 2026-03-25 — ALIGN-111: O2 Succeeded and Verified Opposite-Mode Response Metadata

- **Completed run:**
  - `/ahmed/generate-rejected-opposite-llama-3-3-70b-existing-prompts`
- **Successful child:**
  - `/ahmed/generate-rejected-opposite-llama-3-3-70b-existing-prompts/align-debug_generate_rejected_opposite_llama_3_3_70b_existing_prompts-responses_2f706d39-018b58fa`
- **Logs confirmed the intended codepath:**
  - `Loaded 46 behavior statements for rejected mode (opposite)`
  - `Loaded 67 prompts for batched vLLM serve generation`
  - `vLLM environment ready`
  - one batched `/v1/completions` request for `67` prompts
  - `Wrote 67 records to 1 shards in gs://marin-us-central1/align/debug_generate_rejected_opposite_llama_3_3_70b_existing_prompts/responses-f7b183`
- **Verified output metadata from the written shard:**
  - `response_role = rejected`
  - `behavior_prompt_mode_resolved = opposite`
  - `rejected_prompt_strategy = opposite`
- **Output artifact:**
  - `gs://marin-us-central1/align/debug_generate_rejected_opposite_llama_3_3_70b_existing_prompts/responses-f7b183`
- **Interpretation:**
  - O2 validates the fail-closed rejected-only opposite-mode path in isolation
  - the standalone response primitive is now ready to use inside the full heterogeneous alignment smoke

### 2026-03-25 — ALIGN-112: Launched F2 Heterogeneous `auto` Smoke with Rejected `opposite` Mode

- **New experiment script:**
  - `experiments/align_debug_vllm_70b_mixtral_rejected_opposite.py`
- **Purpose:**
  - skip O3 and go directly to heterogeneous local-local `auto`
  - chosen = Llama 3.3 70B
  - rejected = Mixtral 8x7B Instruct
  - rejected prompting = `opposite`
  - orchestration expectation = separate parallel `chosen` and `rejected` steps
- **Live run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto`
- **Next checks:**
  - executor emits separate `chosen` and `rejected` children
  - no serialized dependency between them
  - chosen uses standard guidance and rejected uses opposite-mode prompting

### 2026-03-25 — ALIGN-113: F2 Hit a Root Preemption During `prompts`, Then Recovered Automatically

- **Affected run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto`
- **What happened:**
  - `spec` succeeded
  - `prompts` started and reached active local `vllm serve` generation
  - root job then incurred a preemption
  - the in-flight `prompts` child was killed with:
    - `Parent task preempted`
- **Important detail:**
  - this was not an application or prompt-construction failure
  - logs before the preemption showed normal prompt-generation progress:
    - `Loaded 46 statements from spec`
    - `Filtered to 1 statements`
    - `vLLM environment ready`
    - batched `/v1/completions` calls for prompt-generation batches
- **Recovery behavior observed:**
  - the root remained alive with `preemption_count=1`
  - Iris restarted the root task automatically
  - the restarted root reused the completed `spec` artifact
  - it then reattached to the still-running `prompts` step state with:
    - `Status ... prompts ...: RUNNING`
    - `Step ... has no active lock, taking over`
    - `Acquired lock for ... prompts ...`
- **Interpretation:**
  - no manual resubmission was needed
  - F2 is still the active run to babysit
- **Next check:**
  - wait for `prompts` to finish under the recovered root
  - then verify heterogeneous `auto` emits separate sibling `chosen` and `rejected` children

### 2026-03-25 — ALIGN-114: F2 Recovered `prompts` Successfully and Reached Parallel `chosen` / `rejected`

- **Affected run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto`
- **Recovered `prompts` outcome:**
  - recovered child succeeded:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto/align-debug_vllm_70b_mixtral_rejected_opposite_auto_smoke-prompts_78d7e3c8-151faf78`
  - output artifact:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/prompts-e27535`
  - logs confirmed:
    - `Wrote 67 records to 1 shards in gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/prompts-e27535`
    - `Saved artifacts for 1 statements ...`
- **Critical orchestration check passed:**
  - after `prompts` succeeded, the executor emitted two separate response children:
    - `chosen`: `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto/align-debug_vllm_70b_mixtral_rejected_opposite_auto_smoke-chosen_b28ffc4b-a01dc0c1`
    - `rejected`: `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto/align-debug_vllm_70b_mixtral_rejected_opposite_auto_smoke-rejected_99ce7554-4b01f0bf`
  - executor logs showed both locks being acquired independently, with no serialized dependency:
    - `Attempting to acquire lock for ... chosen_b28ffc4b`
    - `Attempting to acquire lock for ... rejected_99ce7554`
    - `Acquired lock for ... chosen_b28ffc4b`
    - `Acquired lock for ... rejected_99ce7554`
- **Current state at this checkpoint:**
  - `chosen` is `JOB_STATE_RUNNING`
  - `rejected` is `JOB_STATE_PENDING` on TPU capacity
  - this is still consistent with heterogeneous `auto` parallel scheduling; the delay is scheduler-side, not executor-side
- **Interpretation:**
  - F2 has now validated the intended `auto` orchestration policy for heterogeneous local-local runs
  - the remaining live check is behavioral:
    - chosen should use standard guided prompting
    - rejected should start separately and log opposite-mode prompting once it gets TPU capacity

### 2026-03-25 — ALIGN-115: F2 `chosen` Confirmed Standard Prompting; `rejected` Still Waiting on TPU Capacity

- **Affected run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto`
- **Chosen-side confirmation:**
  - live `chosen` child:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto/align-debug_vllm_70b_mixtral_rejected_opposite_auto_smoke-chosen_b28ffc4b-a01dc0c1`
  - logs now show:
    - `Loaded 46 behavior statements for chosen mode (standard)`
    - `Loaded 67 prompts for batched vLLM serve generation`
    - `Starting vLLM environment`
- **Rejected-side current blocker:**
  - live `rejected` child:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto/align-debug_vllm_70b_mixtral_rejected_opposite_auto_smoke-rejected_99ce7554-4b01f0bf`
  - current state:
    - `JOB_STATE_PENDING`
  - scheduler reason:
    - `Insufficient TPUs (need 4, available 0)`
    - `Insufficient memory (need 256.0GB, available 32.8GB)`
    - autoscaler waiting for `tpu_v5p_8-us-central1-a`
- **Interpretation:**
  - F2 has now proven both:
    - heterogeneous `auto` creates separate sibling response jobs
    - the `chosen` side preserves standard guided prompting under the new opposite-mode plumbing
  - the final remaining F2 validation is the `rejected` child actually starting and logging:
    - rejected role
    - opposite-mode behavior prompt resolution

### 2026-03-25 — ALIGN-116: F2 `rejected` Started and Confirmed Live Opposite-Mode Prompting

- **Affected run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto`
- **Scheduler update:**
  - `rejected` eventually acquired TPU capacity and moved from `JOB_STATE_PENDING` to `JOB_STATE_RUNNING`
- **Critical behavioral confirmation:**
  - logs now show:
    - `Loaded 46 behavior statements for rejected mode (opposite)`
- **What is now validated live inside F2:**
  - `chosen` runs with standard guided prompting
  - `rejected` runs as a separate sibling job with opposite-mode prompting
  - this is happening under heterogeneous `auto`, not serialized fallback
- **Remaining live work:**
  - wait for both response children to finish
  - confirm chosen/rejected output artifacts land successfully
  - then watch judgments and final preference-pair filtering

### 2026-03-25 — ALIGN-117: F2 Both Response Children Succeeded; Pipeline Advanced into `judgments`

- **Affected run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto`
- **Chosen response result:**
  - child:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto/align-debug_vllm_70b_mixtral_rejected_opposite_auto_smoke-chosen_b28ffc4b-a01dc0c1`
  - succeeded
  - output artifact:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/chosen-607ba4`
  - logs showed:
    - `Loaded 46 behavior statements for chosen mode (standard)`
    - one batched `/v1/completions` request for `67` prompts
    - `Wrote 67 records ... chosen-607ba4`
- **Rejected response result:**
  - child:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto/align-debug_vllm_70b_mixtral_rejected_opposite_auto_smoke-rejected_99ce7554-4b01f0bf`
  - succeeded
  - output artifact:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/rejected-67ca91`
  - logs showed:
    - `Loaded 46 behavior statements for rejected mode (opposite)`
    - one batched `/v1/completions` request for `67` prompts
    - `Wrote 67 records ... rejected-67ca91`
- **Downstream progression:**
  - after both response steps succeeded, executor launched:
    - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto/align-debug_vllm_70b_mixtral_rejected_opposite_auto_smoke-judgments_0348aa13-1d4adb2f`
  - judgments output path:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/judgments-aca2ba`
- **Interpretation:**
  - F2 has now validated the full heterogeneous `auto` response stage with opposite-mode rejected generation
  - only judge/filter completion remains before the run can be marked complete

### 2026-03-25 — ALIGN-118: F2 Completed Successfully End-to-End with Heterogeneous `auto` and Rejected `opposite`

- **Completed run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto`
- **Terminal result:**
  - root job `JOB_STATE_SUCCEEDED`
  - note: the earlier `prompts` child kill remained only as historical evidence of a root preemption; the recovered run completed successfully
- **Validated orchestration:**
  - heterogeneous `auto` created separate sibling `chosen` and `rejected` jobs
  - `chosen` ran with standard guided prompting
  - `rejected` ran with opposite-mode prompting
  - both response steps succeeded independently before judgment/filtering
- **Output artifacts:**
  - prompts:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/prompts-e27535`
  - chosen:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/chosen-607ba4`
  - rejected:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/rejected-67ca91`
  - judgments:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/judgments-aca2ba`
  - preference pairs:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke/preference_pairs-52b7d7`
- **Key completion evidence from logs:**
  - `Loaded 46 behavior statements for chosen mode (standard)`
  - `Loaded 46 behavior statements for rejected mode (opposite)`
  - `Wrote 67 records ... chosen-607ba4`
  - `Wrote 67 records ... rejected-67ca91`
  - `Loaded 67 chosen, 67 rejected responses`
  - `Wrote 67 judgment records ... judgments-aca2ba`
  - `Built 67 preference pairs ... preference_pairs-52b7d7`
  - `Wrote 67 records ... preference_pairs-52b7d7/artifacts/filter_decisions`
- **Interpretation:**
  - O2 and F2 together validate the rejected-only opposite-mode design locally and end-to-end
  - the fail-closed plumbing correctly keeps chosen on standard prompting while letting rejected opt into opposite-mode prompting
  - heterogeneous `auto` is now validated on the real Llama-chosen / Mixtral-rejected smoke path
- **Next logical follow-up:**
  - audit one written rejected shard from F2 for metadata fields:
    - `response_role = rejected`
    - `behavior_prompt_mode_resolved = opposite`
    - `rejected_prompt_strategy = opposite`
  - then decide whether to:
    - keep opposite mode as an optional adversarial benchmark only, or
    - route it into the main alignment dataset pipeline by default for rejected generation

### 2026-03-25 — ALIGN-119: Standardized Structured `vllm_metrics.json` Artifacts Across All Local-vLLM Alignment Stages

- **Goal:**
  - stop relying on ad hoc log scraping for local `vllm serve` stages
  - define one shared metrics contract for prompt generation, response generation, and local judging
- **Implementation:**
  - added shared metrics dataclasses and artifact writer in:
    - `lib/marin/src/marin/alignment/batched_vllm_serve.py`
  - instrumented `BatchedVllmServeSession` to accumulate:
    - `tokenizer_load_seconds`
    - `server_start_seconds`
    - per-stage `render_seconds`
    - per-stage `request_seconds`
    - `request_count`
    - `request_prompt_count`
    - `completion_count`
    - `input_token_count`
    - `output_token_count`
    - derived `input_tokens_per_second`
    - derived `output_tokens_per_second`
- **Stage labeling now emitted by the shared wrapper:**
  - prompt generation:
    - `understanding`
    - `concretize`
    - `extract`
  - response generation:
    - `chosen`
    - `rejected`
  - judge:
    - `judge`
- **Artifact locations:**
  - prompt generation:
    - `<prompts_output>/artifacts/vllm_metrics.json`
  - standalone response generation:
    - `<responses_output>/artifacts/vllm_metrics.json`
  - combined chosen/rejected response generation:
    - `<shared_parent>/artifacts/vllm_metrics.json`
  - judgments:
    - `<judgments_output>/artifacts/vllm_metrics.json`
- **Artifact schema:**
  - top level:
    - `logical_stage`
    - `session_count`
    - `sessions`
  - each session:
    - `session_name`
    - `backend = vllm_serve`
    - `model`
    - `tensor_parallel_size`
    - `max_model_len`
    - `tokenizer_load_seconds`
    - `server_start_seconds`
    - `session_enter_seconds`
    - `totals`
    - `stages`
- **Design notes:**
  - same-model reused sessions naturally accumulate metrics across multiple labeled stages in one session artifact
  - split-model flows write multiple named sessions into the same `vllm_metrics.json`
  - token counts are computed with the same tokenizer staged for the local `vllm serve` session
- **Verification:**
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/batched_vllm_serve.py lib/marin/src/marin/alignment/generate_prompts.py lib/marin/src/marin/alignment/generate_responses.py lib/marin/src/marin/alignment/judge.py tests/test_alignment.py`
  - `uv run pytest tests/test_alignment.py -q`
  - result:
    - `86 passed`
- **Interpretation:**
  - the local open-weight alignment pipeline now has one standardized structured metrics story across prompt generation, response generation, and judging
  - Experiment G no longer needs custom one-off timing helpers just to recover basic `vllm` startup/request/token metrics
- **Next logical follow-up:**
  - materialize one fresh end-to-end run after this patch and inspect the new `vllm_metrics.json` artifacts in GCS
  - then use those artifacts as the primary source for Experiment G runtime characterization

### 2026-03-25 — ALIGN-120: Launch Fresh One-Statement End-to-End Opposite-Mode Smoke to Materialize Standardized vLLM Metrics

- **Goal:**
  - rerun the full one-statement open-weight alignment pipeline after `ALIGN-119`
  - verify that the new standardized `artifacts/vllm_metrics.json` files are emitted for:
    - prompt generation
    - chosen response generation
    - rejected response generation
    - judgments
- **Chosen experiment script:**
  - `experiments/align_debug_vllm_70b_mixtral_rejected_opposite.py`
- **Why this script:**
  - already validated end-to-end
  - runs the full pipeline on exactly one statement:
    - `statement_ids = ["ask_clarifying_questions"]`
  - exercises the current intended production shape:
    - chosen = Llama 3.3 70B
    - rejected = Mixtral 8x7B Instruct
    - `response_execution_mode = auto`
    - `rejected_prompt_strategy = opposite`
- **Fresh launch policy:**
  - submit under a new Iris root job name so this rerun is operationally distinct from the earlier F2 smoke
  - allow cached infra/spec/model steps to skip if unchanged, but expect pipeline code changes from `ALIGN-119` to force fresh prompt/response/judge outputs
- **Planned checks after launch:**
  - confirm prompt generation writes `artifacts/vllm_metrics.json`
  - confirm both chosen and rejected write response-side metrics artifacts
  - confirm judgments write `artifacts/vllm_metrics.json`
  - compare emitted stage names and token/sec fields against the new standardized schema

### 2026-03-25 — ALIGN-121: Initial Metrics Rerun Was a Cache Hit; Forced a Fresh One-Statement Experiment Name

- **Affected run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto-metrics`
- **Observed behavior:**
  - root job finished in about `27s`
  - no child jobs were launched
  - this was not a real end-to-end rerun
- **Interpretation:**
  - the previous validated one-statement smoke outputs were reused
  - this did not materialize fresh `vllm_metrics.json` artifacts tied to the new standardized instrumentation patch
- **Recovery:**
  - added a dedicated experiment script:
    - `experiments/align_debug_vllm_70b_mixtral_rejected_opposite_metrics.py`
  - forced a fresh logical align dataset name:
    - `debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics`
  - plan is to relaunch under a new Iris root job name so the entire one-statement pipeline executes again

### 2026-03-25 — ALIGN-122: Fresh One-Statement End-to-End Metrics Run Succeeded and Wrote Standardized `vllm_metrics.json` Artifacts

- **Completed run:**
  - `/ahmed/align-debug-vllm-70b-mixtral-rejected-opposite-auto-metrics-fresh`
- **Terminal state:**
  - root `JOB_STATE_SUCCEEDED`
- **Stage outcomes:**
  - prompts:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics/prompts-de3dec`
  - chosen:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics/chosen-59160b`
  - rejected:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics/rejected-6ff395`
  - judgments:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics/judgments-63ec12`
  - preference pairs:
    - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics/preference_pairs-e83fe6`
- **Observed pipeline shape:**
  - one-statement prompt generation succeeded and wrote `67` prompts
  - heterogeneous `auto` ran separate `chosen` and `rejected` response jobs
  - chosen ran in standard mode
  - rejected ran in opposite mode
  - judgments loaded `67 chosen, 67 rejected responses`
  - final filter step built `67` preference pairs
- **Verified standardized metrics artifacts in GCS:**
  - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics/prompts-de3dec/artifacts/vllm_metrics.json`
  - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics/chosen-59160b/artifacts/vllm_metrics.json`
  - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics/rejected-6ff395/artifacts/vllm_metrics.json`
  - `gs://marin-us-central1/align/debug_vllm_70b_mixtral_rejected_opposite_auto_smoke_metrics/judgments-63ec12/artifacts/vllm_metrics.json`
- **Schema spot-check:**
  - chosen metrics artifact reported:
    - `logical_stage = response_generation`
    - `session_count = 1`
    - `session_names = ["chosen"]`
    - `stage_keys = ["chosen"]`
- **Naming clarification:**
  - the metrics field should be interpreted as “prompts sent to vLLM,” not “final rows written”
  - follow-up code change will rename `prompt_count` to `request_prompt_count` in the emitted schema so prompt-generation artifacts are less easy to misread
- **Interpretation:**
  - the standardized `vllm_metrics.json` contract is now validated on a real end-to-end one-statement open-weight alignment run
  - prompt generation, chosen generation, rejected generation, and local judge all emitted structured metrics artifacts in the intended locations
- **Next logical follow-up:**
  - inspect the actual metric payload values across all four stage artifacts
  - then launch the larger full-spec runtime characterization run using those artifacts as the primary timing source

### 2026-03-25 — ALIGN-123: Renamed Metrics Field to `request_prompt_count` to Avoid Confusing Request Volume with Final Artifact Rows

- **Reason for change:**
  - the original field name `prompt_count` was easy to misread as “rows written”
  - that was especially confusing for prompt generation, where the metrics artifact measures requests sent to `vllm serve`, not final prompts emitted downstream
- **Code change:**
  - renamed the structured metrics schema field from:
    - `prompt_count`
  - to:
    - `request_prompt_count`
  - implementation lives in:
    - `lib/marin/src/marin/alignment/batched_vllm_serve.py`
- **Interpretation:**
  - `request_prompt_count` now unambiguously means the number of prompts sent over the OpenAI-compatible `vllm serve` interface during that stage/session
  - it should not be compared directly with final artifact row counts unless the stage is one-request-per-row by design

### 2026-03-25 — ALIGN-124: Next Major Experiment Is Full-Spec Runtime Characterization on the Refactored Open-Weight Pipeline

- **What is next:**
  - run the entire alignment pipeline over all statements in `openai_model_spec.jsonl`
  - do not limit to a single statement
- **Validated configuration to carry forward:**
  - chosen model:
    - Llama 3.3 70B Instruct
  - rejected model:
    - Mixtral 8x7B Instruct
  - response execution policy:
    - `response_execution_mode = auto`
  - rejected prompt policy:
    - `rejected_prompt_strategy = opposite`
  - local inference backend:
    - batched `vllm serve`
- **Primary objective:**
  - characterize real full-spec cost and timing using the new structured `vllm_metrics.json` artifacts rather than one-off log scraping
- **Key questions for the full-spec run:**
  - total prompt-generation cost and stage breakdown
  - chosen vs rejected startup and request cost
  - overlap behavior of heterogeneous `auto`
  - local judge cost at scale
  - final prompt / judgment / preference-pair counts
- **Planned base script:**
  - `experiments/align_vllm_70b_mixtral_rejected_full_spec.py`

### 2026-03-25 — ALIGN-125: Removed LiteLLM from Alignment and Standardized the API Path on OpenAI + vLLM Only

- **Requested scope:**
  - remove `litellm` from the alignment install surface
  - stop modeling API inference as a generic provider layer
  - support only:
    - OpenAI API
    - local `vllm`
- **Code changes:**
  - renamed the API inference config from:
    - `LiteLLMConfig`
  - to:
    - `OpenAIConfig`
  - rewrote the shared alignment client in:
    - `lib/marin/src/marin/alignment/llm_client.py`
  - old behavior:
    - `litellm.completion(...)`
  - new behavior:
    - `OpenAI().chat.completions.create(...)`
  - removed `litellm` from:
    - `lib/marin/pyproject.toml`
  - updated alignment serialization to use backend tag:
    - `openai`
  - removed Anthropic-specific forwarding from the alignment child-job env var helper
  - updated alignment defaults and OpenAI experiments to use plain model IDs:
    - `gpt-4.1`
    - `gpt-4.1-mini`
    - instead of provider-prefixed strings like `openai/gpt-4.1`
- **Behavioral result:**
  - alignment inference now has only two supported backends:
    - `OpenAIConfig`
    - `VLLMConfig`
  - bare strings in alignment configs are now interpreted as OpenAI model IDs
- **Verification:**
  - `make fix`
  - `uv run pytest tests/test_alignment.py -q`
    - `86 passed`
  - `uv pip uninstall litellm`
  - `uv pip show litellm`
    - package not found in the active workspace venv
- **Lockfile nuance:**
  - the workspace `uv.lock` still mentions `litellm`
  - that remaining reference is transitive from the pinned optional `harbor` dependency, not from `marin`
  - the `marin` package itself no longer declares `litellm` as a dependency
- **Interpretation:**
  - the alignment pipeline is now explicit about API support instead of routing through a generic provider abstraction
  - this narrows the supported surface area and matches the intended product decision:
    - OpenAI API or local `vllm`, nothing else

### 2026-03-25 — ALIGN-126: One-Statement API-Only Iris Smoke Succeeded with `gpt-4.1-mini`

- **Submitted root job:**
  - `/ahmed/iris-run-align_openai_spec_smoke-20260326-002318`
- **Launch shape:**
  - explicit Iris CPU root job via:
    - `uv run iris --controller-url http://127.0.0.1:10000 job run --cpu 4 --memory 8GB --extra cpu --region us-central1 --no-wait -e OPENAI_API_KEY $OPENAI_API_KEY -- python experiments/align_openai_spec_smoke.py`
- **Terminal state:**
  - root `JOB_STATE_SUCCEEDED`
  - child stage jobs all `JOB_STATE_SUCCEEDED`
- **Artifacts:**
  - prompts:
    - `gs://marin-us-central1/align/openai_spec_smoke/prompts-05480a`
  - judgments:
    - `gs://marin-us-central1/align/openai_spec_smoke/judgments-8f7524`
  - preference pairs:
    - `gs://marin-us-central1/align/openai_spec_smoke/preference_pairs-48f62b`
- **Observed counts:**
  - prompts written:
    - `66`
  - judgment records written:
    - `66`
  - final preference pairs built:
    - `19`
- **Important interpretation:**
  - this run proves the OpenAI API key and OpenAI-only alignment path work on Iris end-to-end
  - prompt generation, chosen, rejected, judgments, and final filtering all executed successfully on cluster jobs
  - however, `gpt-4.1-mini` as judge produced many parse failures, which materially reduced final pair count
- **Representative issue signal:**
  - repeated warnings:
    - `Failed to parse judge response: Expecting value ...`
- **Recommendation:**
  - use this run as a successful API/auth/integration smoke
  - do not treat `gpt-4.1-mini` judge quality as production-ready without either stronger parsing constraints or a stronger judge model

### 2026-03-25 — ALIGN-127: Relaunch One-Statement API-Only Iris Smoke with `gpt-4.1`

- **Motivation:**
  - `ALIGN-126` proved auth/integration but showed that `gpt-4.1-mini` was too fragile as the judge, with many parse failures and only `19/66` final pairs
  - next check is the same one-statement Iris smoke with the stronger OpenAI model
- **Code change:**
  - updated:
    - `experiments/align_openai_spec_smoke.py`
  - added a CLI flag:
    - `--model`
  - default remains:
    - `gpt-4.1-mini`
  - launch override for this run:
    - `--model gpt-4.1`
- **Submitted root job:**
  - `/ahmed/align-openai-spec-smoke-gpt41`
- **Launch command:**
  - `source .env && uv run iris --controller-url http://127.0.0.1:10000 job run --cpu 4 --memory 8GB --disk 5GB --extra cpu --region us-central1 --no-wait --job-name align-openai-spec-smoke-gpt41 -e OPENAI_API_KEY "$OPENAI_API_KEY" -- python experiments/align_openai_spec_smoke.py --model gpt-4.1`
- **Immediate goal:**
  - confirm the one-statement end-to-end OpenAI-only path still succeeds on Iris
  - compare judgment robustness and final pair yield against the `gpt-4.1-mini` smoke
- **Status:**
  - running

### 2026-03-25 — ALIGN-128: Fix `gpt-4.1` Smoke Script CLI Handoff and Relaunch

- **Failure on first `gpt-4.1` launch:**
  - root:
    - `/ahmed/align-openai-spec-smoke-gpt41`
  - terminal error:
    - `align_openai_spec_smoke.py: error: unrecognized arguments: --model gpt-4.1`
- **Root cause:**
  - the script added its own `--model` flag, but `executor_main()` also parses CLI flags
  - the first patch used `argparse.parse_args()`, so `--model` was still present in `sys.argv` when `executor_main()` ran
- **Fix:**
  - updated:
    - `experiments/align_openai_spec_smoke.py`
  - now uses:
    - `parse_known_args()`
  - and resets:
    - `sys.argv = [sys.argv[0], *executor_args]`
  - before calling `executor_main()`
- **Local verification:**
  - `uv run python -m py_compile experiments/align_openai_spec_smoke.py`
  - `uv run python experiments/align_openai_spec_smoke.py --model gpt-4.1 --dry_run true`
- **Relaunched root job:**
  - `/ahmed/align-openai-spec-smoke-gpt41-retry`
- **Status:**
  - running

### 2026-03-25 — ALIGN-129: Avoid Cache Hit and Launch Fresh `gpt-4.1` Iris Smoke

- **Issue on the first retry:**
  - root:
    - `/ahmed/align-openai-spec-smoke-gpt41-retry`
  - result:
    - `JOB_STATE_SUCCEEDED`
  - but it was not a real rerun
- **What happened:**
  - the executor reused previously succeeded outputs under:
    - `align/openai_spec_smoke/...`
  - so the root job finished quickly after skipping:
    - `spec`
    - `prompts`
    - `chosen`
    - `rejected`
    - `judgments`
    - `preference_pairs`
- **Fix:**
  - updated:
    - `experiments/align_openai_spec_smoke.py`
  - added a second CLI flag:
    - `--name`
  - this allows forcing a fresh alignment output prefix for reruns with different API models
- **Fresh relaunched root job:**
  - `/ahmed/align-openai-spec-smoke-gpt41-fresh`
- **Launch command:**
  - `source .env && uv run iris --controller-url http://127.0.0.1:10000 job run --cpu 4 --memory 8GB --disk 5GB --extra cpu --region us-central1 --no-wait --job-name align-openai-spec-smoke-gpt41-fresh -e OPENAI_API_KEY "$OPENAI_API_KEY" -- python experiments/align_openai_spec_smoke.py --model gpt-4.1 --name openai_spec_smoke_gpt41_fresh`
- **Current state:**
  - root `JOB_STATE_RUNNING`
  - not a cache hit
  - executor is launching fresh step ids under:
    - `align/openai_spec_smoke_gpt41_fresh/...`

### 2026-03-25 — ALIGN-130: One-Statement API-Only Iris Smoke Succeeded with `gpt-4.1`

- **Root job:**
  - `/ahmed/align-openai-spec-smoke-gpt41-fresh`
- **Terminal state:**
  - root `JOB_STATE_SUCCEEDED`
- **Artifacts:**
  - prompts:
    - `gs://marin-us-central1/align/openai_spec_smoke_gpt41_fresh/prompts-4774a8`
  - chosen:
    - `gs://marin-us-central1/align/openai_spec_smoke_gpt41_fresh/chosen-9558db`
  - rejected:
    - `gs://marin-us-central1/align/openai_spec_smoke_gpt41_fresh/rejected-5bcf9d`
  - judgments:
    - `gs://marin-us-central1/align/openai_spec_smoke_gpt41_fresh/judgments-4e9476`
  - preference pairs:
    - `gs://marin-us-central1/align/openai_spec_smoke_gpt41_fresh/preference_pairs-70992c`
- **Observed counts:**
  - prompts written:
    - `72`
  - judgment records written:
    - `72`
  - final preference pairs built:
    - `26`
- **Runtime notes:**
  - prompt generation succeeded end to end:
    - Stage 1
    - Stage 2
    - Stage 3
  - chosen and rejected ran as parallel OpenAI API Zephyr lanes with:
    - `gpt-4.1`
  - both response lanes completed successfully before judgment
  - executor wall time:
    - `689.76s`
- **Comparison vs `gpt-4.1-mini` smoke:**
  - prior mini run:
    - `66` prompts
    - `66` judgments
    - `19` final pairs
  - `gpt-4.1` run:
    - `72` prompts
    - `72` judgments
    - `26` final pairs
  - importantly, this `gpt-4.1` run did not show the repeated judge parse-failure warnings seen with `gpt-4.1-mini`
- **Interpretation:**
  - the OpenAI-only pipeline is now validated on Iris for both:
    - `gpt-4.1-mini`
    - `gpt-4.1`
  - `gpt-4.1` is materially more robust as the judge in this one-statement smoke

### 2026-03-25 — ALIGN-131: Full-Spec Open-Weight Entrypoint Updated to the Validated Opposite-Mode Configuration

- **Existing script kept, not duplicated:**
  - `experiments/align_vllm_70b_mixtral_rejected_full_spec.py`
- **Why this update was needed:**
  - the full-spec script already existed, but it was stale relative to the validated smoke path
  - specifically, it still used:
    - old rejected behavior
    - older disk settings
    - older prompt-generation token budgets
- **Updated to match the validated one-statement heterogeneous pipeline:**
  - chosen / ideation / extract / judge:
    - Llama 3.3 70B Instruct
  - rejected:
    - Mixtral 8x7B Instruct
  - rejected strategy:
    - `RejectedPromptStrategy.OPPOSITE`
  - response execution policy:
    - `ResponseExecutionMode.AUTO`
  - vLLM resource envelope:
    - `disk="10g"`
    - `ram="256g"`
    - `tpu_type="v5p-8"`
  - prompt-generation settings:
    - `prompt_batch_size=4`
    - `understanding_max_tokens=1024`
    - `concretize_max_tokens=1536`
    - `concretize_max_attempts=5`
    - `extract_max_tokens=1024`
  - judge settings:
    - `judge_batch_size=4`
- **Result:**
  - we now have a full-spec experiment entrypoint for exactly this target configuration:
    - entire OpenAI model spec
    - chosen = Llama 3.3 70B
    - rejected = Mixtral 8x7B Instruct
    - opposite mode as the default rejected prompting policy
- **Suggested submit command:**
  - `source .env && uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name align-vllm-70b-mixtral-rejected-opposite-full-spec --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/align_vllm_70b_mixtral_rejected_full_spec.py`

### 2026-03-25 — ALIGN-132: Launch Full-Spec Open-Weight Runtime Characterization with Opposite-Mode Rejected Responses

- **Goal:**
  - run the entire refactored alignment pipeline over the full OpenAI model spec:
    - `spec`
    - `prompts`
    - `chosen`
    - `rejected`
    - `judgments`
    - `preference_pairs`
- **Configuration:**
  - chosen / ideation / extract / judge:
    - Llama 3.3 70B Instruct
  - rejected:
    - Mixtral 8x7B Instruct
  - rejected prompt strategy:
    - `RejectedPromptStrategy.OPPOSITE`
  - response execution:
    - `ResponseExecutionMode.AUTO`
- **Pre-launch safeguard:**
  - renamed the full-spec experiment namespace from:
    - `debug_vllm_70b_mixtral_rejected_full_spec`
  - to:
    - `debug_vllm_70b_mixtral_rejected_opposite_full_spec`
  - so this launch cannot silently reuse stale executor outputs from the earlier pre-opposite full-spec script
- **Launch command:**
  - `source .env && uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name align-vllm-70b-mixtral-rejected-opposite-full-spec --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/align_vllm_70b_mixtral_rejected_full_spec.py`
- **Status:**
  - launching

### 2026-03-25 — ALIGN-133: Full-Spec Opposite-Mode Run Live on Iris Under Root Job `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec`

- **Live Iris root job:**
  - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec`
- **Current observed child states:**
  - `spec`:
    - succeeded
  - `prompts`:
    - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec/align-debug_vllm_70b_mixtral_rejected_opposite_full_spec-prompts_a9dd9e03-c931aeb3`
    - running on `v5p-8` with `ram=256g`, `disk=10g`
- **Latest healthy signals from logs:**
  - `Loaded 46 statements from spec`
  - `Starting vLLM environment`
- **Latest interpretation:**
  - the full-spec run has cleared scheduler startup and is currently inside the heavy prompt-generation stage
  - no TPU/HBM/OOM/traceback signatures have appeared so far
- **Next babysit checkpoint:**
  - wait for `prompts` to succeed, then verify that hetero `auto` fans out into sibling `chosen` and `rejected` jobs

### 2026-03-25 — ALIGN-134: First Full-Spec Run Failed in Stage 1 Prompt Understanding on `be_kind`

- **Failed root job:**
  - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec`
- **Failed child:**
  - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec/align-debug_vllm_70b_mixtral_rejected_opposite_full_spec-prompts_a9dd9e03-c931aeb3`
- **Failure signature:**
  - `RuntimeError: Stage 1 failed for 1 statement(s): be_kind: Stage1 response missing <variation_axes> block`
- **Observed sequence:**
  - `spec` succeeded
  - `prompts` started normally
  - Llama `vllm serve` reached `vLLM environment ready`
  - Stage 1 understanding began over all `46` statements
  - one statement (`be_kind`) returned a malformed Stage 1 response with no `<variation_axes>` block
  - current code treated that as fatal and aborted the full-spec run
- **Interpretation:**
  - this was not a TPU or infra failure
  - the failure mode is analogous to the Stage 2 partial-return issue we already hardened: one malformed model response should trigger targeted retry, not full-run death

### 2026-03-25 — ALIGN-135: Added Stage 1 Retry-on-Parse-Failure and Prepared Full-Spec Resubmission

- **Code changes:**
  - `generate_prompts.py`
    - added `understanding_max_attempts`
    - Stage 1 understanding now retries only the failed statements instead of failing immediately on the first parse miss
  - `align.py`
    - plumbed `understanding_max_attempts` through `AlignConfig` into `PromptGenConfig`
  - `align_vllm_70b_mixtral_rejected_full_spec.py`
    - explicitly set `understanding_max_attempts=5`
- **Regression coverage:**
  - added a local-vLLM prompt-generation test where the first Stage 1 response omits `<variation_axes>` and the retry succeeds
- **Verification:**
  - `uv run pytest tests/test_alignment.py -q`
    - `87 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py lib/marin/src/marin/alignment/align.py experiments/align_vllm_70b_mixtral_rejected_full_spec.py tests/test_alignment.py`
    - passed
- **Next action:**
  - relaunch `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec` with the patched Stage 1 retry logic and continue babysitting

### 2026-03-25 — ALIGN-136: Same-Name Root Relaunch Failed Opaquely; Switching to a Fresh Iris Root Job Name

- **What happened:**
  - resubmitting the patched code under the same Iris root job name:
    - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec`
  - produced an immediate root-level failure with no fresh child logs and no useful traceback beyond:
    - `RuntimeError: 1 step(s) failed`
- **Interpretation:**
  - this does not look like the Stage 1 bug recurring
  - it looks more like an Iris root-job reuse / stale-state problem from recycling the same fixed job name immediately after a failed run
- **Recovery plan:**
  - keep the patched experiment code and executor namespace
  - switch only the Iris root job name to a fresh retry-specific name
  - continue babysitting the fresh root job until the patched prompt step either succeeds or exposes a new concrete error

### 2026-03-25 — ALIGN-137: Fresh Full-Spec Retry Root Job Running as `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec-retry1-stage1`

- **Fresh Iris root job:**
  - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec-retry1-stage1`
- **Why this root name changed:**
  - avoid the opaque immediate root failure seen when reusing the previous fixed root job name
- **Current state:**
  - root is `JOB_STATE_RUNNING`
  - normal root bootstrap has started:
    - `syncing deps`
    - `installing pip deps`
    - `activating venv`
    - `running user command`
- **Monitoring owner / state file:**
  - `scratch/20260325-1801_monitoring_state.json`
  - updated to point at the fresh retry root job with `restart_count = 2`

### 2026-03-25 — ALIGN-138: Stage 1 Retry Worked Live, Then Full-Spec Stage 2 Hit the 2560-Token Context Ceiling

- **Live evidence that the Stage 1 fix worked:**
  - `be_kind` failed on `Stage1 attempt 1` with missing `<variation_axes>`
  - the run did not abort
  - the failed statement was retried as a singleton
  - the pipeline advanced into:
    - `Stage 2: Concretizing 46 statements`
- **New failure mode in Stage 2:**
  - multiple concretization requests exceeded the local TPU request limit:
    - example:
      - `This model's maximum context length is 2560 tokens. However, your request has 2563 input tokens.`
  - concrete statement examples from logs:
    - `assume_best_intentions`
    - `avoid_overstepping`
    - `avoid_regulated_advice`
    - `avoid_sycophancy`
    - `be_clear`
    - `be_creative`
    - `be_empathetic`
    - `be_engaging`
    - `be_rationally_optimistic`
- **Interpretation:**
  - the old full-spec default of `concretize_batch_size=10` is too large for some full-spec statements under the validated local envelope
  - this is not an infra issue; it is a prompt-size / batch-size mismatch
- **Operational action taken:**
  - stopped the running retry root to avoid burning more TPU time once the failure mode was clear

### 2026-03-25 — ALIGN-139: Added `concretize_batch_size` to `AlignConfig` and Lowered the Full-Spec Run to `4`

- **Code changes:**
  - `align.py`
    - exposed `concretize_batch_size` on `AlignConfig`
    - passed it through into `PromptGenConfig`
  - `align_vllm_70b_mixtral_rejected_full_spec.py`
    - set `concretize_batch_size=4`
- **Why `4`:**
  - this matches the already validated prompt-generation smoke envelope
  - it is small enough to pull Stage 2 concretization prompts back under the observed `2560`-token limit
- **Verification:**
  - `uv run pytest tests/test_alignment.py -q`
    - `87 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/align.py experiments/align_vllm_70b_mixtral_rejected_full_spec.py tests/test_alignment.py`
    - passed
- **Next action:**
  - relaunch the full-spec job under a fresh retry root name with:
    - Stage 1 retry enabled
    - `concretize_batch_size=4`

### 2026-03-25 — ALIGN-140: Retry2 (`batch4`) Is Still Running Cleanly in Prompt Generation

- **Current live root job:**
  - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec-retry2-batch4`
- **Current live prompt child:**
  - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec-retry2-batch4/align-debug_vllm_70b_mixtral_rejected_opposite_full_spec-prompts_a9dd9e03-eb3c86e7`
- **Current observed state:**
  - root: `JOB_STATE_RUNNING`
  - prompt child: `JOB_STATE_RUNNING`
- **Latest positive signals:**
  - Stage 1 again hit the known `be_kind` parse miss, but only as:
    - `Stage1 attempt 1 failed for 'be_kind': Stage1 response missing <variation_axes> block`
  - the run stayed alive and advanced into:
    - `Stage 2: Concretizing 46 statements`
- **Most important current read:**
  - after lowering `concretize_batch_size` from `10` to `4`, the earlier Stage 2 `HTTP 400` / `maximum context length is 2560 tokens` failures have not reappeared in the filtered logs so far
- **Monitoring status:**
  - still actively babysitting this retry root

### 2026-03-25 — ALIGN-141: Planned Semantic Refactor for Stage 2 Concretization: Exactly One `cfg_*` per Request

- **Problem statement:**
  - the current Stage 2 design still uses *semantic batching*:
    - one prompt can ask the model to generate scenarios for multiple covering-array configs at once
  - that creates two distinct problems:
    - prompt-length blowups on local vLLM
    - weaker guarantees that each returned scenario faithfully matches its intended config
- **Design decision:**
  - change Stage 2 so that each covering-array config is its own semantic unit of work:
    - one `cfg_*`
    - one concretization prompt
    - one returned scenario
    - one returned rubric
- **Important distinction to preserve:**
  - remove *semantic batching*
  - keep *transport/runtime batching*
  - meaning:
    - for local vLLM:
      - still send many one-config prompts together via `local_serve_batch_size`
      - let vLLM do continuous batching on the backend
    - for API models:
      - still use normal request parallelism via the threadpool
      - but each request prompt should contain exactly one config
- **Exact implementation plan:**
  - `prompts/concretize.py`
    - simplify `make_concretize_prompt(...)` to take exactly one config instead of a list of indexed configs
    - emit unambiguous single-config output tags:
      - `<scenario_cfg_123>...</scenario_cfg_123>`
      - `<rubric_cfg_123>...</rubric_cfg_123>`
  - `generate_prompts.py`
    - remove Stage 2 multi-config request construction
    - replace `list[list[_ConcretizeConfig]]` request batches with `list[_ConcretizeConfig]`
    - keep `_ConcretizeConfig` as the stable per-config work item and ID source
    - make retries operate on the same single-config items, not re-sliced sub-batches
    - keep `concretization_attempts` diagnostics, but record one config per attempt
  - API path
    - keep threadpool parallelism
    - each future should call concretization for exactly one config
  - local vLLM path
    - keep `session.generate_from_messages(...)`
    - each message batch element should correspond to exactly one config
    - `local_serve_batch_size` remains the runtime batch size knob
- **Config cleanup after the refactor:**
  - remove `concretize_batch_size` from the semantic design
  - if we still want an explicit throughput knob later, name it for what it actually is:
    - e.g. `concretize_request_parallelism` or rely on existing `local_serve_batch_size` / threadpool workers
- **Validation plan after the refactor:**
  - unit tests:
    - one config per prompt in both API and local paths
    - retries still recover missing/malformed single-config outputs
    - `ideation.json` diagnostics remain interpretable
  - one-statement local smoke:
    - confirm Stage 2 no longer needs context tuning by config count
  - full-spec retry:
    - confirm the earlier `maximum context length is 2560 tokens` failures disappear without depending on semantic batch-size tuning
- **Interpretation:**
  - the current `concretize_batch_size=4` retry is only a mitigation
  - the intended final design is:
    - one config per concretization request
    - many requests batched only at the backend/runtime layer

### 2026-03-25 — ALIGN-142: Revised Semantic Refactor Plan: Apply the Same “One Item per Request” Rule to Both Stage 2 and Stage 3

- **Revision to the earlier plan:**
  - the same semantic-batching problem exists in both:
    - Stage 2 concretization
    - Stage 3 extraction
  - so the refactor should cover both stages together, not just concretization
- **Core rule after the refactor:**
  - one semantic work item per request prompt
  - meaning:
    - Stage 2:
      - one covering-array config per concretization prompt
    - Stage 3:
      - one concretized scenario per extraction prompt
- **Why this matters for Stage 3 too:**
  - the current extraction path still asks the model to process multiple scenarios in one prompt
  - that has the same downsides:
    - prompt growth with batch size
    - higher risk of cross-item drift
    - weaker guarantee that each extracted `system_prompt` / `user_message` pair matches the intended scenario cleanly
- **Parser simplification that falls out of this design:**
  - once there is exactly one semantic item per response, we do not need indexed multi-item response parsing
  - Stage 2 can simplify from:
    - many `<scenario_cfg_XXX>` / `<rubric_cfg_XXX>` blocks
  - to:
    - one `<scenario>...</scenario>`
    - one `<rubric>...</rubric>`
  - Stage 3 can simplify from:
    - many `<scenario_i>` blocks
  - to either:
    - one `<system_prompt>...</system_prompt>` and one `<user_message>...</user_message>`
    - or one single `<scenario>` wrapper containing those two tags
  - this is a real reduction in failure surface:
    - no index matching
    - no per-batch missing-block bookkeeping
    - no need to pad or realign results by position inside a multi-item response
- **Performance note for local vLLM:**
  - one-item-per-request is not as wasteful as it might sound because the shared prefix for a statement is largely identical across its configs:
    - behavior understanding
    - scientific motivation
    - axes metadata
    - transcript analyses
  - vLLM prefix caching should let the backend reuse that shared prefix work across many one-config requests for the same statement
  - so the refactor trades prompt ambiguity for cleaner semantics without necessarily paying the full repeated-prefix compute cost
- **Concrete implementation plan:**
  - `prompts/concretize.py`
    - change `make_concretize_prompt(...)` to accept exactly one config
    - change the output contract to one unindexed scenario/rubric pair
  - `generate_prompts.py` Stage 2
    - remove multi-config Stage 2 request construction
    - treat `_ConcretizeConfig` as the one request unit
    - simplify parsing to one scenario/rubric pair per response
    - keep retries on individual configs only
  - `prompts/extract.py`
    - revise the Stage 3 prompt to target exactly one scenario per request
  - `generate_prompts.py` Stage 3
    - remove multi-scenario extraction request construction
    - simplify parsing to one extracted prompt pair per response
    - keep retries/failures at the single-scenario level
  - backend execution model
    - for local vLLM:
      - keep `local_serve_batch_size` as the runtime batching knob across many one-item prompts
    - for API:
      - keep threadpool parallelism
      - but each future/request still represents exactly one semantic item
- **Config cleanup after the refactor:**
  - remove semantic batch-size knobs:
    - `concretize_batch_size`
    - `extract_batch_size`
  - preserve runtime batching knobs only:
    - `local_serve_batch_size`
    - worker counts / concurrency
- **Token-budget follow-up that should ship with the refactor:**
  - once Stage 2 and Stage 3 each generate exactly one item per request, the old multi-item token budgets are too generous
  - after the semantic refactor:
    - revisit `concretize_max_tokens`
    - revisit `extract_max_tokens`
  - expected direction:
    - lower both budgets to reflect one-item outputs rather than batched outputs
  - rationale:
    - reduce latency
    - reduce filler / rambling outputs
    - keep request envelopes tighter under local context limits
  - implementation preference:
    - keep the knobs explicit
    - but update the validated defaults and experiment settings to per-item values rather than leaving the old batched-response budgets in place
- **Validation plan after the refactor:**
  - unit tests:
    - Stage 2 one-config response parsing
    - Stage 3 one-scenario response parsing
    - single-item retry behavior for both stages
  - one-statement local smoke:
    - verify Stage 2 and Stage 3 complete under the current local envelope without semantic batch tuning
  - full-spec retry:
    - verify prompt generation no longer depends on semantic batch-size tuning to stay under context limits
- **Explicitly deferred for now:**
  - OpenAI API rate-limit tuning
  - quality-comparison study between old batched prompts and new one-item prompts

### ALIGN-143 - 2026-03-25 18:31 - Per-item Stage 2/3 token budgets set to 1024

- Narrowed the planned one-item prompt-generation budgets to:
  - `concretize_max_tokens=1024`
  - `extract_max_tokens=1024`
- Applied that choice to the canonical defaults in:
  - `lib/marin/src/marin/alignment/generate_prompts.py`
  - `lib/marin/src/marin/alignment/align.py`
- Also aligned the explicit local-vLLM experiment settings that still pinned Stage 2 at `1536`.
- Rationale:
  - these are now the intended per-item ceilings for the upcoming semantic refactor
  - they stay well below the local prompt-generation context ceiling that previously triggered `HTTP 400` failures
  - they are still generous for one scenario/rubric or one extracted prompt pair
- Important scope note:
  - this does not change the currently running full-spec job in flight
  - it only affects future launches / reruns

### ALIGN-144 - 2026-03-25 18:43 - Added Stage-1-only understanding experiment entrypoint

- Added a dedicated Stage-1-only helper:
  - `generate_understandings_from_spec(...)` in `lib/marin/src/marin/alignment/generate_prompts.py`
- Added a dedicated Iris-friendly experiment script:
  - `experiments/generate_understanding_llama_3_3_70b.py`
- Purpose:
  - isolate prompt-generation Stage 1 concurrency / batching experiments without paying for Stage 2 concretization or Stage 3 extraction
  - keep the artifact shape familiar:
    - sharded Stage-1 output records
    - `artifacts/<statement_id>/understanding.json`
    - `artifacts/summary.json`
    - `artifacts/vllm_metrics.json` for local vLLM runs
- Default behavior:
  - staged regional Llama 3.3 70B checkpoint
  - full OpenAI model spec
  - configurable `--local-serve-batch-size`
  - optional repeated `--statement-id`
- Canonical launch:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name generate-understanding-llama-3-3-70b --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/generate_understanding_llama_3_3_70b.py`
- Verification:
  - `uv run pytest tests/test_alignment.py -q` → `88 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py experiments/generate_understanding_llama_3_3_70b.py tests/test_alignment.py`

### ALIGN-145 - 2026-03-25 21:46 - Implemented one-item Stage 2/3 prompt semantics and collapsed experiments back to one full prompt-generation entrypoint

- Implemented the semantic refactor that was only planned in `ALIGN-141` / `ALIGN-142`:
  - Stage 1 remains one statement per prompt
  - Stage 2 is now one covering-array config per prompt
  - Stage 3 is now one scenario per prompt
- Concretization changes:
  - `prompts/concretize.py` now builds one-config prompts instead of multi-config prompts
  - Stage 2 output contract simplified from indexed tags like:
    - `<scenario_cfg_000>...</scenario_cfg_000>`
    - `<rubric_cfg_000>...</rubric_cfg_000>`
  - to:
    - `<scenario>...</scenario>`
    - `<rubric>...</rubric>`
  - retries are now naturally per-config because each request already represents exactly one config
- Extraction changes:
  - `prompts/extract.py` now builds one-scenario prompts instead of multi-scenario prompts
  - Stage 3 output contract simplified from indexed `<scenario_N>` wrappers to direct:
    - `<system_prompt>...</system_prompt>`
    - `<user_message>...</user_message>`
  - local vLLM still batches many independent single-scenario prompts per transport request via `local_serve_batch_size`
- Config cleanup:
  - removed semantic batch-size knobs from the code path:
    - `concretize_batch_size`
    - `extract_batch_size`
  - `prompt_batch_size` / `local_serve_batch_size` is now the only prompt-generation batching knob for local vLLM
- Experiment cleanup:
  - removed the temporary Stage-1-only experiment entrypoint
  - promoted `experiments/generate_prompts_llama_3_3_70b_refactored.py` into the canonical full Stage 1/2/3 prompt-generation experiment
  - that script is now configurable via:
    - `--name`
    - `--local-serve-batch-size`
    - repeated `--statement-id`
- Metrics note:
  - no new metrics format was needed because the shared-session `vllm_metrics.json` already records per-stage metrics under:
    - `understanding`
    - `concretize`
    - `extract`
  - when ideation and extraction use the same model, all three stages still reuse one `vllm serve` session and one staged model load
- Verification:
  - `uv run pytest tests/test_alignment.py -q` → `87 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py lib/marin/src/marin/alignment/prompts/concretize.py lib/marin/src/marin/alignment/prompts/extract.py lib/marin/src/marin/alignment/align.py experiments/generate_prompts_llama_3_3_70b_refactored.py experiments/align_vllm_70b_mixtral_rejected_full_spec.py tests/test_alignment.py`
- Important scope note:
  - this refactor changes future prompt-generation launches
  - it does not mutate the currently running full-spec root that was launched before the semantic change

### ALIGN-146 - 2026-03-25 21:53 - Launch plan for local-vLLM prompt-generation batch-width sweep

- Objective:
  - probe higher local `vllm serve` client batch widths after the Stage 2/3 one-item semantic refactor
  - keep the same full Stage 1/2/3 prompt-generation experiment and same-model shared-session lifecycle
- Sweep:
  - `local_serve_batch_size=16`
  - `local_serve_batch_size=32`
- Shared constraints:
  - use `experiments/generate_prompts_llama_3_3_70b_refactored.py`
  - request `v5p-8` through the existing `VLLMConfig`
  - keep one-statement scope via the script default (`ask_clarifying_questions`)
  - keep the shared Llama 3.3 70B ideation/extraction session so the resulting `vllm_metrics.json` stays directly comparable

### ALIGN-147 - 2026-03-25 21:45 - Launched local-vLLM prompt-generation batch-width sweep roots

- Launched two Iris root jobs against the refactored full Stage 1/2/3 prompt-generation experiment:
  - `/ahmed/generate-prompts-llama-3-3-70b-bs16-20260325`
    - command payload: `python experiments/generate_prompts_llama_3_3_70b_refactored.py --name debug_generate_prompts_llama_3_3_70b_refactored_bs16_20260325 --local-serve-batch-size 16`
  - `/ahmed/generate-prompts-llama-3-3-70b-bs32-20260325`
    - command payload: `python experiments/generate_prompts_llama_3_3_70b_refactored.py --name debug_generate_prompts_llama_3_3_70b_refactored_bs32_20260325 --local-serve-batch-size 32`
- Resource note:
  - both roots are CPU orchestrators only (`4cpu`, `16 GiB`, `10 GiB disk`)
  - both prompt children request `v5p-8`, `32cpu`, `256 GiB`, `10 GiB disk` through the experiment `VLLMConfig`
- Current state at launch check:
  - both `spec` children already succeeded
  - both `prompts` children exist and are pending on TPU capacity
  - scheduler reason on both prompt children: `Scheduler: Insufficient TPUs (need 4, available 0)`

### ALIGN-148 - 2026-03-25 22:03 - Batch-width sweep result: `local_serve_batch_size=32` beats `16`

- Completed roots:
  - `/ahmed/generate-prompts-llama-3-3-70b-bs16-20260325`
  - `/ahmed/generate-prompts-llama-3-3-70b-bs32-20260325`
- Final prompt artifacts:
  - `bs16` → `gs://marin-us-central1/align/debug_generate_prompts_llama_3_3_70b_refactored_bs16_20260325/prompts-5b1672`
  - `bs32` → `gs://marin-us-central1/align/debug_generate_prompts_llama_3_3_70b_refactored_bs32_20260325/prompts-909a0a`
- Both runs succeeded cleanly:
  - `67` final prompts written
  - no `HTTP 400`
  - no retry warnings
  - shared Llama `vllm serve` session reused across Stage 1/2/3
- `vllm_metrics.json` comparison:
  - total request time:
    - `bs16` → `94.65s`
    - `bs32` → `75.87s`
  - Stage 2 concretize request time:
    - `bs16` → `60.92s` across `5` batched requests
    - `bs32` → `45.58s` across `3` batched requests
  - Stage 3 extract request time:
    - `bs16` → `14.81s` across `5` batched requests
    - `bs32` → `11.06s` across `3` batched requests
  - total throughput:
    - input tokens/sec: `1205.26` (`bs16`) vs `1501.27` (`bs32`)
    - output tokens/sec: `375.55` (`bs16`) vs `464.36` (`bs32`)
- Startup remained the dominant fixed cost for both:
  - `server_start_seconds` ≈ `430s` in both runs
  - end-to-end task runtime was only ~`19.5s` shorter for `bs32`, which matches the reduction in request time
- Resource note:
  - peak memory was effectively unchanged:
    - `bs16` → `183500 MB`
    - `bs32` → `183808 MB`
- Interpretation:
  - after the Stage 2/3 one-item semantic refactor, raising local client batch width from `16` to `32` improves request throughput without introducing new failures or a meaningful memory penalty on this one-statement Llama lane
  - next useful sweep point is `64`, or promoting `32` into the next full-spec prompt-generation retry if we want a higher-throughput default

### ALIGN-149 - 2026-03-25 22:08 - Plan: global item-level local-vLLM stage sweeps with one shared session

- Motivation:
  - current local `vllm` prompt generation still keeps Stage 2 and Stage 3 in per-statement loops
  - that leaves cross-statement parallelism unused even after the one-item semantic refactor
- Refactor target:
  - keep the same stage barriers (`understanding` → `concretize` → `extract`)
  - but within each stage, flatten work into one global item list and batch across the whole spec with one shared `BatchedVllmServeSession`
- Intended local-vLLM architecture:
  - Stage 1:
    - batch over all statements in the filtered spec
  - Stage 2:
    - build one global list of `(statement_id, understanding, axes, cfg)` concretize items across all statements
    - batch those globally by `local_serve_batch_size`
    - parse and regroup by statement after completion
  - Stage 3:
    - build one global list of `(statement_id, variation)` extraction items across all statements
    - batch those globally by `local_serve_batch_size`
    - regroup final prompts by statement only for artifact assembly
- Shared-session constraint:
  - when `ideation_model == extract_model` and both are local, keep exactly one `vllm serve` session alive for all three stages
  - preserve the existing per-stage `vllm_metrics.json` accounting under:
    - `understanding`
    - `concretize`
    - `extract`
- Scope note:
  - this is not a fully streaming cross-stage scheduler yet
  - the immediate change is stage-global batching, not overlapping Stage 2 and Stage 3 while Stage 1 is still running
- Validation plan:
  - update the prompt-generation experiment so the default is full-spec rather than one-statement
  - run new full-spec prompt-generation sweeps at:
    - `local_serve_batch_size=32`
    - `local_serve_batch_size=64`
  - compare request time, throughput, failures, and peak memory via `artifacts/vllm_metrics.json`

### ALIGN-150 - 2026-03-25 22:16 - Landed global local-vLLM stage sweeps for prompt generation

- Refactor landed in `lib/marin/src/marin/alignment/generate_prompts.py`:
  - Stage 1 local path already batched across all filtered statements and remains so
  - Stage 2 local path now flattens one global list of `(statement_id, cfg)` concretize work items across the spec and batches them globally through one shared `BatchedVllmServeSession`
  - Stage 3 local path now flattens one global list of `(statement_id, variation)` extraction work items across the spec and batches them globally through that same shared session
  - same-model reuse is preserved: when ideation and extraction use the same local model, one `vllm serve` session stays live for all three stages
- Experiment entrypoint update:
  - `experiments/generate_prompts_llama_3_3_70b_refactored.py` now defaults to full-spec
  - `--statement-id` remains available as an optional filter
- Test coverage:
  - added a regression test proving local Stage 2 and Stage 3 now batch across statements globally rather than iterating one statement at a time
- Verification:
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py experiments/generate_prompts_llama_3_3_70b_refactored.py tests/test_alignment.py`
  - `uv run pytest tests/test_alignment.py -q` → `88 passed`

### ALIGN-151 - 2026-03-25 22:13 - Launched full-spec prompt-generation sweep after global-stage refactor

- Launched two new Iris root jobs against the refactored full-spec prompt-generation experiment:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs32-20260325`
    - command payload: `python experiments/generate_prompts_llama_3_3_70b_refactored.py --name debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs32_20260325 --local-serve-batch-size 32`
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-20260325`
    - command payload: `python experiments/generate_prompts_llama_3_3_70b_refactored.py --name debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_20260325 --local-serve-batch-size 64`
- Scope:
  - full `openai_model_spec.jsonl`
  - no `statement_id` filter
  - one shared staged Llama 3.3 70B local `vllm serve` session across Stage 1/2/3
- Resource note:
  - roots are CPU orchestrators only (`4cpu`, `16 GiB`, `10 GiB disk`)
  - prompt children will request `v5p-8`, `32cpu`, `256 GiB`, `10 GiB disk` through the experiment `VLLMConfig`
- Current state at launch check:
  - both roots are `running`
  - child-step fanout had not appeared yet at the first post-submit poll

### ALIGN-152 - 2026-03-25 22:39 - Full-spec global-stage sweep progress check

- Current live roots:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs32-20260325`
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-20260325`
- Current live prompt children:
  - `bs32` → `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs32-20260325/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs32_20260325-prompts_b7a734f8-f2144e25`
  - `bs64` → `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-20260325/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_20260325-prompts_b68d4b8a-6c635857`
- Stage status:
  - both runs loaded all `46` statements
  - both runs completed Stage 1 and are currently in Stage 2 concretization
  - neither run had reached Stage 3 at this check
- Notable difference so far:
  - `bs64` hit one Stage 1 retry on `be_kind` missing `<variation_axes>`, then advanced normally into Stage 2
  - `bs32` did not show that warning in the current tail
  - neither run has shown `HTTP 400` or other Stage 2 request failures in the current logs
- Batched request interpretation:
  - lines like `Sending batched vLLM serve request ... for 32 prompts (n=1)` mean:
    - batch width = `32` prompts in that HTTP request
    - `n=1` = request one completion per prompt, not multiple samples

### ALIGN-153 - 2026-03-25 22:44 - Added lightweight `k/N` progress logs for local prompt generation

- Motivation:
  - current babysitting can only infer coarse progress from repeated batched-request logs
  - there was no explicit intra-stage progress signal for Stage 1/2/3
- Change landed in `lib/marin/src/marin/alignment/generate_prompts.py`:
  - Stage 1 local path now logs progress like:
    - `Stage 1 progress: 24/46 (52.2%) [attempt 1]`
  - Stage 2 local path now logs:
    - total concretize queue size up front
    - cumulative successful concretize items during execution
    - pending retry count in the suffix
  - Stage 3 local path now logs:
    - total extraction queue size up front
    - cumulative extracted items during execution
- Overhead guard:
  - logging is emitted only after completed batch groups, not inside the `vllm` request path
  - interval is coarse (`max(local_serve_batch_size, total_items // 20)`), so this does not add per-item logging or alter batching behavior
- Verification:
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py tests/test_alignment.py`
  - `uv run pytest tests/test_alignment.py -q` → `88 passed`
- Scope note:
  - the two currently running full-spec sweep jobs were launched before this change, so they will not emit the new progress lines
  - future launches will

### ALIGN-154 - 2026-03-25 22:51 - `bs64` full-spec run failed in Stage 3 extraction due to malformed outputs

- Failed root:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-20260325`
- Failed prompt child:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-20260325/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_20260325-prompts_b68d4b8a-6c635857`
- Failure stage:
  - Stage 1 succeeded with one recovered retry on `be_kind`
  - Stage 2 succeeded
  - Stage 3 extraction failed
- Exact failure:
  - `RuntimeError: Stage 3 failed: statement 'be_creative' variation 10: Stage3: missing <user_message> block in extraction response; statement 'be_empathetic' variation 46: Stage3: missing <user_message> block in extraction response; statement 'be_thorough_but_efficient' variation 34: Stage3: missing <user_message> block in extraction response; statement 'letter_and_spirit' variation 30: Stage3: missing <user_message> block in extraction response; statement 'support_mental_health' variation 5: Stage3: missing <system_prompt> block in extraction response; statement 'uphold_fairness' variation 34: Stage3: missing <system_prompt> block in extraction response; statement 'uphold_fairness' variation 68: Stage3: missing <system_prompt> block in extraction response`
- Interpretation:
  - this is not a TPU or scheduler failure
  - this is not an `HTTP 400` or context-limit failure
  - the wider `64`-prompt Stage 3 transport batches completed at the server level, but several returned completions did not respect the required extraction tag schema
  - Stage 3 currently has no malformed-output retry path, so a small number of bad completions kills the whole run
- Debug record:
  - `docs/debug-log-prompt-generation-fullspec-bs64-stage3.md`

### ALIGN-155 - 2026-03-25 22:58 - Recommended architecture for stage caching with local vLLM prompt generation

- Core constraint:
  - executor step caching only skips a step after `STATUS_SUCCESS`
  - failed steps rerun on the same output path when re-submitted
  - therefore, the right way to reuse completed prompt-generation stages is to checkpoint stage artifacts inside the `prompts` step and make the function resume from them
- Recommended design:
  - keep one executor `prompts` step
  - keep one shared local `vllm serve` session per attempt when ideation and extraction use the same model
  - write durable stage-boundary artifacts immediately after each stage completes:
    - Stage 1 → persisted understandings
    - Stage 2 → persisted ideations
    - Stage 3 → final prompt shards
  - on rerun of the same failed step:
    - if Stage 1 checkpoint is complete, load it and skip Stage 1
    - if Stage 2 checkpoint is complete, load it and skip Stage 2
    - restart a fresh local `vllm serve` session and continue from the first incomplete stage
- Why this is the best near-term design:
  - reuses Stage 1/2 after a Stage 3 failure
  - does not require passing a live Python session object across executor steps (not possible)
  - does not require a long-lived service job or service discovery layer
  - preserves same-model session reuse within a single attempt
- Explicit non-recommendation:
  - separate executor steps for Stage 1/2/3 would give true executor-native caching, but they would each boot a separate `vllm serve` session
  - passing a “session config” argument to another executor step only recreates a new session; it does not reuse the live one
- Future option if startup dominates too much:
  - introduce a dedicated long-lived prompt-generation service job and have stage workers call it over RPC/HTTP
  - but that is a larger lifecycle/service-discovery problem and is not the first fix

### ALIGN-156 - 2026-03-25 23:14 - Landed stage-boundary prompt-generation checkpoints plus Stage 3 item-level resume

- Change landed in `lib/marin/src/marin/alignment/generate_prompts.py`:
  - prompt generation still runs as one executor `prompts` step
  - Stage 1 now writes a durable checkpoint immediately after success:
    - `artifacts/checkpoints/understandings.jsonl.gz`
    - plus the existing per-statement `artifacts/<sid>/understanding.json`
  - Stage 2 now writes a durable checkpoint immediately after success:
    - `artifacts/checkpoints/ideations.jsonl.gz`
    - plus the existing per-statement `artifacts/<sid>/ideation.json`
  - Stage status is tracked in:
    - `artifacts/checkpoints/stage_status.json`
  - Stage 3 now checkpoints successful extraction items incrementally in:
    - `artifacts/checkpoints/extractions/shard_*.jsonl.gz`
- Resume semantics:
  - on rerun of the same failed `prompts` step, Stage 1 is skipped if the Stage 1 checkpoint is complete and matches the filtered statement set
  - Stage 2 is skipped if the Stage 2 checkpoint is complete and matches the filtered statement set
  - Stage 3 loads any existing item-level extraction checkpoint records and only sends missing `(statement_id, variation_index)` items back to the model
  - if Stage 1 or Stage 2 is recomputed, downstream Stage 3 checkpoint files are cleared so stale extraction items are never mixed with new ideations
- Session/lifecycle behavior:
  - same-model local `vllm` reuse is preserved within each attempt
  - a resumed Stage 3-only rerun boots a fresh local `vllm serve` session, but skips Stage 1/2 completely and only runs the missing extraction items
  - no attempt is made to pass a live session object across executor steps

### ALIGN-157 - 2026-03-25 23:17 - Verified checkpointed resume with a failing Stage 3 local-vLLM regression test

- Added regression coverage in `tests/test_alignment.py` for the exact failure shape we saw at full-spec `bs64`:
  - first run:
    - Stage 1 succeeds
    - Stage 2 succeeds
    - Stage 3 extracts one item successfully and checkpoints it
    - the next Stage 3 item returns malformed output and the run fails
  - second run on the same `output_path`:
    - loads Stage 1 and Stage 2 checkpoints
    - skips both stages
    - opens a fresh session
    - submits only the single missing Stage 3 item
- Verification:
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py tests/test_alignment.py`
  - `uv run pytest tests/test_alignment.py -q` → `89 passed`

### ALIGN-158 - 2026-03-25 23:28 - Both full-spec prompt-generation sweeps failed in Stage 3 extraction due to malformed outputs

- Failed roots:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs32-20260325`
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-20260325`
- `bs32` failure:
  - failed child:
    - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs32-20260325/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs32_20260325-prompts_b7a734f8-f2144e25`
  - exact Stage 3 error:
    - `statement 'no_erotica_or_gore' variation 39: missing <user_message>`
    - `statement 'support_mental_health' variation 52: missing <system_prompt>`
    - `statement 'transformation_exception' variation 34: missing <user_message>`
    - `statement 'uphold_fairness' variation 48: missing <user_message>`
- `bs64` failure:
  - failed child:
    - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-20260325/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_20260325-prompts_b68d4b8a-6c635857`
  - exact Stage 3 error:
    - `statement 'be_creative' variation 10: missing <user_message>`
    - `statement 'be_empathetic' variation 46: missing <user_message>`
    - `statement 'be_thorough_but_efficient' variation 34: missing <user_message>`
    - `statement 'letter_and_spirit' variation 30: missing <user_message>`
    - `statement 'support_mental_health' variation 5: missing <system_prompt>`
    - `statement 'uphold_fairness' variation 34: missing <system_prompt>`
    - `statement 'uphold_fairness' variation 68: missing <system_prompt>`
- Interpretation:
  - both runs reached Stage 3 successfully; neither failed on TPU startup, scheduler behavior, or context length
  - the failure mode is consistent: extraction completions occasionally omit one required tag block
  - wider transport batches increase the failure count (`7` malformed outputs at `64` vs `4` at `32`), but `32` is not sufficient to eliminate the issue
  - this means Stage 3 needs per-item retry even after the semantic one-item-per-request refactor
- Resume applicability:
  - these two failed runs were launched before the checkpoint/resume code landed
  - their prompt-output prefixes contain only `.executor_info` and `.executor_status`, so they cannot be resumed with cached Stage 1/2 work
  - future reruns launched from the new code will persist stage checkpoints and be resumable
- Next fix:
  - add Stage 3 per-item retry on malformed extraction outputs before failing the run
  - keep the new stage-boundary + item-level checkpointing so reruns only redo items that remain missing after in-run retries

### ALIGN-159 - 2026-03-25 23:34 - Added Stage 3 per-item retry on malformed extraction outputs

- Change landed in `lib/marin/src/marin/alignment/generate_prompts.py`:
  - Stage 3 extraction now mirrors Stage 1 and Stage 2 retry semantics
  - both local `vllm` and API extraction paths retry only the items that failed parsing
  - successful extraction items are still checkpointed immediately, so retry and resume stack cleanly:
    - in-run retries handle transient malformed outputs
    - reruns only handle items still missing after all retry attempts
- Config:
  - added `PromptGenConfig.extract_max_attempts`
  - default set to `5`
- Expected behavior:
  - a small number of malformed Stage 3 outputs no longer kills the run immediately
  - only items still missing `<system_prompt>` / `<user_message>` after all attempts will fail the stage
- Verification:
  - added a regression in `tests/test_alignment.py` where a Stage 3 item fails once, succeeds on retry, and the run completes without a rerun
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py tests/test_alignment.py`
  - `uv run pytest tests/test_alignment.py -q` → `90 passed`
- Scope note:
  - the already-failed `bs32` and `bs64` runs cannot benefit retroactively; they were launched before both the checkpoint/resume code and the Stage 3 retry code landed
  - the next full-spec launch should use the new code path and will have both protections

### ALIGN-160 - 2026-03-25 23:39 - Launching fresh full-spec `bs64` prompt-generation rerun with Stage 3 retry + checkpoint resume

- Goal:
  - rerun the full-spec prompt-generation sweep at `local_serve_batch_size=64`
  - use the newly landed protections:
    - stage-boundary checkpointing
    - Stage 3 item-level resume
    - Stage 3 per-item retry
- Planned launch:
  - root job name:
    - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325`
  - experiment name:
    - `debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_retry_stage3retry_20260325`
  - command:
    - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/generate_prompts_llama_3_3_70b_refactored.py --name debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_retry_stage3retry_20260325 --local-serve-batch-size 64`
- Launch result:
  - submitted successfully as `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325`
  - current root state: `running`

### ALIGN-161 - 2026-03-25 23:24 - Fresh `bs64` rerun is live on TPU and has entered prompt generation

- Current live root:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325`
- Current live prompt child:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_retry_stage3retry_20260325-prompts_4a7620d2-ef17be47`
- Status:
  - `spec` succeeded
  - `prompts` is now `JOB_STATE_RUNNING` on `v5p-8`
  - worker logs show:
    - `Loaded 46 statements from spec`
    - `Starting vLLM environment`
    - `Starting vLLM native server`
- Interpretation:
  - the rerun has cleared the initial TPU-capacity wait and is now in the real local-vLLM execution path

### ALIGN-162 - 2026-03-25 23:31 - Fresh `bs64` rerun completed Stage 1 and is actively draining the global Stage 2 concretize queue

- Current live root:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325`
- Current live prompt child:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_retry_stage3retry_20260325-prompts_4a7620d2-ef17be47`
- Latest prompt-generation milestones from logs:
  - `vLLM environment ready`
  - `Stage 1: Generating understanding for 46 statements`
  - `Stage 1 progress: 46/46 (100.0%) [attempt 1]`
  - `Saved artifacts for 46 statements to .../prompts-277d65/artifacts/`
  - `Stage 2: Concretizing 46 statements`
  - `Stage 2 local work queue: 3323 concretize items across 46 statements`
  - first live Stage 2 transport batch submitted as `64 prompts (n=1)`
- Interpretation:
  - the checkpointing path is working: Stage 1 finished and wrote artifacts before Stage 2 began
  - the new global cross-statement Stage 2 queue is active on real full-spec work
  - no Stage 1 retry was needed in this rerun, and no Stage 2 context-limit errors have appeared so far

### ALIGN-163 - 2026-03-25 23:42 - Fresh `bs64` rerun is sustaining wide Stage 2 transport batches without instability

- Current live state:
  - root remains `JOB_STATE_RUNNING`
  - prompt child remains `JOB_STATE_RUNNING` on `v5p-8`
- Latest sustained Stage 2 behavior from logs:
  - Stage 2 started with a global queue of `3323` concretize items
  - over the next ~12 minutes the worker issued more than 30 consecutive `64 prompts (n=1)` concretize batches
  - no `HTTP 400`, context-limit, TPU, or schema-adherence errors appeared during that window
- Interpretation:
  - the semantic one-item-per-request refactor plus the new global Stage 2 queue are holding up under the full-spec `bs64` workload
  - this rerun has already exceeded the stability envelope of the earlier failed `bs64` attempt, which never had checkpointing or Stage 3 retry protection

### ALIGN-164 - 2026-03-25 23:53 - Fixed missing mid-stage progress logging for Stage 2 and tightened all stage progress to batch granularity

- User report:
  - the live logs showed repeated batched vLLM requests but no `Stage 2 progress: k/N` lines
- Root cause:
  - Stage 2 progress was only emitted after the full retry attempt completed, because the logger hook sat outside the local per-batch concretize loop
  - the shared progress interval was also coarse (`~10` updates per stage), which is easy to miss in remote Iris logs
- Fix landed in `lib/marin/src/marin/alignment/generate_prompts.py`:
  - Stage 2 local progress now updates inside the per-batch concretize loop
  - progress logging for all stages is now batch-granular (`interval = batch_size`)
  - this gives remote logs a durable `k/N` progression without relying on interactive `tqdm`
- Verification:
  - added a regression in `tests/test_alignment.py` that proves Stage 2 emits incremental `2/5`, `4/5`, `5/5` progress lines for `local_serve_batch_size=2`
  - `uv run pytest tests/test_alignment.py -q` → `91 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py tests/test_alignment.py docs/debug-log-prompt-generation-fullspec-bs64-stage3.md .agents/logbooks/alignment_function.md` → OK
- Scope note:
  - the already-running `bs64` prompt child will not pick up the new progress logs
  - any fresh relaunch or automatic resubmit from this point forward will

### ALIGN-165 - 2026-03-25 23:52 - Live `bs64` prompt child was preempted once and is currently waiting to resume

- Latest Iris status for the live rerun:
  - root `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325` remains `JOB_STATE_RUNNING`
  - prompt child remains associated with the same job id, but now reports:
    - `preemption_count = 1`
    - `task_state_counts = {"pending": 1}`
- Interpretation:
  - the live worker lost its slot once and Iris is rescheduling the prompt child
  - this is separate from the missing-progress-line bug above; it is a runtime scheduling event
  - if the child restarts from the current workspace bundle, it will pick up the new batch-granular Stage 2 progress logging

### ALIGN-166 - 2026-03-25 23:54 - Live `bs64` rerun completed Stage 2, entered Stage 3, and then restarted after one preemption

- Confirmed from live logs:
  - `Stage 2 progress: 3323/3323 (100.0%) [attempt 1, retries pending=0]`
  - `Saved artifacts for 46 statements to .../prompts-277d65/artifacts/`
  - `Reusing the ideation vLLM serve session for extraction`
  - `Stage 3: Extracting prompts from 46 statements`
  - `Stage 3 local work queue: 3323 pending extraction items (0 already checkpointed) across 46 statements`
  - visible Stage 3 incremental progress:
    - `190/3323 (5.7%)`
    - `382/3323 (11.5%)`
    - `573/3323 (17.2%)`
- Interpretation:
  - Stage 3 progress always worked because its logger was already inside the per-batch extraction loop
  - the bug was specific to Stage 2 progress placement
  - the run then experienced one TPU preemption and the child restarted from worker bootstrap
  - after restart, the same prompt child id remained live and is now back in worker startup

### ALIGN-167 - 2026-03-25 23:55 - Preemption was real; Stage 3 resumed from checkpoints instead of restarting from scratch

- Latest confirmed status:
  - prompt child is still `JOB_STATE_RUNNING`
  - Iris reports `preemption_count = 1` for the prompt child
- Evidence from worker logs:
  - before restart, Stage 3 had reached:
    - `573/3323 (17.2%) [attempt 1, retries pending=3]`
  - after restart, the child re-entered bootstrap and then logged:
    - `Loaded Stage 1 checkpoint for 46 statements`
    - `Loaded Stage 2 checkpoint for 46 statements`
    - `Loaded 765 checkpointed Stage 3 extraction items`
    - `Starting vLLM environment`
- Interpretation:
  - this was a genuine preemption / worker loss event, not just a noisy log gap
  - the new checkpoint/resume path is working as intended:
    - Stage 1 and Stage 2 were skipped via checkpoints
    - Stage 3 resumed from the successfully checkpointed extraction items rather than starting over
  - the resumed child is currently rebuilding the local vLLM server and then should continue Stage 3 from the remaining items

### ALIGN-168 - 2026-03-26 00:00 - Original full end-to-end align run is terminal failed; only the later prompt-generation retry remains live

- Original full-pipeline root:
  - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec-retry2-batch4`
- Current status:
  - root is `JOB_STATE_FAILED`
  - failing child was the prompt-generation step:
    - `/ahmed/align-vllm-70b-mixtral-rejected-opposite-full-spec-retry2-batch4/align-debug_vllm_70b_mixtral_rejected_opposite_full_spec-prompts_a9dd9e03-eb3c86e7`
- Exact terminal error:
  - `RuntimeError: Stage 3 failed for 2 statement(s):`
    - `follow_all_applicable_instructions`: missing `<scenario_49>` block
    - `sexual_content_involving_minors`: missing `<scenario_70>` block
- Interpretation:
  - that end-to-end run never reached chosen / rejected / judge / preference-pair stages
  - it died inside prompt generation under the old pre-refactor extraction code path
- Separate still-live job:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325`
  - this is prompt-generation-only, not the full pipeline
  - it remains `JOB_STATE_RUNNING` with `preemption_count = 2` and is resuming Stage 3 from checkpoints

### ALIGN-169 - 2026-03-26 00:03 - Repeated Stage 3 restarts are genuine TPU preemptions, not babysitter resubmits or fresh Stage 3 crashes

- Current live prompt-generation retry:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325`
- Current status:
  - prompt child still `JOB_STATE_RUNNING`
  - `preemption_count = 2`
  - monitor state file still shows `restart_count = 0`, so the babysitter has not resubmitted anything
- Observed restart sequence in logs:
  - first Stage 3 run progressed to `573/3323 (17.2%)`
  - then worker bootstrap restarted and loaded:
    - Stage 1 checkpoint
    - Stage 2 checkpoint
    - `765` Stage 3 extraction items
  - later, the worker bootstrap restarted again and loaded the same checkpoint counts:
    - Stage 1 checkpoint
    - Stage 2 checkpoint
    - `765` Stage 3 extraction items
- Interpretation:
  - the restarts are internal task preemptions / worker loss events handled by Iris, not new root-job submissions
  - Stage 3 is where this is visible because it is the longest-running phase after Stage 2 completes
  - the second restart appears to have happened before the resumed worker made new extraction progress, since the loaded Stage 3 checkpoint count remained `765`

### ALIGN-170 - 2026-03-26 00:12 - Resumed Stage 3 is advancing steadily after the second preemption

- Latest live status:
  - prompt child remains `JOB_STATE_RUNNING`
  - `preemption_count` is still `2` (no new preemption since the last restart)
- Latest resumed Stage 3 window:
  - worker reached `vLLM environment ready`
  - resumed Stage 3 queue:
    - `2558 pending extraction items (765 already checkpointed)`
  - observed progress:
    - `826/3323 (24.9%)`
    - `1018/3323 (30.6%)`
    - `1209/3323 (36.4%)`
    - `1401/3323 (42.2%)`
    - `1593/3323 (47.9%)`
    - `1784/3323 (53.7%)`
    - `1976/3323 (59.5%)`
    - `2166/3323 (65.2%)`
    - `2358/3323 (71.0%)`
    - `2550/3323 (76.7%)`
- Interpretation:
  - the resumed worker is healthy and checkpoint resume is paying off
  - the job has now recovered most of the Stage 3 work lost to the earlier preemptions

### ALIGN-171 - 2026-03-26 00:13 - Stage 3 attempt 1 is essentially complete; only 8 extraction items remain for retry

- Latest live Stage 3 window:
  - `2741/3323 (82.5%) [attempt 1, retries pending=8]`
  - `2933/3323 (88.3%) [attempt 1, retries pending=8]`
  - `3125/3323 (94.0%) [attempt 1, retries pending=8]`
  - `3315/3323 (99.8%) [attempt 1, retries pending=8]`
- Interpretation:
  - the resumed worker has nearly drained the full Stage 3 queue
  - if the retry logic behaves as intended, the next immediate work should be a small retry round over the remaining 8 malformed extraction items rather than a full-stage restart

### ALIGN-172 - 2026-03-26 00:16 - First checkpointed `bs64` rerun still failed, but only on 6 stubborn Stage 3 items; immediately resubmitted from checkpoints

- Terminal result of the first checkpointed retry:
  - prompt child failed after Stage 3 retries
  - root failed with `RuntimeError: 1 step(s) failed`
- Stage 3 retry trail:
  - attempt 1 left `8` extraction items pending retry
  - attempt 2 left `6`
  - attempts 3, 4, and 5 remained at `6`
- Final stubborn items:
  - `assume_best_intentions` variation `3`: missing `<system_prompt>`
  - `avoid_extremist_content` variation `46`: missing `<user_message>`
  - `be_creative` variation `71`: missing `<user_message>`
  - `do_not_lie` variation `34`: missing `<system_prompt>`
  - `formatting` variation `67`: missing `<user_message>`
  - `protect_privacy` variation `4`: missing `<user_message>`
- Recovery:
  - manually resubmitted the same Iris root job name:
    - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325`
  - updated `scratch/20260325-2319_monitoring_state.json` to `restart_count = 1`
  - new root is already back to `JOB_STATE_RUNNING`
- Expectation:
  - because the output path is unchanged, the new attempt should reload Stage 1 + Stage 2 checkpoints and only retry the remaining Stage 3 items

### ALIGN-173 - 2026-03-26 00:22 - Corrected rerun diagnosis: executor is force-rerunning the failed prompt step; child is pending TPU capacity

- Correction:
  - the second manual resubmit did not die immediately because of executor refusal
  - root logs show:
    - `Force running align/.../prompts_4a7620d2, previous status: FAILED`
- Current live rerun state:
  - root:
    - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325`
    - `JOB_STATE_RUNNING`
  - prompt child:
    - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_retry_stage3retry_20260325-prompts_4a7620d2-73daf1aa`
    - `JOB_STATE_PENDING`
- Pending reason:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - autoscaler waiting on scale group `tpu_v5p_8-us-central1-a`
- Interpretation:
  - checkpoint resume is still the active recovery path
  - the only blocker at this moment is TPU capacity, not a new code failure

### ALIGN-174 - 2026-03-26 00:25 - Prompt child acquired `v5p-8` capacity and is back to running

- Latest scheduler state:
  - prompt child `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_retry_stage3retry_20260325-prompts_4a7620d2-73daf1aa`
  - `JOB_STATE_RUNNING`
  - task state counts currently show `building=1`, so the new worker is still finishing startup/bootstrap
- Interpretation:
  - TPU capacity is no longer the blocker
  - next expected logs are the checkpoint loads and Stage 3 resume path

### ALIGN-175 - 2026-03-26 00:27 - Stage 3 resume is now down to the final 6 extraction items

- Latest child logs:
  - `Loaded 3317 checkpointed Stage 3 extraction items`
  - followed immediately by fresh `vllm` startup
- Interpretation:
  - the restarted prompt child correctly recovered almost the entire Stage 3 queue from checkpoint
  - only the last 6 stubborn extraction items remain to be retried under the new worker

### ALIGN-176 - 2026-03-26 00:35 - Second checkpointed rerun failed again, but the stubborn Stage 3 set shrank from 6 items to 4; resubmitted immediately

- Terminal result of the second checkpointed retry:
  - prompt child `...prompts_4a7620d2-73daf1aa` failed
  - root `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325` failed
- Latest remaining Stage 3 failures:
  - `assume_best_intentions` variation `3`: missing `<system_prompt>`
  - `avoid_extremist_content` variation `46`: missing `<user_message>`
  - `be_creative` variation `71`: missing `<user_message>`
  - `do_not_lie` variation `34`: missing `<system_prompt>`
- Cleared since the prior rerun:
  - `formatting` variation `67`
  - `protect_privacy` variation `4`
- Recovery:
  - immediately resubmitted the same root job name again:
    - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325`
  - updated `scratch/20260325-2319_monitoring_state.json` to `restart_count = 3`
- Interpretation:
  - item-level Stage 3 checkpointing is working
  - repeated reruns are monotonically shrinking the stuck extraction set, but we still need to babysit until those last 4 items clear or stabilize

### ALIGN-177 - 2026-03-26 00:36 - Fixed-root resubmit path became unreliable; switched recovery to fresh Iris root ids while keeping the same experiment output path

- After the immediate fixed-root resubmit:
  - root `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-retry-stage3retry-20260325` failed in about 9 seconds
  - `job list` showed only the root, with no new child jobs visible
  - direct root logs did not show a fresh prompt-child launch
- Recovery change:
  - launched a fresh Iris root id instead:
    - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-r4-20260326-0037`
  - kept the same experiment name:
    - `--name debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_retry_stage3retry_20260325`
  - this preserves the same prompt-step output path and therefore the same Stage 1/2 checkpoints plus Stage 3 item checkpoints
- Monitoring state update:
  - rewrote `scratch/20260325-2319_monitoring_state.json`
  - `job_id = /ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-r4-20260326-0037`
  - `restart_count = 4`
  - `resubmit_command` now omits `--job-name` so future recoveries get fresh root ids by default

### ALIGN-178 - 2026-03-26 00:38 - Fresh-root recovery succeeded in launching a new prompt child against the same checkpointed output path

- New live root:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-r4-20260326-0037`
  - `JOB_STATE_RUNNING`
- New live prompt child:
  - `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-r4-20260326-0037/align-debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_retry_stage3retry_20260325-prompts_4a7620d2-0ee03307`
  - `JOB_STATE_RUNNING`
- Interpretation:
  - the fresh-root strategy avoids the flaky immediate-fail behavior of the fixed-root resubmit path
  - checkpointed recovery of the same prompt-step output path is still working under the new root id

### ALIGN-179 - 2026-03-26 00:39 - New prompt child resumed from `3319` checkpointed Stage 3 items; only 4 extraction items remain

- Latest child logs from `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-r4-20260326-0037/...-prompts_4a7620d2-0ee03307`:
  - `Loaded 3319 checkpointed Stage 3 extraction items`
  - then fresh `vllm` startup
- Interpretation:
  - the new rerun has preserved all progress from the prior attempts
  - only the final 4 stubborn extraction items remain to be retried under this worker

### ALIGN-180 - 2026-03-26 00:48 - Fresh-root rerun also failed on the same 4 Stage 3 extraction items; progress has now plateaued

- Terminal result of `/ahmed/generate-prompts-llama-3-3-70b-fullspec-bs64-r4-20260326-0037`:
  - prompt child `...prompts_4a7620d2-0ee03307` failed
  - root failed with `RuntimeError: 1 step(s) failed`
- Resume behavior worked as designed:
  - loaded `3319 checkpointed Stage 3 extraction items`
  - Stage 3 queue was only `4 pending extraction items`
  - all five retry attempts still left `4` items pending
- Remaining deterministic failures:
  - `assume_best_intentions` variation `3`: missing `<system_prompt>`
  - `avoid_extremist_content` variation `46`: missing `<user_message>`
  - `be_creative` variation `71`: missing `<user_message>`
  - `do_not_lie` variation `34`: missing `<system_prompt>`
- Interpretation:
  - repeated reruns are no longer shrinking the stuck set
  - the current recovery path has reached a plateau, so another plain rerun is unlikely to help without changing Stage 3 behavior for these items

### ALIGN-181 - 2026-03-26 00:51 - Stage 2 inspection points to a Stage 2 / Stage 3 contract mismatch, with one genuinely bad Stage 2 record

- Inspected Stage 2 ideation artifacts at:
  - `gs://marin-us-central1/align/debug_generate_prompts_llama_3_3_70b_refactored_fullspec_bs64_retry_stage3retry_20260325/prompts-277d65/artifacts/<statement_id>/ideation.json`
- Key finding for the four stuck items:
  - `assume_best_intentions/3`: rich scenario prose with a quoted user request, but no explicit deployment-style system prompt in the description
  - `be_creative/71`: same pattern, quoted user request but no explicit system prompt in the description
  - `do_not_lie/34`: same pattern, quoted user request but no explicit system prompt in the description
  - `avoid_extremist_content/46`: worse than the others
    - rubric is empty
    - scenario is framed as passive consumption and already includes an assistant message
    - there is no clean user turn to extract into `<user_message>`
- Broader Stage 2 quality signal:
  - `ideations.jsonl.gz` contains `3323` total variations
  - `142` of those already have empty rubrics
- Interpretation:
  - the main issue is not that all four Stage 2 records are garbage
  - the real mismatch is that Stage 3 prompt wording says the scenario "contains a system prompt ... usually in quotes near the start", but many Stage 2 variations are narrative scenarios that do not actually contain one
  - `avoid_extremist_content/46` is a true Stage 2 defect or under-specified case; the other three look more like valid narrative scenarios that Stage 3 is being asked to "extract" from a representation that does not literally contain the fields it expects

### ALIGN-182 - 2026-03-26 00:55 - Refined diagnosis: the problem is narrower than a general Stage 2 / Stage 3 mismatch

- Important correction:
  - Stage 3 uses only `variation.description`, not `rubric`, so empty-rubric Stage 2 records are a quality smell but not the direct cause of Stage 3 parse failures
- Strong counterexamples from nearby successful checkpointed extractions:
  - `be_creative/70` succeeded even though its Stage 2 description has no explicit `system prompt` phrase and no quoted user utterance
  - `avoid_extremist_content/47` succeeded even though it is also a narrative scenario without an explicit literal system prompt in the description
- Updated interpretation:
  - the earlier "Stage 2 / Stage 3 contract mismatch" is real but not sufficient to explain the plateau by itself
  - Stage 3 can often synthesize prompt pairs from narrative Stage 2 descriptions
  - the remaining 4 items are more likely *specific hard cases* for the current Stage 3 prompt/parsing setup
  - among them, `avoid_extremist_content/46` still stands out as the most under-specified Stage 2 input because it is passive-consumption framed and already includes an assistant message instead of a clean user turn

### ALIGN-183 - 2026-03-26 09:32 - Plan new `gpt-oss-120b` thread: dual-region staging, then prompt-generation validation, then full open-weight pipeline

- Motivation:
  - current `Llama-3.3-70B-Instruct` prompt-generation thread has a stubborn Stage 3 tail, and the working hypothesis is that `openai/gpt-oss-120b` may follow the Stage 1/2/3 formatting instructions more reliably
  - user wants a staged migration path instead of a one-shot model swap
- Working assumptions to validate early:
  - `openai/gpt-oss-120b` should fit and serve on `v5p-8`
  - we want the model staged in both `us-central1-a` and `us-east5-a`
  - we should pin an exact Hugging Face revision before any download or runtime test; do not use a floating branch
- Implementation plan:
  1. Register the model in [experiments/models.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/models.py)
     - add a new `download_model_step(ModelConfig(...))` entry for:
       - `hf_repo_id="openai/gpt-oss-120b"`
       - `hf_revision="<pinned commit>"`
     - keep the output naming consistent with existing staged-model paths:
       - `models/openai--gpt-oss-120b--<revision>`
  2. Stage the model in both target regions
     - launch a `download_model_step` run against the `marin-us-central1` bucket so the staged artifact lands under:
       - `gs://marin-us-central1/models/openai--gpt-oss-120b--<revision>`
     - launch the same download flow against the `marin-us-east5` bucket so the staged artifact lands under:
       - `gs://marin-us-east5/models/openai--gpt-oss-120b--<revision>`
     - verify both regions have complete staged artifacts before any runtime experiment
  3. Validate raw `vllm` bring-up on `v5p-8`
     - before touching the full pipeline, run a minimal one-request smoke in each zone:
       - `us-central1-a`
       - `us-east5-a`
     - success criteria:
       - `vllm serve` boots
       - tokenizer loads
       - one batched completion request succeeds
       - no immediate TPU memory failure or model-load crash
  4. Run prompt-generation-only Stage 1/2/3 on one statement
     - base this on [generate_prompts_llama_3_3_70b_refactored.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/generate_prompts_llama_3_3_70b_refactored.py), but point it at the staged `gpt-oss-120b` artifact
     - keep the current architecture:
       - one shared `vllm` session
       - Stage 1/2 checkpoints
       - Stage 3 item-level resume and retry
       - `vllm_metrics.json` emitted for `understanding`, `concretize`, and `extract`
     - start with one statement only:
       - `ask_clarifying_questions`
     - start conservatively on batch width, then widen only after a clean pass
  5. Run prompt-generation-only Stage 1/2/3 on the entire spec
     - once the one-statement run is clean, rerun the same prompt-generation experiment over the full `openai_model_spec.jsonl`
     - use the current checkpoint/resume path from the start
     - primary goal is to learn:
       - Stage 3 failure rate under `gpt-oss-120b`
       - prompt-generation wall-clock and `vllm` throughput metrics
       - whether the prior Llama Stage 3 tail largely disappears
  6. Run the full open-weight alignment pipeline with `gpt-oss-120b` as the chosen model
     - after prompt generation is stable, update the full pipeline experiment derived from [align_vllm_70b_mixtral_rejected_full_spec.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/align_vllm_70b_mixtral_rejected_full_spec.py)
     - target initial config:
       - prompt generation on `gpt-oss-120b`
       - chosen response model = `gpt-oss-120b`
       - rejected response model = `Mixtral-8x7B-Instruct-v0.1`
       - rejected prompt strategy = `opposite`
       - response execution mode = `auto`
     - preserve the existing opposite-mode and structured metrics path
- Planned experiment sequence:
  - `GOSS-1`: register and download `gpt-oss-120b` in `us-central1-a`
  - `GOSS-2`: register and download `gpt-oss-120b` in `us-east5-a`
  - `GOSS-3`: `v5p-8` local-serve smoke in `us-central1-a`
  - `GOSS-4`: `v5p-8` local-serve smoke in `us-east5-a`
  - `GOSS-5`: Stage 1/2/3 one-statement prompt-generation run on staged `gpt-oss-120b`
  - `GOSS-6`: Stage 1/2/3 full-spec prompt-generation run on staged `gpt-oss-120b`
  - `GOSS-7`: full open-weight alignment run with `gpt-oss-120b` chosen and `Mixtral` rejected-opposite
- Success criteria for the thread:
  - model is staged and runnable in both target regions
  - one-statement Stage 1/2/3 is clean
  - full-spec Stage 1/2/3 completes with materially fewer deterministic Stage 3 tail failures than the current Llama path
  - the first full open-weight pipeline run reaches `preference_pairs`

### ALIGN-184 - 2026-03-26 01:16 - Registered pinned `gpt-oss-120b` and added the `GOSS` experiment entrypoints

- Resolved Hugging Face model metadata for:
  - `openai/gpt-oss-120b`
  - full SHA: `b5c939de8f754692c1647ca79fbf85e8c1e70f8a`
  - staged short revision used in Marin model paths: `b5c939d`
- Sanity checks against the published model files:
  - repo is public and ungated
  - tokenizer exposes a working runtime chat template even though `tokenizer_config.json` advertises `chat_template = null`
  - config reports:
    - `model_type = gpt_oss`
    - `initial_context_length = 4096`
    - `max_position_embeddings = 131072`
- Code changes:
  - registered `gpt_oss_120b` in [experiments/models.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/models.py)
  - added [download_gpt_oss_120b.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/download_gpt_oss_120b.py)
  - added [gpt_oss_120b_vllm_smoke.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/gpt_oss_120b_vllm_smoke.py)
  - added [generate_prompts_gpt_oss_120b_refactored.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/generate_prompts_gpt_oss_120b_refactored.py)
  - added [align_gpt_oss_120b_mixtral_rejected_full_spec.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py)
- Validation before launch:
  - targeted pre-commit passed on the new `gpt-oss` files
  - `py_compile` passed on the new scripts
- Next action:
  - launch `GOSS-1` and `GOSS-2` to stage `gpt-oss-120b` into `gs://marin-us-central1/...` and `gs://marin-us-east5/...`

### ALIGN-185 - 2026-03-26 01:18 - Launched `GOSS-1` and `GOSS-2` staging jobs against the live Iris controller with explicit zone constraints

- Important execution detail:
  - the live Iris controller at `http://127.0.0.1:10000` already exposes both `us-central1-a` and `us-east5-a` `v5p` scale groups
  - the earlier attempt to use `infra/*.yaml` directly with `iris job run --config ...` failed because those files are Ray cluster configs, not Iris controller configs
  - the correct launch path is:
    - `--controller-url http://127.0.0.1:10000`
    - plus explicit `--region` and `--zone` constraints
- `GOSS-1` launch:
  - root job: `/ahmed/goss1-download-gpt-oss-120b-central1`
  - constraints:
    - `--region us-central1`
    - `--zone us-central1-a`
  - output prefix:
    - `gs://marin-us-central1`
  - live Zephyr child:
    - `/ahmed/goss1-download-gpt-oss-120b-central1/zephyr-download-hf-abc6e028-p0-a0`
- `GOSS-2` launch:
  - root job: `/ahmed/goss2-download-gpt-oss-120b-east5`
  - constraints:
    - `--region us-east5`
    - `--zone us-east5-a`
  - output prefix:
    - `gs://marin-us-east5`
  - live Zephyr child:
    - `/ahmed/goss2-download-gpt-oss-120b-east5/zephyr-download-hf-60a509d3-p0-a0`
- Shared live state from logs:
  - both runs identified `37` files totaling `182.32 GB`
  - both runs started Zephyr download pipelines successfully
  - target staged paths are:
    - `gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d`
    - `gs://marin-us-east5/models/openai--gpt-oss-120b--b5c939d`
- Next action:
  - babysit both regional staging jobs to completion before launching `GOSS-3` / `GOSS-4`

### ALIGN-186 - 2026-03-26 01:20 - Both regional `gpt-oss-120b` staging runs are healthy; they are in the large-shard transfer phase rather than stuck

- Current live roots:
  - `/ahmed/goss1-download-gpt-oss-120b-central1`
  - `/ahmed/goss2-download-gpt-oss-120b-east5`
- Current live Zephyr coordinators:
  - `/ahmed/goss1-download-gpt-oss-120b-central1/zephyr-download-hf-abc6e028-p0-a0`
  - `/ahmed/goss2-download-gpt-oss-120b-east5/zephyr-download-hf-60a509d3-p0-a0`
- Latest coordinator progress:
  - both runs reached `7/37 complete`
  - both still have `2 in-flight`, `28 queued`, `2/2 workers alive`, `0 dead`
- Why this is not considered stalled:
  - both regional GCS prefixes now show `11` materialized files
  - worker-pool task inspection shows all four worker tasks are still `TASK_STATE_RUNNING`
  - worker resource usage is nonzero on all active workers:
    - central1 workers: ~`16%` CPU, disk rising into `3.8 GB` / `6.4 GB`
    - east5 workers: ~`5-7%` CPU, disk rising into `23 GB`
- Interpretation:
  - the download pipelines are in the "few giant safetensor shards dominate wall clock" phase
  - no recovery action is warranted yet

### ALIGN-187 - 2026-03-26 01:22 - Worker heartbeats confirm the current long pole is `metal/model.bin`; downloads are active, not wedged

- Direct worker-task log inspection shows the current in-flight shard on both regions is:
  - `openai/gpt-oss-120b@b5c939d/metal/model.bin`
- Latest worker heartbeats:
  - central1:
    - `2632.0 MiB` in `60.2s` (`43.75 MiB/s`)
    - `5240.0 MiB` in `120.2s` (`43.58 MiB/s`)
    - `7840.0 MiB` in `180.3s` (`43.49 MiB/s`)
  - east5:
    - `1096.0 MiB` in `60.2s` (`18.21 MiB/s`)
    - `2216.0 MiB` in `120.7s` (`18.35 MiB/s`)
    - `3272.0 MiB` in `180.9s` (`18.09 MiB/s`)
- Interpretation:
  - there is no Zephyr deadlock or worker hang
  - regional throughput is meaningfully better in `us-central1-a` than `us-east5-a`
  - if staging time later becomes a recurring bottleneck, the first optimization to consider is excluding `metal/model.bin` from the staged artifact if `vllm` does not need it

### ALIGN-188 - 2026-03-26 01:27 - Both `gpt-oss-120b` staging runs are still healthy; `us-central1-a` is clearly ahead of `us-east5-a`

- Live roots remain:
  - `/ahmed/goss1-download-gpt-oss-120b-central1`
  - `/ahmed/goss2-download-gpt-oss-120b-east5`
- Live Zephyr coordinators remain:
  - `/ahmed/goss1-download-gpt-oss-120b-central1/zephyr-download-hf-abc6e028-p0-a0`
  - `/ahmed/goss2-download-gpt-oss-120b-east5/zephyr-download-hf-60a509d3-p0-a0`
- Latest coordinator progress:
  - central: `10/37 complete`, `2 in-flight`, `25 queued`, `2/2 workers alive`, `0 dead`
  - east: `8/37 complete`, `2 in-flight`, `27 queued`, `2/2 workers alive`, `0 dead`
- Direct worker logs show the current long poles precisely:
  - central worker `0` is still streaming:
    - `openai/gpt-oss-120b@b5c939d/metal/model.bin`
    - latest heartbeat: `18304.0 MiB` in `420.9s` (`43.49 MiB/s`)
  - central worker `1` has already completed:
    - `model-00000-of-00014.safetensors`
    - `model-00001-of-00014.safetensors`
    - `model-00002-of-00014.safetensors`
    - `model-00003-of-00014.safetensors`
    - and is now on `model-00004-of-00014.safetensors`
  - east worker `0` is still streaming:
    - `openai/gpt-oss-120b@b5c939d/metal/model.bin`
    - latest heartbeat: `6560.0 MiB` in `361.5s` (`18.15 MiB/s`)
  - east worker `1` has completed:
    - `model-00000-of-00014.safetensors`
    - and is now on `model-00001-of-00014.safetensors`
- Interpretation:
  - the pipelines are still healthy; the slow coordinator counters are explained by a few giant files rather than a Zephyr control-plane problem
  - `us-central1-a` is downloading this model at roughly `2.3x` the shard throughput of `us-east5-a`
  - central should reach `GOSS-3` materially earlier than east reaches `GOSS-4`

### ALIGN-189 - 2026-03-26 01:31 - Central is still the lead region; one transient coordinator-RPC timeout did not interrupt shard streaming

- Updated staging progress:
  - central: `12/37 complete`, `17` materialized files in `gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d`
  - east: `9/37 complete`, `14` materialized files in `gs://marin-us-east5/models/openai--gpt-oss-120b--b5c939d`
- Central worker details:
  - worker `0` is still streaming `metal/model.bin`
    - latest observed heartbeat: `20960.0 MiB` in `481.0s` (`43.58 MiB/s`)
  - worker `1` completed:
    - `model-00000-of-00014.safetensors`
    - `model-00001-of-00014.safetensors`
    - `model-00002-of-00014.safetensors`
    - `model-00003-of-00014.safetensors`
    - and is now actively streaming `model-00005-of-00014.safetensors`
    - latest observed heartbeat: `2632.0 MiB` in `60.1s` (`43.81 MiB/s`)
- East worker details:
  - worker `0` is still on `metal/model.bin`
  - worker `1` completed `model-00001-of-00014.safetensors` and moved on to `model-00002-of-00014.safetensors`
- Control-plane note:
  - central worker `1` logged a transient `connectrpc.errors.ConnectError: Request timed out` while heartbeating to the Zephyr coordinator
  - the same worker resumed normal progress logs immediately afterward and kept streaming the shard, so this does not currently look like a fatal worker or coordinator failure
- Immediate next action:
  - keep babysitting both staging jobs
  - launch `GOSS-3` in `us-central1-a` the moment the central staging root succeeds, without waiting for east to catch up

### ALIGN-190 - 2026-03-26 01:33 - Central staging hit a TPU-slice failure, but the root auto-retried cleanly and resumed from the already staged files

- Failure event on the original central Zephyr child:
  - old child: `/ahmed/goss1-download-gpt-oss-120b-central1/zephyr-download-hf-abc6e028-p0-a0`
  - worker pool outcome:
    - one worker died with `Worker ... failed`
    - sibling worker was then killed with `Parent task preempted`
  - root task error context:
    - `Worker marin-tpu-v5p-32-us-central1-a-20260325-2105-982b3329-worker-3 failed: sibling worker ... worker-0 failed, slice terminated`
- Recovery behavior:
  - the root executor step stayed alive and retried automatically
  - new child:
    - `/ahmed/goss1-download-gpt-oss-120b-central1/zephyr-download-hf-bd4db731-p0-a0`
  - root logs show the executor reclaimed the running step lock and relaunched Zephyr rather than marking the model step failed
- Evidence that the retry is resuming effectively:
  - the new central child reached `13/37 tasks completed` almost immediately after worker registration
  - it is now at `14/37 complete`, `2 in-flight`, `21 queued`, `2/2 workers alive`, `0 dead`
  - the central GCS prefix already contains:
    - metadata files
    - `model-00000` through `model-00006`
  - so the second attempt is benefiting from the existing staged files rather than paying full cost from scratch
- East status at the same checkpoint:
  - `/ahmed/goss2-download-gpt-oss-120b-east5/zephyr-download-hf-60a509d3-p0-a0`
  - still healthy at `10/37 complete`, `2 in-flight`, `25 queued`
- Immediate next action:
  - continue babysitting the new central child and the original east child
  - keep the `GOSS-3` central smoke launch ready for the first successful completion of `/ahmed/goss1-download-gpt-oss-120b-central1`

### ALIGN-191 - 2026-03-26 01:35 - Added per-file skip-on-size-match to the Hugging Face download path for future retries and relaunches

- Motivation:
  - the live central retry demonstrated that reusing already staged files matters for large model staging
  - the previous `download_hf` path wrote success metrics shards, but it did not explicitly skip already-present target files by checking GCS object size before streaming
- Code change:
  - updated [download_hf.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/download/huggingface/download_hf.py)
  - new behavior:
    - if the target object already exists and its size matches the expected Hugging Face size, the file is skipped instead of being re-downloaded
    - if the object exists with a mismatched size, the file is re-downloaded and a warning is logged
  - the log summary now reports how many files were skipped as already present
- Validation:
  - targeted `pre-commit` passed
  - `py_compile` passed
- Important scope note:
  - this change does **not** affect the already-running `GOSS-1` / `GOSS-2` bundles
  - it will apply to any future resubmission or new staging launch

### ALIGN-192 - 2026-03-26 01:38 - Central retry is steadily advancing; both regional staging jobs remain healthy

- Central current child:
  - `/ahmed/goss1-download-gpt-oss-120b-central1/zephyr-download-hf-bd4db731-p0-a0`
  - progress:
    - `18/37 complete`
    - `2 in-flight`
    - `17 queued`
    - `2/2 workers alive`
    - `0 dead`
  - central staged file count is now `22`
- East current child:
  - `/ahmed/goss2-download-gpt-oss-120b-east5/zephyr-download-hf-60a509d3-p0-a0`
  - progress:
    - `12/37 complete`
    - `2 in-flight`
    - `23 queued`
    - `2/2 workers alive`
    - `0 dead`
  - east staged file count is now `16`
- Interpretation:
  - the central retry is no longer in a fragile recovery phase; it is progressing normally through the large shard band
  - east remains slower but healthy
  - central is still the first likely path to `GOSS-3`

### ALIGN-193 - 2026-03-26 01:43 - Central is not smoke-ready yet; the final model shard, index, and tokenizer files are still missing

- I diffed the central staged path against the upstream HF file list for `openai/gpt-oss-120b@b5c939d`.
- Upstream file count is `37`.
- Current central staged payload count, excluding executor metadata and Zephyr metrics sidecars, is `20`.
- The central bucket already has:
  - `model-00000-of-00014.safetensors` through `model-00013-of-00014.safetensors`
  - `chat_template.jinja`
  - `config.json`
  - `generation_config.json`
  - the basic license/readme/policy files
- But it is still missing files that the local `vllm` smoke genuinely needs:
  - `model-00014-of-00014.safetensors`
  - `model.safetensors.index.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json`
- It is also still missing the noncritical extras:
  - `metal/model.bin`
  - the whole `original/` subtree
- Decision:
  - do **not** launch `GOSS-3` early yet
  - wait for the central staging root to materialize at least the final numbered shard, the shard index, and the tokenizer files

### ALIGN-194 - 2026-03-26 01:53 - Switched the `gpt-oss-120b` serving path to a dedicated vLLM-minimal artifact so GOSS execution is no longer blocked on `metal/` and `original/`

- Updated [experiments/models.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/models.py):
  - added `hf_urls_glob` and `output_name_suffix` support to `ModelConfig`
  - registered a new staged artifact:
    - `gpt_oss_120b_vllm`
    - output path shape:
      - `models/openai--gpt-oss-120b-vllm--b5c939d`
    - filtered files:
      - `chat_template.jinja`
      - `config.json`
      - `generation_config.json`
      - `model-*.safetensors`
      - `model.safetensors.index.json`
      - `special_tokens_map.json`
      - `tokenizer.json`
      - `tokenizer_config.json`
- Added [download_gpt_oss_120b_vllm.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/download_gpt_oss_120b_vllm.py) to stage the serving subset explicitly.
- Updated downstream `gpt-oss` execution entrypoints to use the vLLM-minimal artifact:
  - [gpt_oss_120b_vllm_smoke.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/gpt_oss_120b_vllm_smoke.py)
  - [generate_prompts_gpt_oss_120b_refactored.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/generate_prompts_gpt_oss_120b_refactored.py)
  - [align_gpt_oss_120b_mixtral_rejected_full_spec.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py)
- Validation:
  - targeted `pre-commit` passed
  - `py_compile` passed
- Rationale:
  - the full-repo staging roots remain useful for the complete repository download goal
  - but the GOSS smokes and downstream alignment pipeline only need the top-level weights plus tokenizer sidecars
  - this removes the current dependency on the `metal/` and `original/` subtrees

### ALIGN-195 - 2026-03-26 01:54 - Prepared the next execution branch: launch minimal-serving staging in `us-central1-a` and `us-east5-a`, then chain GOSS-3/GOSS-4 onto those artifacts

- Next launch commands:
  - central minimal staging:
    - root job name:
      - `/ahmed/goss1b-download-gpt-oss-120b-vllm-central1`
  - east minimal staging:
    - root job name:
      - `/ahmed/goss2b-download-gpt-oss-120b-vllm-east5`
- Execution intent:
  - treat the vLLM-minimal artifacts as the staged inputs for `GOSS-3` through `GOSS-7`
  - keep the original full-repo `GOSS-1` / `GOSS-2` roots running in parallel to satisfy the broader "download the model in both regions" goal

### ALIGN-196 - 2026-03-26 01:50 - Launched the vLLM-minimal `gpt-oss-120b` staging roots in both target zones

- Central minimal staging:
  - root:
    - `/ahmed/goss1b-download-gpt-oss-120b-vllm-central1`
  - constraints:
    - `--region us-central1`
    - `--zone us-central1-a`
  - output prefix:
    - `gs://marin-us-central1`
- East minimal staging:
  - root:
    - `/ahmed/goss2b-download-gpt-oss-120b-vllm-east5`
  - constraints:
    - `--region us-east5`
    - `--zone us-east5-a`
  - output prefix:
    - `gs://marin-us-east5`
- Command details:
  - both launches exported `HF_TOKEN`
  - both ran [download_gpt_oss_120b_vllm.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/download_gpt_oss_120b_vllm.py)
- Next action:
  - babysit both new roots to first healthy Zephyr child creation
  - launch `GOSS-3` as soon as the central minimal artifact is complete

### ALIGN-197 - 2026-03-26 01:51 - Both minimal-serving staging roots started healthy Zephyr pipelines; each is downloading a 22-file, 60.79 GB serving subset

- Central minimal root:
  - `/ahmed/goss1b-download-gpt-oss-120b-vllm-central1`
  - coordinator child:
    - `/ahmed/goss1b-download-gpt-oss-120b-vllm-central1/zephyr-download-hf-9f015b25-p0-a0`
- East minimal root:
  - `/ahmed/goss2b-download-gpt-oss-120b-vllm-east5`
  - coordinator child:
    - `/ahmed/goss2b-download-gpt-oss-120b-vllm-east5/zephyr-download-hf-d27a2b28-p0-a0`
- Root-log confirmation from both regions:
  - `Total number of files to process: 22 (60.79 GB); skipping 0 already-present files with matching sizes`
  - Zephyr coordinator submission succeeded immediately in both regions
- Interpretation:
  - the minimal-serving path is now the active critical path for `GOSS-3` through `GOSS-7`
  - this is materially smaller than the original `37`-file / `182.32 GB` full-repo staging path

### ALIGN-198 - 2026-03-26 01:53 - Minimal serving downloads are healthy at `3/22`; central full download is farther along but still not smoke-ready because tokenizer sidecars are absent

- Minimal-serving progress:
  - central minimal metrics:
    - `3/22`
  - east minimal metrics:
    - `3/22`
  - first landed files in both minimal prefixes:
    - `chat_template.jinja`
    - `config.json`
    - `generation_config.json`
  - worker pools:
    - both coordinators started `4` workers cleanly
- Direct worker-log confirmation on central minimal:
  - shard `0` completed:
    - `chat_template.jinja`
  - shard `1` completed:
    - `config.json`
  - shard `2` completed:
    - `generation_config.json`
  - later shards now correspond to the first top-level model shards
- Parallel status on the original full central download:
  - central full metrics are already at `26/37`
  - central full now has:
    - all top-level `model-00000` through `model-00014`
    - `model.safetensors.index.json`
  - but it still lacks the tokenizer sidecars needed for serving:
    - `special_tokens_map.json`
    - `tokenizer.json`
    - `tokenizer_config.json`
- Decision:
  - keep the full-download roots alive
  - keep using the minimal-serving roots as the actual GOSS unblock path

### ALIGN-199 - 2026-03-26 01:59 - Central minimal serving artifact is smoke-ready and now unblocks `GOSS-3`

- Central minimal progress advanced from `11/22` to `21/22`, then reached full serving completeness.
- Final central minimal readiness check:
  - required serving set present in one directory:
    - `chat_template.jinja`
    - `config.json`
    - `generation_config.json`
    - `model-00000` through `model-00014`
    - `model.safetensors.index.json`
    - `special_tokens_map.json`
    - `tokenizer.json`
    - `tokenizer_config.json`
- Important nuance:
  - tokenizer sidecars landed before all top-level model shards, so a simple "tokenizers exist" gate was not sufficient
  - the final launch gate for `GOSS-3` is now "all 22 serving files are present"
- Next action:
  - launch the central `v5p-8` local-serve smoke immediately against:
    - `gs://marin-us-central1/models/openai--gpt-oss-120b-vllm--b5c939d`

### ALIGN-200 - 2026-03-26 02:00 - Launched `GOSS-3` central `v5p-8` smoke against the completed minimal serving artifact

- Root job:
  - `/ahmed/goss3-gpt-oss-120b-vllm-smoke-central1`
- Constraints:
  - `--region us-central1`
  - `--zone us-central1-a`
- Script:
  - [gpt_oss_120b_vllm_smoke.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/gpt_oss_120b_vllm_smoke.py)
- Experiment name:
  - `goss3_gpt_oss_120b_vllm_smoke_central1`
- Input model path:
  - `gs://marin-us-central1/models/openai--gpt-oss-120b-vllm--b5c939d`
- Expected success condition:
  - one `v5p-8` worker boots `gpt-oss-120b`
  - `/v1/completions` returns a valid completion
  - `artifacts/vllm_metrics.json` is written

### ALIGN-201 - 2026-03-26 02:15 - `GOSS-3` failed on central `v5p-8` because the top-level `gpt-oss-120b` artifact resolves to MXFP4 weights that current TPU vLLM cannot initialize

- Failed root:
  - `/ahmed/goss3-gpt-oss-120b-vllm-smoke-central1`
- Failed child:
  - `/ahmed/goss3-gpt-oss-120b-vllm-smoke-central1/align-goss3_gpt_oss_120b_vllm_smoke_central1-smoke_85c27db2-9c4423a4`
- Authoritative failure signal from the child logs:
  - vLLM resolved:
    - `architecture = GptOssForCausalLM`
    - `quantization = mxfp4`
  - TPU runtime died during engine bring-up with:
    - `jax.errors.JaxRuntimeError: INVALID_ARGUMENT: Element type F4E2M1FN is not supported on TPU`
- Current staging state after the failure:
  - `GOSS-1` central full download is now complete enough to expose:
    - `gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d/original/model.safetensors.index.json`
  - `GOSS-2` east full download is still in progress; `original/` is not complete yet in `us-east5`
  - `GOSS-1b` central minimal-serving download succeeded
  - `GOSS-2b` east minimal-serving download is still running
- Interpretation:
  - the top-level staged artifact is not a valid TPU-serving input for `gpt-oss-120b`
  - the only remaining plausible TPU path is to retry from the full-download `original/` subtree with an explicit tokenizer path pointing at the parent root

### ALIGN-202 - 2026-03-26 02:15 - Added explicit tokenizer override support to local `vllm serve` and repointed the `gpt-oss-120b` experiment entrypoints at `original/`

- Code changes:
  - [inference_config.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/inference_config.py)
    - added `VLLMConfig.tokenizer`
  - [batched_vllm_serve.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/batched_vllm_serve.py)
    - local tokenizer loading now uses `config.tokenizer` when provided
    - `ModelConfig.engine_kwargs` now carries the tokenizer override
  - [vllm_server.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/inference/vllm_server.py)
    - `engine_kwargs["tokenizer"]` now becomes `--tokenizer ...` on the vLLM CLI
  - [llm_client.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/llm_client.py)
    - non-batched local vLLM also forwards `tokenizer`
- Experiment entrypoint changes:
  - [gpt_oss_120b_vllm_smoke.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/gpt_oss_120b_vllm_smoke.py)
  - [generate_prompts_gpt_oss_120b_refactored.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/generate_prompts_gpt_oss_120b_refactored.py)
  - [align_gpt_oss_120b_mixtral_rejected_full_spec.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py)
- New serving layout for `gpt-oss-120b`:
  - `model = <staged_root>/original`
  - `tokenizer = <staged_root>`
- Validation:
  - `./infra/pre-commit.py --fix ...` passed on the touched files
  - `uv run pytest tests/test_alignment.py -q` passed with `94 passed`
- Interpretation:
  - the codebase is now capable of testing the only credible TPU-serving retry path without inventing a flattened artifact

### ALIGN-203 - 2026-03-26 02:15 - Relaunch plan for `GOSS-3`: retry the central `v5p-8` smoke against `original/` immediately, then only proceed to `GOSS-4` if TPU bring-up succeeds

- Launch gate:
  - `GOSS-1` central full artifact is ready for `original/`-based serving
  - `GOSS-2` east full artifact is not ready yet, so the east retry is gated behind the central result
- Immediate next action:
  - submit a fresh central smoke using the patched [gpt_oss_120b_vllm_smoke.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/gpt_oss_120b_vllm_smoke.py)
  - if it boots, continue to:
    - `GOSS-4` east smoke when east `original/` completes
    - `GOSS-5` one-statement Stage 1/2/3
    - `GOSS-6` full-spec Stage 1/2/3
    - `GOSS-7` full open-weight alignment run
  - if it fails with the same low-precision TPU incompatibility, treat that as a hard blocker for `gpt-oss-120b` on the current `v5p-8` local-vLLM stack

### ALIGN-204 - 2026-03-26 02:16 - Launched the `original/`-based central `GOSS-3` retry on `v5p-8`

- Root:
  - `/ahmed/goss3b-gpt-oss-120b-vllm-smoke-central1-original`
- Constraints:
  - `--region us-central1`
  - `--zone us-central1-a`
- Script:
  - [gpt_oss_120b_vllm_smoke.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/gpt_oss_120b_vllm_smoke.py)
- Experiment name:
  - `goss3b_gpt_oss_120b_vllm_smoke_central1_original`
- Serving layout under test:
  - `model = gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d/original`
  - `tokenizer = gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d`
- Success condition:
  - the smoke child boots `gpt-oss-120b` on `v5p-8`
  - `/v1/completions` returns one valid response
  - `artifacts/vllm_metrics.json` is written

### ALIGN-205 - 2026-03-26 02:21 - The `original/`-based central retry failed before TPU math because vLLM could not recognize `original/config.json` as a Hugging Face `gpt_oss` config

- Failed root:
  - `/ahmed/goss3b-gpt-oss-120b-vllm-smoke-central1-original`
- Failed child:
  - `/ahmed/goss3b-gpt-oss-120b-vllm-smoke-central1-original/align-goss3b_gpt_oss_120b_vllm_smoke_central1_original-smoke_a8475d69-ab6d09d6`
- Important difference from `ALIGN-201`:
  - the retry did **not** die on the earlier TPU `F4E2M1FN` quantization error
  - it failed earlier in model config resolution
- Exact server launch under test:
  - `vllm serve gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d/original --tokenizer gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d --load-format runai_streamer ...`
- Exact failure signal from stderr:
  - `ValidationError: Unrecognized model in /root/.cache/vllm/assets/model_streamer/...`
- Local artifact diagnosis:
  - [README.md](/Users/ahmed/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/README.md) explicitly routes `original/*` through the separate `gpt_oss` reference package:
    - `python -m gpt_oss.chat model/`
  - the staged `original/model.safetensors.index.json` uses `block.*` keys rather than the top-level Transformers-style `model.layers.*` keys
  - `original/config.json` omits the HF architecture metadata that vLLM expects:
    - no `model_type`
    - no `architectures`
- Interpretation:
  - top-level checkpoint layout is vLLM-recognizable but TPU-quantization-incompatible
  - `original/` layout avoids that first failure mode but is not self-identifying enough for vLLM to treat it as `gpt_oss`

### ALIGN-206 - 2026-03-26 02:24 - Added `hf_overrides` support to local vLLM so the next central retry can force `original/` to resolve as `gpt_oss`

- Code changes:
  - [inference_config.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/inference_config.py)
    - added `VLLMConfig.hf_overrides`
    - added a custom `__hash__` so JSON-like overrides remain cache-safe
  - [batched_vllm_serve.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/batched_vllm_serve.py)
    - forwards `hf_overrides` into `ModelConfig.engine_kwargs`
  - [vllm_server.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/inference/vllm_server.py)
    - emits `--hf-overrides <json>`
  - [llm_client.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/llm_client.py)
    - forwards `hf_overrides` to direct local-vLLM construction too
- Experiment entrypoint changes:
  - [gpt_oss_120b_vllm_smoke.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/gpt_oss_120b_vllm_smoke.py)
  - [generate_prompts_gpt_oss_120b_refactored.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/generate_prompts_gpt_oss_120b_refactored.py)
  - [align_gpt_oss_120b_mixtral_rejected_full_spec.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py)
- Override payload now used for `gpt-oss-120b` on `original/`:
  - `{"model_type":"gpt_oss","architectures":["GptOssForCausalLM"]}`
- Validation:
  - `./infra/pre-commit.py --fix ...` passed
  - `uv run pytest tests/test_alignment.py -q` passed with `95 passed`
- Parallel staging note:
  - `GOSS-2b` east minimal-serving download has now succeeded
  - `GOSS-2` east full download is still running, so the east `original/` retry is still gated

### ALIGN-207 - 2026-03-26 02:25 - Launched the third central `GOSS-3` retry with `original/`, tokenizer override, and explicit `hf_overrides`

- Root:
  - `/ahmed/goss3c-gpt-oss-120b-vllm-smoke-central1-original-hfoverrides`
- Child:
  - `/ahmed/goss3c-gpt-oss-120b-vllm-smoke-central1-original-hfoverrides/align-goss3c_gpt_oss_120b_vllm_smoke_central1_original_hfoverrides-smoke_0bfbea65-02aa74f6`
- Serving layout under test:
  - `model = gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d/original`
  - `tokenizer = gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d`
  - `hf_overrides = {"model_type":"gpt_oss","architectures":["GptOssForCausalLM"]}`
- Early child logs confirm the patched CLI reached vLLM:
  - `--tokenizer gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d`
  - `--hf-overrides {"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss"}`
- Current status:
  - the first attempt did not return a model-level pass/fail yet
  - Iris marked the worker attempt as:
    - `TASK_STATE_WORKER_FAILED`
    - `Worker marin-tpu-v5p-8-us-central1-a-20260326-0916-ead38b95-worker-0 failed: Request timed out`
  - the child is now pending automatic retry:
    - `Retrying (attempt 1, last: task_state_worker_failed)`
- Interpretation:
  - the `hf_overrides` path remains unresolved
  - the latest interruption is infrastructure-level, not evidence for or against the model-layout fix

### ALIGN-208 - 2026-03-26 02:33 - `hf_overrides` did not rescue the `original/` layout; the third central `GOSS-3` retry failed with the same vLLM model-recognition error after a clean second attempt

- Final root:
  - `/ahmed/goss3c-gpt-oss-120b-vllm-smoke-central1-original-hfoverrides`
- Final child:
  - `/ahmed/goss3c-gpt-oss-120b-vllm-smoke-central1-original-hfoverrides/align-goss3c_gpt_oss_120b_vllm_smoke_central1_original_hfoverrides-smoke_0bfbea65-02aa74f6`
- Attempt history:
  - attempt `0`:
    - worker timeout on `marin-tpu-v5p-8-us-central1-a-20260326-0916-ead38b95-worker-0`
  - attempt `1`:
    - clean bring-up on `marin-tpu-v5p-8-us-central1-a-20260326-0928-0e54ffa0-worker-0`
    - reached the patched vLLM CLI with:
      - `--tokenizer gs://marin-us-central1/models/openai--gpt-oss-120b--b5c939d`
      - `--hf-overrides {"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss"}`
    - still failed with:
      - `ValidationError: Unrecognized model in /root/.cache/vllm/assets/model_streamer/4b2b46f8`
- Updated interpretation:
  - we have now exercised both available staged checkpoint layouts on central `v5p-8`:
    - top-level Transformers-format checkpoint:
      - vLLM recognizes it
      - TPU runtime rejects the low-precision weight format (`F4E2M1FN` / MXFP4 path)
    - `original/` reference checkpoint:
      - vLLM does not accept it as a serveable HF model layout under `runai_streamer`
      - explicit tokenizer override and `hf_overrides` do not change that
- Practical conclusion:
  - `gpt-oss-120b` is currently blocked on the Marin `v5p-8` local-vLLM stack
  - `GOSS-4` through `GOSS-7` cannot proceed on the current architecture without a deeper unblocking change, such as:
    - upstream TPU support for the top-level MXFP4 checkpoint in vLLM
    - a conversion path from the `original/` checkpoint into a vLLM-compatible TPU format
    - a different local inference backend that supports the `original/` checkpoint directly
- Parallel staging state at this checkpoint:
  - `GOSS-1` central full download: complete
  - `GOSS-1b` central minimal-serving download: complete
  - `GOSS-2b` east minimal-serving download: complete
  - `GOSS-2` east full download: still running; `original/` is not complete yet in `us-east5`

### ALIGN-209 - 2026-03-26 02:45 - Official support posture confirms the TPU path is blocked upstream, so the next GOSS attempt should pivot to GPU-backed `vllm` rather than more `v5p-8` retries

- External support evidence checked during this session:
  - OpenAI's official `gpt-oss` vLLM guide targets dedicated NVIDIA GPU servers and explicitly recommends:
    - `vllm==0.10.1+gptoss`
    - H100-class hardware
  - vLLM's official TPU hardware-supported-models page for `v0.11.1` does **not** list `GptOssForCausalLM` / `openai/gpt-oss-120b` under supported TPU models
- Combined interpretation with the live failures above:
  - the earlier `v5p-8` failures are not a Marin orchestration bug
  - they are consistent with the current upstream support matrix:
    - top-level checkpoint:
      - model recognized
      - TPU runtime rejects MXFP4 path
    - `original/` checkpoint:
      - reference runtime layout
      - not loadable by current local-vLLM path
- Execution decision:
  - preserve the overall `GOSS` goal, but pivot the active local-vLLM runtime from TPU-native to GPU-backed Docker `vllm`
  - keep the already-completed dual-region staging work; the GPU runs can use the completed top-level `gpt_oss_120b_vllm` artifacts in both regions

### ALIGN-210 - 2026-03-26 02:45 - Extended the alignment local-vLLM path to support GPU-backed execution and repointed the `gpt-oss-120b` experiments at the supported top-level staged artifact

- Code changes:
  - [inference_config.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/inference_config.py)
    - `VLLMConfig` now supports:
      - `gpu_type`
      - `gpu_count`
      - `cpu`
      - `serve_mode`
      - `docker_image`
    - added:
      - `pip_dependency_groups`
      - `resolved_serve_mode`
    - GPU-backed configs now resolve to:
      - GPU resources via `ResourceConfig.with_gpu(...)`
      - Docker-backed local vLLM by default
  - [batched_vllm_serve.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/batched_vllm_serve.py)
    - shared batched session now passes `resolved_serve_mode` and `docker_image` into `VllmEnvironment`
  - [align.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/align.py)
    - prompt/judge/response child jobs now derive pip dependency groups from the selected inference backend instead of assuming `["vllm", "tpu"]`
  - [llm_client.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/marin/src/marin/alignment/llm_client.py)
    - direct in-process vLLM construction now fails fast for Docker-backed configs, because that path is only valid for native local engines
- Experiment entrypoint changes:
  - [gpt_oss_120b_vllm_smoke.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/gpt_oss_120b_vllm_smoke.py)
  - [generate_prompts_gpt_oss_120b_refactored.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/generate_prompts_gpt_oss_120b_refactored.py)
  - [align_gpt_oss_120b_mixtral_rejected_full_spec.py](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py)
- New runtime layout for `gpt-oss-120b`:
  - use the top-level staged `gpt_oss_120b_vllm` artifact again
  - run it on GPU-backed `vllm`
  - stop forcing the TPU-only `original/` checkpoint path
- Validation:
  - `./infra/pre-commit.py --fix ...` passed
  - `uv run pytest tests/test_alignment.py -q` passed with `98 passed`
- Immediate next action:
  - launch:
    - `GOSS-3` central GPU smoke
    - `GOSS-4` east GPU smoke
  - if those succeed, proceed directly to:
    - `GOSS-5` one-statement Stage 1/2/3
    - `GOSS-6` full-spec Stage 1/2/3
    - `GOSS-7` full open-weight pipeline

### ALIGN-211 - 2026-03-26 02:46 - Launched the GPU-backed `gpt-oss-120b` smoke jobs in both staged regions to advance `GOSS-3` and `GOSS-4` in parallel

- `GOSS-3` central GPU smoke:
  - root:
    - `/ahmed/goss3d-gpt-oss-120b-vllm-smoke-central1-h100`
  - command:
    - `python experiments/gpt_oss_120b_vllm_smoke.py --prefix gs://marin-us-central1 --name goss3d_gpt_oss_120b_vllm_smoke_central1_h100 --gpu-type H100 --gpu-count 1`
  - staged model root under test:
    - `gs://marin-us-central1/models/openai--gpt-oss-120b-vllm--b5c939d`
- `GOSS-4` east GPU smoke:
  - root:
    - `/ahmed/goss4-gpt-oss-120b-vllm-smoke-east5-h100`
  - command:
    - `python experiments/gpt_oss_120b_vllm_smoke.py --prefix gs://marin-us-east5 --name goss4_gpt_oss_120b_vllm_smoke_east5_h100 --gpu-type H100 --gpu-count 1`
  - staged model root under test:
    - `gs://marin-us-east5/models/openai--gpt-oss-120b-vllm--b5c939d`
- Shared success condition:
  - child smoke job lands on H100
  - GPU-backed `vllm serve` boots the staged top-level checkpoint
  - `/v1/completions` returns one valid completion
  - `artifacts/vllm_metrics.json` is written

### ALIGN-212 - 2026-03-26 02:49 - The current local Iris controller has no GPU scale groups, and the checked-in CoreWeave H100 config is not usable from this machine because the kubeconfig is missing

- What the local controller reported:
  - [cluster status] showed zero GPU scale groups entirely; only CPU + TPU groups are configured on the active controller
  - both just-launched smoke children stayed `JOB_STATE_PENDING` with:
    - `Autoscaler: Unsatisfied autoscaler demand: no_matching_group: no groups with device gpu:h100`
- What I checked next:
  - [lib/iris/examples/coreweave.yaml](/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/lib/iris/examples/coreweave.yaml)
    - this repo does include a separate H100-backed CoreWeave Iris configuration
  - but local access is blocked:
    - `uv run iris --config lib/iris/examples/coreweave.yaml cluster status`
    - failed to establish the controller tunnel because:
      - `~/.kube/coreweave-iris` does not exist on this machine
    - environment variables for the CoreWeave path are also absent:
      - no `KUBECONFIG`
      - no `R2_ACCESS_KEY_ID`
      - no `R2_SECRET_ACCESS_KEY`
- Cleanup action:
  - stopped the two doomed pending H100 roots:
    - `/ahmed/goss3d-gpt-oss-120b-vllm-smoke-central1-h100`
    - `/ahmed/goss4-gpt-oss-120b-vllm-smoke-east5-h100`
- Updated execution state:
  - `GOSS-1` central full staging: complete
  - `GOSS-1b` central minimal-serving staging: complete
  - `GOSS-2b` east minimal-serving staging: complete
  - `GOSS-2` east full staging: still running
  - `GOSS-3` through `GOSS-7`: blocked in the current environment because:
    - TPU path is unsupported for `gpt-oss-120b` on the current local-vLLM stack
    - the active controller has no GPU groups
    - the available repo GPU cluster config is inaccessible from this workstation
- Practical unblock requirements from here:
  - either:
    - access to a GPU-backed Iris cluster config from this machine
  - or:
    - a new TPU-compatible `gpt-oss-120b` local-vLLM path that does not exist today

### ALIGN-213 - 2026-03-26 02:52 - The BF16 path is real, but it is a GPU path rather than a TPU fallback, so it does not unblock GOSS execution from the current machine by itself

- What I checked after the new direction "`use bf16` / `don't use TPU`":
  - the official `openai/gpt-oss-120b` model card and README
  - the staged `original/dtypes.json`
- Confirmed facts:
  - the reference `original/*` checkpoint is BF16-based:
    - [original/dtypes.json](/Users/ahmed/.cache/huggingface/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/original/dtypes.json)
    - sample entries are `BF16`
  - the published top-level serving checkpoint remains MXFP4-based for single-GPU inference
  - the official guidance for the BF16 reference path is still GPU-centric:
    - `gpt-oss-120b` "fits into a single 80GB GPU" in the top-level quantized path
    - the reference PyTorch implementation described in the `gpt-oss` repo upcasts weights to BF16 and targets multi-H100-class setups
- Local environment check:
  - this workstation only has the built-in Apple GPU
  - there is still no reachable Iris GPU cluster from this machine:
    - local controller: no GPU scale groups
    - CoreWeave config: inaccessible because `~/.kube/coreweave-iris` is absent
- Updated conclusion:
  - "`use bf16`" is the correct runtime direction for the reference checkpoint
  - but it still requires external H100-class GPU access
  - therefore `GOSS-3` through `GOSS-7` remain blocked from this workstation until GPU-cluster access is available

### ALIGN-214 - 2026-03-26 05:29 - Found a viable TPU path in our `tpu-inference` fork: route GPT-OSS to the JAX bootstrap path and skip TPU-side re-quantization so MXFP4 weights dequantize into BF16 at load time

- What I found in `/Users/ahmed/code/tpu-inference`:
  - the fork already contains:
    - `tpu_inference/models/jax/gpt_oss.py`
    - GPT-OSS registration in `tpu_inference/models/common/model_loader.py`
    - GPT-OSS Qwix defaults in `tpu_inference/models/jax/utils/qwix/qwix_utils.py`
  - the blocker was not "missing model code"; it was loader policy:
    - GPT-OSS defaulted to the vLLM/Torch wrapper path
    - `abstract_load` refused any `hf_config.quantization_config`
    - GPT-OSS MXFP4 checkpoints therefore could not opt into the JAX bootstrap path
- Why this matters on `v5p-8`:
  - the TPU quantization/Qwix path uses `float4_e2m1fn`, and the quantization tests gate MXFP4 support to TPU v7+
  - but the JAX GPT-OSS loader already knows how to decode MXFP4 checkpoint tensors and dequantize them to BF16 when the target params are regular tensors instead of QArrays
- Implemented in the fork:
  - created branch:
    - `ahmed/gpt-oss-tpu-bringup`
  - committed and pushed:
    - `7fedd109` — `Enable GPT-OSS JAX bootstrap on TPU`
  - code changes:
    - `GptOss` is now eligible for abstract bootstrap routing
    - `GptOssForCausalLM` is allowed to reroute from `MODEL_IMPL_TYPE=auto` to the JAX path when `prefer_jax_for_bootstrap=true`
    - `abstract_load` now allows `hf_config.quantization_config={"quant_method":"mxfp4"}` specifically for GPT-OSS when `skip_quantization=true`
- Validation on this workstation:
  - `python3 -m py_compile ...` passed for the changed fork files
  - full fork unit tests could not be run locally because the current workstation env here does not have `torch` / `vllm` installed

### ALIGN-215 - 2026-03-26 05:34 - Wired Marin so TPU alignment jobs can install the patched `tpu-inference` branch and pass TPU-specific GPT-OSS bootstrap config into `vllm serve`

- Marin changes:
  - `lib/marin/src/marin/execution/remote.py`
    - added `pip_packages` passthrough for remote Fray jobs
  - `lib/marin/src/marin/alignment/inference_config.py`
    - `VLLMConfig` now supports:
      - `additional_config`
      - `model_impl_type`
      - `extra_pip_packages`
  - `lib/marin/src/marin/alignment/batched_vllm_serve.py`
    - forwards `additional_config` into the vLLM model config
    - forwards `model_impl_type` into `VllmEnvironment` env overrides
  - `lib/marin/src/marin/inference/vllm_server.py`
    - supports `--additional-config`
    - supports explicit env overrides for native/docker TPU startup
  - `lib/marin/src/marin/alignment/align.py`
    - now forwards per-model `pip_packages` into remote prompt/response/judge steps
  - `experiments/gpt_oss_120b_vllm_smoke.py`
    - repointed from the dead GPU path to a TPU/JAX-bootstrap smoke
- TPU smoke config now in the experiment:
  - TPU type: `v5p-8`
  - tensor parallel size: `8`
  - model artifact: staged top-level `gpt_oss_120b_vllm`
  - `MODEL_IMPL_TYPE=auto`
  - `additional_config`:
    - `skip_quantization=true`
    - `tpu_bootstrap.model_bootstrap=abstract_load`
    - `tpu_bootstrap.prefer_jax_for_bootstrap=true`
    - `tpu_bootstrap.weight_loader=fsspec_streamer`
  - extra pip package installed in the remote job:
    - `tpu_inference @ git+https://github.com/marin-community/tpu-inference.git@ahmed/gpt-oss-tpu-bringup`
- Validation:
  - `uv run pytest tests/test_alignment.py -q` -> `98 passed`
  - targeted `./infra/pre-commit.py --fix ...` -> OK
- Next action:
  - launch the TPU smoke and use that as the first real signal for `GOSS-3`

### ALIGN-216 - 2026-03-26 05:31 - Launched the first TPU GPT-OSS smoke on `v5p-8`; it is past capacity wait and building on a TPU worker

- Launched root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax`
- Root behavior:
  - cache-hit the staged top-level GPT-OSS artifact:
    - `models/openai--gpt-oss-120b-vllm--b5c939d`
  - launched the smoke child with output:
    - `gs://marin-us-central1/align/debug_gpt_oss_120b_vllm_smoke_tpu/smoke-bdf5c7`
- Child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax/align-debug_gpt_oss_120b_vllm_smoke_tpu-smoke_a11df351-04621696`
  - requested:
    - `v5p-8`
    - `32 CPU`
    - `256 GiB RAM`
    - `100 GiB disk`
  - current controller state:
    - `JOB_STATE_RUNNING`
    - task state `building`
    - `failure_count=0`
    - `preemption_count=0`
- Interpretation:
  - this is the first concrete sign that the GPT-OSS TPU/JAX bootstrap path is being exercised on a real TPU worker
  - the remaining critical signal is child-side runtime logs:
    - branch install of `tpu_inference`
    - `MODEL_IMPL_TYPE=auto` path selection
    - `vllm serve` bootstrap outcome

### ALIGN-217 - 2026-03-26 05:35 - First GPT-OSS TPU smoke failed at TPU mesh creation because the `v5p-8` slice only exposed 4 JAX devices

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax`
- Failed child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax/align-debug_gpt_oss_120b_vllm_smoke_tpu-smoke_a11df351-04621696`
- Important positive signal:
  - the patched TPU/JAX bootstrap path was exercised successfully enough to:
    - resolve `GptOssForCausalLM`
    - enter `tpu_inference`
    - start `TPUModelRunner`
- Terminal error:
  - `ValueError: Number of devices 4 must be >= the product of mesh_shape (1, 8)`
- Interpretation:
  - this was a slice-layout mismatch, not a GPT-OSS format incompatibility
  - `v5p-8` on this stack presents 4 JAX devices, so `tensor_parallel_size=8` was invalid
- Action taken:
  - lowered GPT-OSS TPU defaults to `tensor_parallel_size=4`
  - kept the TPU/JAX bootstrap configuration unchanged otherwise

### ALIGN-218 - 2026-03-26 05:35 - Relaunched the GPT-OSS TPU smoke with `tensor_parallel_size=4`

- Launched root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax-tp4`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-jax-tp4 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_tp4`
- Code changes used by this relaunch:
  - `experiments/gpt_oss_120b_tpu.py`: default `tensor_parallel_size=4`
  - `experiments/gpt_oss_120b_vllm_smoke.py`: default CLI `--tensor-parallel-size=4`
- Immediate next signal:
  - whether `vllm serve` can complete engine initialization on TPU once the mesh shape matches the slice

### ALIGN-219 - 2026-03-26 05:38 - `us-central1` TPU capacity remained blocked, so I opened an east-region hedge using the east-staged GPT-OSS artifact

- Observed state for `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax-tp4`:
  - child remained `JOB_STATE_PENDING`
  - scheduler reason: waiting for `tpu_v5p_8-us-central1-a`
- Verified staged artifacts exist in both regions:
  - `gs://marin-us-central1/models/openai--gpt-oss-120b-vllm--b5c939d`
  - `gs://marin-us-east5/models/openai--gpt-oss-120b-vllm--b5c939d`
- Decision:
  - launch the same TPU smoke in `us-east5` with `--prefix gs://marin-us-east5` so the worker reads the east-local GPT-OSS weights instead of cross-region

### ALIGN-220 - 2026-03-26 05:40 - The corrected `us-central1` GPT-OSS TPU smoke reached real `vllm serve` startup; the east hedge is still only a capacity hedge

- `us-central1` root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax-tp4`
- Live child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax-tp4/align-debug_gpt_oss_120b_vllm_smoke_tpu_tp4-smoke_032d9627-8c3419b5`
- Current child status:
  - `JOB_STATE_RUNNING`
  - worker logs reached:
    - `marin.inference.vllm_server Starting vLLM environment`
    - `marin.inference.vllm_server Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
- `us-east5` hedge:
  - root `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax-tp4-east`
  - child `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax-tp4-east/align-debug_gpt_oss_120b_vllm_smoke_tpu_tp4_east-smoke_7d3293b8-ed88000f`
  - still `JOB_STATE_PENDING` on `tpu_v5p_8-us-east5-a`
- Interpretation:
  - central is now the primary signal
  - the next critical result is whether engine initialization succeeds after the corrected `(1, 4)` mesh

### ALIGN-221 - 2026-03-26 05:42 - The corrected mesh was not enough; GPT-OSS still resolved to the `vllm` wrapper path and hit unsupported MXFP4 TPU post-processing

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax-tp4`
- Failed child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax-tp4/align-debug_gpt_oss_120b_vllm_smoke_tpu_tp4-smoke_032d9627-8c3419b5`
- New decisive evidence:
  - the mesh mismatch is gone
  - the child successfully reached model loading with:
    - `Init mesh | mesh=Mesh('data': 1, 'model': 4, ...)`
  - but then logged:
    - `Loading model with MODEL_IMPL_TYPE=auto ...`
    - `Resolved MODEL_IMPL_TYPE 'auto' to 'vllm'`
- Terminal error:
  - `jax.errors.JaxRuntimeError: INVALID_ARGUMENT: Element type F4E2M1FN is not supported on TPU.`
  - traceback ended in:
    - `tpu_inference/layers/vllm/quantization/mxfp4.py`
- Interpretation:
  - `skip_quantization=True` alone is insufficient if GPT-OSS still goes through the vLLM wrapper load path
  - for bring-up, the safest move is to force `MODEL_IMPL_TYPE=flax_nnx` explicitly so GPT-OSS uses the JAX BF16 bootstrap path and bypasses vLLM MXFP4 post-load processing entirely
- Action taken:
  - changed the shared Marin GPT-OSS TPU helper to set `model_impl_type=\"flax_nnx\"`

### ALIGN-222 - 2026-03-26 05:43 - Relaunched the GPT-OSS TPU smoke with `MODEL_IMPL_TYPE=flax_nnx`

- Stopped stale east hedge that still carried the old `auto` routing config:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-jax-tp4-east`
- Fresh root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4`
- Intent:
  - force the GPT-OSS TPU smoke through the JAX `flax_nnx` loader path
  - bypass the vLLM MXFP4 post-load quantization hook that triggered `F4E2M1FN is not supported on TPU`

### ALIGN-223 - 2026-03-26 05:46 - The `flax_nnx` attempt proved the JAX loader route is viable, but the worker likely reused a stale `tpu_inference` branch snapshot

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4`
- Important positive signal:
  - the worker logged:
    - `Loading model with MODEL_IMPL_TYPE=flax_nnx`
    - `Architecture lookup: ... 'GptOssForCausalLM'`
  - so the explicit model-impl override worked
- Terminal error:
  - `ValueError: abstract_load is not supported for architecture 'GptOss'`
- Interpretation:
  - local `tpu_inference` source already includes `GptOss` in `_ABSTRACT_BOOTSTRAP_ARCHITECTURES`
  - the worker behavior matches an older package snapshot, not the pushed `7fedd109` commit
  - the most likely cause is pip reusing a cached branch install from `git+...@ahmed/gpt-oss-tpu-bringup`
- Action taken:
  - pinned the worker package URL to the exact pushed commit:
    - `7fedd1097c2384140d5dbcf6ea65e7c622ca9df1`
  - next smoke should force a fresh install of the intended `tpu_inference` code

### ALIGN-224 - 2026-03-26 05:46 - Relaunched the forced-`flax_nnx` smoke with the exact `tpu_inference` SHA pinned

- Fresh root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-sha`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-sha --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_sha`
- Expected discriminant:
  - if the worker now loads the intended `tpu_inference` snapshot, `abstract_load` should be accepted for `GptOss`
  - if it still errors with `abstract_load is not supported for architecture 'GptOss'`, the issue is not pip-branch staleness

### ALIGN-225 - 2026-03-26 05:48 - The raw-SHA package URL failed to build, so I published the same `tpu_inference` snapshot under a fresh branch name for cache-busting

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-sha`
- Build failure:
  - worker could not resolve raw git revision `7fedd1097c2384140d5dbcf6ea65e7c622ca9df1` during pip install
- New cache-busting branch:
  - `ahmed/gpt-oss-tpu-bringup-v2`
  - points to commit `7fedd1097c2384140d5dbcf6ea65e7c622ca9df1`
- Action taken:
  - repointed Marin worker package install to:
    - `git+https://github.com/marin-community/tpu-inference.git@ahmed/gpt-oss-tpu-bringup-v2`
  - this preserves the intended `tpu_inference` code while forcing a different package source URL than the original stale branch install

### ALIGN-226 - 2026-03-26 05:56 - The worker is now on the intended JAX GPT-OSS path; the remaining blocker was the TPU-quantization bootstrap guard itself

- What changed in `tpu-inference`:
  - added `_ABSTRACT_BOOTSTRAP_TPU_QUANTIZATION_ALLOWLIST`
  - allowed `GptOss` `abstract_load` to proceed when:
    - `skip_quantization=True`
    - TPU quantization metadata is MXFP4 / `tpu-mxfp4`
- Files changed in `/Users/ahmed/code/tpu-inference`:
  - `tpu_inference/models/common/model_loader.py`
  - `tests/models/common/test_model_loader.py`
- Commit:
  - `31af4ddb` — `Allow GPT-OSS TPU bootstrap with skip-quantization`
- New cache-busting worker branch:
  - `ahmed/gpt-oss-tpu-bringup-v3`
- Validation:
  - `python3 -m py_compile tpu_inference/models/common/model_loader.py tests/models/common/test_model_loader.py`
  - local full pytest for `tpu-inference` was blocked by missing `torch` in the available local env

### ALIGN-227 - 2026-03-26 05:56 - Repointed Marin to the new `tpu_inference` branch and relaunched the GPT-OSS TPU smoke

- Updated Marin helper:
  - `experiments/gpt_oss_120b_tpu.py`
  - worker package URL now points to:
    - `git+https://github.com/marin-community/tpu-inference.git@ahmed/gpt-oss-tpu-bringup-v3`
- Fresh root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v3`
- Fresh child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v3/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v3-smoke_f9592b7c-487695e3`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v3 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v3 --tensor-parallel-size 4`
- Live status at log time:
  - root `JOB_STATE_RUNNING`
  - child `JOB_STATE_RUNNING`
  - no preemptions
  - child has reached:
    - `marin.inference.vllm_server Starting vLLM environment`
    - `marin.inference.vllm_server Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
- Next discriminant:
  - whether the child gets past the old `abstract_load is incompatible with TPU quantization` failure and proceeds into real GPT-OSS abstract-load / mesh initialization

### ALIGN-228 - 2026-03-26 05:58 - The `v3` smoke got much deeper into GPT-OSS TPU init, but the worker still imported an older cached `tpu_inference` wheel

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v3`
- Failed child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v3/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v3-smoke_f9592b7c-487695e3`
- Important positive signals:
  - reached vLLM API server start
  - resolved GPT-OSS architecture
  - initialized TPU mesh:
    - `Mesh('data': 1, 'model': 4, ...)`
  - forced `MODEL_IMPL_TYPE=flax_nnx`
  - entered `get_flax_model(...)`
- Terminal error:
  - `ValueError: abstract_load is incompatible with TPU quantization`
- Key diagnosis:
  - the worker traceback reported the raise from `tpu_inference/models/common/model_loader.py` line `142`
  - in the patched local branch, that TPU-quantization raise had already moved to line `148`
  - so the worker was still importing an older cached `tpu_inference` wheel even though the job used the new branch URL
- New blocker:
  - worker package override / refresh semantics, not the GPT-OSS JAX loader logic itself

### ALIGN-229 - 2026-03-26 06:01 - Patched Iris worker env setup to force-refresh direct `tpu_inference` git refs, then relaunched the TPU smoke as `v4`

- Iris runtime change:
  - `lib/iris/src/iris/cluster/runtime/entrypoint.py`
  - when `pip_packages` includes a direct `tpu_inference @ git+...` ref, worker setup now adds:
    - `--refresh-package tpu_inference`
    - `--reinstall-package tpu_inference`
- Test:
  - `uv run pytest lib/iris/tests/cluster/runtime/test_entrypoint.py -q`
  - `9 passed`
- Formatting / checks:
  - `./infra/pre-commit.py --fix lib/iris/src/iris/cluster/runtime/entrypoint.py lib/iris/tests/cluster/runtime/test_entrypoint.py experiments/gpt_oss_120b_tpu.py`
- Fresh root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v4`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v4 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v4 --tensor-parallel-size 4`
- Expected discriminant:
  - if the reinstall/refresh fix works, the worker should finally import the `31af4ddb` `tpu_inference` loader and clear the old TPU-quantization bootstrap guard

### ALIGN-230 - 2026-03-26 06:04 - The Iris reinstall/refresh flags were not enough; the worker still imported the stale `site-packages` `tpu_inference`

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v4`
- Failed child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v4/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v4-smoke_06b0ffe3-41ffcd96`
- Important result:
  - even after adding `--refresh-package tpu_inference --reinstall-package tpu_inference`, the runtime still failed at:
    - `tpu_inference/models/common/model_loader.py`, line `142`
  - that is still the old pre-patch line number for the TPU-quantization raise
- Interpretation:
  - worker package refresh/reinstall is still not enough to force the patched `tpu_inference` loader to win over the bundled TPU wheel
  - continuing to fight the pip layer is likely wasted effort
- New plan:
  - vendor the patched `tpu_inference/` source tree directly into the Marin workspace so `/app/tpu_inference/...` shadows `site-packages` on `sys.path`

### ALIGN-231 - 2026-03-26 06:06 - Vendored `tpu_inference/` into the Marin workspace and launched a fresh `v5` smoke that bundles the patched source directly

- Workspace overlay:
  - copied `/Users/ahmed/code/tpu-inference/tpu_inference/` into:
    - `/Users/ahmed/code/marin/.claude/worktrees/gentle-twirling-avalanche/tpu_inference/`
  - bundle size increased from `5.3 MB` to `5.7 MB`, confirming the source overlay is included in the Iris bundle
- Patch location in workspace overlay:
  - `tpu_inference/models/common/model_loader.py`
  - TPU quantization guard now sits at local lines `145-149`
- Fresh root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v5`
- Fresh child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v5/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v5-smoke_3a84a567-3b7cb9f3`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v5 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v5 --tensor-parallel-size 4`
- Live status at log time:
  - root `JOB_STATE_RUNNING`
  - child `JOB_STATE_PENDING`
  - pending on `tpu_v5p_8-us-central1-a` capacity
- Next discriminant:
  - once scheduled, the child traceback should stop referring to stale `site-packages` line `142` if the workspace overlay truly wins on import order

### ALIGN-232 - 2026-03-26 06:18 - `v5` proved the vendored overlay works; the next GPT-OSS TPU blocker is streamed-weight consumption in the JAX loader

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v5`
- Failed child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v5/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v5-smoke_3a84a567-3b7cb9f3`
- Important result:
  - imports now came from `/app/tpu_inference/...`, so the workspace overlay won
  - GPT-OSS got through:
    - `Loading model with MODEL_IMPL_TYPE=flax_nnx`
    - `Abstract load bootstrap for GptOss`
    - `[phase] abstract_load: eval_shape 1.0s`
  - the new failure was during `model.load_weights(rng)`:
    - `RuntimeError: Cannot find any *.safetensors files in /root/.cache/vllm/assets/model_streamer/273f77cf.`
- Diagnosis:
  - `model_loader._build_abstract_model_and_load_weights(...)` was already attaching `model_weights_iterator`
  - GPT-OSS JAX loading did not consume it; it still called `model_weights_generator(model_name_or_path=...)` and fell back to local file discovery
  - under `fsspec_streamer`, weights arrive as streamed `(name, jax.Array)` pairs, so GPT-OSS also needs to tolerate `jax.Array` tensors instead of assuming only `torch.Tensor`
- Source patch prepared:
  - real repo: `/Users/ahmed/code/tpu-inference/`
    - `tpu_inference/models/jax/utils/weight_utils.py`
    - `tpu_inference/models/jax/gpt_oss.py`
    - `tests/models/jax/test_weight_loading.py`
  - workspace overlay synced with the same loader changes:
    - `tpu_inference/models/jax/utils/weight_utils.py`
    - `tpu_inference/models/jax/gpt_oss.py`
- Exact loader change:
  - `model_weights_generator(...)` now accepts an optional `weights_iterator`
  - GPT-OSS now prefers `model_config.model_weights_iterator` when present
  - GPT-OSS weight conversion now accepts streamed `jax.Array` tensors as well as file-backed `torch.Tensor` tensors
- Validation:
  - `python3 -m py_compile` passed for the patched source files and the new focused test
  - focused pytest from the Marin env was blocked by missing local `vllm` imports, so live TPU smoke remains the decisive validation
- Next action:
  - push a fresh `tpu-inference` branch `ahmed/gpt-oss-tpu-bringup-v4`
  - repoint `experiments/gpt_oss_120b_tpu.py`
  - launch a fresh `v6` TPU smoke against the synced workspace overlay

### ALIGN-233 - 2026-03-26 06:21 - Pushed the streamed-weight fix as `ahmed/gpt-oss-tpu-bringup-v4` and prepared a fresh `v6` TPU smoke

- Source repo:
  - `/Users/ahmed/code/tpu-inference`
- Commit:
  - `384f37289117f2cffde341281d2b580ca7fa59fe` — `Support GPT-OSS streamed TPU weight loading`
- Remote branch:
  - `ahmed/gpt-oss-tpu-bringup-v4`
- Marin experiment config updated:
  - `experiments/gpt_oss_120b_tpu.py`
  - `GPT_OSS_TPU_INFERENCE_PACKAGE = git+...@ahmed/gpt-oss-tpu-bringup-v4`
- Workspace overlay still contains the same loader patch, so the live worker should be correct even if pip caching tries to reappear
- Next action:
  - submit `v6`:
    - root name: `gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v6`
    - experiment name: `debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v6`
  - babysit until it either clears weight loading or exposes the next real GPT-OSS TPU incompatibility

### ALIGN-234 - 2026-03-26 06:22 - Launched `v6`; the new GPT-OSS TPU smoke is waiting only on `v5p-8` capacity

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v6`
- Child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v6/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v6-smoke_dca862f8-4d1a6cd2`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v6 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v6 --tensor-parallel-size 4`
- Current status:
  - root `JOB_STATE_RUNNING`
  - smoke child `JOB_STATE_PENDING`
- Pending reason:
  - `Scheduler: Insufficient TPUs (need 4, available 0)`
  - autoscaler waiting on scale group `tpu_v5p_8-us-central1-a`
- Important status note:
  - the staged top-level GPT-OSS artifact was cache-hit and skipped successfully
  - once the child schedules, the next discriminant is whether GPT-OSS logs:
    - `Loading GPT-OSS weights from model_weights_iterator`
  - and clears the old `Cannot find any *.safetensors files ... model_streamer/...` failure

### ALIGN-235 - 2026-03-26 09:49 - `v6` scheduled and failed later than `v5`; the old streamed-JAX loader error is gone, but vLLM now dies in safetensors metadata lookup for the local model-streamer cache path

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v6`
- Failed child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v6/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v6-smoke_dca862f8-4d1a6cd2`
- Runtime:
  - queued for TPU, then ran for about `7m46s`
- Important progress:
  - the child got past the previous `Cannot find any *.safetensors files in /root/.cache/vllm/assets/model_streamer/...` failure
  - `vllm serve` resolved architecture `GptOssForCausalLM`
  - TPU engine startup progressed into `EngineCore_DP0`
- New blocker from `iris job bug-report`:
  - API-server preflight tries to retrieve safetensors metadata using the local model-streamer cache dir as though it were a Hugging Face repo id:
    - `Error retrieving safetensors: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/root/.cache/vllm/assets/model_streamer/273f77cf'`
  - then the engine dies with:
    - `RuntimeError: Engine core initialization failed.`
- Interpretation:
  - the GPT-OSS JAX streamed-weight patch is effective enough to remove the old loader/file-glob failure
  - the next failure is earlier in vLLM's model-introspection / safetensors-metadata path, before the iterator-based GPT-OSS load can finish
  - likely next patch surface is in Marin's TPU vLLM launch path or upstream vLLM's safetensors inspection, not the JAX GPT-OSS loader itself
- Next action:
  - inspect the vLLM / Marin server startup path that triggers `repo_utils.py` on the local model-streamer cache directory
  - either bypass safetensors metadata probing for this TPU bootstrap path or point it at a canonical repo id/model_weights path instead of `/root/.cache/vllm/assets/model_streamer/...`

### ALIGN-236 - 2026-03-26 09:56 - Implemented a Marin-side workaround: stage remote model dirs to local worker disk before launching `vllm serve`, and enabled it for GPT-OSS TPU

- Code changes:
  - `lib/marin/src/marin/alignment/inference_config.py`
    - added `VLLMConfig.stage_remote_model_locally: bool`
  - `lib/marin/src/marin/alignment/batched_vllm_serve.py`
    - forwards the new flag into `ModelConfig.engine_kwargs`
  - `lib/marin/src/marin/inference/vllm_server.py`
    - added remote-dir staging helper using `url_to_fs(...).find(...)` + `fs.get(...)`
    - `VllmEnvironment` now:
      - stages remote model dirs to local temp storage when requested
      - stages remote tokenizer dirs too when explicitly configured
      - drops `runai_streamer` once launching from the staged local path
      - cleans staged temp dirs on close
  - `experiments/gpt_oss_120b_tpu.py`
    - `stage_remote_model_locally=True`
- Tests:
  - `uv run pytest tests/test_vllm_server.py tests/test_alignment.py -q`
  - `101 passed`
  - `./infra/pre-commit.py --fix ...`
  - passed
- Interpretation:
  - instead of patching upstream vLLM safetensors metadata probing directly, the smoke will now launch against a real local checkpoint directory
  - this should avoid both:
    - the HF repo-id confusion on `/root/.cache/vllm/assets/model_streamer/...`
    - the model-streamer local-cache incompatibility we saw in earlier failures
- Next action:
  - relaunch as `v7`
  - inspect whether local staging gets GPT-OSS TPU through engine-core model loading

### ALIGN-237 - 2026-03-26 10:02 - Launched `v7` with local model staging enabled; the child is currently running on `v5p-8`

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v7`
- Child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v7/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v7-smoke_b8411830-fd363e59`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v7 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v7 --tensor-parallel-size 4`
- Current state:
  - root `JOB_STATE_RUNNING`
  - child `JOB_STATE_RUNNING`
- Early runtime signal:
  - the child has progressed past environment boot and dependency setup
  - no immediate startup failure yet
  - the most likely current phase is remote-model copy to local disk before `vllm serve` startup, since no `Starting vLLM environment` line has appeared yet
- Next discriminant:
  - first child log lines after local staging completes
  - then whether the engine starts from a real local model path instead of `runai_streamer`

### ALIGN-238 - 2026-03-26 10:03 - `v7` has not reproduced the old safetensors/model-streamer failure yet, but the child has been preempted once and then hit worker timeouts twice before reaching `vllm` startup

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v7`
- Child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v7/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v7-smoke_b8411830-fd363e59`
- Current state from `iris job list --json --prefix ...`:
  - root `JOB_STATE_RUNNING`
  - child `JOB_STATE_RUNNING`
  - child `preemption_count = 1`
  - child task is currently `pending`
  - pending reason from bug report:
    - `Retrying (attempt 2, last: task_state_worker_failed)`
- Last concrete task failures from `iris job bug-report`:
  - attempt `0`:
    - worker `marin-tpu-v5p-8-us-central1-a-20260326-1608-5c6aadbc-worker-0`
    - `Worker ... failed: Request timed out`
  - attempt `1`:
    - worker `marin-tpu-v5p-8-us-central1-a-20260326-1500-3adeabba-worker-0`
    - `Worker ... failed: Request timed out`
- Observed child logs before the timeout:
  - only worker bootstrap / dependency setup / datasets-import lines
  - still no `Starting vLLM environment`
  - still no evidence yet that the old `/root/.cache/vllm/assets/model_streamer/...` safetensors failure reproduced
- Interpretation:
  - `v7` is now blocked earlier than `v6`, at the worker/runtime layer rather than a deterministic GPT-OSS loader exception
  - this may be TPU worker instability or a long blocking phase before the first `vllm` startup log, likely related to the new remote-directory local staging path
  - the local-staging workaround has not been disproven yet; the run has simply not progressed far enough to exercise it
- Next action:
  - keep babysitting the child until attempt `2` either reaches `vllm` startup or fails again
  - if repeated worker timeouts continue before `Starting vLLM environment`, instrument the staging path with earlier log lines so the next retry reveals where startup is hanging

### ALIGN-239 - 2026-03-26 10:07 - Measured the staged GPT-OSS artifact and patched the local-staging path for earlier visibility plus parallel shard copy; `v7` itself is now stale because the root was preempted and restarted with the old bundle

- Measured staged model artifact:
  - `gs://marin-us-central1/models/openai--gpt-oss-120b-vllm--b5c939d`
  - `48` files total
  - about `60.79 GiB`
  - core weights are `14` safetensor shards, mostly `3.8-4.3 GiB` each
- Interpretation:
  - `100 GiB` child disk is not the immediate blocker
  - the more plausible failure mode is the current serial `fs.get(...)` staging path spending too long before any useful log line, which looks like a worker timeout from Iris
- Code change:
  - `lib/marin/src/marin/inference/vllm_server.py`
    - added early staging logs:
      - before remote listing
      - after file discovery with total size
      - during copy progress
      - after local staging completes
    - changed local staging from serial copy to parallel shard copy with `ThreadPoolExecutor(max_workers=4)`
    - progress logs now report copied-file count and GiB copied
- Validation:
  - `./infra/pre-commit.py --fix lib/marin/src/marin/inference/vllm_server.py tests/test_vllm_server.py`
  - passed
  - `uv run pytest tests/test_vllm_server.py -q`
  - `3 passed`
- Updated runtime state for `v7`:
  - child is now `JOB_STATE_KILLED` with `Parent task preempted`
  - root is still `JOB_STATE_RUNNING` but rebuilding / restarting
  - because the root restarted from the previous submitted bundle, `v7` does **not** include the new parallel-staging patch
- Next action:
  - stop the stale `v7` root and relaunch `v8` from the patched workspace
  - inspect whether the new staging progress logs appear before `vllm` startup, and whether the worker-timeout failure disappears

### ALIGN-240 - 2026-03-26 10:08 - Stopped stale `v7` and launched fresh `v8` from the patched workspace

- Stopped stale run:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v7`
  - terminated child:
    - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v7/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v7-smoke_b8411830-a3fcbac3`
- Fresh launch:
  - root:
    - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v8`
  - command:
    - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v8 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v8 --tensor-parallel-size 4`
- Current state:
  - root `JOB_STATE_RUNNING`
  - root task state `building`
- Expected new discriminant in child logs:
  - `Staging remote vLLM directory locally from ...`
  - `Resolved ... remote files totaling ... GiB ...`
  - `Remote staging progress ...`
  - then `Starting vLLM environment`

### ALIGN-241 - 2026-03-26 10:09 - `v8` confirms the patched local-staging path is live inside the child; we are now inside staged model copy rather than dying before any useful startup signal

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v8`
- Child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v8/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v8-smoke_de164efa-c711ca79`
- Current state:
  - root `JOB_STATE_RUNNING`
  - child `JOB_STATE_RUNNING`
  - no preemptions yet
- Key new child log lines:
  - `Starting vLLM environment`
  - `Staging remote vLLM directory locally from gs://marin-us-central1/models/openai--gpt-oss-120b-vllm--b5c939d`
  - `Listing remote files under ...`
  - `Resolved 48 remote files totaling 60.79 GiB for local staging ...`
  - repeated `Remote staging progress ...`
- Interpretation:
  - the previous silent pre-`vllm` timeout window is now observable
  - the patched bundle is definitely active on the worker
  - the next gate is whether staging finishes cleanly and `vllm serve` starts, or whether the worker still times out during / after the large-weight copy
- Next action:
  - keep babysitting `v8`
  - if it clears local staging, inspect whether it also clears the old safetensors/model-streamer initialization failure

### ALIGN-242 - 2026-03-26 10:10 - `v8` fully cleared local staging and has now started the native TPU `vllm` server

- Key new child log lines:
  - `Remote staging progress ... 48/48 files (100.0%), 60.79/60.79 GiB copied`
  - `Finished local staging of gs://marin-us-central1/models/openai--gpt-oss-120b-vllm--b5c939d into /tmp/marin-vllm-model-.../model`
  - `Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
- Interpretation:
  - the parallel local-staging workaround is effective enough to survive the previous worker-timeout window
  - we have now definitively moved beyond the old `model_streamer`/silent-staging failure class
  - the next discriminant is whether engine initialization now succeeds, or whether a later GPT-OSS TPU / `vllm` compatibility failure appears after local staging
- Next action:
  - keep babysitting `v8` through engine startup
  - if it fails, compare the new failure against `ALIGN-235`; the old repo-id / model-streamer error should be gone if the workaround fully worked

### ALIGN-243 - 2026-03-26 10:19 - Confirmed the current staged GPT-OSS artifact is still MXFP4 and pivoted the codebase to `unsloth/gpt-oss-120b-BF16`

- Diagnosis from the failed TPU engine startup:
  - `jax.errors.JaxRuntimeError: INVALID_ARGUMENT: Element type F4E2M1FN is not supported on TPU`
  - inspected staged config:
    - `gs://marin-us-central1/models/openai--gpt-oss-120b-vllm--b5c939d/config.json`
    - `quantization_config = {"quant_method": "mxfp4", ...}`
  - conclusion:
    - the checkpoint itself is still MXFP4 / FP4 quantized
    - `skip_quantization=True` does not convert the checkpoint to BF16; it only alters the TPU quantization/bootstrap policy path
- Replacement artifact verified:
  - HF repo:
    - `unsloth/gpt-oss-120b-BF16`
  - pinned revision:
    - `e7523373bc44b42296b43202e265a1eebf2ee16f`
  - verified properties:
    - `architectures = ["GptOssForCausalLM"]`
    - `torch_dtype = "bfloat16"`
    - `quantization_config = None`
  - verified file set:
    - all required tokenizer/config files present
    - `73` model safetensor shards
  - verified size:
    - `233,658,313,344` bytes total
    - about `217.61 GiB`
- Code changes:
  - `experiments/models.py`
    - repointed both `gpt_oss_120b` and `gpt_oss_120b_vllm` to `unsloth/gpt-oss-120b-BF16@e7523373bc44b42296b43202e265a1eebf2ee16f`
  - `experiments/gpt_oss_120b_tpu.py`
    - updated docstring to BF16 wording
    - increased TPU worker disk from `100g` to `350g` to fit local staging of the BF16 checkpoint
  - `experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py`
    - switched tokenizer override to `unsloth/gpt-oss-120b-BF16`
- Validation:
  - `./infra/pre-commit.py --fix experiments/models.py experiments/gpt_oss_120b_tpu.py experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py`
  - passed
  - `uv run pytest tests/test_vllm_server.py tests/test_alignment.py -q`
  - `101 passed`
- Runtime note:
  - the prior `v8` smoke is no longer useful for this line of inquiry because it targeted the MXFP4 artifact
- Next action:
  - launch `v9` using the BF16 Unsloth checkpoint and re-test TPU local-vLLM bring-up

### ALIGN-244 - 2026-03-26 10:19 - Launched fresh BF16 smoke `v9`

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v9-bf16`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v9-bf16 --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v9_bf16 --tensor-parallel-size 4`
- Current state:
  - root `JOB_STATE_RUNNING`
  - no child yet at the first poll
- Key expectations for this run:
  - model dependency step should now target `unsloth/gpt-oss-120b-BF16@e7523373bc44b42296b43202e265a1eebf2ee16f`
  - TPU worker disk should be `350g` once the smoke child launches
  - the old `F4E2M1FN` TPU error should disappear if the BF16 checkpoint is wired correctly

### ALIGN-245 - 2026-03-26 10:22 - Launch BF16 vLLM-subset download in `us-east5` so the same executor model step exists in both regions

- User request:
  - materialize the same BF16 `gpt_oss_120b_vllm` model step in `gs://marin-us-east5/...`
- Planned launch:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name download-gpt-oss-120b-vllm-east5-bf16 --cpu 4 --memory 16GB --disk 10GB --region us-east5 -- python experiments/download_gpt_oss_120b_vllm.py`
- Expected artifact path shape:
  - same logical executor output path as central:
    - `models/unsloth--gpt-oss-120b-BF16-vllm--e7523373bc44b42296b43202e265a1eebf2ee16f`
  - under the east5 bucket prefix

### ALIGN-246 - 2026-03-26 11:39 - Submitted the east5 BF16 vLLM-subset download job

- Root:
  - `/ahmed/download-gpt-oss-120b-vllm-east5-bf16`
- Exact launch:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name download-gpt-oss-120b-vllm-east5-bf16 --cpu 4 --memory 16GB --disk 10GB --region us-east5 -- python experiments/download_gpt_oss_120b_vllm.py --prefix gs://marin-us-east5`
- Expected executor artifact:
  - `gs://marin-us-east5/models/unsloth--gpt-oss-120b-BF16-vllm--e7523373bc44b42296b43202e265a1eebf2ee16f`

### ALIGN-247 - 2026-03-26 11:53 - Refined diagnosis of the old `model_streamer` failure: it was specific to the OpenAI MXFP4 GPT-OSS artifact, not obviously a generic `model_streamer` failure for all large models

- Traced relevant `vllm` startup path:
  - `arg_utils.py` resolves `model` via `get_model_path(...)`
  - GPT-OSS architecture setup later calls `transformers_utils/model_arch_config_convertor.py:get_torch_dtype(...)`
  - because the OpenAI checkpoint config had:
    - `torch_dtype = None`
    - `dtype = None`
    - `quantization_config.quant_method = "mxfp4"`
    - `get_torch_dtype(...)` falls back to `get_safetensors_params_metadata(model_id, revision=...)`
  - `get_safetensors_params_metadata(...)` in `transformers_utils/config.py`:
    - uses local `*.safetensors` metadata if `Path(model).exists()`
    - otherwise calls HF safetensors metadata APIs
- Why the old failure happened:
  - under the `runai_streamer` / `model_streamer` path, the `model` string seen by this dtype-metadata path was `/root/.cache/vllm/assets/model_streamer/<hash>`
  - at that point it was not being treated as an existing local directory, so `vllm` fell into the HF metadata branch
  - HF metadata lookup then rejected the cache path as an invalid repo id, producing the `Repo id must be in the form ...` error from `ALIGN-235`
- Why other large models can work:
  - many models have `torch_dtype` declared directly in `config.json`, so they never need this safetensors-metadata fallback during startup
  - the OpenAI GPT-OSS MXFP4 artifact was unusual because it lacked an explicit dtype and forced the metadata-probe path
- Important implication:
  - the BF16 Unsloth checkpoint has `torch_dtype = "bfloat16"` and `quantization_config = None`
  - therefore the specific old `model_streamer` failure class may no longer apply to the BF16 artifact
- Next action:
  - after the current BF16 staged run resolves, test whether BF16 can launch directly via `model_streamer` with `stage_remote_model_locally=False`, which would avoid the expensive full local copy

### ALIGN-248 - 2026-03-26 12:07 - BF16-specific `tpu-inference` bring-up plan after reading the Unsloth HF config

- Verified `unsloth/gpt-oss-120b-BF16` config at pinned revision `e7523373bc44b42296b43202e265a1eebf2ee16f`:
  - `architectures = ["GptOssForCausalLM"]`
  - `model_type = "gpt_oss"`
  - `torch_dtype = "bfloat16"`
  - `dtype = null`
  - `quantization_config = null`
- Implication:
  - the old OpenAI MXFP4 failure chain does not apply directly to this BF16 artifact
  - BF16 GPT-OSS should not need the old safetensors-metadata fallback that was triggered by missing dtype
  - BF16 GPT-OSS should also bypass the GPT-OSS MXFP4 Qwix quantization rules in `tpu-inference`
- `tpu-inference` code facts relevant to the fix:
  - `tpu_inference/models/common/model_loader.py` already has the intended streamed bootstrap path for TPU abstract-load:
    - if loader is `RunaiModelStreamerLoader`, it sets `model_weights_iterator = loader._get_weights_iterator(...)`
    - then `model.load_weights(rng)` consumes streamed weights
  - `tpu_inference/models/jax/gpt_oss.py` already consumes `model_weights_iterator`
  - `tpu_inference/models/jax/utils/qwix/qwix_utils.py` only injects default GPT-OSS quantization when:
    - `model_type == "gpt_oss"`
    - and `quant_method == "mxfp4"`
  - so BF16 should naturally take the non-quantized path if we preserve the HF config accurately
- Plan:
  1. Remove the Marin-side local-staging workaround for the BF16 smoke path:
     - set `stage_remote_model_locally=False` in `experiments/gpt_oss_120b_tpu.py`
     - keep `load_format=runai_streamer`, `model_bootstrap=abstract_load`, `prefer_jax_for_bootstrap=True`
     - this directly tests whether BF16 now works with the native streamer path
  2. If BF16 direct-stream still fails, patch `tpu-inference` rather than reintroducing local staging:
     - make GPT-OSS abstract-load explicitly prefer `model_weights_iterator` / streamer weights for BF16
     - ensure no GPT-OSS bootstrap path re-derives dtype or quantization from local safetensors metadata when `hf_config.torch_dtype` is already set
     - fail fast if a BF16 GPT-OSS run accidentally sees non-null `quantization_config`
  3. Add focused regression coverage in `tpu-inference`:
     - BF16 GPT-OSS + `runai_streamer` + `abstract_load` is accepted
     - BF16 GPT-OSS does not inject default GPT-OSS MXFP4 Qwix quantization
     - streamed `model_weights_iterator` path is used end-to-end for GPT-OSS abstract-load
  4. Only if streamer still fails due to a `vllm` metadata path outside `tpu-inference`:
     - patch the `vllm`/TPU integration path instead of keeping full local model staging as the default
- Bottom line:
  - the right next experiment is a BF16 smoke with native `runai_streamer` restored
  - the `tpu-inference` fix target is to make BF16 GPT-OSS a first-class non-quantized streamed abstract-load path, not to keep compensating with full local copies

### ALIGN-249 - 2026-03-26 12:14 - Confirmed the BF16 local-staging workaround fails with host OOM, so the next smoke must use native `runai_streamer`

- Confirmed failing child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v9-bf16/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v9_bf16-smoke_7398692e-482cc765`
- Iris job state summary:
  - `Exit code 1: OOM killed (container exceeded memory limit)`
- Failure sequence from logs:
  - BF16 model staging started from:
    - `gs://marin-us-central1/models/unsloth--gpt-oss-120b-BF16-vllm--e7523373bc44b42296b43202e265a1eebf2ee16f`
  - worker resolved:
    - `164` files
    - `217.64 GiB`
  - local staging completed enough to reach engine startup / APIServer initialization
  - then the container was OOM-killed by the kernel
- Interpretation:
  - for BF16 GPT-OSS, the Marin-side `stage_remote_model_locally=True` workaround is now a dead end
  - even with `350 GiB` disk, the host memory envelope is not sufficient for full local copy + vLLM/TPU bootstrap overhead
- Immediate change:
  - flipped `experiments/gpt_oss_120b_tpu.py` back to:
    - `stage_remote_model_locally=False`
  - next run should exercise the native `runai_streamer` path with the BF16 artifact

### ALIGN-250 - 2026-03-26 12:23 - Patched `tpu-inference` for BF16 GPT-OSS unquantized bootstrap and pushed a new package ref

- Root BF16-native fixes in real source repo `/Users/ahmed/code/tpu-inference`:
  - `tpu_inference/models/jax/gpt_oss.py`
  - `tpu_inference/models/jax/utils/qwix/qwix_utils.py`
  - `tpu_inference/models/common/model_loader.py`
  - tests:
    - `tests/models/common/test_model_loader.py`
    - `tests/layers/jax/test_qwix.py`
    - `tests/models/jax/test_weight_loading.py`
- Functional changes:
  - GPT-OSS weight loading now treats `hf_config.quantization_config = None` as unquantized instead of indexing into `None`
  - Qwix default-config selection now treats missing/`None` HF quantization config as unquantized
  - dummy/random-init Qwix path now normalizes `quantization_config` to `{}` when absent
- Source commit:
  - `38cf26ac`
  - message: `Handle GPT-OSS BF16 native streamer bootstrap`
- Pushed package branch:
  - `ahmed/gpt-oss-tpu-bringup-v5`
- Workspace mirrored the GPT-OSS/Qwix loader changes into the vendored overlay under `tpu_inference/...`
- Experiment config changes:
  - `experiments/gpt_oss_120b_tpu.py`
    - package ref -> `ahmed/gpt-oss-tpu-bringup-v5`
    - removed explicit `weight_loader = "fsspec_streamer"` so bootstrap uses the native `RunaiModelStreamerLoader` path
    - kept `stage_remote_model_locally=False`
    - reduced smoke disk request from `350g` to `120g`
- Validation:
  - `python3 -m py_compile` passed on all changed source/workspace files
  - full `tpu-inference` pytest was not runnable from this workstation because the available env here does not include that repo's `torch`/`vllm`/`qwix` test dependencies

### ALIGN-251 - 2026-03-26 12:28 - Launched the first BF16 native-`runai_streamer` smoke after removing local staging

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v10-native-streamer`
- Exact launch:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v10-native-streamer --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v10_native_streamer`
- Important config changes active in this run:
  - BF16 checkpoint from `unsloth/gpt-oss-120b-BF16`
  - `stage_remote_model_locally=False`
  - native `runai_streamer` bootstrap path via `RunaiModelStreamerLoader`
  - `tpu_inference` package ref pinned to `ahmed/gpt-oss-tpu-bringup-v5`
- Initial state:
  - root is `running`
  - logs show normal worker bootstrap (`syncing deps`, `installing pip deps`, `running user command`)
- Next discriminant:
  - whether the TPU child starts native `vllm` without the old MXFP4/dtype or local-staging failure classes

### ALIGN-252 - 2026-03-26 12:31 - `v10` confirms the native-streamer path is active; the run is now inside native TPU `vllm` startup

- Current running child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v10-native-streamer/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v10_native_streamer-smoke_0e02b9c4-e85e9f4f`
- Resource signal:
  - child is running on `v5p-8` with `32cpu`, `256 GiB RAM`, `120 GiB disk`
  - this is important because the old local-staging path required `350 GiB` disk
- Log signal:
  - there are no `Staging remote vLLM directory locally ...` lines in the child logs
  - the child reached:
    - `marin.inference.vllm_server Starting vLLM environment`
    - `marin.inference.vllm_server Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
- Interpretation:
  - the BF16 smoke is no longer taking the Marin local-staging workaround
  - the native `runai_streamer` / TPU bootstrap path is now the live codepath under test
  - native streamer is therefore working at least through environment setup and handoff into native TPU `vllm`
- Still not proven yet:
  - we have not yet seen engine initialization complete or a served completion response

### ALIGN-253 - 2026-03-26 12:34 - Full detail checkpoint for the BF16 native-streamer pivot

- BF16 config facts that drove the pivot:
  - repo:
    - `unsloth/gpt-oss-120b-BF16`
  - pinned revision:
    - `e7523373bc44b42296b43202e265a1eebf2ee16f`
  - relevant HF config fields:
    - `architectures = ["GptOssForCausalLM"]`
    - `model_type = "gpt_oss"`
    - `torch_dtype = "bfloat16"`
    - `dtype = null`
    - `quantization_config = null`
- Why this matters:
  - the old OpenAI GPT-OSS MXFP4 artifact failed because it ultimately drove TPU into the unsupported `F4E2M1FN` path
  - the BF16 artifact should not take that quantized path, so the native streamer route is worth reviving
- `v9` failure that forced the pivot:
  - failing child:
    - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v9-bf16/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v9_bf16-smoke_7398692e-482cc765`
  - job summary:
    - `Exit code 1: OOM killed (container exceeded memory limit)`
  - logs showed:
    - local staging of `164` files totaling `217.64 GiB`
    - source artifact:
      - `gs://marin-us-central1/models/unsloth--gpt-oss-120b-BF16-vllm--e7523373bc44b42296b43202e265a1eebf2ee16f`
    - engine/APIServer startup began
    - then the kernel OOM-killed the container
  - conclusion:
    - Marin-side local staging is a dead end for BF16 GPT-OSS
- Exact `tpu-inference` source changes that followed in `/Users/ahmed/code/tpu-inference`:
  - `tpu_inference/models/jax/gpt_oss.py`
    - added safe helper for reading HF quantization method
    - GPT-OSS BF16 no longer assumes `quantization_config` is a dict
  - `tpu_inference/models/jax/utils/qwix/qwix_utils.py`
    - missing/`None` HF quantization config now cleanly means "unquantized"
  - `tpu_inference/models/common/model_loader.py`
    - random-init/Qwix helper path now normalizes absent HF quantization config to `{}`
  - tests edited:
    - `tests/models/common/test_model_loader.py`
    - `tests/layers/jax/test_qwix.py`
    - `tests/models/jax/test_weight_loading.py`
- Source-control state:
  - local source commit:
    - `38cf26ac`
  - local source branch:
    - `ahmed/gpt-oss-tpu-bringup`
  - pushed remote package branch:
    - `ahmed/gpt-oss-tpu-bringup-v5`
- Workspace-side integration changes:
  - mirrored GPT-OSS/Qwix loader fixes into vendored overlay:
    - `tpu_inference/models/jax/gpt_oss.py`
    - `tpu_inference/models/jax/utils/qwix/qwix_utils.py`
  - `experiments/gpt_oss_120b_tpu.py` now:
    - points to package ref `ahmed/gpt-oss-tpu-bringup-v5`
    - has `stage_remote_model_locally=False`
    - no longer forces `weight_loader = "fsspec_streamer"`
    - requests `120g` disk instead of `350g` for the smoke
- Validation status:
  - `python3 -m py_compile` passed on all changed source and workspace files
  - full `tpu-inference` pytest was not runnable from this workstation because the currently available env lacks `torch`, `vllm`, and `qwix` for that repo's test suite
- Current active run:
  - root:
    - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v10-native-streamer`
  - child:
    - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v10-native-streamer/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v10_native_streamer-smoke_0e02b9c4-e85e9f4f`
  - root resources:
    - `4cpu`, `16 GiB`, `10 GiB disk`
  - child resources:
    - `32cpu`, `256 GiB`, `120 GiB disk`, `v5p-8`
- Strongest current evidence that native streamer is active:
  - child logs do **not** contain:
    - `Staging remote vLLM directory locally ...`
  - child logs **do** contain:
    - `marin.inference.vllm_server Starting vLLM environment`
    - `marin.inference.vllm_server Starting vLLM native server with TPU_MIN_LOG_LEVEL=3 TPU_STDERR_LOG_LEVEL=3`
- Current interpretation:
  - the BF16 smoke is no longer failing in the old local-staging/OOM path
  - the live failure surface has moved forward into native TPU `vllm` initialization
  - native `runai_streamer` is therefore working at least through worker bootstrap and handoff into native `vllm`

---

The following entries were originally recorded in `.agents/logbooks/alignment_function_claude.md` and are merged here for continuity.

### ALIGN-254 - 2026-03-26 19:42 - `v10` BF16 native-streamer smoke: TPU engine init succeeded, failed at chat template application

- Failed root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v10-native-streamer`
- Failed child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v10-native-streamer/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v10_native_streamer-smoke_0e02b9c4-e85e9f4f`
- Runtime:
  - 24 minutes total
  - ~23 minutes of silent model loading via native `runai_streamer` (217 GiB BF16)
  - then failed immediately on the first smoke request
- **Critical positive signal:**
  - vLLM TPU engine initialization **succeeded** for GPT-oss-120B BF16 on `v5p-8`
  - the entire v1-v9 failure chain (MXFP4 quantization, model recognition, local staging OOM, weight streaming, pip caching) is resolved
  - the failure is at the **application layer**, not the infrastructure layer
- Terminal error:
  - `ValueError: Cannot use chat template functions because tokenizer.chat_template is not set and no template argument was passed!`
  - traceback:
    - `batched_vllm_serve.py:311` -> `render_messages` -> `tokenizer.apply_chat_template(...)`
    - `transformers/tokenization_utils_base.py:1825` -> `get_chat_template` -> `raise ValueError`
- Root cause analysis:
  - `unsloth/gpt-oss-120b-BF16` has `chat_template: null` in `tokenizer_config.json`
  - it ships a separate `chat_template.jinja` file that newer transformers auto-loads, but only if present in the local directory
  - `_load_tokenizer` in `batched_vllm_serve.py:234` calls `LMEvaluationHarnessEvaluator._stage_remote_tokenizer_dir(...)` to copy tokenizer files from GCS to a temp dir
  - `_stage_remote_tokenizer_dir` iterates `TOKENIZER_FILENAMES` (class constant on `LMEvaluationHarnessEvaluator`)
  - `TOKENIZER_FILENAMES` did **not** include `chat_template.jinja`
  - so `chat_template.jinja` was present in the staged GCS artifact but never copied to the temp dir
  - `AutoTokenizer.from_pretrained(temp_dir)` found no chat template -> crash on `apply_chat_template`
- Observability gap identified:
  - vLLM subprocess stdout/stderr was redirected to temp files on disk (`subprocess.Popen(..., stderr=stderr_f)`)
  - Iris only captures the parent Python process's stdout/stderr
  - result: 23 minutes of zero visibility during model loading

### ALIGN-255 - 2026-03-26 19:45 - Two fixes applied for chat template and vLLM log visibility

- Fix 1 - chat template staging:
  - file: `lib/marin/src/marin/evaluation/evaluators/lm_evaluation_harness_evaluator.py`
  - added `"chat_template.jinja"` to `TOKENIZER_FILENAMES`
  - verified `chat_template.jinja` exists in the staged BF16 artifact:
    - `gs://marin-us-central1/models/unsloth--gpt-oss-120b-BF16-vllm--e7523373bc44b42296b43202e265a1eebf2ee16f/chat_template.jinja`
- Fix 2 - vLLM subprocess stderr visibility:
  - file: `lib/marin/src/marin/inference/vllm_server.py`
  - changed native `vllm serve` subprocess from `stderr=stderr_f` (silent file) to `stderr=subprocess.PIPE`
  - added a daemon thread that tees each stderr line to both the log file (preserving `_native_logs_tail`) and `sys.stderr` (so Iris captures it)
- Validation:
  - `uv run pytest tests/test_vllm_server.py -q` -> 3 passed
  - `./infra/pre-commit.py --fix` -> passed on both files
- Next action:
  - launch `v11` smoke with both fixes

### ALIGN-256 - 2026-03-26 20:04 - Launched `v11` with chat template fix and stderr tee

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11-chat-template`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11-chat-template --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v11_chat_template --tensor-parallel-size 4`
- Bundle size: 5.7 MB
- Fixes active in this bundle:
  - `chat_template.jinja` added to `TOKENIZER_FILENAMES`
  - vLLM subprocess stderr teed to `sys.stderr` for Iris visibility
- Expected differences from v10:
  - `chat_template.jinja` will now be staged into the tokenizer temp dir
  - `AutoTokenizer.from_pretrained` will pick up the chat template
  - vLLM model loading progress will be visible in Iris logs
- Success condition:
  - smoke child boots GPT-oss-120B BF16 on `v5p-8`
  - `/v1/completions` returns a valid response
  - `artifacts/vllm_metrics.json` is written
- Status: killed - child blocked on disk (autoscaler provisions 100 GiB, job requested 120 GiB)

### ALIGN-257 - 2026-03-26 20:09 - Lowered disk to 80g and relaunched as `v11b`

- v11 failure:
  - `Autoscaler: insufficient_resources: disk: need 128849018880 (120 GiB), available 107374182400 (100 GiB)`
  - the `120g` disk request was a leftover from the local-staging era (ALIGN-243)
  - native `runai_streamer` does not stage to local disk, so 80 GiB is sufficient
- Fix:
  - `experiments/gpt_oss_120b_tpu.py`: default disk `120g` -> `80g`
- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11b-chat-template`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11b-chat-template --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/gpt_oss_120b_vllm_smoke.py --name debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v11b_chat_template --tensor-parallel-size 4`
- Status: **SUCCEEDED**

### ALIGN-258 - 2026-03-26 20:26 - `v11b` GPT-oss-120B BF16 smoke **PASSED** - first successful TPU serve + completion

- Root:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11b-chat-template`
- Child:
  - `/ahmed/gpt-oss-120b-vllm-smoke-tpu-flax-tp4-v11b-chat-template/align-debug_gpt_oss_120b_vllm_smoke_tpu_flax_tp4_v11b_chat_template-smoke_43e1d596-290a77ad`
- Child state: `JOB_STATE_SUCCEEDED`
- Timeline:
  - 20:12:41 - worker bootstrap
  - 20:12:57 - `Starting vLLM native server`
  - 20:14:14 - `Loading safetensors using Runai Model Streamer: 21%` (first visible progress - stderr tee working)
  - 20:14:41 - `100% Completed | 615/615 [00:24, 24.83it/s]` - weights loaded in 24 seconds
  - 20:23:48 - `vLLM environment ready` - API server started after ~9 min XLA compilation
  - 20:23:48 - `Rendering 1 chat prompts` - chat template applied successfully
  - 20:23:48 - `Sending batched vLLM serve request to /v1/completions for 1 prompts`
  - job succeeded
- Key metrics:
  - weight loading: 615 safetensor shards, 24 seconds via `runai_streamer` (~25 shards/s)
  - XLA compilation: ~9 minutes (first-run cost)
  - total child wall-clock: ~14 minutes
- Fixes validated:
  - `chat_template.jinja` in `TOKENIZER_FILENAMES` -> tokenizer now has chat template
  - stderr tee -> full vLLM progress visible in Iris logs
  - disk `80g` -> autoscaler can provision new v5p-8 slices
- This is the first successful GPT-oss-120B serve on TPU in this project
- GOSS-3 is now complete; the path is unblocked for:
  - GOSS-5: one-statement prompt generation
  - GOSS-6: full-spec prompt generation
  - GOSS-7: full open-weight alignment pipeline

### ALIGN-259 - 2026-03-26 20:30 - Raised `max_model_len` and `max_tokens` to 4096 for reasoning model thinking budget

- Context:
  - GPT-oss-120B is a reasoning model with a chain-of-thought `analysis` channel
  - the chat template accepts `reasoning_effort` kwarg (defaults to `"medium"`)
  - thinking tokens count against both `max_model_len` (engine context) and `max_tokens` (per-request completion budget)
  - previous values were `max_model_len=2048` and `max_tokens=1024` for prompt generation stages - far too small for a reasoning model that may spend hundreds of tokens thinking before producing the actual response
- Changes:
  - `experiments/gpt_oss_120b_tpu.py`: `max_model_len` 2048 -> 4096
  - `experiments/generate_prompts_gpt_oss_120b_refactored.py`: `understanding_max_tokens`, `concretize_max_tokens`, `extract_max_tokens` all 1024 -> 4096
  - `experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py`: `understanding_max_tokens`, `concretize_max_tokens`, `extract_max_tokens`, `teacher_max_tokens`, `rejected_max_tokens` all -> 4096
  - smoke `max_tokens=64` left as-is (sufficient for trivial validation)
- Note on `reasoning_effort`:
  - the chat template supports `reasoning_effort` as a kwarg to `apply_chat_template`
  - valid values: `"low"`, `"medium"`, `"high"` (defaults to `"medium"`)
  - not yet wired through the alignment pipeline - can be added later per-stage
- Validation:
  - `./infra/pre-commit.py --fix` -> passed on all three files

### ALIGN-260 - 2026-03-26 20:40 - Plan: single-statement end-to-end alignment pipeline with GPT-oss-120B for all roles

- **Goal:** Run the complete alignment pipeline on one statement (`ask_clarifying_questions`) with GPT-oss-120B serving every model role, validating the full data path from spec -> preference pairs.
- **Why one statement:** Minimizes TPU wall-clock while exercising every pipeline stage. If this passes, the full 46-statement run is a scale-up, not a new integration risk.
- **Why Mixtral for rejected:** Mixtral-8x7B-Instruct is a weaker model that naturally produces lower-quality responses without spec guidance, making it a better source of rejected responses. Using a different model for rejected also avoids the risk that GPT-oss's reasoning capability makes it too good at following even "opposite" instructions, collapsing the chosen/rejected margin.

#### Model roles and config

| Role | Model | Config source |
|---|---|---|
| Ideation (Stage 1/2) | GPT-oss-120B BF16 | `gpt_oss_120b_tpu_vllm_config()` |
| Extract (Stage 3) | GPT-oss-120B BF16 | same |
| Teacher (chosen) | GPT-oss-120B BF16 | same |
| Rejected | Mixtral-8x7B-Instruct-v0.1 | `VLLMConfig` on `v5p-8` (opposite prompting) |
| Judge | GPT-oss-120B BF16 | `gpt_oss_120b_tpu_vllm_config()` |

#### Pipeline stages exercised

1. **Spec loading** - read `openai_model_spec.jsonl`, filter to `statement_ids=["ask_clarifying_questions"]`
2. **Stage 1 (Understanding)** - GPT-oss generates variation axes for the statement
3. **Stage 2 (Concretize)** - GPT-oss generates concrete scenarios from covering array configs
4. **Stage 3 (Extract)** - GPT-oss extracts system_prompt + user_message from scenarios
5. **Chosen response generation** - GPT-oss generates spec-guided responses (teacher role)
6. **Rejected response generation** - GPT-oss generates opposite-mode responses (rejected role)
7. **Judge** - GPT-oss scores both chosen and rejected responses against rubrics
8. **Preference pair construction** - filter by score thresholds, build chosen/rejected pairs
9. **Output** - sharded `.jsonl.gz` preference dataset

#### Experiment script

- New file: `experiments/align_gpt_oss_120b_e2e_one_statement.py`
- Based on `align_gpt_oss_120b_mixtral_rejected_full_spec.py` but with:
  - `rejected_model=mixtral_vllm` (Mixtral-8x7B-Instruct, opposite prompting)
  - `statement_ids=["ask_clarifying_questions"]`
  - `dpo_config=None` (stop at preference pairs - DPO training is a separate validation)
  - `covering_strength=2` (pairwise - fewer prompts, faster)
  - relaxed judge thresholds for validation: `judge_min_chosen_score=1.0`, `judge_min_gap=0.0`
  - name: `goss_e2e_one_statement`

#### Resource expectations

- Two vLLM sessions: GPT-oss for prompt gen / chosen / judge, Mixtral for rejected
- Both on `v5p-8` with `tp=4`, `max_model_len=4096`, `max_tokens=4096`
- Estimated prompts per statement with 2-way covering: ~200-500
- All response/judge work is sequential within the shared session

#### Success criteria

- All 5 pipeline stages complete without error
- `prompts/` artifact contains extracted eval prompts for `ask_clarifying_questions`
- `chosen/` and `rejected/` artifacts contain response JSONL
- `preference_pairs/` artifact contains at least 1 valid preference pair
- `artifacts/vllm_metrics.json` emitted for each stage

#### Risk factors

- `max_model_len=4096` may be too small for the judge stage (prompt + rubric + response can be long)
- GPT-oss reasoning overhead means each request may take significantly longer than a non-reasoning model
- Mixtral needs its own staged artifact on `v5p-8` - need to verify `mixtral_8x7b_instruct` is staged in `us-central1`

#### Next steps after this experiment

- If it passes: launch GOSS-6 (full-spec prompt generation) and GOSS-7 (full pipeline with Mixtral rejected)
- If `max_model_len` is a bottleneck: raise to 8192 and re-test
- If opposite-mode rejected is weak: consider using `unguided` strategy instead

### ALIGN-261 - 2026-03-26 20:54 - Launched single-statement E2E pipeline (`goss-e2e-one-statement`)

- Script: `experiments/align_gpt_oss_120b_e2e_one_statement.py`
- Root: `/ahmed/goss-e2e-one-statement`
- Launch command:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name goss-e2e-one-statement --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/align_gpt_oss_120b_e2e_one_statement.py`
- Pre-launch checks:
  - Mixtral staged in us-central1: confirmed at `gs://marin-us-central1/models/mistralai--Mixtral-8x7B-Instruct-v0-1--eba9230/`
  - GPT-oss BF16 staged: confirmed (used by v11b smoke)
  - pre-commit: passed
- Config summary:
  - statement: `ask_clarifying_questions` only
  - GPT-oss: ideation, extract, teacher (chosen), judge
  - Mixtral: rejected (opposite prompting)
  - `covering_strength=2`, `max_model_len=4096`, `max_tokens=4096`
  - `dpo_config=None` -> stops at preference pairs
- Status: **FAILED** - `max_model_len=0` reported by vLLM engine

### ALIGN-262 - 2026-03-26 21:27 - E2E pipeline failed in Stage 1 understanding: vLLM reports `max_context_length=0`

- Failed root: `/ahmed/goss-e2e-one-statement`
- Failed child: `/ahmed/goss-e2e-one-statement/align-goss_e2e_one_statement-prompts_0b3b54c4-77d3bea9`
- Timeline:
  - spec step succeeded
  - prompts child: weights loaded (615/615 in 30s), XLA compiled (~13 min), API server started
  - Stage 1 understanding request sent (944 input tokens)
  - vLLM returned: `"This model's maximum context length is 0 tokens"`
- Error:
  - `requests.HTTPError: 400 Client Error: Bad Request; response body: {"error":{"message":"This model's maximum context length is 0 tokens. However, your request has 944 input tokens."}}`
- Analysis:
  - `max_model_len=4096` is set in `gpt_oss_120b_tpu_vllm_config()` and forwarded as `--max-model-len 4096` to `vllm serve`
  - vLLM engine reports 0 anyway - likely the model config's `initial_context_length=4096` or another field is being misread by the TPU engine
  - the v11b smoke passed with `max_model_len=2048` and a 64-token request, so either the engine was silently capping, or the `max_model_len` change to 4096 exposed a new issue
- Investigation:
  - launched smoke with same `max_model_len=4096` -> **PASSED** (root `/ahmed/gpt-oss-smoke-maxlen-4096`)
  - XLA compilation took ~17 min (longer than v11b's 9 min due to larger KV cache)
  - the E2E `max_context_length=0` was NOT caused by the `max_model_len=4096` value itself
  - the first E2E child was preempted, and the retry ran on a different worker
  - hypothesis: the preemption + retry may have caused corrupted vLLM state or stale worker config
- Next action: relaunch the E2E pipeline

### ALIGN-263 - 2026-03-26 21:59 - Relaunched E2E pipeline as `goss-e2e-one-statement-v2`

- Root: `/ahmed/goss-e2e-one-statement-v2`
- Same script: `experiments/align_gpt_oss_120b_e2e_one_statement.py`
- Rationale: smoke with `max_model_len=4096` succeeded, so the v1 failure was likely caused by preemption + retry worker state corruption, not a config issue
- Status: **FAILED** - same `max_context_length=0` error, reproducible

### ALIGN-264 - 2026-03-26 22:42 - `goss-e2e-one-statement-v2` failed with identical `max_context_length=0` - this is a reproducible E2E-only bug

- Failed root: `/ahmed/goss-e2e-one-statement-v2`
- Failed child: `/ahmed/goss-e2e-one-statement-v2/align-goss_e2e_one_statement-prompts_0b3b54c4-17d0b74e`
- Error: same as v1 - `"This model's maximum context length is 0 tokens. However, your request has 944 input tokens."`
- Worker: `marin-tpu-v5p-8-us-central1-a-20260326-2051-c7ee3911-worker-0` (clean worker, no preemption)
- Key finding: **this is NOT a preemption fluke - it is a reproducible bug specific to the E2E pipeline path**
- Evidence:
  - smoke with `max_model_len=4096` and `max_tokens=64` -> **PASSED** (`/ahmed/gpt-oss-smoke-maxlen-4096`)
  - E2E pipeline v1 with `max_model_len=4096` -> **FAILED** (`max_context_length=0`)
  - E2E pipeline v2 with `max_model_len=4096` -> **FAILED** (`max_context_length=0`)
- What's different between smoke and E2E:
  - smoke: `gpt_oss_120b_vllm_smoke.py` -> constructs `VLLMConfig` at module level -> passes directly to `BatchedVllmServeSession` -> runs in the same executor step
  - E2E: `align_gpt_oss_120b_e2e_one_statement.py` -> `align()` -> creates `ExecutorStep` with `PromptGenConfig(ideation_model=gpt_oss_vllm)` -> executor serializes config -> `remote()` -> worker deserializes -> `generate_prompts_from_spec()` -> `BatchedVllmServeSession(config.ideation_model)`
- Working hypothesis:
  - the executor serializes `PromptGenConfig` (which contains a nested `VLLMConfig`) to transport it to the worker
  - during deserialization, `VLLMConfig` may lose its type and become a plain dict or generic `InferenceConfig`
  - if the deserialized `ideation_model` is a dict, `_build_model_config(config)` would fail accessing `config.max_model_len`, but the error is `max_context_length=0`, not `AttributeError`
  - alternatively, the deserialized config might reconstruct `VLLMConfig` with missing fields, and `max_model_len` might default to 0 or be dropped
- Investigation in progress:
  - tracing how the marin executor serializes nested `VLLMConfig` dataclasses inside `PromptGenConfig`
  - need to check if `max_model_len` survives the round-trip

### ALIGN-265 - 2026-03-26 22:57 - Diagnosed the E2E `max_context_length=0` failure as an impossible completion budget, not executor config corruption

- Result of tracing the serialization hypothesis:
  - `instantiate_config()` in `lib/marin/src/marin/execution/executor.py` recursively rebuilds nested dataclasses with `replace(...)`
  - `PromptGenConfig.ideation_model` therefore survives executor instantiation as a `VLLMConfig`
  - the earlier serialization-loss hypothesis from `ALIGN-264` is not supported by the code path
- Root cause:
  - the GPT-OSS E2E scripts were setting:
    - `max_model_len=4096`
    - `understanding_max_tokens=4096`
    - `concretize_max_tokens=4096`
    - `extract_max_tokens=4096`
    - and for end-to-end runs, `teacher_max_tokens=4096`, `rejected_max_tokens=4096`
  - for vLLM OpenAI completions, completion tokens consume the same context window as the prompt
  - with `max_model_len=4096` and `max_tokens=4096`, the available prompt budget is exactly `0`
  - this matches the failing Stage 1 error verbatim:
    - `This model's maximum context length is 0 tokens. However, your request has 944 input tokens.`
- Why the smoke still passed:
  - the passing smoke `/ahmed/gpt-oss-smoke-maxlen-4096` used `max_tokens=64`
  - that left prompt room, so the same `max_model_len=4096` server could answer successfully
- Code changes made:
  - `lib/marin/src/marin/alignment/batched_vllm_serve.py`
    - added a local preflight guard that tokenizes rendered prompts before the HTTP request
    - now raises a clear `ValueError` when `max_tokens` leaves no prompt budget or when `prompt_tokens + max_tokens > max_model_len`
  - `experiments/gpt_oss_120b_tpu.py`
    - centralized GPT-OSS TPU defaults:
      - `GPT_OSS_TPU_MAX_MODEL_LEN = 4096`
      - `GPT_OSS_TPU_DEFAULT_MAX_TOKENS = 2048`
  - updated GPT-OSS experiment scripts to use `GPT_OSS_TPU_DEFAULT_MAX_TOKENS` instead of `4096`:
    - `experiments/generate_prompts_gpt_oss_120b_refactored.py`
    - `experiments/align_gpt_oss_120b_e2e_one_statement.py`
    - `experiments/align_gpt_oss_120b_mixtral_rejected_full_spec.py`
  - added a debug log:
    - `docs/debug-log-gpt-oss-e2e-max-context.md`
- Validation:
  - `uv run pytest tests/test_alignment.py tests/test_vllm_server.py -q` -> `103 passed`
  - `./infra/pre-commit.py --fix ...` passed on all touched files
- New interpretation:
  - the current blocker is not config transport
  - the next real discriminant is whether the one-statement GPT-OSS E2E run succeeds with a non-zero prompt budget
- Next action:
  - relaunch `experiments/align_gpt_oss_120b_e2e_one_statement.py`
  - if later stages still hit context pressure, raise `max_model_len` further or lower per-stage `max_tokens` selectively rather than setting them equal

### ALIGN-266 - 2026-03-26 22:59 - Launched a one-statement GPT-OSS prompt-generation validation run after the completion-budget fix

- Purpose:
  - validate the corrected prompt-generation budget on the cheapest path that reproduces the prior failure surface
  - this checks Stage 1/2/3 prompt generation before paying for another full chosen/rejected/judge end-to-end run
- Root:
  - `/ahmed/gpt-oss-promptgen-one-statement-budget-fix`
- Exact launch:
  - `uv run iris --controller-url http://127.0.0.1:10000 job run --no-wait --job-name gpt-oss-promptgen-one-statement-budget-fix --cpu 4 --memory 16GB --disk 10GB --region us-central1 -- python experiments/generate_prompts_gpt_oss_120b_refactored.py --name debug_generate_prompts_gpt_oss_120b_budget_fix --statement-id ask_clarifying_questions --tensor-parallel-size 4`
- Active config in this run:
  - GPT-OSS BF16 TPU local-vLLM via `gpt_oss_120b_tpu_vllm_config()`
  - one statement only:
    - `ask_clarifying_questions`
  - corrected prompt-generation budgets:
    - `understanding_max_tokens=2048`
    - `concretize_max_tokens=2048`
    - `extract_max_tokens=2048`
  - `max_model_len` remains `4096`
- Initial state:
  - root is `JOB_STATE_RUNNING`
  - no child logs yet at the first poll
- Next discriminant:
  - whether Stage 1 understanding now clears instead of failing immediately with `max_context_length=0`

### ALIGN-267 - 2026-03-26 23:25 - The prompt-generation validation run cleared the zero-context bug and exposed the next blocker: GPT-OSS Stage 1 is not emitting the required XML schema

- Failed root:
  - `/ahmed/gpt-oss-promptgen-one-statement-budget-fix`
- Failed child:
  - `/ahmed/gpt-oss-promptgen-one-statement-budget-fix/align-debug_generate_prompts_gpt_oss_120b_budget_fix-prompts_3ebfd52b-f34f8727`
- Terminal error:
  - `RuntimeError: Stage 1 failed for 1 statement(s): ask_clarifying_questions: Stage1 response missing <variation_axes> block`
- What this proves:
  - the earlier `max_context_length=0` failure mode is resolved
  - the child successfully:
    - started native TPU `vllm`
    - loaded `615/615` BF16 safetensor shards via `runai_streamer`
    - reached `vLLM environment ready`
    - entered Stage 1 understanding
    - issued multiple understanding requests without the old context-window crash
- Timing signal from logs:
  - `23:13:20` - `vLLM environment ready`
  - `23:13:20` - `Stage 1: Generating understanding for 1 statements`
  - repeated Stage 1 request attempts at roughly:
    - `23:13:20`
    - `23:14:39`
    - `23:15:57`
    - `23:17:16`
    - `23:18:34`
  - terminal parse failure surfaced at `23:19:56`
- Interpretation:
  - this is now a prompt-following / parser-compatibility problem in GPT-OSS Stage 1, not an infra or context-budget problem
  - the current understanding prompt/parser expects a response containing a `<variation_axes>` block
  - GPT-OSS did not satisfy that schema on any of the retry attempts for `ask_clarifying_questions`
- Operational decision:
  - did **not** auto-resubmit
  - this does not look like a tiny syntax/import bug; it needs inspection of the raw Stage 1 outputs and likely a prompt/parser adaptation for GPT-OSS
- Next action:
  - inspect raw Stage 1 completions for GPT-OSS
  - decide whether to:
    - strengthen the understanding prompt to force the XML block
    - add a GPT-OSS-specific parser fallback
    - or lower reasoning / change generation settings if the model is drifting into non-XML reasoning output

### ALIGN-268 - 2026-03-26 23:58 - Added Stage 1 raw-attempt checkpointing and partial Stage 1 resume so failed reruns only repeat still-unparsed statements

- Problem:
  - Stage 1 understanding failures were opaque because raw completions were discarded before parse
  - resume logic could only restart Stage 1 from scratch because the success checkpoint was written only after the full stage completed
- Change:
  - Stage 1 now writes every raw completion attempt to:
    - `artifacts/checkpoints/understanding_attempts/shard_*.jsonl.gz`
  - this happens before `_parse_understanding_response(...)` in both execution paths:
    - local batched vLLM
    - API / threaded path
  - reruns now scan those saved raw attempts, re-parse them, and recover any statements whose prior raw completion is already valid
  - only statements that still do not have a parseable Stage 1 output are retried
  - the clean Stage 1 success checkpoint remains separate:
    - `artifacts/checkpoints/understandings.jsonl.gz`
- Why this shape:
  - failed raw completions should not contaminate the success checkpoint used for clean resume
  - a separate attempts artifact preserves debuggability while keeping resume semantics strict
- Validation:
  - added regression coverage in `tests/test_alignment.py` for:
    - persisting Stage 1 raw attempts before parse
    - resuming from saved Stage 1 raw attempts and rerunning only pending statements
  - `uv run pytest tests/test_alignment.py -q` -> `101 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/alignment/generate_prompts.py tests/test_alignment.py` -> `OK`
- Practical effect on the next GPT-OSS retry:
  - if one statement already produced a parseable Stage 1 raw completion, the rerun will skip regenerating it
  - if a statement only produced malformed Stage 1 outputs, only that statement will be retried

### ALIGN-269 - 2026-03-27 00:11 - Fresh GPT-OSS one-statement rerun confirmed the new Stage 1 attempt artifacts work and showed the real failure: raw completions are gibberish, not merely slightly off-schema

- Fresh rerun:
  - root:
    - `/ahmed/gpt-oss-promptgen-one-statement-attempts-20260326-164627`
  - experiment record:
    - `gs://marin-us-central1/experiments/generate_prompts_gpt_oss_120b_refactored-9a9639.json`
  - fresh prompt output path:
    - `gs://marin-us-central1/align/debug_generate_prompts_gpt_oss_120b_attempts_20260326-164627/prompts-3b47da`
- Runtime timeline:
  - `23:50:49 UTC` - started vLLM native TPU server
  - `23:52:44 UTC` - completed `615/615` BF16 safetensor shard load
  - `00:01:34 UTC` - `vLLM environment ready`
  - `00:01:34 UTC` - Stage 1 started
  - `00:02:54`, `00:04:13`, `00:05:34`, `00:06:53`, `00:08:12` UTC - five failed Stage 1 attempts
  - terminal error stayed the same:
    - `Stage1 response missing <variation_axes> block`
- New artifact result:
  - the patched Stage 1 checkpointing worked
  - `artifacts/checkpoints/understanding_attempts/` exists and contains:
    - `shard_00000.jsonl.gz`
    - `shard_00001.jsonl.gz`
    - `shard_00002.jsonl.gz`
    - `shard_00003.jsonl.gz`
    - `shard_00004.jsonl.gz`
- What the saved raw completions show:
  - this is **not** a near-miss XML-formatting problem
  - all five responses are long, mostly nonsensical token soup with fragments like:
    - `higlt rested rund/s GABEadura/...`
    - `rundtinlesslybe/le duly nob nob erstapid ...`
    - repeated random mixed subwords, code-ish fragments, and malformed tokens
  - there is no latent `<behavior_understanding>` / `<variation_axes>` structure to salvage with a parser tweak
- Interpretation:
  - the current GPT-OSS local-vLLM prompt-generation path is producing corrupted or severely misaligned text at Stage 1
  - this points upstream of the parser:
    - tokenizer/model mismatch
    - broken prompt rendering / wrong chat template application
    - decode/config issue specific to this local completion path
  - it does **not** look like “tighten the XML prompt and retry”
- Next action:
  - debug the actual served text path with a minimal one-prompt smoke against the same deployed config
  - compare:
    - rendered prompt
    - tokenizer
    - chat template
    - raw completion text
  - until that is fixed, parser fallbacks for Stage 1 are unlikely to help

### ALIGN-270 - 2026-03-27 00:28 - Switched the next debugging loop to GPT-OSS 20B, confirmed the pinned vLLM-serving subset download, and prepared a side-by-side Harmony smoke to test whether Marin's current `/v1/completions` path is the actual integration bug

- Why switch to 20B:
  - the 120B reruns already proved the failure shape
  - the next discriminator is serving-path correctness, not capacity
  - 20B is cheaper and faster for repeated TPU-vLLM startup/debug iterations
- Source review:
  - OpenAI Harmony docs say GPT-OSS should work through provider-backed inference solutions like vLLM, which should handle Harmony formatting automatically
  - the GPT-OSS model cards also say the models were trained on Harmony and should not be used without it
  - this means “GPT-OSS works with vLLM” and “our current Marin GPT-OSS path is broken” are not contradictory
- Current hypothesis:
  - Marin is not using a pure provider-managed vLLM path for GPT-OSS today
  - the current local batched alignment path:
    - locally renders the Hugging Face chat template
    - sends the rendered prompt to `/v1/completions`
    - treats the return value as plain assistant text
  - for GPT-OSS, that hybrid may be invalid even though vLLM itself is supported
- New assets prepared:
  - pinned 20B model entries in `experiments/models.py`
  - `experiments/download_gpt_oss_20b_vllm.py`
  - `experiments/gpt_oss_20b_tpu.py`
  - `experiments/gpt_oss_20b_harmony_compare_smoke.py`
- Download result:
  - Iris job:
    - `/ahmed/download-gpt-oss-20b-vllm-20260326-172105`
  - terminal state:
    - `JOB_STATE_SUCCEEDED`
  - output model path:
    - `gs://marin-us-central1/models/unsloth--gpt-oss-20b-BF16-vllm--cc89b3e7fd423253264883a80a4fa5abc619649f`
- Immediate next action:
  - run the 20B Harmony comparison smoke on the same Stage 1 understanding prompt
  - capture:
    - locally rendered prompt
    - current Marin `/v1/completions` output
    - provider-managed `/v1/chat/completions` output
  - if `/v1/chat/completions` is coherent while `/v1/completions` is gibberish, the next fix is to add a GPT-OSS-aware chat path to batched local serving

### ALIGN-271 - 2026-03-27 00:39 - The 20B Harmony comparison smoke narrowed the failure further: `/v1/completions` is gibberish, but `/v1/chat/completions` is also not usable because the current vLLM TPU server is not launching with GPT-OSS's dedicated Harmony reasoning parser

- Successful comparison run:
  - root:
    - `/ahmed/gpt-oss-20b-harmony-compare-20260326-1728`
  - output path:
    - `gs://marin-us-central1/align/debug_gpt_oss_20b_harmony_compare_20260326_1728/compare-ecc59c`
- What the smoke captured:
  - `rendered_prompt.txt`
  - `completions_path.json`
  - `chat_completions_path.json`
  - `artifacts/vllm_metrics.json`
- Key results:
  - the rendered prompt is a Harmony conversation ending in `<|start|>assistant`
  - `/v1/completions` still returns long incoherent token soup, confirming the existing Marin path is invalid for GPT-OSS as currently implemented
  - `/v1/chat/completions` is **not** returning a normal answer yet either:
    - `finish_reason = "length"`
    - `completion_tokens = 1024`
    - `message.content = null`
    - `message.reasoning_content = null`
- New diagnosis:
  - this is not just an endpoint-selection bug
  - Marin currently launches `vllm serve` for GPT-OSS without `--reasoning-parser openai_gptoss`
  - vLLM has a dedicated GPT-OSS reasoning parser for Harmony outputs, and omitting it explains why:
    - raw completions look uninterpreted/broken
    - chat completions consume tokens but fail to surface usable `content` / `reasoning_content`
- Code change prepared:
  - `lib/marin/src/marin/inference/vllm_server.py` now auto-adds:
    - `--reasoning-parser openai_gptoss`
    - for GPT-OSS model paths / ids
  - added regression coverage in `tests/test_vllm_server.py`
- Validation:
  - `uv run pytest tests/test_vllm_server.py -q` -> `4 passed`
  - `./infra/pre-commit.py --fix lib/marin/src/marin/inference/vllm_server.py tests/test_vllm_server.py` -> `OK`
- Immediate next action:
  - rerun the exact same 20B Harmony comparison smoke with the parser-enabled vLLM launch
  - if `/v1/chat/completions` becomes coherent, move GPT-OSS alignment off the current `/v1/completions` path

### ALIGN-272 - 2026-03-27 00:49 - The parser-enabled rerun ruled out the cleanest server-side fix: adding `--reasoning-parser openai_gptoss` alone does not change the broken payload shape, so the next discriminator is lowering GPT-OSS reasoning effort

- Parser-enabled rerun:
  - root:
    - `/ahmed/gpt-oss-20b-harmony-compare-parser-20260326-1739`
  - output path:
    - `gs://marin-us-central1/align/debug_gpt_oss_20b_harmony_compare_parser_20260326_1739/compare-c031e0`
- What changed:
  - `lib/marin/src/marin/inference/vllm_server.py` now auto-adds:
    - `--reasoning-parser openai_gptoss`
    - for GPT-OSS model ids/paths
- What did **not** change:
  - `/v1/completions` is still the same gibberish token soup
  - `/v1/chat/completions` is still:
    - `finish_reason = "length"`
    - `completion_tokens = 1024`
    - `message.content = null`
    - `message.reasoning_content = null`
- Interpretation:
  - the missing reasoning parser was a plausible gap, but it is not sufficient to make GPT-OSS usable on this path
  - the stronger live hypothesis is now:
    - GPT-OSS is spending the whole budget in the hidden Harmony reasoning channel and never reaching a final answer
    - the raw `/v1/completions` path is simply exposing that hidden, non-user-facing token stream as garbage text
- Why this fits the evidence:
  - the chat template defaults GPT-OSS to `Reasoning: medium`
  - the chat completion consumes exactly the whole `1024` token budget yet exposes no final content
  - parser support being present but still seeing `content = null` is consistent with “no final channel produced before truncation”
- Next action:
  - patch the comparison smoke to let us set `reasoning_effort`
  - rerun with:
    - `reasoning_effort = low`
  - if low effort produces a normal final answer on `/v1/chat/completions`, then the near-term GPT-OSS integration path is:
    - use a GPT-OSS-aware chat path
    - explicitly set low reasoning effort for prompt generation

### ALIGN-273 - 2026-03-27 00:57 - Lowering GPT-OSS reasoning effort to `low` still did not produce a usable final answer on the TPU-native vLLM path, so the current problem is deeper than endpoint choice, missing parser, or hidden reasoning budget alone

- Low-reasoning rerun:
  - root:
    - `/ahmed/gpt-oss-20b-harmony-compare-low-20260326-1750`
  - output path:
    - `gs://marin-us-central1/align/debug_gpt_oss_20b_harmony_compare_low_20260326_1750/compare-81c174`
- What changed:
  - the comparison smoke now explicitly threads:
    - `reasoning_effort`
  - this rerun used:
    - `reasoning_effort = low`
- Result:
  - `/v1/completions` is still gibberish token soup
  - `/v1/chat/completions` is still:
    - `finish_reason = "length"`
    - `completion_tokens = 1024`
    - `message.content = null`
    - `message.reasoning_content = null`
- Interpretation:
  - the remaining issue is not just “GPT-OSS spends too many tokens in hidden reasoning at medium effort”
  - the current TPU-native `vllm serve` HTTP path remains non-viable for GPT-OSS Stage 1 even after:
    - using `/v1/chat/completions`
    - enabling `--reasoning-parser openai_gptoss`
    - lowering `reasoning_effort` to `low`
- Additional side result:
  - the `v6e-8` fallback attempt failed immediately due region overlap constraints between the `us-central1` model artifact and TPU-capable DAG regions for that family, so it did not change the diagnosis
- Current best diagnosis:
  - this is now most likely a deeper GPT-OSS incompatibility/bug in the TPU-native `vllm-tpu` path, a tokenizer/decode mismatch on this BF16 checkpoint, or another missing GPT-OSS-specific serving feature beyond the reasoning parser
- Recommended next move:
  - stop treating the current batched local TPU vLLM HTTP path as the likely production answer for GPT-OSS
  - either:
    - compare against a known-good GPT-OSS serving path outside this TPU-native stack
    - or use a direct Harmony-aware render/generate/parse path rather than relying on the OpenAI-compatible HTTP layer

### ALIGN-274 - 2026-03-27 01:16 - The GPT-OSS TPU jobs are not "plain vLLM" under the hood: they already run through `tpu_inference`'s JAX `flax_nnx` model path beneath `vllm serve`, which means a Harmony-aware fix can change the serving layer without abandoning TPU-backed inference

- Key architectural finding:
  - the GPT-OSS TPU experiment configs pin:
    - `model_impl_type = "flax_nnx"`
    - and install:
      - `tpu_inference @ git+https://github.com/marin-community/tpu-inference.git@ahmed/gpt-oss-tpu-bringup-v5`
- Marin carries that override into the local serve environment via:
  - `MODEL_IMPL_TYPE=flax_nnx`
- In `tpu_inference.models.common.model_loader.get_model(...)`:
  - GPT-OSS is registered as:
    - `GptOssForCausalLM -> GptOss`
  - `MODEL_IMPL_TYPE=flax_nnx` selects:
    - `get_flax_model(...)`
  - not:
    - the generic vLLM PyTorch model-wrapper path
- In `tpu_inference.runner.tpu_runner.TPUModelRunner`:
  - TPU mesh setup
  - model execution
  - KV-cache management
  - and token sampling
  - are all implemented inside `tpu_inference`
- Practical interpretation:
  - the active stack is:
    - Marin batched client
    - `vllm serve`
    - `vllm-tpu` scheduler / OpenAI-compatible server
    - `tpu_inference` TPU runner
    - `tpu_inference` JAX `GptOss`
  - so "use a direct Harmony-aware path" does **not** imply "no TPU inference"
  - the smaller-change option is:
    - keep the current TPU execution engine
    - but replace or bypass the GPT-OSS prompt/render/parse layer above it
  - the larger-change option is:
    - bypass `vllm serve` entirely
    - and drive `tpu_inference` more directly
- Immediate next implication:
  - the likely failure layer is above core TPU execution:
    - tokenizer/rendering
    - decode/output shaping
    - or the OpenAI-compatible HTTP response path
  - rather than:
    - "GPT-OSS cannot run on this TPU inference stack at all"

### ALIGN-275 - 2026-03-30 21:05 - The GPT-OSS alignment branch is now rebased onto `main` semantically, validated on the current TPU/vLLM path without the custom `tpu_inference` fork, and reduced to one shared helper plus two canonical E2E entrypoints

- `main` merge and serving-path resolution:
  - merged `origin/main` into the branch and resolved the only hard overlap in:
    - `lib/marin/src/marin/inference/vllm_server.py`
  - kept the `main` structure for:
    - shared readiness polling
    - consolidated vLLM env defaults
    - server cleanup/finally behavior
  - re-applied only the GPT-OSS-critical deltas on top:
    - `env_overrides` plumbing
    - CLI forwarding for:
      - `tensor_parallel_size`
      - `additional_config`
      - `hf_overrides`
      - `tokenizer`
    - GPT-OSS auto-injection of:
      - `--reasoning-parser openai_gptoss`
    - opt-in native stderr teeing via config, default off
  - deleted the old remote local staging path completely:
    - no `stage_remote_model_locally`
    - no `_stage_remote_directory()`
- Post-merge validation:
  - a fresh one-statement GPT-OSS 20B E2E smoke succeeded end-to-end after the
    merge with `main`
  - this covered:
    - spec
    - prompts
    - chosen
    - rejected
    - judgments
    - preference pairs
  - the merged `vllm_server.py` path remained functional for:
    - local vLLM serve on TPU
    - GPT-OSS `/v1/chat/completions`
    - reasoning parser injection
    - chosen/rejected generation
    - pair construction
- `tpu_inference` de-fork result:
  - removed the GPT-OSS TPU helper dependency on the git-ref fork of
    `marin-community/tpu-inference`
  - moved the untracked local `tpu_inference/` workspace overlay out of the job
    bundle path so it no longer shadowed the installed package
  - reran the smallest GPT-OSS 20B one-statement E2E smoke against the stock
    lockfile `tpu-inference`
  - validated under stock `tpu-inference`:
    - prompt generation
    - chosen generation
    - rejected generation
  - conclusion:
    - the old custom `tpu_inference` fork is no longer required for the current
      GPT-OSS TPU local-vLLM path
    - the current supported path is:
      - `model_impl_type="vllm"`
      - `/v1/chat/completions`
      - top-level `reasoning_effort`
      - `--reasoning-parser openai_gptoss`
- Final experiment surface cleanup:
  - collapsed the old model-specific helper files into one shared module:
    - `experiments/gpt_oss_tpu.py`
  - canonical kept entrypoints:
    - `experiments/align_gpt_oss_20b_e2e_one_statement.py`
    - `experiments/align_gpt_oss_120b_full_spec_e2e.py`
  - the shared helper now owns:
    - GPT-OSS TPU defaults for 20B and 120B
    - rejected-model presets for:
      - Mixtral-8x7B-Instruct
      - Heretic GPT-OSS 20B
  - rejected-model selection is now a preset knob inside the canonical scripts,
    rather than separate duplicate experiment files
  - deleted duplicate GPT-OSS experiment entrypoints, scratch smokes, download
    helpers, and fork-era debug clutter that were no longer part of the minimal
    supported surface
- Final status:
  - `origin/main` is merged into this branch
  - the branch keeps only one GPT-OSS TPU helper and two canonical GPT-OSS E2E
    scripts
  - the GPT-OSS TPU path no longer depends on a custom `tpu_inference` fork
  - the remaining intentionally-kept variants are the canonical smoke and full
    E2E entrypoints with Mixtral/Heretic rejected selection handled in-code
