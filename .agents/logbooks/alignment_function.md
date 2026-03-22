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
- **Results (confidence: replicated — ran multiple times during debugging):**
  - Stage 1: 8 variation axes (6 behavior-specific + 2 demographic)
  - Stage 2: 72 pairwise covering configs, 688/688 tuples covered
  - Stage 3: 74 prompts extracted
  - Chosen: 74 responses (with spec guidance)
  - Rejected: 74 responses (without spec guidance)
  - Judge: 15 preference pairs after filtering (20% pass rate)
  - All artifacts saved to GCS: understanding.json, ideation.json, summary.json
  - Output: `gs://marin-us-central1/align/openai_spec_smoke/`
- **Bugs found and fixed during validation:**
  1. `_load_behavior_statements` crashed on JSONL spec files (`json.load` vs JSONL)
  2. All file I/O used `Path()` which doesn't work with `gs://` paths — switched to `iris.marin_fs.url_to_fs` + `zephyr.write_jsonl_file`
  3. Spec file not uploaded to GCS — added spec upload ExecutorStep
  4. `_load_responses` indentation bug → `UnboundLocalError` on empty shards
  5. `model_id=` kwarg not renamed to `config=` after InferenceConfig refactor (4 call sites)
  6. `tenacity` missing in remote Iris jobs — added as explicit dep
  7. `OPENAI_API_KEY` not forwarded to child Iris jobs — added `_llm_env_vars()` to `@remote` env_vars
- **Confidence:** `replicated` — pipeline ran successfully multiple times after fixes

### 2026-03-21 — ALIGN-007: VLLMConfig for All Model Roles
- **Action:** Extended InferenceConfig support to ideation/extract/judge models (not just teacher/rejected). Added `vllm_engine()` context manager for efficient engine reuse across multiple calls within a step. Auto-selects TPU resources and single-threaded execution for vLLM.
- **Commit:** 2a493aa6a
- **Debug script:** `experiments/align_debug_vllm.py` — uses Llama 3.1 8B Instruct via vLLM for all roles
- **Status:** Not yet validated — no TPU capacity available during testing. Job queued but killed.

### 2026-03-21 — ALIGN-008: Intermediate Artifact Persistence
- **Action:** Pipeline now saves per-statement artifacts alongside prompts:
  - `artifacts/<stmt>/understanding.json` — Stage 1 output (axes, understanding, motivation)
  - `artifacts/<stmt>/ideation.json` — Stage 2 output (covering plan, scenarios, rubrics)
  - `artifacts/summary.json` — overview with axis names, config counts, coverage stats
- **Commit:** 9bdb3b892
- **Verified:** Artifacts correctly written to GCS and contain full pipeline state
- **Next action:** Validate vLLM path when TPU capacity available. Run full 46-statement pipeline.

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
- **Results:** 71 prompts → 71 chosen → 71 rejected → 13 preference pairs. All artifacts present.

### 2026-03-21 — ALIGN-010: Simplify Round 2
- **Action:** Second review pass:
  - Replaced manual `__enter__`/`__exit__` with `contextlib.ExitStack` (fixed buggy identity check)
  - Fixed quadratic per-axis counting in `compute_coverage_stats` → single pass O(C*A)
  - Renamed `_get_or_create_vllm_engine` to public (was private but imported cross-module)
  - Removed stale "Ported from bloom" comments from 7 files
  - Extracted hardcoded `max_tokens=16000` into config fields (`concretize_max_tokens`, `extract_max_tokens`)
  - Moved all non-guard imports to top of file
- **Validation job:** `/ahmed/iris-run-align_openai_spec_smoke-20260322-020230` (us-central1)
- **Results:** 72 prompts → 72 chosen → 72 rejected → 26 preference pairs (36% pass rate). All artifacts present.
- **Confidence:** `replicated` — pipeline validated 3 times across refactors with consistent results
- **Next action:** Validate vLLM path when TPU capacity available. Run full 46-statement pipeline.
