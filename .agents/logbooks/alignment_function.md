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
