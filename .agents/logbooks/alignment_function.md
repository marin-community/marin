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
