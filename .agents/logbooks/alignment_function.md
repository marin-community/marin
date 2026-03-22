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
