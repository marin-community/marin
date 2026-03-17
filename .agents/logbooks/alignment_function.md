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
