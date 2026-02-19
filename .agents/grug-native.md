# Grug Native Plan (for `grugformer_speedrun.py`)

## Goal
Make `experiments/speedrun/grugformer_20260129/grugformer_speedrun.py` run as a **grug-native** training entrypoint:
- no `LmConfig`
- no `GrugWrapper`
- no `default_speedrun()` / `default_train()` dependency chain
- keep core Levanter utilities that are still high leverage (mesh setup, trackers, data/mixture ingestion, eval logic, TensorStore checkpoints)

Critical constraint:
- Build on `codex/grug-lm-example-data-wrap` data-path work; do not re-invent parallel adapters.

## Scope and Non-Goals
In scope:
- Native training loop for grug arrays + pytrees.
- Native dataset batches (`jax.Array`), including mixture schedules.
- Perplexity eval parity with `lib/levanter/src/levanter/eval.py`.
- LM harness integration (loglikelihood-only in v1).
- Preemption-safe checkpoint/resume.
- Marin executor integration as the end state (with `uv run -m` bring-up allowed first).
- Per-model training loop ownership (no generic multi-model config registry requirement).

Out of scope for now:
- HF import/export path.
- Generation-heavy eval support.
- Full removal of Levanter data internals on day 1.
- Gradient accumulation (explicitly skipped in v1).

## Current Coupling Map
Today `grugformer_speedrun.py` depends on:
1. `LmConfig` subclass in experiment file.
2. `GrugWrapper` to satisfy `LmHeadModel`.
3. `marin.speedrun.default_speedrun()` -> `experiments.default_train()` -> `levanter.main.train_lm`.
4. `LmExample`/NamedArray contracts in eval and training stack.

This is the main cut to make: **from model-interface-native (`LmHeadModel`) to batch/protocol-native (plain arrays).**

## Existing Work To Reuse (`codex/grug-lm-example-data-wrap`)
Concrete commits to treat as baseline:
- `c51026e82` ("Refactor LM data path around GrugLmExample")
- `002eb2c66` ("Normalize eval datasets before batching")

What those commits already provide:
1. `lib/levanter/src/levanter/data/text/examples.py`
   - `GrugLmExample` (raw-array example protocol)
   - named<->grug conversion helpers
2. `lib/levanter/src/levanter/data/text/datasets.py`
   - datasets produce `GrugLmExample` internally
   - `NamedLmDataset` wraps back to `LmExample` at train/eval boundaries where needed
3. `lib/levanter/src/levanter/eval.py`
   - eval accepts both `LmExample` and `GrugLmExample` inputs
4. tests in `lib/levanter/tests/test_text.py` and `lib/levanter/tests/test_eval.py`

Plan implication:
- Phase 1 should start from this `GrugLmExample` path, not from a fresh `GrugBatch` invention.

## Resolved Decisions
1. Bring-up can start with `uv run -m`, but Marin executor integration is the target.
2. Skip gradient accumulation in v1.
3. LM harness v1 is loglikelihood-only (no `generate_until` requirement).
4. EMA is desired if implementation stays simple and does not complicate the native loop.

## Guiding Constraint
We should prefer a new/hacked model-specific loop over extending Levanter’s generic config-heavy model instantiation framework. Grug-native is intentionally explicit and model-owned.

## Keep These Features (and one more)
Keep (requested):
- easy mesh init
- tracker integration
- mixture datasets from current pipeline
- perplexity eval path
- lm-eval-harness support (ideally loglikelihood path)
- preemption robustness

Also keep:
- deterministic resume semantics for data (`iter_from_step` equivalent behavior)
- throughput/perf metrics logging (step time, loading time, tokens/sec)

## Architecture Direction
Use a small protocol layer around the grug core:

```python
from levanter.data.text.examples import GrugLmExample

# Optional naming convenience only; do not create a competing structure.
GrugBatch = GrugLmExample

@dataclass(frozen=True)
class GrugTrainState:
    step: int
    params: GrugModelParameters
    opt_state: optax.OptState
    training_key: jax.Array

class CheckpointIO(Protocol):
    def save(self, path: str, state: GrugTrainState, *, metadata: dict) -> None: ...
    def restore_latest(self, base_path: str, state_shape: GrugTrainState) -> GrugTrainState | None: ...
```

Design rule: grug core remains in `levanter.grug.*`; integration adapters live outside core.

Secondary rule: avoid introducing a new generic trainer abstraction for multiple model families. Keep the runner Grug-specific and copy/hack-friendly.

Data protocol rule:
- Prefer `GrugLmExample` from `levanter.data.text.examples` as the canonical batch/example type.
- If `GrugBatch` exists, make it an alias/thin view over `GrugLmExample` fields rather than a competing protocol.

## Proposed File Plan
Add:
- `lib/levanter/src/levanter/grug_native/config.py`
- `lib/levanter/src/levanter/grug_native/data.py` (thin orchestration over `GrugLmExample` loaders)
- `lib/levanter/src/levanter/grug_native/train.py`
- `lib/levanter/src/levanter/grug_native/checkpoint.py`
- `lib/levanter/src/levanter/grug_native/eval_hooks.py` (thin wiring over existing `levanter.eval.TaggedEvaluator`)
- `lib/levanter/src/levanter/grug_native/eval_harness.py` (optional in phase 2)

Change:
- `experiments/speedrun/grugformer_20260129/grugformer_speedrun.py` to call grug-native runner config directly.

Keep temporarily:
- `lib/levanter/src/levanter/models/grug_wrapper.py` as compatibility bridge for legacy runs.

## Migration Phases

## Phase 0: Land/Adopt Data-Wrap Baseline
Deliverables:
1. Ensure `c51026e82` and `002eb2c66` semantics are present in the target branch (merge/cherry-pick/equivalent).
2. Confirm tests covering `GrugLmExample` conversions + eval compatibility pass.

Acceptance checks:
- `lib/levanter/src/levanter/data/text/examples.py` exists with `GrugLmExample`.
- `LmDataConfig` path emits grug-native examples internally.
- `TaggedEvaluator` can consume `GrugLmExample` datasets.

## Phase 1: Native Data Batches (No Wrapper in Train Path)
Deliverables:
1. Reuse `GrugLmExample` as native batch protocol in the grug-native training loop.
2. Loader path that reuses `LmDataConfig`/mixture logic and directly consumes `train_sets(...)` grug examples.
3. Stable resume cursor by step (same semantics as `iter_from_step`).

Implementation sketch:

```python
train_sets = data_cfg.train_sets(Pos, initial_batch_size=batch_size, key=data_key)  # GrugLmExample datasets
mixture = MixtureDataset(..., datasets=train_sets, ...)
batch: GrugLmExample = next(loader)
```

Acceptance checks:
- mixture stage transitions remain correct
- dataset weights logged as before
- no `GrugWrapper` in train path

## Phase 2: Native Trainer + Checkpoint Protocol
Deliverables:
1. Minimal `GrugTrainer` loop with optax + explicit mesh.
2. Tracker hooks via `levanter.tracker` API.
3. Checkpoint/resume via TensorStore serialization with atomic step dirs.
4. Optional EMA path if it is straightforward (no broad trainer abstraction work).
5. Explicit optimizer config surface in native config (default AdamW, easy swap to Muon/etc.).

Implementation sketch:

```python
def create_mesh(global_batch_size: int, axes=("data", "model")) -> Mesh:
    # explicit-axis mesh init with divisibility guard
    ...

@jax.jit
def train_step(state: GrugTrainState, batch: GrugLmExample, cfg: GrugModelConfig):
    def loss_fn(params):
        mask = batch.attn_mask
        return grug_loss_fn(params, batch.tokens, batch.loss_weight, cfg, mask=mask, reduction="mean")
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    updates, opt_state = optimizer.update(grads, state.opt_state, state.params)
    params = optax.apply_updates(state.params, updates)
    return dataclasses.replace(state, step=state.step + 1, params=params, opt_state=opt_state), loss
```

Optimizer placement:
- `GrugNativeRunConfig` should include `optimizer: GrugOptimizerConfig`.
- `GrugTrainerConfig` remains loop/timing/checkpoint knobs.
- `GrugOptimizerConfig.build(num_steps)` returns the concrete optax transform.

Checkpoint protocol requirements:
- `step-{n}` directories
- temporary checkpoint + periodic permanent checkpoint
- metadata includes run id, config hash, and data offset
- recovery chooses latest valid checkpoint and ignores partial/incomplete ones

EMA requirement (conditional):
- Include EMA parameters in checkpoint if enabled.
- Eval path can choose current params vs EMA params via a small flag/config.
- If EMA adds substantial complexity, defer to a follow-up PR after baseline native parity.

## Phase 3: Perplexity Eval Parity
Deliverables:
1. Reuse the existing hybrid `levanter.eval.TaggedEvaluator` path from 2727 (`GrugLmExample` + `LmExample` support).
2. Add only thin native-runner hook/wiring for invoking tagged eval.
3. Keep existing tagging conventions so dashboards do not break.

Implementation strategy:
- Do not fork/re-port evaluator logic unless blocked.
- Use the 2727 hybrid `TaggedEvaluator` directly from native runner callbacks.
- Keep key names aligned with current eval logs (`eval/loss`, `eval/macro_loss`, `eval/*/bpb`).

Acceptance checks:
- metric parity against current eval callback on same checkpoint and eval sets
- same aggregate tags/hierarchy behavior

## Phase 4: LM Harness (Loglikelihood First)
Deliverables:
1. `grug_native/eval_harness.py` entry for loglikelihood tasks.
2. No generation requirement for v1.

Strategy:
- Reuse request packing logic from `levanter.eval_harness` where possible.
- Implement worker evaluation against native batch protocol.
- Defer `generate_until` unless needed for a concrete task list.

Acceptance checks:
- can run a known non-generation harness subset end-to-end
- logs land in tracker with existing metric naming pattern

## Phase 5: Convert `grugformer_20260129` to Native
Deliverables:
1. Replace `LmConfig` subclass with native run config dataclass.
2. Remove `GrugWrapper` import/usage.
3. Replace `default_speedrun(...)` call with native runner invocation.
4. Provide Marin executor step wrapper for this native runner (if bring-up started with `uv run -m`).

Target shape:

```python
cfg = GrugNativeRunConfig(
    model=GrugModelConfig(...),
    data=existing_lm_data_config,
    trainer=GrugTrainerConfig(...),
    optimizer=GrugOptimizerConfig(...),
    eval=GrugEvalConfig(...),  # wired to existing TaggedEvaluator
)
run_grug_native(cfg)
```

## PR Breakdown (recommended)
1. Land/adopt `codex/grug-lm-example-data-wrap` data/eval baseline (`GrugLmExample` path).
2. Protocol + config skeleton (`grug_native/config.py`) and mesh helper.
3. Data orchestration over `GrugLmExample` loaders (no duplicate adapters).
4. Trainer loop + tracker + checkpointing (+ EMA if simple) + explicit optimizer config.
5. Native eval wiring using existing hybrid `TaggedEvaluator`.
6. Harness loglikelihood integration.
7. Switch `grugformer_20260129` script to native path and remove wrapper dependency there.
8. Marin executor integration for native runner (if not already done in step 7).

## Concrete Checklist

### PR1: Adopt 2727 Data/Eval Baseline (No New Native Loop Yet)
Goal:
- Make this branch match the `GrugLmExample` + hybrid `TaggedEvaluator` behavior from `codex/grug-lm-example-data-wrap`.

Files to add/change:
1. `lib/levanter/src/levanter/data/text/examples.py`
   - Add:
     - `GrugLmExample`
     - `grug_attention_mask_from_named`
     - `named_attention_mask_from_grug`
     - `grug_lm_example_from_named`
     - `named_lm_example_from_grug`
2. `lib/levanter/src/levanter/data/text/datasets.py`
   - Add:
     - `_single_cpu_sharding()`
     - `NamedLmDataset`
   - Update:
     - `CausalLmDataset`, `PrebuiltLmDataset`, `PackedTokenDataset`, `ChatDataset` to emit `GrugLmExample` internally.
     - `LmDataConfig.train_set()` and `LmDataConfig.validation_sets()` to return named wrappers where required.
3. `lib/levanter/src/levanter/data/text/__init__.py`
   - Export new `examples.py` public API.
4. `lib/levanter/src/levanter/eval.py`
   - Ensure hybrid path:
     - `LmEvalExample = LmExample | GrugLmExample`
     - `_ensure_named_lm_example(...)`
     - `_ensure_grug_lm_example(...)`
     - `TaggedEvaluator` normalizes datasets before batching.
5. Tests:
   - `lib/levanter/tests/test_text.py`:
     - `test_unnamed_lm_example_parity_with_named`
     - `test_named_unnamed_lm_example_roundtrip`
     - `test_train_set_last_mile_wraps_to_named`
   - `lib/levanter/tests/test_eval.py`:
     - `test_tagged_evaluator_accepts_grug_lm_examples`

Validation commands:
```bash
uv run pytest lib/levanter/tests/test_text.py -k "unnamed_lm_example or roundtrip or last_mile"
uv run pytest lib/levanter/tests/test_eval.py -k "grug_lm_examples"
uv run pytest lib/levanter/tests/grug -q
```

Exit criteria:
- Data path is grug-native internally (`GrugLmExample`), with compatibility wrappers at legacy call sites.
- Tagged eval accepts both example types without forks.

### PR2: Native Grug Runner for `grugformer_20260129`
Goal:
- Run the speedrun without `LmConfig`, `GrugWrapper`, or `default_speedrun()` while preserving data/trackers/eval/checkpoint behavior.

Files to add:
1. `lib/levanter/src/levanter/grug_native/config.py`
   - Add dataclasses:
     - `GrugOptimizerConfig` (default AdamW; `build(num_steps)` -> optax transform)
     - `GrugTrainerConfig` (steps, batch, log/eval/checkpoint intervals, EMA toggle)
     - `GrugEvalConfig` (tagged eval settings + optional harness config)
     - `GrugNativeRunConfig` (model, data, optimizer, trainer, eval, resources/output)
2. `lib/levanter/src/levanter/grug_native/data.py`
   - Add:
     - `build_train_dataset(cfg, Pos, key) -> AsyncDataset[GrugLmExample]`
     - `build_train_loader(dataset, mesh, batch_size, axis_mapping)`
     - `iter_from_step(loader, start_step)` helper for resume behavior parity
3. `lib/levanter/src/levanter/grug_native/checkpoint.py`
   - Add:
     - `restore_latest_or_initialize(...)`
     - `save_step_checkpoint(...)`
     - optional `save_temporary_checkpoint(...)` policy helper
4. `lib/levanter/src/levanter/grug_native/eval_hooks.py`
   - Add:
     - `build_tagged_eval_callback(...)` that uses existing `levanter.eval.cb_tagged_lm_evaluate`.
5. `lib/levanter/src/levanter/grug_native/train.py`
   - Add:
     - `create_mesh(...)`
     - `init_train_state(...)`
     - `train_step(...)`
     - `run_grug_native(cfg)` main loop
6. Optional (same PR if small):
   - `lib/levanter/src/levanter/grug_native/eval_harness.py`
     - thin wrapper to call existing loglikelihood harness path.

Files to change:
1. `experiments/speedrun/grugformer_20260129/grugformer_speedrun.py`
   - Replace `LmConfig`/`GrugWrapper`/`default_speedrun` usage with `GrugNativeRunConfig` + `run_grug_native`.
2. (If needed for launch ergonomics) `lib/levanter/src/levanter/grug_native/main.py`
   - CLI entrypoint for `uv run -m ...`.

Core functions to implement in PR2 (minimum set):
- `GrugOptimizerConfig.build`
- `run_grug_native`
- `train_step`
- `build_tagged_eval_callback`
- `restore_latest_or_initialize`

Validation commands:
```bash
uv run pytest lib/levanter/tests/grug -q
uv run pytest lib/levanter/tests/test_eval.py -k "grug_lm_examples"
uv run python -m experiments.speedrun.grugformer_20260129.grugformer_speedrun
```

Exit criteria:
- `experiments/speedrun/grugformer_20260129/grugformer_speedrun.py` imports none of:
  - `levanter.models.grug_wrapper`
  - `levanter.models.lm_model.LmConfig`
  - `marin.speedrun.speedrun.default_speedrun`
- End-to-end train + tagged perplexity eval + resume works in native runner.

## Test Plan
Unit:
- `levanter.data.text.examples`: conversion correctness (`tokens`, `loss_weight`, mask/segment ids) [baseline tests]
- `grug_native/data`: loader/orchestration correctness on `GrugLmExample`
- `grug_native/train`: one-step update and checkpoint roundtrip
- `grug_native/eval_hooks`: callback wiring tests around existing `TaggedEvaluator`

Integration:
- 2-step smoke train on synthetic + real token cache
- resume from checkpoint after forced stop
- one lm-harness loglikelihood task (if phase 4 enabled)

Regression:
- Existing `lib/levanter/tests/grug/*` unchanged and passing.

## Definition of Done for “Fully Grug Native” Speedrun
- `experiments/speedrun/grugformer_20260129/grugformer_speedrun.py` has zero imports of:
  - `levanter.models.grug_wrapper`
  - `levanter.models.lm_model.LmConfig`
  - `marin.speedrun.speedrun.default_speedrun`
- Training, eval perplexity, and checkpoint resume run end-to-end with native runner.
- LM harness loglikelihood eval runs end-to-end from the native path.
- Mixture datasets from current pipeline are supported without bespoke per-experiment wrappers.
- Run can be launched through Marin executor (direct `uv run -m` path may still exist for debugging).
