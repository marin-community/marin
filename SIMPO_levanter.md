# SIMPO Implementation Plan for Levanter (Reference‑Free)

## Goal & Constraints
- **Implement SimPO (reference‑free)** exactly as described in `SIMPO.md`: length‑normalized log‑probabilities and margin term `gamma_beta_ratio` only (core sigmoid loss).
- **Remove DPO entirely**: SimPO is the only preference‑optimization entrypoint.
- **Keep data type names** (`DpoExample`, `PreferencePairDataset`) to avoid API/cache churn.
- **Use SimPO.md Llama‑3 defaults** in new YAMLs (`beta=2.0`, `gamma_beta_ratio=0.5`, `learning_rate=6e-7`).
- **Keep `DPO.md` / `DPO_claude.md`** as historical docs.
- **Revert Haliax back to `main`** (remove all DPO‑driven patches).
- **Tag runs as `dpo`** (legacy tagging convention): do **not** tag runs as `simpo`.

## Confirmed Decisions (2026-01-24)
- **Haliax**: revert to `main` (remove the 3 diffs listed below).
- **Tagging**: keep `dpo` tags (even though the algorithm is SimPO).
- **DPO removal**: delete DPO entrypoints and all wiring/imports (no compat shims).

---

## Haliax: Revert Back to `main`
These are the **only** Haliax diffs vs `main`; we can fully revert them because SimPO is reference‑free and doesn’t need any of the DPO‑specific workarounds.

### 1) `lib/haliax/src/haliax/nn/scan.py`
**Change**: Remove the `auto_sharded` call added after `vmap` inside `Stacked.init`.  
**Reason**: This was added to reduce OOMs for **two‑model DPO**; SimPO runs a single model and should match `main`.

```python
# remove this line to match main
stacked = haliax.auto_sharded(stacked)
```

### 2) `lib/haliax/src/haliax/partitioning.py`
**Changes**:
- Remove the `batch_dim` early‑return for vmap tracers.
- Remove the `None` guard in `pspec_for` for `NamedArray(array=None)`.

**Reason**: Both were introduced to tolerate DPO’s reference‑model partitioning and sharding‑inside‑vmap. SimPO doesn’t trigger those code paths; revert to `main`.

```python
# remove this early return
if getattr(named.array, "batch_dim", None) is not None:
    return named

# remove this None guard
if isinstance(node, NamedArray):
    if not is_jax_array_like(node.array):
        return None
```

### 3) `lib/haliax/src/haliax/quantization.py`
**Changes**:
- Remove `NamedArray` from the `is_leaf` in `partition_for_grad_overwrite`.
- Remove `update is None` handling and the `NamedArray` leaf special‑case in `apply_updates`.

**Reason**: These were added to prevent `NamedArray(None)` placeholders from DPO’s frozen reference model. With SimPO (single model, no reference), revert to `main`.

```python
def is_leaf(v):
    return isinstance(v, OverwriteWithGradient)  # remove NamedArray

def _apply_update(tree, update, overwrite):
    if overwrite is not None:
        return overwrite
    return eqx.apply_updates(tree, update)  # remove update is None branch
```

---

## Levanter: New SimPO Entrypoint (No Reference Model)

### 4) **Rename** `lib/levanter/src/levanter/main/train_dpo.py` → `train_simpo.py`
**Reason**: You requested the SimPO entrypoint to replace DPO.  
**Net effect**: A new `TrainSimpoConfig`, SimPO loss function, and removal of all reference‑model logic.

#### 4.1 `TrainSimpoConfig` (formerly `TrainDpoConfig`)
**Changes**:
- Remove `reference_model_path`, `reference_is_hf`.
- Keep model initialization fields (`initialize_from_hf`, `use_hf_model_config`, etc.).
- Add SimPO parameters:
  - `beta: float`
  - `gamma_beta_ratio: float` (this is the `γ/β` parameter in the paper + reference code)

```python
@dataclass
class TrainSimpoConfig:
    data: Union[SingleDatasetLMConfig, LMMixtureDatasetConfig] = field(default_factory=UrlSingleDatasetLMConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    model: LmConfig = field(default_factory=LlamaConfig)
    train_seq_len: int | None = None
    optimizer: OptimizerConfig = field(default_factory=AdamConfig)

    beta: float = 2.0
    gamma_beta_ratio: float = 0.5
    validation_split_fraction: float | None = 0.1
    ...
```

#### 4.2 Replace DPO loss with SimPO loss (length‑normalized)
**Core requirement from `SIMPO.md`**: use **average log‑prob** per completion token, not sum.

```python
def _average_logp(model: LmHeadModel, example: LmExample, *, key=None) -> hax.NamedArray:
    nll = model.compute_next_token_loss(example, reduction=None, reduction_axis=(), key=key)
    Pos = example.tokens.resolve_axis("position")
    logp_sum = -hax.sum(nll, axis=Pos)
    denom = hax.sum(example.loss_weight, axis=Pos)
    zeros = hax.zeros_like(logp_sum)
    return hax.where(denom != 0, logp_sum / denom, zeros)


def simpo_loss_from_logps(
    avg_logp_chosen: hax.NamedArray | jnp.ndarray,
    avg_logp_rejected: hax.NamedArray | jnp.ndarray,
    *,
    beta: float,
    gamma_beta_ratio: float,
) -> tuple[jnp.ndarray, dict[str, Metric]]:
    logits = (avg_logp_chosen - avg_logp_rejected) - gamma_beta_ratio
    loss = hax.mean(hax.nn.softplus(-beta * logits)).scalar()
    metrics = {
        "simpo_loss": Metric.from_value(loss, ReductionType.MEAN),
        "simpo_margin": Metric.from_value(hax.mean(avg_logp_chosen - avg_logp_rejected).scalar(), ReductionType.MEAN),
        "simpo_accuracy": Metric.from_value(hax.mean(logits > 0).scalar(), ReductionType.MEAN),
    }
    return loss, metrics
```

**Reason**: This directly matches Eq. 6 in `SIMPO.md` and the Princeton implementation (`simpo_trainer.py`).

#### 4.3 Training loop changes (remove reference model)
**Remove**:
- `DpoModel` wrapper.
- `reference_model_path` validation.
- All reference model loading and casting.
- `trainable_filter` for policy vs reference.
- `dpo_*` metrics (reference rewards).

**Keep**:
- Preference dataset validation.
- Dataset split logic and evaluation hooks.
- HF checkpoint save logic, but now save **the single model**.

---

## Data & Dataset (keep as‑is; no reference model required)
These are DPO‑specific *data* changes that SimPO **still needs**, so **no revert here**.

### 5) `lib/levanter/src/levanter/data/text.py`
**Keep**:
- `PreferenceChatProcessor`, `PreferencePairDataset`, `DpoExample`
**Reason**: SimPO still trains on preference pairs; only the loss changes.

**Optional micro‑change (not required)**:
- Update the log message `"DPO preference batch..."` to `"Preference batch..."` for clarity.

### 6) `lib/levanter/src/levanter/data/packing.py`
**Keep** the `slice_strategy="drop"` support.  
**Reason**: SimPO uses the preference dataset packing stack; this is still useful and not DPO‑specific.

---

## Experiments, Configs, and Marin Integration

### 7) `experiments/simple_dpo_config.py` → `experiments/simple_simpo_config.py`
**Change**: Rename class to `SimpleSimPOConfig`, remove reference model fields, add `gamma_beta_ratio`.  
**Reason**: This is the canonical experiment config API for SimPO.

```python
@dataclass(frozen=True)
class SimpleSimPOConfig:
    resources: ResourceConfig
    train_batch_size: int | IntSchedule = 128
    num_train_steps: int = 10000
    learning_rate: float = 6e-7
    beta: float = 2.0
    gamma_beta_ratio: float = 0.5
    ...
```

### 8) `experiments/defaults.py`
**Changes**:
- Update imports to `TrainSimpoConfig`, `SimpleSimPOConfig`, `run_levanter_train_simpo`.
- Rename `default_dpo(...)` → `default_simpo(...)`.
- Remove `reference_model_path` logic.
- Keep tags as `"dpo"` (legacy) and do **not** add `"simpo"`.

**Reason**: The experiment framework should call the SimPO entrypoint.

### 9) `experiments/exp2101_dpo_ultrafeedback.py` → `exp2101_simpo_ultrafeedback.py`
**Changes**:
- Update imports to SimPO versions.
- Remove reference model fields.
- Set `beta` + `gamma_beta_ratio` from `SIMPO.md`.
- Update tags to `"dpo"` (legacy; no `"simpo"` tag).

### 10) Levanter configs
**Rename**:
- `lib/levanter/config/dpo_ultrafeedback_llama3_8b.yaml` → `simpo_ultrafeedback_llama3_8b.yaml`
- `lib/levanter/config/dpo_tiny_gpt2.yaml` → `simpo_tiny_gpt2.yaml`

**Changes**:
- Update command comments to `train_simpo.py`.
- Remove `reference_model_path` / `reference_is_hf`.
- Add `gamma_beta_ratio` and set `beta` to SimPO defaults.
- Set `optimizer.learning_rate` to SimPO defaults (e.g. `6e-7` for Llama 3 per `SIMPO.md`).
- Keep tags as `"dpo"` (legacy; no `"simpo"` tag).

### 11) Marin training integration
**Update**:
- `lib/marin/src/marin/training/training.py`
  - Rename `TrainDpoOnPodConfig` → `TrainSimpoOnPodConfig`
  - Rename `run_levanter_train_dpo` → `run_levanter_train_simpo`
  - Update entrypoint to `train_simpo.main`, name `"train_simpo"`
- `lib/marin/src/marin/training/__init__.py` exports

**Reason**: Keep the executor framework aligned with the new entrypoint.

---

## Tests
### 12) `lib/levanter/tests/test_dpo.py` → `test_simpo.py`
**Remove** DPO‑specific and Haliax‑specific tests that no longer apply:
- Trainable filter excluding reference model
- `NamedArray(None)` placeholder tests
- Haliax pspec/quantization behaviors tied to DPO

**Add/keep** SimPO‑relevant tests:
1. **Loss monotonicity**:
   ```python
   loss_small, _ = simpo_loss_from_logps(avg_small, avg_rejected, beta=1.0, gamma_beta_ratio=0.0)
   loss_large, _ = simpo_loss_from_logps(avg_large, avg_rejected, beta=1.0, gamma_beta_ratio=0.0)
   assert loss_large < loss_small
   ```
2. **Metrics are `Metric` objects**.
3. **Dropout key passthrough** for `_average_logp`.
4. **Preference dataset** tests (processor + dataset) stay as‑is.

**Reason**: Keep test coverage on the new loss while removing tests tied to the removed reference model and reverted Haliax behavior.

---

## Validation Plan (after implementation)
- `uv run pytest lib/levanter/tests/test_simpo.py -v`
- `uv run pytest lib/levanter/tests -m "not entry and not slow and not ray"`
- (Optional) run a tiny SimPO config:  
  `uv run python lib/levanter/src/levanter/main/train_simpo.py --config_path lib/levanter/config/simpo_tiny_gpt2.yaml`

---

## Summary of All Files to Change
**Haliax (revert to main):**
- `lib/haliax/src/haliax/nn/scan.py`
- `lib/haliax/src/haliax/partitioning.py`
- `lib/haliax/src/haliax/quantization.py`

**Levanter (SimPO entrypoint & tests):**
- `lib/levanter/src/levanter/main/train_simpo.py` (rename + rewrite)
- `lib/levanter/src/levanter/main/train_dpo.py` (remove)
- `lib/levanter/tests/test_simpo.py` (rename + rewrite)
- `lib/levanter/tests/test_dpo.py` (remove)
- `lib/levanter/config/simpo_ultrafeedback_llama3_8b.yaml` (rename + update)
- `lib/levanter/config/simpo_tiny_gpt2.yaml` (rename + update)

**Experiments:**
- `experiments/simple_simpo_config.py` (rename + update)
- `experiments/simple_dpo_config.py` (remove)
- `experiments/defaults.py` (update)
- `experiments/exp2101_simpo_ultrafeedback.py` (rename + update)
- `experiments/exp2101_dpo_ultrafeedback.py` (remove)

**Marin:**
- `lib/marin/src/marin/training/training.py` (update)
- `lib/marin/src/marin/training/__init__.py` (update)

---

This plan reflects the confirmed decisions above; implementation should follow it as-written.
