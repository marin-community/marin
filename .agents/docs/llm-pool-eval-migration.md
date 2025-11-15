# LM Evaluation Migration to Fray Pool + Controller Mode

## Overview

This migration updates the LM evaluation experiments to use the new Fray-based inference pool architecture with load-balanced VLLM servers and OpenAI-compatible API.

**Current problems:**
- Mixed configurations: Old Ray-style ResourceConfig coexists with new Fray ResourceConfig
- Incomplete migration: `lib/marin/src/marin/evaluation/` has new InferencePool system, but `experiments/evals/` still uses old patterns
- Redundant fields: EvaluationConfig has obsolete fields that belong in ModelConfig or should be removed
- No region support: Can't easily target eu-west4 or other regions

**Benefits of new architecture:**
- Fray-based scheduling with proper TPU/GPU resource specifications and region support
- Load-balanced pool: Multiple VLLM servers behind OpenAI-compatible proxy
- Clean configs: ModelConfig (model details) + InferencePoolConfig (pool settings) + EvaluationConfig (eval settings)
- Evaluator simplicity: Evaluators just call OpenAI API, don't manage VLLM directly

## Target Architecture

### Configuration Structure

After migration, evaluations will use three clean config objects:

```python
# 1. ModelConfig - Model details and inference settings
model_config = ModelConfig(
    name="model-name",
    path="path/to/model",
    device="tpu",  # or "cuda", "cpu", "auto"
    engine_kwargs={"max_model_len": 2048},
    apply_chat_template=False,
    generation_params={"temperature": 0.7},  # Optional, for Alpaca etc.
)

# 2. InferencePoolConfig - Pool configuration
# Note: Number of VLLM workers is determined by resource_config.replicas
pool_config = InferencePoolConfig(
    resource_config=ResourceConfig(
        cpu=16,
        ram="128g",
        disk="200g",
        device=TpuConfig(type="v5e-16", count=16),
        replicas=1,  # Number of VLLM server workers
        regions=["eu-west4"],
    ),
    model_config=model_config,
)

# 3. EvaluationConfig - Eval-specific settings
eval_config = EvaluationConfig(
    evaluator="lm_evaluation_harness",
    pool_config=pool_config,
    model_name="model-name",
    model_path="path/to/model",
    evaluation_path="gs://bucket/output",
    evals=[EvalTaskConfig(name="mmlu", num_fewshot=5)],
    max_eval_instances=None,
    discover_latest_checkpoint=True,
    wandb_tags=["tag1", "tag2"],
)
```

### Helper Function Pattern

**Before:**
```python
def evaluate_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    resource_config: ResourceConfig | None = None,  # OLD Ray-style
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness/{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="lm_evaluation_harness",
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
            max_eval_instances=max_eval_instances,
            launch_with_ray=True,  # OBSOLETE
            discover_latest_checkpoint=discover_latest_checkpoint,
            engine_kwargs=engine_kwargs,  # WRONG PLACE
            resource_config=resource_config,  # WRONG TYPE
            apply_chat_template=apply_chat_template,
            wandb_tags=wandb_tags,
        ),
    )
```

**After:**
```python
def evaluate_lm_evaluation_harness(
    model_name: str,
    model_path: str,
    evals: list[EvalTaskConfig],
    resource_config: ResourceConfig,  # NEW: Fray type
    max_eval_instances: int | None = None,
    engine_kwargs: dict | None = None,
    apply_chat_template: bool = False,
    wandb_tags: list[str] | None = None,
    discover_latest_checkpoint: bool = True,
) -> ExecutorStep:
    from marin.evaluation.evaluation_config import InferencePoolConfig, ModelConfig

    # Auto-detect device from resource config
    device = infer_device_from_resource_config(resource_config)

    # Build ModelConfig
    model_config = ModelConfig(
        name=model_name,
        path=model_path,
        device=device,
        engine_kwargs=engine_kwargs or {},
        apply_chat_template=apply_chat_template,
    )

    # Build InferencePoolConfig
    pool_config = InferencePoolConfig(
        resource_config=resource_config,
        model_config=model_config,
    )

    # Build clean EvaluationConfig
    return ExecutorStep(
        name=f"evaluation/lm_evaluation_harness/{model_name}",
        fn=evaluate,
        config=EvaluationConfig(
            evaluator="lm_evaluation_harness",
            pool_config=pool_config,
            model_name=model_name,
            model_path=model_path,
            evaluation_path=this_output_path(),
            evals=evals,
            max_eval_instances=max_eval_instances,
            discover_latest_checkpoint=discover_latest_checkpoint,
            wandb_tags=wandb_tags,
        ),
    )
```

## Core Infrastructure Changes

These changes must happen **first** before migrating individual evaluators, as they affect all evaluation code.

### 1. Clean Up EvaluationConfig

**File:** `lib/marin/src/marin/evaluation/evaluation_config.py`

**Remove obsolete fields:**
- `launch_with_ray: bool` - Always uses Fray now
- `generation_params: dict | None` - Moved to ModelConfig
- `engine_kwargs: dict[str, Any]` - Moved to ModelConfig
- `resource_config: ResourceConfig` - Wrong type (now in pool_config)
- `apply_chat_template: bool` - Moved to ModelConfig

**Keep only:**
- `evaluator: str`
- `pool_config: InferencePoolConfig`
- `model_name: str | None`
- `model_path: str | None`
- `evaluation_path: str`
- `evals: list[EvalTaskConfig]`
- `discover_latest_checkpoint: bool`
- `max_eval_instances: int | None`
- `wandb_tags: list[str] | None`

### 2. Remove _impute_model_config

**File:** `lib/marin/src/marin/evaluation/run.py`

Delete the entire `_impute_model_config` function. The `evaluate()` function should directly use `config.pool_config.model_config` instead of calling `_impute_model_config(config)`.

### 3. Add Device Auto-Detection Helper

**File:** `lib/marin/src/marin/evaluation/evaluation_config.py` or new utils file

```python
from fray.cluster.base import CpuConfig, GpuConfig, ResourceConfig, TpuConfig

def infer_device_from_resource_config(resource_config: ResourceConfig) -> str:
    """Infer device type from ResourceConfig.device."""
    if isinstance(resource_config.device, TpuConfig):
        return "tpu"
    elif isinstance(resource_config.device, GpuConfig):
        return "cuda"
    elif isinstance(resource_config.device, CpuConfig):
        return "cpu"
    else:
        return "auto"
```

## Evaluation Migration Pattern

For each evaluator helper function in `experiments/evals/evals.py`, apply this pattern:

1. **Update signature:** Accept Fray `ResourceConfig` (required parameter, not optional)
2. **Auto-detect device:** Call `infer_device_from_resource_config(resource_config)`
3. **Build ModelConfig:** Include model details, engine_kwargs, device, apply_chat_template
4. **Build InferencePoolConfig:** Include resource_config and model_config (number of workers determined by resource_config.replicas)
5. **Build clean EvaluationConfig:** Include pool_config, no obsolete fields

### Special Cases

**Alpaca Eval:** Builds `generation_params` dict and includes it in ModelConfig:
```python
generation_params = {
    "temperature": temperature,
    "presence_penalty": presence_penalty,
    "frequency_penalty": frequency_penalty,
    "repetition_penalty": repetition_penalty,
    "top_p": top_p,
    "top_k": top_k,
    "stop_token_ids": stop_token_ids,
}
model_config = ModelConfig(..., generation_params=generation_params)
```

**Levanter Eval:** Does NOT use inference pool (Levanter-native evaluation). Update signature to accept Fray ResourceConfig for consistency, but don't create InferencePoolConfig.

## Migration Order

Migrate evaluators in this order (easiest first):

1. **HELM** (`evaluate_helm`, `evaluate_helm_on_step`)
   - Simplest evaluator
   - Already has pool support in lib/marin
   - Good first test case

2. **LM Evaluation Harness** (`evaluate_lm_evaluation_harness`)
   - Most commonly used
   - Straightforward migration

3. **Alpaca Eval** (`evaluate_alpaca_eval`)
   - Requires generation_params handling

4. **High-level helpers** (migrate after individual evals work)
   - `default_eval`
   - `default_key_evals`
   - `default_base_eval`
   - `default_sft_eval`

5. **Levanter Eval** (`evaluate_levanter_lm_evaluation_harness`)
   - Special case: doesn't use pool
   - Just update signature, don't create pool

## Testing Strategy

### Local Testing

Use `tests/evals/test_inference_pool.py` as a reference for local testing patterns.

**Run existing tests:**
```bash
# See what breaks after infrastructure changes
uv run pytest tests/evals/test_inference_pool.py -v

# Run all evaluation tests
uv run pytest lib/marin/tests/test_evaluation*.py -v -x
```

**Create test scripts for each migrated evaluator:**
```python
# Example: experiments/evals/test_helm_migration.py
from fray.cluster.base import ResourceConfig, TpuConfig
from experiments.evals.evals import evaluate_helm
from experiments.evals.task_configs import EvalTaskConfig
from marin.execution.executor import executor_main

# Local test resource config
local_config = ResourceConfig(
    cpu=2,
    ram="8g",
    replicas=1,
)

step = evaluate_helm(
    model_name="test-baby-llama",
    model_path="timinar/baby-llama-58m",
    evals=[EvalTaskConfig(name="mmlu", num_fewshot=0)],
    resource_config=local_config,
    max_eval_instances=10,
)

if __name__ == "__main__":
    executor_main(steps=[step])
```

### Remote Testing (eu-west4)

After local testing passes, test on remote cluster:

```bash
# Test on eu-west4 TPU
uv run lib/marin/src/marin/run/ray_run.py \
  --auto-stop \
  --cluster eu-west4 \
  --extra=eval,tpu \
  -- python ./experiments/evals/test_helm_migration.py --force_run_failed true --prefix=gs://marin-eu-west4/tmp/power
```

**Success criteria:**
- Pool starts successfully
- Model loads on VLLM servers
- Evaluation runs to completion
- Results uploaded to output path
- No errors in logs

## Progress Checklist

### Core Infrastructure
- [x] Clean up EvaluationConfig (remove obsolete fields)
- [x] Remove _impute_model_config from run.py
- [x] Add infer_device_from_resource_config helper
- [x] Fix any broken tests after infrastructure changes

### Individual Evaluators
- [ ] Migrate evaluate_helm
  - [ ] Update function signature and implementation
  - [ ] Create test_helm_migration.py
  - [ ] Test locally
  - [ ] Test on eu-west4
- [ ] Migrate evaluate_helm_on_step (if different from evaluate_helm)
- [ ] Migrate evaluate_lm_evaluation_harness
  - [ ] Update function signature and implementation
  - [ ] Create test_lm_eval_migration.py
  - [ ] Test locally
  - [ ] Test on eu-west4
- [ ] Migrate evaluate_alpaca_eval
  - [ ] Update function signature and implementation
  - [ ] Handle generation_params properly
  - [ ] Create test_alpaca_migration.py
  - [ ] Test locally
  - [ ] Test on eu-west4
- [ ] Migrate evaluate_levanter_lm_evaluation_harness (special case)
  - [ ] Update signature to accept Fray ResourceConfig
  - [ ] Do NOT create InferencePoolConfig
  - [ ] Test locally
  - [ ] Test on eu-west4

### High-Level Helpers
- [ ] Migrate default_eval
- [ ] Migrate default_key_evals
- [ ] Migrate default_base_eval
- [ ] Migrate default_sft_eval

### Runner Scripts
- [ ] Update experiments/evals/run_helm.py
- [ ] Update experiments/evals/run_key_evals.py
- [ ] Update experiments/evals/run_base_model_evals.py
- [ ] Update experiments/evals/run_sft_model_evals.py
- [ ] Update experiments/evals/run_alpaca_eval.py
- [ ] Update experiments/evals/run_on_gpu.py
- [ ] Update experiments/evals/exp1602_lm_eval_harness.py
- [ ] Update experiments/evals/exp1600_uncheatable_evals.py

### Documentation & Cleanup
- [ ] Update docs/tutorials/run-lm-evals.md
- [ ] Update docstrings in evals.py
- [ ] Clean up obsolete code
- [ ] Final test pass on all runner scripts

## Runner Script Migration

After all helpers are migrated and tested, update runner scripts to use Fray ResourceConfigs:

**Pattern:**
```python
# OLD:
from experiments.evals.resource_configs import SINGLE_TPU_V6E_8

step = evaluate_lm_evaluation_harness(
    ...,
    resource_config=SINGLE_TPU_V6E_8,
)

# NEW:
from fray.cluster.base import ResourceConfig, TpuConfig

eu_west4_tpu = ResourceConfig(
    cpu=16,
    ram="128g",
    disk="200g",
    device=TpuConfig(type="v5e-16", count=16),
    replicas=1,
    regions=["eu-west4"],
)

step = evaluate_lm_evaluation_harness(
    ...,
    resource_config=eu_west4_tpu,
)
```

Test each runner script after updating:
```bash
uv run lib/marin/src/marin/run/ray_run.py \
  --auto-stop \
  --cluster eu-west4 \
  -e HF_TOKEN=$HF_TOKEN \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  --extra=eval,tpu \
  -- python ./experiments/evals/run_<script>.py
```

## Design Decisions & Potential Issues

### Design Decisions

1. **No backward compatibility**: Aggressively update all callers rather than maintaining compatibility layer. Keeps codebase clean and prevents technical debt.

2. **InferencePoolConfig as primary interface**: Pool config contains everything needed for inference (model + resources + pool settings). Centralizes inference configuration.

3. **Device auto-detection**: Infer device from ResourceConfig.device type (TpuConfig → "tpu", GpuConfig → "cuda"). Makes configuration simpler.

4. **Worker count from replicas**: Number of VLLM workers determined by resource_config.replicas. Removed num_servers field to avoid redundancy.

5. **Region-aware configs**: Each ResourceConfig specifies regions, making it easy to target specific clusters (us-central2, us-east5, eu-west4).

### Potential Issues

1. **Resource specification mapping**: Old TPU types like "TPU-v4-8" need careful mapping to new format (type="v4-8", count=4). Must verify chip counts are correct.

2. **Engine kwargs defaults**: Need to ensure engine_kwargs defaults are preserved. Check experiments/evals/engine_configs.py for canonical defaults.

3. **Generation params**: These were previously in EvaluationConfig, now in ModelConfig. Make sure all evaluators that need them (Alpaca) get them properly.

4. **Levanter vs VLLM**: Some evaluators use Levanter-native evaluation, others use VLLM via pool. Need to preserve this distinction.

5. **Checkpoint discovery**: The discover_latest_checkpoint logic must still work after migration.

### Success Criteria

1. All runner scripts in experiments/evals/ work with new configs
2. Can successfully run evals on eu-west4 cluster
3. No references to old ResourceConfig (num_tpu, tpu_type, strategy)
4. All tests pass
5. Documentation updated

### Future Improvements

1. **Region-specific defaults**: Could have region-specific default configs that users just select by name
2. **Pool reuse**: For multiple evals on same model, could reuse the inference pool instead of recreating

## Completed Infrastructure Fixes (2025-11-24 and 2025-11-25)

### Fix 3: RuntimeContext dashboard_url Access (2025-11-25)

**Problem:** When `RayCluster` is instantiated inside a Ray worker (e.g., during evaluation tasks), `ray.get_runtime_context()` does not have a `dashboard_url` attribute, causing `AttributeError`.

**Root Cause:** Ray's API change - `RuntimeContext` object doesn't expose `dashboard_url` when running inside a worker process. The attribute is only available in certain contexts.

**Solution:** Fall back to `gcs_address` if available, or use the configured address:

```python
# lib/fray/src/fray/cluster/ray/cluster.py:401-416
def _get_dashboard_address(self) -> str:
    """Get Ray dashboard address for job submission.

    When running inside a Ray worker, we can't access the dashboard URL
    directly, so we fall back to using the Ray GCS address or the
    configured address.
    """
    if ray.is_initialized():
        try:
            # Try to get the GCS address, which we can use for job submission
            ctx = ray.get_runtime_context()
            if hasattr(ctx, "gcs_address"):
                return ctx.gcs_address
        except Exception:
            pass
    return self._address
```

**Note:** For TPU jobs (which use `_launch_tpu_job`), the job submission client is not used, so the exact dashboard address is not critical.

**Commit:** `8785565dd` - Fix RuntimeContext dashboard_url access for Ray workers

### Testing Status Update (2025-11-25)

**Local CPU test:** ✅ **PASSED**
- Job submission: ✅
- Evaluation task launch: ✅
- Fray namespace creation: ✅
- Inference pool job launch: ✅
- Proxy server startup: ✅
- Health check: ✅
- Test completion request: ✅

**Remote eu-west4 TPU test:** ❌ **BLOCKED** by infrastructure issue
- Job submission and Ray cluster connection: ✅
- Evaluation task launch: ✅
- Fray namespace creation: ✅
- Inference pool job launch: ✅
- TPU slice actor pool initialization: ✅
- Isolated venv creation and package installation: ✅
- vLLM-TPU model loading: ❌ **FAILED** - libtpu compatibility error

### Known Issue: vLLM-TPU libtpu Version Incompatibility

**Error:**
```
RuntimeError: Pallas TPU requires a libtpu version that's at most a month old.
Found version string:
  PJRT C API
  TFRT TPU v5 lite
  Built on Jun 12 2025 16:39:57 (1749771597) cl/769337304
```

**Root Cause:** The vLLM-TPU package's Pallas/JAX backend has strict version requirements for libtpu (TPU firmware). The cluster's libtpu version doesn't satisfy these requirements.

**Impact:** TPU-based evaluations cannot currently run. CPU-based evaluations work correctly.

**Workarounds:**
1. **Short-term:** Use CPU backend for testing (confirmed working)
2. **Medium-term:** Pin vllm-tpu to older version that's compatible with cluster libtpu
3. **Long-term:** Update cluster TPU firmware/libtpu

**Test Model Change:** Switched from `timinar/baby-llama-58m` to `HuggingFaceTB/SmolLM2-135M` due to rotary embedding `inv_freq` buffer compatibility issues with vLLM-TPU weight loader. (Commit: `a930159ba`)

## Completed Infrastructure Fixes (2025-11-24)

### Issue 1: Ray runtime_env Error for Nested Remote Calls

**Problem:** When launching TPU jobs via Fray's `_launch_tpu_job`, nested `ray.remote()` calls failed with:
```
ValueError: /tmp/ray/.../working_dir_files/... is not a valid URI.
Passing directories or modules to be dynamically uploaded is only supported at the job level (i.e., passed to `ray.init`).
```

**Root Cause:** Ray's API constraint - `working_dir`, `excludes`, and `config` can only be set at job level via `ray.init()`, not in task-level `ray.remote()` calls. The `_launch_tpu_job` method was passing the full runtime_env (including these job-level keys) to nested remote function decorators.

**Solution:** Filter out job-level keys from runtime_env when creating nested `ray.remote()` calls:

```python
# lib/fray/src/fray/cluster/ray/cluster.py:331-333
nested_runtime_env = {k: v for k, v in runtime_env.items()
                     if k not in ["working_dir", "excludes", "config"]}
```

**Commit:** `3b2945419` - Fix Ray runtime_env and vLLM torchvision issues for TPU jobs

### Issue 2: vLLM Dependency Conflicts on TPU

**Problem:** Installing vllm alongside torch_xla in the Ray worker environment caused:
1. Initial attempt: torchvision version mismatch (0.23.0 built for torch 2.7.0, but torch_xla pins torch 2.8.0)
   - Error: `RuntimeError: operator torchvision::nms does not exist` → Segfault (exit code 139)
2. Second attempt: Missing dependencies when installing vllm with `--no-deps`
   - Error: `ModuleNotFoundError: No module named 'cbor2'`

**Root Cause:** Dependency conflicts between Ray's environment and vllm's requirements. Installing vllm's dependencies into the shared Ray worker environment creates version conflicts.

**Solution:** Create an isolated uv virtual environment specifically for vllm:

```python
# lib/marin/src/marin/evaluation/backends/vllm.py:156-167
vllm_venv = tempfile.mkdtemp(prefix="vllm_venv_")
subprocess.check_call(["uv", "venv", vllm_venv])
vllm_python = os.path.join(vllm_venv, "bin", "python")

# Install in isolated venv
subprocess.check_call(["uv", "pip", "install", "--python", vllm_python,
                      "torch_xla[tpu, pallas]==2.8.0"])
subprocess.check_call(["uv", "pip", "install", "--python", vllm_python,
                      "vllm==0.11.0"])

# Run vllm from isolated venv
vllm_cmd = os.path.join(vllm_venv, "bin", "vllm")
```

**Benefits:**
- Complete isolation of vllm dependencies from Ray environment
- No version conflicts with torch, torchvision, ray, etc.
- All vllm dependencies installed correctly via normal dependency resolution
- Clean, simple approach vs. manual dependency management

**Commits:**
- `6e669b2eb` - Simplify vLLM TPU dependency handling
- `64ad27442` - Use isolated uv venv for vLLM to avoid dependency conflicts
- `0bb2659af` - Refactor: use vllm_venv instead of vllm_python for clarity

### Testing Status

**Infrastructure verified on eu-west4:**
- ✅ Job submission and Ray cluster connection
- ✅ Evaluation task launch
- ✅ Fray namespace creation
- ✅ Inference pool job launch
- ✅ TPU slice actor pool initialization
- ✅ Isolated venv creation and package installation

**Next steps for HELM migration:**
- Debug vLLM server startup timeout on TPU (health check timing out)
- Complete end-to-end HELM evaluation test
- Migrate remaining evaluators following the same pattern
