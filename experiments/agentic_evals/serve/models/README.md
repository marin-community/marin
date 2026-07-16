# model_config/ — unified per-model vLLM serve config

The **single source of truth** for "what vLLM parameters does model X require." Replaces
the two divergent sources (the eval-listener mono-file `eval/configs/model_configs.yaml`
and the datagen per-model files under `hpc/datagen_yaml/`) with one file-per-model home.

## Layout

```
model_config/
├── _patterns.yaml          # regex fallbacks (size-inference defaults) for models with no file
├── <org>/<model-slug>.yaml # one file per model (e.g. Qwen/Qwen3-32B.yaml)
└── resolver.py             # the resolution function all entrypoints call
```

## File schema (layered)

```yaml
model: Qwen/Qwen3-32B              # canonical HF id (for matching)
# --- base intrinsics (small: the constants, same regardless of who's serving) ---
trust_remote_code: true
max_model_len: 32768
tool_call_parser: hermes           # absent on Pattern-D models (deliberate)
reasoning_parser: qwen3
hf_overrides: ...
limit_mm_per_prompt: ...
# --- subsystem overlays (eval / datagen / iris): the workflow-specific stuff ---
subsystems:
  eval:                           # the eval listener / Iris
    tensor_parallel_size: 2       # (parity geometry)
    swap_space: 32
    agent_kwargs: [extra_body={...enable_thinking:true}]
    extra_args: --enable-prefix-caching
    variants:                     # hardware-geometry overrides (merged last, win)
      gh200: {tensor_parallel_size: 1}
      gh200-65k: {tensor_parallel_size: 1}
  datagen:                        # trace generation (later migration)
    extra_args: --dtype bfloat16 ...
    swap_space: 12
```

**Resolution merge order (later wins):** base intrinsics → subsystem(s) → hardware variant.

A field goes to **base** only if it's model-intrinsic (the same regardless of subsystem).
Intentional divergences (e.g. GLM-4.7 Pattern D omits `tool_call_parser` for eval; datagen
needs it) live under the relevant `subsystems:` overlay — not in base.

## API

```python
from model_config import resolve_model_config

# "What vLLM params does Qwen3-32B need for an eval run on a multi-GPU node?"
cfg = resolve_model_config("Qwen/Qwen3-32B", subsystem="eval", hardware="multi_gpu")
# -> {trust_remote_code: true, max_model_len: 32768, tool_call_parser: hermes,
#     tensor_parallel_size: 2, swap_space: 32, ...}

# Stack multiple subsystems (named overrides, later wins):
cfg = resolve_model_config("QuantTrio/GLM-4.7-AWQ", subsystem="eval",
                           subsystems=["eval", "pattern_d"])
```

## Migration status

- ✅ **`model_config/` is the sole editable source.** `eval/configs/model_configs.yaml`
  is now a **GENERATED build artifact** — produced by
  `scripts/generate_eval_registry.py` from the per-model files. Do NOT hand-edit the
  mono-file; edit the per-model files and re-run the generator. `--check` mode is the
  CI drift gate (`python scripts/generate_eval_registry.py --check` exits nonzero if
  the committed mono-file is stale). Verified: generated mono-file is parsed-data-
  identical to the original (52 models, 6 patterns, zero mismatches); the listener's
  `load_model_registry` is unchanged.
- ⏳ **Datagen YAMLs** (`hpc/datagen_yaml/`, 167 files) → not yet migrated; stale files
  to be retired first. Later stage.
- ⏳ **Iris wiring** → `resolve_model_config` available; Iris integration is the next step.
- ⏳ **Listener direct wiring (next step #2)** → the listener still reads the GENERATED
  mono-file (zero risk). Wiring it directly to `resolve_model_config` removes the
  generation step entirely and gives the listener the resolver's "always-apply
  intrinsics" improvement (vs the old registry's profile-exclusion quirk). This is a
  behavioral change to gate carefully — the resolver currently does NOT replicate the
  old `_included_on_profile` exclusion (bare entries excluded on non-default hardware
  profiles). Either (a) add that exclusion to the resolver for parity, or (b) accept
  the improved behavior and regenerate the serve-parity golden. Tracked as the
  follow-up after this merge.

## Adding a new model

Create `model_config/<org>/<slug>.yaml` with the base intrinsics + an `eval` subsystem
overlay. The slug is the model name with non-`[\w.\-]` chars replaced by `_`.
