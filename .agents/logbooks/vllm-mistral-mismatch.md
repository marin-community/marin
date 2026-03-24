# vLLM Mistral Architecture Mismatch Investigation

**Branch:** `vllm-mistral-mismatch`
**Experiment ID prefix:** `VMIS`

## Problem Statement

When vLLM loads a Llama model checkpoint from a GCS path (e.g. `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`), the on-disk `config.json` correctly declares `"architectures": ["LlamaForCausalLM"]` and `"model_type": "llama"`, but vLLM's runtime resolves the architecture as `MistralForCausalLM`. This triggers a cascade:

1. `tpu_inference` does not have `MistralForCausalLM` registered in its JAX-native path.
2. vLLM falls back to PyTorch-native model definition.
3. RL hot-reload (`_sync_weights`) expects JAX `nnx.State` but the fallback runner stores state as a plain dict.
4. First weight update crashes with `AttributeError: 'dict' object has no attribute 'flat_state'`.

The mismatch has been independently observed and documented in two separate workstreams.

## Prior Art

### From `alignment_function` branch

**Logbook:** `.agents/logbooks/alignment_function.md`

- **ALIGN-052** (line ~2177): "GCS Load Path Verified; vLLM Still Resolves the Llama 3.3 Checkpoint as Mistral on TPU"
  - Model: Llama 3.3 70B Instruct
  - GCS path: `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b`
  - Both `config.json` files under the prefix declare `architectures: ["LlamaForCausalLM"]`, `model_type: "llama"`
  - vLLM logged: `Resolved architecture: MistralForCausalLM`
  - TPU model loading warned: `Model architectures ['MistralForCausalLM'] not registered in tpu-inference. Falling back to vLLM-native Pytorch definition.`
  - Context: direct alignment vLLM path using `load_format='runai_streamer'` for batch inference
  - Status at time of writing: flagged as the next debugging target; no root cause identified

### From `iris_rl` branch

**Logbook:** `.agents/logbooks/iris-rl-codex.md`

The RL workstream hit the same mismatch repeatedly across multiple debugging sessions. Key entries:

- **2026-03-24T07:56Z — `r9`** (line ~1521): First observation in RL context
  - Model: Llama 3.1 8B Instruct
  - GCS artifact `config.json` confirmed correct (`architectures: ["LlamaForCausalLM"]`, `model_type: "llama"`)
  - vLLM resolved as `MistralForCausalLM`, triggering PyTorch fallback
  - Downstream crash: `AttributeError: 'dict' object has no attribute 'flat_state'` in `transfer_state_with_mappings`
  - Root cause narrowed: `tgt_state` (the model runner's own state) was dict-like, not JAX `nnx.State`, because PyTorch fallback path was used

- **2026-03-24T08:18Z — source inspection** (line ~1727): Stronger root-cause diagnosis
  - Confirmed `transfer_state_with_mappings` calls `tgt_state.flat_state()` on the model runner's state
  - If vLLM falls back to PyTorch runner, that state is a plain dict, not `nnx.State`
  - Best hypothesis at this point: "GCS-backed `runai_streamer` model loading is causing `vllm-tpu` to infer the wrong architecture from the streamed local cache path"
  - Proposed fix: stage only HF metadata files locally, use `load_format="dummy"` for RL rollout startup

- **2026-03-24T08:24Z — metadata staging patch** (line ~1800): First fix attempt
  - Patched `rollout_worker.py` to copy only small HF metadata files from GCS to local temp dir
  - Used `load_format="dummy"` so startup doesn't depend on object-store weight loading
  - Tests passed locally (20 passed)

- **2026-03-24T08:29Z — `r11` live result** (line ~1836): Fix attempt failed
  - Local metadata staging was confirmed active in logs
  - vLLM **still** resolved `MistralForCausalLM` despite loading from a local directory with correct Llama config
  - This disproved the hypothesis that `runai_streamer` or GCS streaming was the primary cause
  - Revised diagnosis: the mismatch is in vLLM's architecture detection / model-registry interaction inside `vllm-tpu` + `tpu_inference`, not in the storage backend

- **Ranked hypotheses after `r11`** (line ~1876):
  1. Most likely: Marin's local patching of the TPU inference registry is incomplete; `MistralForCausalLM` needs explicit aliasing to the JAX Llama implementation
  2. Possible: vLLM-side config canonicalization rewrites Llama metadata to Mistral before `tpu_inference` sees it
  3. Less likely: model registry is correct but rollout inflight worker constructs the wrong target state object

## Key Facts Established

| Fact | Source |
|---|---|
| GCS `config.json` is correct for Llama | Both logbooks, verified via `gcloud storage cat` |
| Loading by HF repo ID resolves `LlamaForCausalLM` correctly | iris-rl-codex (line ~1795) |
| Loading from GCS resolves `MistralForCausalLM` | Both logbooks |
| Local metadata staging + `load_format="dummy"` does NOT fix it | iris-rl-codex `r11` (line ~1836) |
| `MistralForCausalLM` is not registered in `tpu_inference` | Both logbooks |
| PyTorch fallback produces dict state, not `nnx.State` | iris-rl-codex (line ~1745) |
| Trainer side (Levanter/JAX) is unaffected | iris-rl-codex (line ~1874) |
| Marin patches `Qwen2ForCausalLM` into registry but not `MistralForCausalLM` | iris-rl-codex (line ~1882) |

## Affected Models

- `meta-llama/Llama-3.3-70B-Instruct` (alignment workstream)
- `meta-llama/Llama-3.1-8B-Instruct` (RL workstream)

Both exhibit the same behavior when loaded from regional GCS artifacts.

## Root Cause Analysis (2026-03-24)

### The remapping chain (confirmed via source inspection)

The architecture mismatch is caused by a **three-step failure cascade**:

#### Step 1: vLLM internally remaps `LlamaForCausalLM` → `MistralForCausalLM`

In vLLM's PyTorch model registry (`vllm/model_executor/models/registry.py`):
- `LlamaForCausalLM` maps to `("llama", "LlamaForCausalLM")`
- `MistralForCausalLM` maps to `("mistral", "MistralForCausalLM")`
- `MistralForCausalLM` is a direct subclass of `LlamaForCausalLM` (`vllm/model_executor/models/mistral.py:230`)

vLLM considers these interchangeable. Somewhere in its config resolution chain (`config/model.py:530` calls `registry.inspect_model_cls(architectures, self)` which logs `"Resolved architecture: %s"` at line 533), the resolved architecture becomes `MistralForCausalLM`.

One confirmed vector: vLLM's Mistral config parser (`vllm/transformers_utils/configs/mistral.py:53`) unconditionally sets `config_dict["architectures"] = ["MistralForCausalLM"]` when `config_format == "mistral"`. The Mistral format is detected when `consolidated*.safetensors` files exist in the repo AND `params.json` exists. However, the `r11` experiment disproved that this is the only path — the remapping also happens with local metadata directories that have no `consolidated*` or `params.json` files.

This means there is likely a second remapping path inside vLLM's model registry resolution where the `LlamaForCausalLM` → `MistralForCausalLM` swap happens as part of the `_normalize_arch` or `inspect_model_cls` flow. The exact mechanism within the installed `vllm-tpu==0.13.2.post6` was not fully traced due to the package only being available on TPU workers.

**Key insight**: From vLLM's perspective, this is not a bug — Llama and Mistral share the same PyTorch implementation. The problem is only visible to `tpu-inference`, which has separate JAX implementations.

#### Step 2: `tpu-inference`'s JAX registry doesn't have `MistralForCausalLM`

In the installed `tpu-inference` (bundled with `vllm-tpu==0.13.2.post6`), `_MODEL_REGISTRY` contains `LlamaForCausalLM` but NOT `MistralForCausalLM`. When `_get_model_architecture()` iterates over `hf_config.architectures` (which now contains `["MistralForCausalLM"]`), it fails to find a match and raises `UnsupportedArchitectureError`.

#### Step 3: PyTorch fallback breaks RL weight sync

`tpu-inference`'s `get_model()` catches `UnsupportedArchitectureError` and falls back to `get_vllm_model()`, which returns a PyTorch model with dict-like state. But the RL hot-reload path calls `_sync_weights()` → `transfer_state_with_mappings(src_state, tgt_state)` where `tgt_state.flat_state()` is called. The PyTorch fallback state is a plain dict, not `nnx.State`, causing:

```
AttributeError: 'dict' object has no attribute 'flat_state'
```

### What disproved earlier hypotheses

- **`r11`** (iris-rl-codex, line ~1836): Local metadata staging + `load_format="dummy"` still resolved `MistralForCausalLM`. This rules out `runai_streamer` and GCS streaming as the cause.
- The remapping is internal to vLLM's architecture resolution, not dependent on the storage backend.

### Existing fix in `tpu-inference` fork (proven but not on `main`)

The `marin-community/tpu-inference` fork at `/Users/ahmed/code/tpu-inference` already has the fix:

**`tpu_inference/models/common/model_loader.py:237`**:
```python
_MODEL_REGISTRY["MistralForCausalLM"] = LlamaForCausalLM
```

Plus a `model_type` fallback at lines 255-265 and `_ABSTRACT_BOOTSTRAP_ARCHITECTURES` registration at line 52.

This was deployed via two wheel updates on `vllm_load_fast`:
- Commit `298b5d9e4` — `marin-b2c90c99` wheel: `model_type` fallback
- Commit `47f6ef4d8` — `marin-a74f6142` wheel: `MistralForCausalLM` alias

The latest wheel is `marin-4abb68f4` (pinned in `vllm_load_fast`'s `pyproject.toml`). None of these have reached `main`.

### Current state on `main`

- `vllm-tpu==0.13.2.post6` bundles an old `tpu-inference` **without** the Mistral alias
- `lib/marin/pyproject.toml:172` pins `vllm-tpu==0.13.2.post6`
- No `tpu-inference` override in root `pyproject.toml`
- Marin's `_patch_tpu_inference_registry()` at `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:117-129` only registers `Qwen2ForCausalLM`

## Proposed Fix

### Approach: Extend Marin's `_patch_tpu_inference_registry()`

Add `MistralForCausalLM` → `LlamaForCausalLM` alias to the existing registry patch. This matches exactly what the fork does at `model_loader.py:237`.

**Why this approach**:
- Minimal change (~5 lines) in one file
- Matches the proven fork fix
- Does not require pulling the entire fork wheel (which carries unvetted changes: abstract bootstrap, fsspec streaming, mesh fixes)
- Guarded by `if "MistralForCausalLM" not in _MODEL_REGISTRY` — becomes a no-op once the fork wheel lands on `main`
- vLLM intentionally treats Llama/Mistral as interchangeable, so there's no upstream fix to wait for

**File to modify**: `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`

**Change**: In `_patch_tpu_inference_registry()` (lines 117-129), after the existing Qwen2 registration, add:

```python
# vLLM may remap LlamaForCausalLM → MistralForCausalLM (they share
# the same PyTorch implementation). Register the alias so tpu-inference
# uses its JAX path instead of falling back to PyTorch.
if "MistralForCausalLM" not in model_loader._MODEL_REGISTRY:
    logger.info("Patching tpu_inference: MistralForCausalLM → LlamaForCausalLM (JAX alias)")
    from tpu_inference.models.jax.llama3 import LlamaForCausalLM
    model_loader._MODEL_REGISTRY["MistralForCausalLM"] = LlamaForCausalLM
```

Uses direct `_MODEL_REGISTRY` assignment (not `register_model()`) because we only need the JAX registry entry. vLLM's PyTorch registry already has `MistralForCausalLM` mapped correctly.

### Key references

| What | Where |
|------|-------|
| Current registry patch (Qwen2 only) | `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:117-129` |
| Fork's proven fix | `/Users/ahmed/code/tpu-inference/tpu_inference/models/common/model_loader.py:237` |
| JAX LlamaForCausalLM class | `tpu_inference.models.jax.llama3:350` |
| vLLM's MistralForCausalLM (subclass of Llama) | `vllm/model_executor/models/mistral.py:230` |
| vLLM "Resolved architecture" log | `vllm/config/model.py:533` |
| vLLM Mistral config parser (one remapping vector) | `vllm/transformers_utils/configs/mistral.py:53` |

### Verification plan

1. **Pre-commit**: `./infra/pre-commit.py --fix` on the changed file
2. **On-cluster (RL)**: Submit a short RL run with Llama-3.1-8B from GCS. Verify logs show:
   - `"Patching tpu_inference: MistralForCausalLM → LlamaForCausalLM (JAX alias)"`
   - No `"Falling back to vLLM-native Pytorch definition"`
   - No `flat_state` crash
   - Rollout completes first weight update
3. **On-cluster (alignment)**: Submit a short eval with Llama-3.3-70B from GCS. Verify no PyTorch fallback warning.

### Risks

- **Import may not exist in older `tpu-inference`**: `tpu_inference.models.jax.llama3` should be present in `vllm-tpu==0.13.2.post6` (the existing `qwen2` import from the same package works). If absent, the `except ImportError` block catches it.
- **Becomes no-op once fork lands**: The guard ensures this is safe when the fork replaces `vllm-tpu`.

## Chronological Record

### 2026-03-24 — Branch created, logbooks copied, investigation started

- Created `vllm-mistral-mismatch` branch based on `main`
- Copied `.agents/logbooks/alignment_function.md` from `alignment_function` branch
- Copied `.agents/logbooks/iris-rl-codex.md` from `iris_rl` branch
- Created this logbook cross-referencing both sources
- Performed deep source inspection of vLLM (`vllm/transformers_utils/configs/mistral.py`, `vllm/config/model.py`, `vllm/model_executor/models/registry.py`) and `tpu-inference` fork (`tpu_inference/models/common/model_loader.py`)
- Confirmed root cause: vLLM internal Llama→Mistral remap + missing JAX registry entry
- Confirmed existing fix in fork at `model_loader.py:237`
- Designed minimal Marin-side fix: extend `_patch_tpu_inference_registry()` with MistralForCausalLM alias
