# Llama-as-Mistral Misidentification: Full Technical Analysis

**Date**: 2026-04-03
**Context**: Investigation triggered by marin-community/marin#4356 and the
`worktree-tpu-dep-hell` branch which moves Marin to upstream `main` of both
`marin-community/vllm` and `marin-community/tpu-inference`.

**Question**: Does moving to upstream `main` fix the Llama-as-Mistral
misidentification bug, or are we still vulnerable?

**Answer**: Still vulnerable. Upstream `main` has no fix. The old fork's `marin`
branch had a partial fix (JAX registry alias) that is not present in upstream.
And the problem is worse than originally understood -- it's not just a TPU
performance issue, it's a correctness issue on both GPU and TPU.

---

## 1. Background: Two Execution Paths on TPU

Before understanding the bug, you need to understand that `tpu-inference` (the
library that adapts vLLM for TPU) offers two completely different ways to run a
model on TPU hardware.

### Path A: Native JAX/Flax (fast path)

`tpu-inference` maintains hand-written JAX/Flax implementations of specific
model architectures. These are purpose-built for TPU: they use JAX's native
sharding, `jax.jit` compilation, and Flax NNX modules. The weights are loaded
directly as `jax.Array` objects and placed on TPU devices.

The architectures with native JAX implementations (as of upstream `main`):

```
LlamaForCausalLM
Llama4ForCausalLM
DeepseekV3ForCausalLM
Qwen3ForCausalLM
Qwen3MoeForCausalLM
Qwen2ForCausalLM
Qwen2_5_VLForConditionalGeneration
Eagle3LlamaForCausalLM
GptOssForCausalLM
Llama4ForConditionalGeneration (-> LlamaGuard4)
```

These are registered in an exact-match dictionary called `_MODEL_REGISTRY` in
`tpu_inference/models/common/model_loader.py`. The lookup is by architecture
string from `config.architectures[0]`. No aliases, no fuzzy matching, no
inheritance fallback.

### Path B: PyTorch-on-torchax (slow fallback)

For any architecture NOT in the JAX registry, `tpu-inference` falls back to a
bridge layer called `VllmModelWrapper`. This works as follows:

1. Loads the standard vLLM PyTorch model onto **CPU** using vLLM's stock model
   loader (`vllm_get_model()`)
2. Wraps it in a `_VllmRunner(torch.nn.Module)` that delegates `forward()` to
   the vLLM model
3. Shards the PyTorch weights to TPU via `shard_model_to_tpu()`, converting
   them to JAX arrays through `torchax.interop.jax_view()` (zero-copy view)
4. Monkey-patches `torch.nn.functional.scaled_dot_product_attention` to route
   through a JAX flash attention kernel
5. JIT-compiles the entire forward pass: `torch.func.functional_call()` runs
   inside `torchax.default_env()`, which routes every PyTorch op through
   JAX/XLA, then `@jax.jit` compiles the whole thing to a single TPU HLO
   program

In other words: the PyTorch model's op graph gets traced through `torchax`
into JAX/XLA IR, compiled, and executed on TPU. It works for any architecture
vLLM supports on GPU, but:

- Compilation is slower (the tracing overhead of going PyTorch -> torchax ->
  XLA adds time)
- The generated HLO may be less optimal than hand-written JAX (no manual
  sharding annotations, no custom TPU kernels)
- It's less tested on TPU than the native JAX path
- It's the path that `tpu-inference` uses as a catch-all fallback

### How the dispatch works

```python
# tpu_inference/models/common/model_loader.py

def get_model(vllm_config, rng, mesh, is_draft_model=False):
    impl = envs.MODEL_IMPL_TYPE  # default: "auto"
    if impl == "auto":
        impl = resolve_model_architecture(vllm_config, is_draft_model)
        # returns "flax_nnx" unless arch is in _VLLM_PREFERRED_ARCHITECTURES

    match impl:
        case "flax_nnx":
            try:
                return get_flax_model(...)     # tries JAX registry lookup
            except UnsupportedArchitectureError:
                return get_vllm_model(...)     # falls back to PyTorch-on-torchax
        case "vllm":
            return get_vllm_model(...)         # forced PyTorch-on-torchax
```

Despite the two `case` branches, every model ends up in one of two places:
`get_flax_model()` (native JAX) or `get_vllm_model()` (PyTorch-on-torchax).
There is no third execution path. The `case "vllm"` branch is just a way to
**skip the JAX registry lookup entirely** and go straight to the torchax bridge.
It is reached either by setting `MODEL_IMPL_TYPE=vllm` manually, or by
`resolve_model_architecture()` returning `"vllm"` for architectures in
`_VLLM_PREFERRED_ARCHITECTURES`. These are architectures that have JAX
implementations in the registry but are routed to the torchax path anyway:

- `GptOssForCausalLM` -- torchax path is faster than the current JAX impl
  (PR #1255, described as temporary: "for now")
- `Qwen3MoeForCausalLM` -- no explicit rationale given, but added during a
  period of known MoE JAX compilation regressions (PR #1773)
- `Gemma4ForConditionalGeneration` -- JAX impl only covers the language model,
  vision/multimodal support is missing (PR #2130)

`get_flax_model()` calls `_get_model_architecture()` which does an exact lookup
in `_MODEL_REGISTRY`. If the architecture string isn't found, it raises
`UnsupportedArchitectureError`, which is caught and silently falls through to
`get_vllm_model()`. The only log evidence is a warning message.

---

## 2. How the Misidentification Happens

### The trigger: flattened checkpoint layout

vLLM auto-detects "Mistral-format" checkpoints by looking for two things:

1. Files matching `consolidated*.safetensors` in the model directory
2. A `params.json` file in the same directory

This detection lives in `vllm/transformers_utils/repo_utils.py`:

```python
def is_mistral_model_repo(model_name_or_path, ...):
    return any_pattern_in_repo_files(
        model_name_or_path=model_name_or_path,
        allow_patterns=["consolidated*.safetensors"],
        ...
    )
```

And is invoked in `vllm/transformers_utils/config.py`:

```python
if is_mistral_model_repo(...) and file_or_path_exists(..., config_name="params.json", ...):
    config_format = "mistral"
```

**The problem**: Meta distributes Llama models with both HuggingFace-format
files (`config.json`, `model-*.safetensors`) and Mistral-native-format files
(`params.json`, `consolidated.*.pth`/`.safetensors`). In the correct layout,
the Mistral-native files live under an `original/` subdirectory. But some
historical Marin GCS caches were downloaded with a flattened layout where
`params.json` and `consolidated.*.pth` ended up at the model root alongside
the HF files.

When vLLM sees `consolidated*.safetensors` + `params.json` at the root, it
triggers the Mistral detection path. This happens **before** it even looks at
`config.json`, where `architectures` correctly says `["LlamaForCausalLM"]`.

### What `MistralConfigParser` does to the config

Once vLLM decides it's a "Mistral-format" repo, `MistralConfigParser` takes
over. It reads `params.json` (not `config.json`) and runs `adapt_config_dict()`:

**Architecture assignment** -- unconditional for non-MoE, non-MLA, non-mamba:

```python
# vllm/transformers_utils/configs/mistral.py
config_dict["architectures"] = ["MistralForCausalLM"]
```

This overwrites whatever `config.json` said. A Llama model is now labeled
Mistral.

**Config key remapping** -- Mistral-native keys are translated to HF keys:

```python
config_mapping = {
    "dim": "hidden_size",
    "norm_eps": "rms_norm_eps",
    "n_kv_heads": "num_key_value_heads",
    "n_layers": "num_hidden_layers",
    "n_heads": "num_attention_heads",
    "hidden_dim": "intermediate_size",
}
```

**Forced defaults**:

```python
"tie_word_embeddings": ("tied_embeddings", False),        # forced False
"max_position_embeddings": ("max_position_embeddings", 128_000),  # forced 128k
```

If the Llama model's actual `tie_word_embeddings` is `True`, or its actual
`max_position_embeddings` differs from 128k, the Mistral parser overrides it.

---

## 3. The Correctness Bugs

### Bug 1: Q/K weight permutation (the worst one)

`MistralForCausalLM.load_weights()` wraps every `(name, weight)` pair through
`maybe_remap_mistral()`. This method:

1. Checks if the weight name contains `wq` or `wk` (Mistral-native key names)
2. If so, applies a permutation that reorders the rotary embedding dimensions:

```python
def permute(w, n_heads, attn_out):
    attn_in = self.config.head_dim * n_heads
    return (
        w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
        .transpose(1, 2)
        .reshape(attn_in, attn_out)
    )
```

This permutation converts from Mistral's split-real-imaginary RoPE layout to
HF's interleaved layout. It exists because real Mistral checkpoints store Q/K
weights in a different rotary embedding dimension order than HuggingFace
convention.

**When this hits Llama**: Meta's original `.pth` checkpoint files use the same
key names as Mistral (`wq`, `wk`, `wv`, `wo`) because Mistral's format
descends from Meta's original release format. So when a flattened Llama
checkpoint gets loaded through `MistralForCausalLM`, `maybe_remap_mistral`
**matches** the `wq`/`wk` keys and applies the permutation.

If the Llama model's Q/K weights are already in HF-interleaved order (as they
are in HuggingFace-hosted copies), this permutation **corrupts** the attention
computation. The model will produce wrong outputs with no error or warning.

If the Llama model's Q/K weights are in Meta's original order (as they are in
`consolidated.*.pth` files), the permutation may or may not be correct
depending on whether Meta and Mistral use the same RoPE dimension ordering.
They are closely related but not guaranteed identical across all model versions.

**Impact**: Silent correctness failure. Attention patterns are wrong. Outputs
are garbage or subtly degraded, especially at longer context lengths where
rotary position encoding matters most. No error, no crash, no warning.

### Bug 2: Sliding window attention (conditional)

`_remap_mistral_sliding_window` in `adapt_config_dict()` checks for a
`sliding_window` field in `params.json`:

```python
def _remap_mistral_sliding_window(config):
    if sliding_window := config.get("sliding_window"):
        if isinstance(sliding_window, int) and config.get("layer_types") is None:
            config["layer_types"] = ["sliding_attention"] * config["num_hidden_layers"]
```

When `layer_types` is set to `"sliding_attention"` on every layer, the
attention backend uses `FlashAttentionImpl` with `window_size=(sw - 1, 0)`.
This is a **hard truncation** -- queries physically cannot attend to keys
outside the window. Old KV entries are evicted.

**When this hits Llama**: Standard Llama `params.json` does NOT contain a
`sliding_window` field, so this is a no-op for most Llama checkpoints.
However:

- If the HF `config.json` has `"sliding_window": null` (which some Llama
  models do), the `setdefault` merge at the end of `adapt_config_dict`
  copies it into the config. `None` is falsy, so `_remap_mistral_sliding_window`
  skips it. Safe.
- If someone ships a Llama checkpoint with a non-null `sliding_window` in
  `params.json` (unusual but possible), it WOULD trigger SWA on all layers,
  destroying long-context capability.

**Impact**: Usually safe for standard Llama checkpoints. But fragile -- one
unexpected field in `params.json` and you silently lose long context.

### Bug 3: Config value corruption

Even without weight or attention bugs, the Mistral config parser forces:

- `tie_word_embeddings = False` (Llama 3.x uses `False` so this is usually
  safe, but Llama 2 and some variants use `True`)
- `max_position_embeddings = 128000` (Llama 3.1+ is 128k so this is usually
  safe, but Llama 2 is 4096 and Llama 3.0 is 8192)
- `model_type = "transformer"` instead of `"llama"`

**Impact**: For Llama 3.1+ specifically, these defaults happen to be close
enough. For older Llama variants, the forced 128k context length would cause
incorrect positional encoding behavior.

---

## 4. Impact on TPU Specifically

On GPU, all three bugs above apply. On TPU, the situation is compounded by two
separate issues: the JAX registry gap AND a Marin-side override.

### Issue 1: `MistralForCausalLM` is not in the JAX registry

`tpu-inference`'s JAX model registry has only 10 exact-match entries, and
Mistral is not one of them. In principle, the default `MODEL_IMPL_TYPE=auto`
flow would try the JAX path first, fail with `UnsupportedArchitectureError`
for `MistralForCausalLM`, and fall back to the torchax path.

### Issue 2: Marin forces `MODEL_IMPL_TYPE=vllm`

But in practice, Marin doesn't even get to the JAX registry lookup.
`lib/marin/src/marin/inference/vllm_server.py:774` sets:

```python
("MODEL_IMPL_TYPE", "vllm"),
```

with the comment:

> tpu_inference defaults MODEL_IMPL_TYPE=auto, which selects flax_nnx for many
> architectures. flax_nnx currently fails without an auto mesh context, so
> default to the vllm implementation unless the user overrides it.

This means **every model in Marin goes through the PyTorch-on-torchax path**,
not just misidentified ones. The JAX registry is never consulted. Even a
correctly-identified `LlamaForCausalLM` gets loaded via vLLM's PyTorch model
loader, traced through torchax, and compiled to TPU HLO.

### Combined effect

For a misidentified Llama model running on TPU through Marin:

1. `MODEL_IMPL_TYPE=vllm` skips the JAX registry entirely
2. `get_vllm_model()` loads the PyTorch `MistralForCausalLM` class
3. `MistralForCausalLM.load_weights()` runs `maybe_remap_mistral`, which
   permutes Q/K weights (bug 1)
4. The model runs on the torchax path (slower than native JAX, but this is
   what Marin uses for ALL models today, not just misidentified ones)

So on TPU through Marin, the Q/K corruption is the main additional damage from
misidentification. The torchax-vs-JAX performance difference is not caused by
the misidentification itself -- Marin forces the torchax path for everything.

### Why Marin forces the vllm path

The comment says `flax_nnx` fails without an "auto mesh context." This refers
to a JAX mesh initialization issue where `tpu-inference`'s JAX path expects a
mesh to be set before model loading, but the vLLM engine startup doesn't
provide one in the right scope. The old fork's `marin` branch fixed several
mesh context bugs (see section 3 of the `tpu-dep-hell` logbook), but those
fixes are not in upstream `main`, so Marin sidesteps the whole issue by
staying on the torchax path.

### What it would take to use the JAX path

To benefit from the native JAX path (and the alias fix), Marin would need to:

1. Fix or work around the mesh context issue in upstream `tpu-inference`
2. Remove the `MODEL_IMPL_TYPE=vllm` override
3. Add the `MistralForCausalLM` alias to the JAX registry (for defense
   against misidentification)

Until then, the JAX registry alias alone wouldn't help -- Marin never reaches
the registry lookup.

---

## 5. What the Old Fork Fixed (and What Upstream Doesn't)

The old `marin-community/tpu-inference` fork on the `marin` branch (commit
`a74f6142`) added an explicit alias in the JAX model registry:

```python
_MODEL_REGISTRY["MistralForCausalLM"] = LlamaForCausalLM  # alias
```

Plus a `model_type` fallback lookup so that even if the architecture string
was wrong, the loader could find the right JAX class by checking
`hf_config.model_type == "llama"`.

**What this fixed**: Both the TPU performance problem AND the Q/K weight
corruption -- because the JAX path bypasses `maybe_remap_mistral` entirely.

The Q/K permutation lives in `MistralForCausalLM.load_weights()`, which is
vLLM PyTorch code. When the JAX registry alias routes the model to the native
JAX `LlamaForCausalLM`, weights are loaded through `tpu-inference`'s own JAX
weight loading code, which has no concept of Mistral-native key remapping or
Q/K permutation. It just loads HF-format weights directly as `jax.Array`
objects. So the alias fix solved two problems in one shot:

1. No Q/K corruption (JAX weight loader doesn't call `maybe_remap_mistral`)
2. Fast native JAX execution (no torchax overhead)

**What this did NOT fix**: The config mangling. `MistralConfigParser` runs
during vLLM's config resolution phase, before `tpu-inference` ever sees the
model. Both the JAX and torchax paths receive the already-corrupted config
(forced `architectures`, `tie_word_embeddings`, `max_position_embeddings`).
The alias fixes weight loading and dispatch, but not config parsing.

**GPU is still affected regardless**: On GPU, vLLM doesn't go through
`tpu-inference` at all. The PyTorch `MistralForCausalLM` class runs
`maybe_remap_mistral` during weight loading, so the Q/K corruption applies
on GPU with no workaround short of fixing the checkpoint layout or the
detection logic in vLLM itself.

**Upstream `main` has none of these fixes.** No alias, no model_type fallback,
no awareness of the problem at all.

---

## 6. Does This Branch Fix It?

**No.** The `worktree-tpu-dep-hell` branch moves Marin to upstream `main` of
both `marin-community/vllm` and `marin-community/tpu-inference`. Since upstream
`main` has:

- The same `is_mistral_model_repo()` detection logic
- The same `MistralConfigParser` with architecture forcing and config mangling
- The same `maybe_remap_mistral` Q/K permutation
- No `MistralForCausalLM` entry in the JAX registry
- No alias or fallback logic

...the full chain of bugs is still present. A flattened Llama checkpoint would
still be misidentified as Mistral, still get its Q/K weights permuted, and
still fall back to the slow PyTorch-on-torchax path on TPU.

---

## 7. The Actual Fix and What Should Be Upstreamed

The GCS cache repair (marin-community/marin#4356) was the correct immediate
fix: delete the flattened prefixes and re-download so that `params.json` and
`consolidated*.pth` live under `original/` where vLLM's Mistral detection
doesn't see them.

For defense in depth, two things should be upstreamed:

1. **To `vllm-project/tpu-inference`**: Register `MistralForCausalLM` as an
   alias for `LlamaForCausalLM` in the JAX registry. This prevents the
   performance degradation on TPU even when misidentification occurs. One line:
   ```python
   _MODEL_REGISTRY["MistralForCausalLM"] = LlamaForCausalLM
   ```

2. **To `vllm-project/vllm`**: Make `is_mistral_model_repo()` more
   discriminating. Currently it only checks for `consolidated*.safetensors`.
   It could additionally verify that no `config.json` with
   `"architectures": ["LlamaForCausalLM"]` exists, or check that the files
   are not under an `original/` subdirectory. This would prevent the
   misidentification at the source.

Neither of these is present in upstream today.

---

## 8. Affected Prefixes (from #4356)

Broken (flattened layout, vulnerable to misidentification):
- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B--d04e592/`
- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B--main/`
- `gs://marin-us-central1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/`
- `gs://marin-us-central1/models/meta-llama--Llama-3-2-1B--4e20de3/`
- `gs://marin-us-central1/models/meta-llama--Llama-3-2-1B--main/`
- `gs://marin-us-central1/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b/`
- `gs://marin-us-central2/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/`
- `gs://marin-us-central2/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b/`
- `gs://marin-us-east1/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/`
- `gs://marin-us-east5/models/meta-llama--Llama-3-2-1B--main/`

Already repaired:
- `gs://marin-us-east5/models/meta-llama--Llama-3-1-8B-Instruct--0e9e39f/`
- `gs://marin-us-east5/models/meta-llama--Llama-3-3-70B-Instruct--6f6073b/`

---

## 9. Proposed Fix: Trust `config.json` When It Exists

### The problem with the current detection

The misidentification happens in `vllm/transformers_utils/config.py`, in the
`get_config()` function. The auto-detection logic is:

```python
if config_format == "auto":
    if is_mistral_model_repo(...) and file_or_path_exists(..., "params.json"):
        config_format = "mistral"
```

This commits to the Mistral config path the moment it sees
`consolidated*.safetensors` + `params.json`, without checking whether a
perfectly good `config.json` already exists with the correct architecture.
Everything downstream — the architecture forcing, the config key remapping,
the Q/K weight permutation — flows from this one decision.

### Why `MistralConfigParser` exists at all

`MistralConfigParser` exists for a narrow use case: model repos distributed
**exclusively** in Mistral-native format, with no HuggingFace `config.json`.
These repos only have `params.json` (Mistral's config format) and
`consolidated*.safetensors` (Mistral's weight format). Without
`MistralConfigParser`, vLLM wouldn't know how to load them.

But most models on HuggingFace — including all Meta Llama models and all
Mistral models hosted on HF — ship with a `config.json` that already contains
the correct `architectures` field. For these repos, `MistralConfigParser` is
not needed, and when triggered incorrectly (by flattened checkpoint layouts),
it actively corrupts the config.

### The fix: prefer `config.json` when it has valid architectures

The change is in `get_config()`. Before committing to the Mistral path, check
whether `config.json` exists and already declares architectures. If it does,
trust it.

```python
# Current code (vllm/transformers_utils/config.py, in get_config):
if config_format == "auto":
    if is_mistral_model_repo(...) and file_or_path_exists(..., "params.json"):
        config_format = "mistral"

# Proposed replacement:
if config_format == "auto":
    if is_mistral_model_repo(...) and file_or_path_exists(..., "params.json"):
        # Check if a valid HF config.json already exists. If it declares
        # architectures, trust it — MistralConfigParser is only needed for
        # repos that lack a usable HF config entirely.
        try:
            hf_dict, _ = PretrainedConfig.get_config_dict(
                model, revision=revision
            )
            if hf_dict.get("architectures"):
                config_format = "hf"
            else:
                config_format = "mistral"
        except OSError:
            config_format = "mistral"
```

### Why this works for every case

**Flattened Llama checkpoint** (the bug we're fixing):
- `consolidated*.safetensors` at root: `is_mistral_model_repo()` → `True`
- `params.json` at root: exists
- `config.json` at root: exists, `architectures: ["LlamaForCausalLM"]`
- Result: `config_format = "hf"` → loads as Llama → correct

**HF-hosted Mistral model** (e.g. `mistralai/Mistral-7B-v0.1`):
- `consolidated*.safetensors`: may or may not exist
- `params.json`: may or may not exist
- `config.json`: exists, `architectures: ["MistralForCausalLM"]`
- Result: `config_format = "hf"` → loads as Mistral via the standard HF path
- `MistralForCausalLM` is still the class, `load_weights()` still calls
  `maybe_remap_mistral`, but since HF-format weights use `q_proj`/`k_proj`
  keys (not `wq`/`wk`), the permutation logic doesn't match any keys and is
  a no-op. Correct.

**Mistral-native-only repo** (no `config.json` at all):
- `consolidated*.safetensors`: exists
- `params.json`: exists
- `config.json`: does not exist → `PretrainedConfig.get_config_dict()` raises
  `OSError`
- Result: `config_format = "mistral"` → `MistralConfigParser` handles it →
  correct (this is the one case where `MistralConfigParser` is actually needed)

**Mistral-native repo with incomplete `config.json`** (exists but no
`architectures`):
- `config.json`: exists but `architectures` is missing or empty
- Result: `config_format = "mistral"` → `MistralConfigParser` handles it →
  correct

**Correctly-structured Llama checkpoint** (`original/` subdirectory intact):
- `consolidated*.safetensors` at root: does not exist (only under `original/`)
- `is_mistral_model_repo()` → `False`
- The entire `if` block is skipped → normal HF loading → correct
- (This case already works today; the fix doesn't change it.)

### What this fixes

This single change in vLLM's config resolution eliminates the entire
misidentification chain:

1. **No architecture forcing** — `MistralConfigParser` never runs, so
   `architectures` is never overwritten to `["MistralForCausalLM"]`
2. **No Q/K permutation** — since the model loads as `LlamaForCausalLM`,
   `maybe_remap_mistral` is never called. Even on the PyTorch-on-torchax path
   (which Marin uses today with `MODEL_IMPL_TYPE=vllm`), the weight loading
   goes through `LlamaForCausalLM.load_weights()` which does not permute
   anything.
3. **No config mangling** — `tie_word_embeddings`, `max_position_embeddings`,
   and `model_type` come from the real `config.json` values, not Mistral
   defaults
4. **No sliding window risk** — `_remap_mistral_sliding_window` never runs
5. **Works on both GPU and TPU** — the fix is in config resolution, upstream
   of both the JAX and torchax paths
6. **Works regardless of `MODEL_IMPL_TYPE`** — whether Marin stays on
   `MODEL_IMPL_TYPE=vllm` or eventually switches to `auto`, the config is
   correct before model dispatch happens
7. **Real Mistral models still work** — HF-hosted Mistral repos have
   `config.json` with `MistralForCausalLM`, which is trusted and loaded
   correctly through the standard HF path

### What this does NOT fix

- **Already-corrupted GCS caches** — if weights were previously loaded and
  cached with permuted Q/K values, the cache needs to be rebuilt. The config
  fix prevents future corruption but doesn't repair past damage.
- **`tpu-inference` JAX registry gap** — `MistralForCausalLM` is still not in
  the JAX registry. If someone runs a real Mistral model with
  `MODEL_IMPL_TYPE=auto`, it will still fall back to the torchax path. This is
  a separate issue from misidentification.
- **Marin's `MODEL_IMPL_TYPE=vllm` override** — Marin still forces the torchax
  path for all models. This is a separate performance concern unrelated to
  correctness.

### Where to make the change

This fix belongs in `vllm-project/vllm` (upstream), since the detection logic
is in vLLM core, not in `tpu-inference`. It could be prototyped first on
`marin-community/vllm`'s `main` branch and validated with a smoke test using a
flattened Llama checkpoint, then submitted as an upstream PR.

The change is ~10 lines in one function in one file
(`vllm/transformers_utils/config.py`), with no new dependencies.
