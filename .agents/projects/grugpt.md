# Grug: Explicit-Sharding LM Trainer

## Background

Inspired by [grugbrain.dev](https://grugbrain.dev/) and Andrej Karpathy’s [NanoGPT](https://github.com/karpathy/nanoGPT), we want a “grug-simple” causal LM trainer that showcases JAX’s explicit sharding mode while relying only on primitive building blocks: `jax`, `einops`, and JAX dataclasses (via `jax.tree_util.register_dataclass`). Training utilities stay minimal—optax for optimization, Levanter’s data loading/trackers for ingestion + logging, HuggingFace tokenizers (or Levanter’s serializer) for text, and TensorStore serialization for checkpoints. We explicitly do **not** want Haliax abstractions or class-heavy APIs—every computation lives in straightforward top-level functions so the trainer reads like a notebook.


## Grug Principles

### Software Principles

- **Few methods, keep arrays out of them inside jit.** We still prefer top-level functions, but lightweight dataclass helpers/property accessors—and any method that only touches metadata, not arrays—are fine even inside jit-traced regions. Once a method would manipulate `jax.Array`s, implement it as a standalone function instead.
- **Keep dependencies small.** Prefer `einshard`, `einops`, `optax`, JAX core libs, HF tokenizers, and Levanter’s `data` + `tracker` + TensorStore serialization APIs. Grug doesn't want to reinvent wheels, but we also don't want heavy frameworks obscuring the core logic.
- **Fast attention kernels are allowed, but keep the surface simple.** For now, we standardize on `ejkernel` block-sparse attention (`ejkernel.modules.operations.blocksparse_attention`) with a reference fallback.
- **Serializable state.** Trainer state must round-trip through `levanter.tensorstore_serialization.tree_{de,}serialize_leaves_tensorstore` with no custom logic.
- **Consumption-ready.** Provide a `uv run python -m marin.grugpt.train` entrypoint plus tests. Tracking hooks log loss/perplexity through Levanter’s tracker interface.

### JAX/ML Principles

- **Mesh-first mental model.** We always create one mesh axis per logical unit. For now, `['replica', 'data', 'tensor']`. Work must still run on a single GPU (mesh axes collapse to size 1) but should seamlessly extend to 4-device TPU v4-8 pods for CI. Mesh creation and validation live in one place.
- **Use good kernels when available.** Grug is happy to call out to other people's fast attention kernels (ejkernel blocksparse today; Tokamax later) as long as the surface stays simple and the fallback reference path remains.
- **Explicit sharding everywhere.** Arrays carry `PartitionSpec`s in their types using `jax.set_mesh`, `AxisType.Explicit`, and `einshard` helpers. Any op with ambiguous shardings must either supply `out_sharding=` or run inside `auto_axes`.

### Code Organization Principles

- **Keep `levanter.grug` core-only.** The `levanter.grug` package should stay “notebook-like”: raw `jax.Array` inputs/outputs, top-level functions, small dataclasses, and minimal plumbing.
- **Adapters live outside grug.** Integration with Levanter model interfaces (`LmHeadModel`, `NamedArray`, etc.) lives in `levanter.models`, currently `levanter.models.grug_wrapper`.
- **Mask spec is simple.** Grug’s attention mask is a small spec (`levanter.grug.attention.AttentionMask`) storing only raw JAX arrays/ints (no NamedArray fields and no explicit dense masks yet).


## Proposed Directory Layout

| Path | Purpose |
| --- | --- |
| `lib/grugpt/pyproject.toml` | `uv`-managed package metadata; declares dependencies on `jax`, `einshard`, `einops`, `optax`, `levanter`. |
| `lib/grugpt/src/grugpt/__init__.py` | Re-export configs and public helpers for downstream scripts. |
| `lib/grugpt/src/grugpt/config.py` | Pure dataclasses defining model/data/runtime configs; registered as pytrees so they can flow through `jax.jit`. |
| `lib/grugpt/src/grugpt/attention.py` | Reference attention + rotary helpers + backend dispatcher (splash stubbed). |
| `lib/grugpt/src/grugpt/model.py` | Parameter init + forward pass using `jnp.einsum`, `einops.rearrange`, and `jax.sharding.PartitionSpec` to place weights. |
| `lib/grugpt/src/grugpt/data.py` | Glue around Levanter datasets/tokenizers; config-driven creation of iterable batches that already respect the mesh sharding. |
| `lib/grugpt/src/grugpt/train.py` | Training loop + CLI: builds mesh, data, optimizer, trackers, compiles `train_step`, handles checkpoints. |
| `tests/grugpt/test_model.py` | Shape & sharding tests for forward pass. |
| `tests/grugpt/test_train_step.py` | Golden-step test covering loss, grads, and tracker logging. |

## Detailed Design

### Configuration & Dataclasses

- `ModelConfig`, `DataConfig`, `OptimizerConfig`, `RunConfig` defined in `config.py` using `@dataclass(frozen=True)`. After definition we call `register_dataclass(ModelConfig)` so configs can live in pytrees.
- Each config only stores data—no helper methods. Convenience functions live in `config.py` (e.g., `def validate_config(cfg)`), returning new configs instead of mutating.
- `TrainingState` dataclass contains `(params, opt_state, step, rng)` and is registered as a pytree. Parameters are themselves a dataclass (`ModelParameters`) mirroring module structure (token embedding, attention weights, etc.).
- Provide `default_run_config()` helper that builds a small LM spec for smoke tests (e.g., 4 layers, 256 hidden size).
- Attention backend selection is a single string: `AttentionBackend = Literal["reference", "blocksparse"]`. This keeps the surface small and avoids runtime config objects in hot paths.

- `mesh.py` exposes `def create_mesh(cfg: RunConfig) -> Mesh` which:
  1. Discovers available devices.
  2. Reshapes them into `(replica, data, tensor)` dims specified by `cfg.mesh_shape` (default `(1, 1, num_devices)`).
  3. Calls `jax.make_mesh(shape, axis_names=("replica", "data", "tensor"), axis_types=(AxisType.Explicit, ...)` and `jax.set_mesh(mesh)`.
- We attach shardings explicitly via `jax.sharding.PartitionSpec` during parameter init (e.g., embeddings on `('data', None)`, attention weights on `('data','tensor')`). No separate sharding helper module is needed; everything flows through `jax.sharding.reshard`.

### Llama-Style Model Sketch

```python
@dataclass
class GrugAttentionParams:
    w_q: jax.Array  # [hidden, num_heads * head_dim]
    w_k: jax.Array  # [hidden, num_kv_heads * head_dim]
    w_v: jax.Array
    w_o: jax.Array  # [num_heads * head_dim, hidden]

@dataclass
class GrugBlockParams:
    attn: GrugAttentionParams
    rms_attn: jax.Array  # RMSNorm weights (bias optional)
    rms_mlp: jax.Array
    mlp_gate: jax.Array  # [hidden, intermediate]
    mlp_up: jax.Array
    mlp_down: jax.Array  # [intermediate, hidden]

@dataclass
class GrugModelParameters:
    token_embed: jax.Array  # [vocab, hidden]
    output_proj: jax.Array  # shared with token_embed if tie_embeddings
    blocks: tuple[GrugBlockParams, ...]
    final_norm: jax.Array

def rms_norm(x, weight, eps):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    return x * weight / jnp.sqrt(var + eps)

def apply_attention(params, hidden, mask, rotary, backend):
    q = einsum("b s h, h (n d) -> b s n d", hidden, params.attn.w_q)
    k = einsum("b s h, h (m d) -> b s m d", hidden, params.attn.w_k)
    v = einsum("b s h, h (m d) -> b s m d", hidden, params.attn.w_v)
    q, k = apply_rotary_embedding(q, k, rotary)
    attn = attention(q, k, v, mask, backend=backend)
    return einsum("b s n d, (n d) h -> b s h", attn, params.attn.w_o)

def transformer_block(block, hidden, mask, rotary, runtime, eps):
    attn_in = rms_norm(hidden, block.rms_attn, eps)
    attn_out = apply_attention(block, attn_in, mask, rotary, runtime)
    hidden = hidden + attn_out
    mlp_in = rms_norm(hidden, block.rms_mlp, eps)
    gate = einsum("b s h, h m -> b s m", mlp_in, block.mlp_gate)
    up = einsum("b s h, h m -> b s m", mlp_in, block.mlp_up)
    act = jnp.silu(gate) * up
    mlp_out = einsum("b s m, m h -> b s h", act, block.mlp_down)
    return hidden + mlp_out

def forward(params, tokens, mask, rotary, runtime, eps):
    hidden = params.token_embed[tokens]
    for block in params.blocks:
        hidden = transformer_block(block, hidden, mask, rotary, runtime, eps)
    hidden = rms_norm(hidden, params.final_norm, eps)
    logits = hidden @ params.output_proj.T
    return logits
```

- Use `einops.rearrange` for `[batch, seq, heads, head_dim]` manipulations where it improves clarity.
- Use `jnp.einsum` and explicit `PartitionSpec` annotations to keep matmuls consistent with the mesh layout.
- Attention backend selection lives in `attention.py` as a single `attention(q, k, v, mask, *, backend=...)` function:
  - `backend="blocksparse"` uses ejkernel blocksparse attention.
  - `backend="reference"` uses `einsum` + softmax.
  - The reference path exists for debugging; blocksparse is the default for real runs.
- Attention masks are a small spec dataclass (`levanter.grug.attention.AttentionMask`) rather than a dense mask/bias array. Grug defaults to causal masking.
- Layer norm: implement epsilon-stabilized RMSNorm using `jnp.mean/var` and explicit shardings.
- Residual dropouts remain optional and implemented via `jax.random.bernoulli` (with `out_sharding=P(None)` to keep them replicated).

### Loss & Training Step

- `loss_fn(params, batch, *, cfg, mesh)` runs forward pass and computes token-level cross-entropy (optionally ignoring padding tokens). Output sharding uses `(Batch @ data, None)` so reductions stay deterministic.
- Optimizer: `optax.adamw` built from `TrainingConfig`. `TrainingState` carries `opt_state`. We wrap the optimizer in `optax.apply_updates`.
- `@jax.jit` (or `jax.jit(static_argnames=("cfg",))`) compiled `train_step(state, batch, tracker)`:
  1. `loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)`
  2. `grads = reshard(grads, param_spec)` so updates stay model-sharded.
  3. `updates, new_opt_state = optimizer.update(grads, state.opt_state, state.params)`
  4. `new_params = optax.apply_updates(state.params, updates)`
  5. `tracker.log({"train/loss": loss, "train/ppl": jnp.exp(loss)}, step=state.step)`
  6. Return updated `TrainingState(step + 1, rng=next_rng)`
- Provide `eval_step` mirroring `train_step` without optimizer updates for validation.

### Data & Tokenization

- `DataConfig` picks between HuggingFace tokenizer (`transformers.AutoTokenizer`) or a Levanter `TreeCache` path. We default to HF and fall back to Levanter’s `deserialize_leaves_tensorstore` when resuming vocab.
- `data.py` exposes `build_dataloader(cfg, mesh)` that:
  - Builds/loads tokenizer.
  - Constructs a `levanter.data.text.TokenSeqDataset` (or other AsyncDataset) using seq length + sharding info.
  - Wraps it in a lightweight async background iterator returning dictionaries `{"input_ids": sharded_array, "labels": sharded_array}` already resharded to `(Batch@data, Seq)`.
- Because Levanter’s loader internally uses Haliax, grugpt never imports haliax directly—all adapters convert outputs to plain `jax.Array`s via `jnp.asarray` before feeding the model.

### Checkpointing & Tracking

- `checkpoint_path = cfg.run_dir / f"step_{state.step:07d}"`.
- Saving uses `tree_serialize_leaves_tensorstore(path, (state, cfg))` inside host 0 guard; loading uses matching `tree_deserialize...`.
- Trackers: `levanter.tracker` configs are CLI-selectable (noop vs wandb). `train.py` enters `with current_tracker(tracker):` before training.
- On resume, we log hyperparameters once and emit `train/time_per_step`, `train/tok_per_sec` metrics every `cfg.log_every` steps.

### CLI Entrypoint

- `src/marin/grugpt/train.py` implements `def main(argv=None): ...` using `argparse` (no click). Steps:
  1. Parse YAML/JSON or CLI flags into `RunConfig` (we can reuse Levanter’s draccus parser if convenient).
  2. Build mesh and set it globally.
  3. Initialize tokenizer + dataloader + tracker.
  4. Initialize or restore `TrainingState`.
  5. Compile `train_step`/`eval_step` once per process.
  6. Run training loop with periodic evaluation/checkpointing.
- Provide `if __name__ == "__main__": uvicorn.main()` guard and register console script in `pyproject.toml`.

- `tests/grugpt/test_model.py` creates a toy mesh (1×1×1), initializes params, runs forward pass, and asserts:
  - logits shape `[batch, seq, vocab]`
  - `jax.typeof(hidden).sharding` matches expected `PartitionSpec`
- `tests/grugpt/test_mesh.py` spins up synthetic mesh metadata (e.g., 4-device v4-8) to ensure axis naming/resizing works when devices >= 4.
- `tests/grugpt/test_train_step.py` runs two train steps on random tokens, verifying loss decreases and tracker receives entries (with splash attention disabled for determinism).
- All tests run under `uv run pytest tests/grugpt -n auto`; TPU-specific splash kernels get smoke-tested behind a flag if CI has TPU runtime, otherwise stub out gracefully.

### Current Status

- The current prototype lives under `lib/levanter/src/levanter/grug/` (not yet extracted to `lib/grugpt/`).
- Attention uses ejkernel blocksparse by default; reference attention remains available for debugging.
- Levanter integration lives in `lib/levanter/src/levanter/models/grug_wrapper.py` (kept out of `levanter.grug` intentionally).

## Implementation Tasks

1. **Data + tracker integration.** Swap the synthetic batch generator for a real tokenizer/dataset pipeline (leveraging Levanter loaders) and pipe metrics through the tracker API.
2. **Attention backends.** Implement the TPU splash backend (or other hardware-accelerated kernels) and keep the reference path as a fallback.
3. **Checkpointing.** Hook `tree_{de}serialize_leaves_tensorstore` into the training loop so state/rng/optimizer snapshots can be saved/restored.
4. **Evaluation + CLI polish.** Add validation hooks, flag parsing (YAML/Draccus), and a proper entry point under `uv run` instead of the current hard-coded config.
5. **Testing.** Port the sketched unit tests (`test_model`, `test_mesh`, `test_train_step`) into `tests/grugpt/` and wire them into CI.

## Future Integration with Levanter Trainer

To plug a “grug-defined” model into Levanter’s trainer (`lib/levanter/src/levanter/trainer.py`), we’d need:

1. **Model interface parity.** Implement a wrapper that presents `forward(batch, *, train)` and exposes parameter/state PyTrees the way Levanter expects (see `lib/levanter/src/levanter/models/lm_model.py`). Our current dataclasses are close; we’d just need adapters for attention masks, loss computation, and optional KV cache state.
2. **Loss adapter.** The trainer takes arbitrary batches and a loss function. We’d supply a simple callable that takes the trainer’s batch (whatever structure we choose) plus model params/rng and returns `(loss, metrics)`. No need to emit `LmExample`s if we keep the batch format consistent.
3. **Config plumbing.** Expose the grug model config via Levanter’s config system (Draccus). That means a small adapter class registered under `LmConfig` that instantiates `GrugModelParameters` and returns the forward/loss function pointers. See `lib/levanter/src/levanter/models/llama.py` for reference.
4. **Optimizer/state hooks.** Levanter’s trainer manages optimizer state, gradient accumulation, and checkpointing. Our `TrainingState` would need to slot into that (likely by reusing Levanter’s `OptimizerState` structures) so serialization and resume work seamlessly.
5. **Sharding compatibility.** Ensure our explicit `PartitionSpec`s line up with Levanter’s mesh/resource mapping. Either reuse Levanter’s mapping utilities or provide a translation layer so both sides agree on axis names.

With those pieces, defining a “grug” model could be as simple as providing the dataclasses + forward function, and letting Levanter’s trainer handle data flow, logging, checkpointing, and multi-host orchestration.

## Open Questions & Follow-Ups

- Do we want gradient accumulation/microbatching in v1, or stick to per-step full batches?
- Should we support tensor parallel axes beyond `('model',)` at launch, or gate behind a flag until we verify the explicit-sharding rules?
- For tokenizer caching, is it acceptable to rely solely on HF local cache, or should we add explicit `deserialize_leaves_tensorstore` support for tokenizers stored in checkpoints?
- What minimum TPU/GPU topology should the CI tests assume (2 devices vs 4)?
