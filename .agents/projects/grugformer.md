# Grug: Explicit-Sharding LM Trainer

## Background

Inspired by [grugbrain.dev](https://grugbrain.dev/) and Andrej Karpathy’s [NanoGPT](https://github.com/karpathy/nanoGPT), we want a “grug-simple” causal LM trainer that showcases JAX’s explicit sharding mode while relying only on primitive building blocks: `jax`, `einops`, and JAX dataclasses (via `jax.tree_util.register_dataclass`). Training utilities stay minimal—optax for optimization, Levanter’s data loading/trackers for ingestion + logging, HuggingFace tokenizers (or Levanter’s serializer) for text, and TensorStore serialization for checkpoints. We explicitly do **not** want Haliax abstractions or class-heavy APIs—every computation lives in straightforward top-level functions so the trainer reads like a notebook.


## Grug Principles

### Software Principles

- **Few methods**: prefer top level functions to lots of methods (i.e. torch-style class hierarchies). Use `@dataclass`es for parameter/state containers, but keep logic in functions. Accessors are okay for small helpers (e.g. `def get_attention_params(model_params): ...`).
- **No array-involved methods in jit**: avoid class methods that operate on `jax.Array` fields inside `@jax.jit`-compiled functions. There are a few corner cases where they behave surprisingly. (Mostly we need to avoid `jit(module.compute_loss)`).
- **Keep dependencies small.** Prefer `einshard`, `optax`, JAX core libs, HF tokenizers, and Levanter’s `data` + `tracker` + TensorStore serialization APIs. Grug doesn't want to reinvent wheels, but we also don't want heavy frameworks obscuring the core logic.
- **Fast kernels are allowed, but keep the surface simple.** For now, grug core hard-depends on `ejkernel` for block-sparse attention (`ejkernel.modules.operations.blocksparse_attention`) and keeps a separate reference implementation for debugging/regressions.
- **Serializable state.** Trainer state must round-trip through `levanter.tensorstore_serialization.tree_{de,}serialize_leaves_tensorstore` with no custom logic.
- **Consumption-ready.** Provide a `uv run python -m marin.grugpt.train` entrypoint plus tests. Tracking hooks log loss/perplexity through Levanter’s tracker interface.
- **Limit config knobs.** In general, it's better to copy-paste experimental code than to bake every possible option into the core. Keep the config surface minimal and stable. As an example, attention sinks should be added by copy-pasting/editing the attention callsite in a speedrun script, not by adding a `cfg.use_attention_sinks` flag. But changing the number of layers etc. should be easy via config. This is a judgment call.
  An exception to this principle is when constructing an A/B test in a speedrun file (see docs/recipes/change_grug.md)—in that case, it's okay (even preferred) to add temporary flags to your copy-pasted grug core to toggle between two behaviors for comparison.

## Working Agreement: How Grug Evolves

- Canonical “best guess” lives in `lib/levanter/src/levanter/grug/`.
- The evolving speedrun entrypoint is `experiments/speedrun/grugformer_starter/grugformer_speedrun.py`.
- One-off speedruns under `experiments/speedrun/…` are snapshots/edit surfaces; they should not silently become the source of truth.
- When we upstream a successful experiment, we update tests and record/clean up old experiments per `docs/recipes/change_grug.md`. This involves deletion of incompatible experiments but leaving a trail.

### JAX/ML Principles

- **Mesh-first mental model.** We always create one mesh axis per logical unit. For now, `['replica_dcn', 'replica', 'data', 'model']`. Work must still run on a single GPU (mesh axes collapse to size 1) but should seamlessly extend to 4-device TPU v4-8 pods for CI. Mesh creation and validation live in one place.
- **Use good kernels when available.** Grug is happy to call out to other people's fast attention kernels (ejkernel blocksparse today) as long as the surface stays simple and the fallback reference path remains.
- **Explicit sharding everywhere.** Arrays carry `PartitionSpec`s in their types. By using `set_mesh` with `AxisType.Explicit` we'll always know every Array's sharding. Any op with ambiguous shardings must either supply `out_sharding=` or run inside `auto_axes`. Prefer the former.

### Code Organization Principles

- **Keep `levanter.grug` core-only.** The `levanter.grug` package should stay “notebook-like”: raw `jax.Array` inputs/outputs, top-level functions, small dataclasses, and minimal plumbing.
- **Adapters live outside grug.** Integration with Levanter model interfaces (`LmHeadModel`, `NamedArray`, etc.) lives in `levanter.models`, currently `levanter.models.grug_wrapper`.
- **Mask spec is simple.** Grug’s attention mask is a small spec (`levanter.grug.attention.AttentionMask`) storing only raw JAX arrays/ints (no NamedArray fields and no explicit dense masks yet).

## Misc Style
- Use `einops.rearrange` for `[batch, seq, heads, head_dim]` manipulations where it improves clarity.
- Use `jnp.einsum` and explicit `PartitionSpec` annotations to keep matmuls consistent with the mesh layout.
- Grug core attention is a single `attention(q, k, v, mask)` that calls ejkernel block-sparse attention; `reference_attention(...)` exists for debugging/regressions but is not selected at runtime.
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
2. **Attention + masks.** Extend the structured mask surface (segments/windows) and keep the reference implementation available for debugging/regressions.
3. **Checkpointing.** Hook `tree_{de}serialize_leaves_tensorstore` into the training loop so state/rng/optimizer snapshots can be saved/restored.
4. **Evaluation + CLI polish.** Add validation hooks, flag parsing (YAML/Draccus), and a proper entry point under `uv run` instead of the current hard-coded config.
5. **Testing.** Port the sketched unit tests (`test_model`, `test_mesh`, `test_train_step`) into `tests/grugpt/` and wire them into CI.

## Next Milestones (Current Plan)

1) **PR what we have**
   - Keep it focused: grug core simplicity + ejkernel blocksparse + adapter placement + doc updates.

2) **Speedrun: hackable single-file gauntlet**
   - Target: `experiments/speedrun/hackable_transformer_starter/hackable_transformer_attn_sink.py` (similar to `experiments/hackable_transformer_starter_template.py`).
   - Copy-paste the grug core into the file and define a small “gauntlet API” (`init_fn`, `fwd_fn`, `loss_fn`, `train_step`).
   - Include both `reference_attention` and ejkernel block-sparse paths and a tiny correctness check comparing them on small shapes.

3) **Native datasets for grug**
   - Add “grug-native” dataset analogs for `CausalLmDataset` and `MultiturnChatDataset` that yield plain `jax.Array` batches (`tokens`, `labels`, optional `segment_ids`).
   - Reuse Levanter ingestion/tokenization internally via adapters until we decide to fully decouple.

4) **Preemption + resume in the grug trainer**
   - Follow the patterns in `lib/levanter/src/levanter/main/train_lm.py`: periodic checkpoints, resume-from-latest, and safe/atomic step dirs.
   - Checkpoint `(state, cfg, tokenizer metadata)` with Levanter’s TensorStore serialization.

5) **Lock the grug core surface with tests**
   - Shapes + `jax.jit` compile sanity for `init_parameters`/`forward`.
   - Sharding sanity: `PartitionSpec` expectations don’t explode when a mesh is active.
   - Mask semantics for causal + sliding window + segment ids.
   - (Optional) blocksparse vs reference numerical check on tiny shapes.

## Speedrun Milestone: Hackable Transformer Gauntlet

Goal: use Grug as a “copy-pasteable” reference implementation inside a single speedrun script (e.g.
`experiments/speedrun/hackable_transformer_starter/hackable_transformer_attn_sink.py`), so the workflow is:

1) copy/paste the Grug core into the file
2) modify it (e.g. add attention sinks)
3) run it through a standard gauntlet (correctness + compile + throughput + memory)

### Constraints

- Single-file friendly: minimal imports, no class-heavy APIs, top-level functions.
- Keep an accelerated path (ejkernel blocksparse) + a reference fallback path.
- Make the “hack points” obvious: attention sinks, masks, sharding, and config.

### Plan (concrete)

1) **Define a “Grug-In-File” template section in the speedrun script**
   - Put it near the top of the file under a clear header like `# === GRUG CORE (COPY-PASTE) ===`.
   - Include only:
     - param dataclasses (`GrugAttentionParams`, `GrugBlockParams`, `GrugModelParameters`)
     - `init_parameters(cfg, *, key)`
     - `forward(params, tokens, cfg, *, mask=None)`
     - `attention(q, k, v, mask)`
     - `AttentionMask` (Grug-local, raw arrays only)

2) **Make sinks a deliberate edit point**
   - Do not bake “sinks”/`softmax_aux` into the stable grug API.
   - In the hackable speedrun script, add sinks by directly editing the ejkernel block-sparse attention callsite (or by copy-pasting a slightly modified `attention()` helper into the file).

3) **Keep the mask surface minimal and explicit**
   - Use the Grug-local `AttentionMask` spec (causal + sliding window + segment ids).
   - Avoid accepting dense boolean masks in accelerated mode (raise loudly).
   - Default to causal masking in `forward()` when `mask is None`.

4) **Define the gauntlet API in the speedrun script**
   - Standardize a small set of functions the harness calls:
     - `build_cfg()` (or constants) that define model hyperparams.
     - `init_fn(key) -> params`
     - `fwd_fn(params, batch) -> logits`
     - `loss_fn(logits, labels) -> loss`
     - `train_step(state, batch) -> (new_state, metrics)`
   - Keep all state as plain pytrees so it’s easy to serialize/inspect.

5) **Gauntlet checks to run every time**
   - **Correctness sanity** (small shapes):
     - reference vs blocksparse forward (match within a tolerance)
     - gradients exist and are finite
   - **Compile sanity**:
     - time-to-first-step (TTFS) for `train_step`
   - **Throughput sanity**:
     - tokens/sec and step time for a fixed number of warmup+iters
   - **Memory sanity**:
     - optional: track max HBM/VMEM if available in the speedrun harness

6) **Keep Levanter adapters out of the speedrun file**
   - Do not rely on `LmHeadModel`/`NamedArray` in the hackable file.
   - If you need to run the same model inside Levanter, use the adapter in `levanter.models.grug_wrapper`.

7) **Document the expected “edit surface” inside the file**
   - Put a short list at the top:
     - “To add sinks: edit the ejkernel block-sparse attention call in `attention()`.”
     - “To change masking: edit `AttentionMask` / `mask` construction in `forward()`.”
     - “To change sharding: edit `PartitionSpec`s in init + `out_sharding=` callsites.”

## Integration Direction

Grug currently integrates with Levanter via a thin wrapper (`levanter.models.grug_wrapper`). That’s fine as an adapter, but it should not become a long-term architectural dependency.

Longer term, we want one of:

1) **A grug-native trainer** (preferred for “hackable” workflows)
   - Minimal surface: plain pytrees + explicit mesh + small config.
   - Owns data loading, checkpointing, and evaluation in a way that’s easy to copy/paste into speedrun scripts.

2) **Evolve Levanter/Marin to support grug natively**
   - Make Levanter’s trainer accept a “grug-style” model (pure functions + pytrees) without requiring a wrapper that reifies NamedArray/Haliax concepts.
   - Goal: the core stays `jax.Array`-first, and the training stack becomes more flexible rather than forcing everything into `LmHeadModel`-style interfaces.

The wrapper remains a pragmatic bridge, but the intended direction is to shrink/remove it over time.

## Open Questions & Follow-Ups

- Do we want gradient accumulation/microbatching in v1, or stick to per-step full batches?
- Should we support tensor parallel axes beyond `('model',)` at launch, or gate behind a flag until we verify the explicit-sharding rules?
- For tokenizer caching, is it acceptable to rely solely on HF local cache, or should we add explicit `deserialize_leaves_tensorstore` support for tokenizers stored in checkpoints?
- What minimum TPU/GPU topology should the CI tests assume (2 devices vs 4)?
- Loss: the current blockwise CE (`cross_entropy_block_size`) is a tradeoff; in the 125M speedrun we saw MFU jump from ~20 -> ~40 when disabling chunking (`block_size=None`). We need a better large-vocab loss kernel eventually.
