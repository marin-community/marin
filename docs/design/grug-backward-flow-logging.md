# Design: Grug Backward Flow Logging

## Problem

Grug already has clean module boundaries in `experiments/grug/base/model.py:78-195`
via `@named_call`, and the train loop already supports sampled auxiliary logging in
`experiments/grug/base/train.py:330-396`. What it does not surface is the thing you
usually want when a run goes numerically bad: which module's backward-gradient scale is
vanishing or blowing up, and how that module sits in the surrounding dataflow.

Today the closest built-in signal is parameter/update watching. That lives in
`levanter.callbacks.watch.compute_watch_stats(...)` and is wired from
`experiments/grug/base/train.py:369-382`, but it says nothing about the cotangent
flow through intermediate activations. If a linear-attention block is unstable, the
parameter norm alone is usually too late and too indirect.

We also already have a precedent for trace artifacts. The generic trainer writes JAXpr
and HLO artifacts in `lib/levanter/src/levanter/trainer.py:761-777`. Grug can reuse
that basic idea, but with a graph collapsed onto named module scopes and annotated
with sampled backward-flow stats.

## Goals

- Log sampled activation and backward-gradient RMS statistics at named module boundaries,
  explicitly separating module inputs from module outputs
- Produce a graphical DAG view that preserves layer identity (`block_0`, `block_1`, ...)
- Keep reusable mechanics and common residual-stream display inference in `levanter`,
  with only thin Grug-specific wiring
- Leave the default training path unchanged when the feature is disabled

**Non-goals**: full-step tracing on every iteration, a general-purpose debugger for all
JAX programs, perfect support for every nested JAX control-flow primitive, or a new
trainer-wide framework abstraction

## Proposed Solution

Add a small reusable module in `levanter.analysis.backward_flow` and wire it into the
canonical Grug base template.

### 1) Reusable backward marker

`lib/levanter/src/levanter/analysis/backward_flow.py:88-145` adds:

- `capture_backward_flow(...)`: a tracing-time context that turns logging on and can
  carry an optional gradient scale
- `log_backward_activation(x, site=...)`: a `custom_vjp` identity that leaves the
  forward value unchanged and logs activation stats in the forward pass and gradient
  stats in the backward pass
- `normalize_name_stack(...)`: removes transform wrappers such as `jvp(...)` and
  `transpose(...)` so metric keys stay stable

The marker uses the current JAX `name_stack` rather than hard-coding a separate naming
scheme. That lets the graph and the scalar metrics line up without inventing another
registry.

```python
@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def _tagged_identity(metric_prefix: str, site: str, x: jax.Array) -> jax.Array:
    return x

def _tagged_identity_fwd(metric_prefix: str, site: str, x: jax.Array):
    levanter.tracker.jit_log(_tensor_metrics(metric_prefix, x, site=site, kind="activation"), step=None)
    return x, None

def _tagged_identity_bwd(metric_prefix: str, site: str, _residual: None, cotangent: jax.Array):
    levanter.tracker.jit_log(_tensor_metrics(metric_prefix, cotangent, site=site, kind="gradient"), step=None)
    return (cotangent,)

def log_backward_activation(x: jax.Array, *, site: str = "out") -> jax.Array:
    context = _ACTIVE_CONTEXT.get()
    if context is None:
        return x
    name_stack = normalize_name_stack(str(source_info_util.current_name_stack()))
    return _tagged_identity(f"{context.prefix}/{name_stack}", site, x)
```

### 2) JAXpr-to-DAG projection

`lib/levanter/src/levanter/analysis/backward_flow.py:147-214` walks a traced JAXpr,
recurses into single-child nested JAXprs (enough for Grug's checkpointed blocks), and
builds a graph over normalized scope names instead of primitive names.

The graph is then collapsed onto the nodes that actually emitted backward-flow metrics.
That keeps the rendered artifact readable even when the underlying JAXpr contains many
unnamed primitive groups.

### 3) Small Grug wiring

`experiments/grug/base/model.py:78-195` gets explicit markers at the module
boundaries we care about:

- `CausalSelfAttention.__call__` input and output
- `MLP.__call__` input and output
- `Block.__call__` residual stream points: `resid_in`, `resid_post_attn`, and
  `resid_out`
- `Transformer.__call__` for embeddings and final hidden state

The transformer loop also adds `jax.named_scope(f"block_{i}")` at
`experiments/grug/base/model.py:191-193` so repeated layers do not collapse into a
single node in the graph.

`Block` itself is treated as a visual plate/container in the renderer rather than a
metric-bearing flow node. The actual dataflow is represented as residual-stream nodes
inside the block plate, with attention and MLP as sibling module nodes connected through
that stream.

### 4) Sampled train-step path

`experiments/grug/base/train.py:58-68` adds a `BackwardFlowConfig` field to the
trainer config. The reusable `BackwardFlowConfig()` default remains disabled with
`interval=0`, but base Grug chooses `interval=50` so important runs get sampled
backward-flow artifacts without extra launch wiring. Set `interval=0` in a run config to
disable it.

When `compute_backward_flow` is true, the jitted train step in
`experiments/grug/base/train.py:330-396` runs `jax.value_and_grad(...)` inside both:

- `capture_backward_flow(...)` to activate the module markers
- `levanter.tracker.defer_tracker_for_jit()` to smuggle the per-node stats out of JIT

Grug passes `sum(batch.loss_weight)` as the backward-flow gradient scale. That keeps raw
cotangent RMS available, while also logging `*_gradient_rms_scaled` on the unreduced loss
scale. This avoids coloring every mean-reduced LM gradient as vanishing just because the
loss was averaged over batch and sequence positions.

That produces a separate sampled compile path, which is acceptable because the feature
is explicitly infrequent.

### 5) Artifact emission

On sampled steps, `experiments/grug/base/train.py:571-586`:

1. logs the scalar backward-flow metrics
2. traces one grad JAXpr for the current batch shape
3. collapses it onto the tracked nodes
4. writes an HTML artifact under `.../artifacts/backward_flow/step_<n>.html`
5. logs the same HTML as `backward_flow/dag` when the active tracker supports native
   HTML media, such as W&B

The graph trace also runs under `capture_backward_flow(...)`. That matters for
identity-only residual probes such as `resid_in`: without active probes, those scopes
emit scalar metrics during the train step but do not leave a JAXpr node for the renderer
to collapse onto.

The generic renderer in `lib/levanter/src/levanter/analysis/backward_flow.py` is
deliberately self-contained HTML+SVG so it works in artifact viewers without requiring
external JS. It accepts optional lane hints, display edges, and visual plates, and when
display edges are provided, renderer depth is inferred from those edges.

`infer_backward_flow_render_hints(...)` provides the default "sensible if not perfect"
view for residual-stream models using only runtime names. If a graph contains probes
named `resid_in`, `resid_post_*`, and `resid_out` under block-like scopes, it infers
ordered residual anchors, branch edges through attention/MLP-like nodes, residual skip
edges, inter-block flow, and block plates. Grug calls that reusable inference in
`experiments/grug/base/train.py`, so the template only needs stable named scopes and
unobtrusive activation markers. Nodes are colored by scaled backward-gradient RMS when it
is available, falling back to raw gradient RMS otherwise. Raw gradient RMS and norms
remain available in scalar metrics and node labels for shape-aware debugging.

## Implementation Outline

1. Add reusable backward-flow context, marker, graph extraction, collapse, and HTML rendering in `levanter.analysis`
2. Add a minimal `BackwardFlowConfig` to Grug trainer config, defaulting base Grug to 50-step sampling
3. Mark Grug module outputs and add per-layer block scopes in the transformer loop
4. Sample the backward-flow path in `_make_train_step(...)` and emit HTML artifacts from the outer train loop
5. Test name normalization, activation/gradient metric capture, checkpointed graph extraction, graph collapse, residual-stream hint inference, and HTML rendering

## Notes

- This first pass relies on `jax._src.source_info_util.current_name_stack()` at
  `lib/levanter/src/levanter/analysis/backward_flow.py:17`. That is an internal API.
  The design keeps that dependency isolated in one file so it is easy to replace if
  JAX eventually exposes a public equivalent.
- The train-step contract stays intact. `_make_train_step(...)` still returns the same
  three top-level values that the Grug contract tests expect; backward-flow payloads are
  tucked under a reserved metrics key and peeled back out in the outer loop.
- The graph artifact is built lazily on the first sampled step because the structure is
  shape-dependent but not step-dependent.
- The current implementation is array-first. That matches Grug today. Generalizing to
  `NamedArray` or richer pytree outputs is future work, not part of the first pass.

## Future Work

- Move the sampling policy into generic trainer/watch infrastructure once the interface
  stabilizes
- Add richer node summaries (percentiles, sign skew, per-head reductions, update RMS)
- Support stacked/scanned models with better nested-JAXpr handling
- Add a tracker-native panel instead of HTML artifacts only
- Optionally emit Graphviz/DOT alongside the HTML for external tooling
