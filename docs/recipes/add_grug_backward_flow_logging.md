# Recipe: Add Grug Backward Flow Logging

Use this when a Grug variant needs sampled backward activation logging and a DAG artifact
showing which module outputs are carrying vanishing or exploding backward-gradient scale.

The reusable machinery now lives in
`lib/levanter/src/levanter/analysis/backward_flow.py`. The Grug-specific work is just:

1. turn the sampled path on in the train loop
2. mark the module boundaries and residual stream points you care about
3. make repeated layers distinct with explicit scopes

## Canonical example

The base template is the reference implementation:

- `experiments/grug/base/model.py`
- `experiments/grug/base/train.py`

If you are modifying another variant, copy the same pattern into that variant instead of
adding a new shared Grug framework layer.

## 1) Add the config knob

Expose `BackwardFlowConfig` from the variant's trainer config and keep it disabled by
default:

```python
from levanter.analysis.backward_flow import BackwardFlowConfig

@dataclass(frozen=True)
class GrugTrainerConfig:
    ...
    backward_flow: BackwardFlowConfig = field(default_factory=BackwardFlowConfig)
```

`interval=0` disables the feature. Set a positive interval only when you actually want
to sample it.

## 2) Mark module outputs

At each module boundary you want in the graph, wrap the returned activation with
`log_backward_activation(..., site="out")`. For modules where you want to see what
backward is sending *into* the module, also mark the input with `site="in"`:

```python
from levanter.analysis.backward_flow import log_backward_activation

@named_call
def __call__(self, x):
    x = log_backward_activation(x, site="in")
    out = ...
    return log_backward_activation(out, site="out")
```

Good default patch points for a transformer-like Grug model:

- attention input/output
- MLP input/output
- residual stream points inside each block, such as `resid_in`, `resid_post_attn`,
  and `resid_out`
- embeddings / final hidden state

Do not blanket-wrap every tiny helper. Start at the module boundaries you would actually
want to read in a graph.

The reusable hint inference recognizes this residual naming convention and treats
repeated blocks as plates/containers. The residual stream nodes carry metrics and edges;
the block container only groups the attention, MLP, and residual stream nodes for
readability. For a custom architecture, start by naming the key stream anchors rather
than adding bespoke renderer code.

## 3) Give repeated layers stable names

`@named_call` gives you module names, but not layer indices from a Python loop. Add an
explicit scope around each repeated block:

```python
for i, block in enumerate(self.blocks):
    with jax.named_scope(f"block_{i}"):
        hidden = eqx.filter_checkpoint(block)(hidden, mask)
```

Without this, all blocks collapse into the same node in the DAG.

## 4) Sample the backward-flow path in the train step

When the interval fires, run the gradient computation inside both
`capture_backward_flow(...)` and `levanter.tracker.defer_tracker_for_jit()`. For
mean-reduced LM losses, pass `sum(batch.loss_weight)` as the gradient scale so the
renderer colors gradients on the unreduced loss scale:

```python
if backward_flow_config is not None and compute_backward_flow:
    gradient_scale = jnp.sum(jnp.asarray(batch.loss_weight, dtype=jnp.float32))
    with capture_backward_flow(backward_flow_config, gradient_scale=gradient_scale):
        with levanter.tracker.defer_tracker_for_jit() as backward_flow_stats:
            loss, grads = jax.value_and_grad(loss_fn)(state.params)
else:
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
```

Keep `compute_backward_flow` as a static JIT argument. That gives you a separate sampled
compile path and keeps the normal training path untouched.

## 5) Log scalars and write the artifact outside JIT

Use the returned metric dict to:

- log `backward_flow/<scope>/{in,out}_{activation,gradient}_{norm,rms,mean_abs,max_abs,finite_fraction}`
  plus `{in,out}_gradient_rms_scaled` when a gradient scale is configured
- build a graph once from a traced grad JAXpr
- call `infer_backward_flow_render_hints(...)` before rendering so residual-stream
  anchors become a readable display graph
- write the HTML artifact for the sampled step
- log the same HTML to `backward_flow/dag` with `tracker.log_html(...)` so W&B renders
  it as an inline media panel

Read scaled gradient RMS first when it exists. Raw gradient norms are still useful for
shape-aware debugging, but raw gradient RMS from a mean-reduced loss is dominated by the
loss denominator.

The base template writes artifacts to:

`<trainer.log_dir>/<run_id>/artifacts/backward_flow/step_<step>.html`

## Notes

- The graph is only as good as the names you provide. If a custom kernel or fused helper
  hides too much of the structure, add a slightly larger surrounding scope instead of
  trying to instrument every primitive.
- This is intentionally sampled. If you run it every step, the cost and artifact volume
  stop being reasonable.
- If a variant has different key module boundaries than base, patch those locally in the
  variant rather than extending the reusable logging core.
