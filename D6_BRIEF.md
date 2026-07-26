# EP25-D6 — latent MoE on top of the EP25 stack

Worktree: `/home/marin/projects/marin/.worktrees/ep25-d6-latent`, branch `agent/ep25-d6-latent`,
base `agent/ep25-d1-adjoint` @ f53f781ce (custom scatter-add adjoint + gather dispatch + drops
metric + same-step spill + capacity knob).

## Why this direction

EP25 established that the step is **collective-volume-bound**: with collectives async, exposed
collective time (4.29 s) almost exactly fills compute idle (4.44 s of a 33.2 s span). The whole
scheduling family (rotation, prefetch, token-chunk pipelining, PGLE, overlap limit) measured null or
negative. **Reducing collective bytes is the only remaining lever.**

fp8 on the permutation legs attacked exactly that and validated the byte thesis — 936 ms/step of
exposure recovered — but still lost 1.6–2.3pp, because quantization compute cost more than the bytes
saved, and the amax attribution for that compute was falsified at 17x its own threshold.

**Latent MoE halves the dispatched activation width structurally, with no quantization compute at
all.** It is the fp8 thesis without the mechanism that killed fp8.

## Prior work you are building on (read it first)

Branch `research/mcwitt/7279-latent-moe`, logbook `.agents/logbooks/7279-latent-moe.md`, commits
`50fa034cd` (implementation) and `44844df1c` (results). Access it with `git show`, e.g.
`git show 50fa034cd -- experiments/grug/moe/standalone/grug_moe_mfu.py`.

That work implemented latent MoE in the **standalone** harness only
(`experiments/grug/moe/standalone/grug_moe_mfu.py`), at a commit predating every EP25 win. Measured
at d6144, 64 GPU, `ring_cute` EP4:

| arm | tok/s | arch-aware MFU |
|---|--:|--:|
| baseline 4-of-128 · i3072 | 220,648 | 19.686% |
| latent L3072 · I3072 · e256 (routed params preserved) | 256,059 | 19.401% |

+16.0% tok/s at preserved routed params, but arch-aware MFU 0.28pp *lower*, because analytic
FLOPs/token fell 15.1%. Three of four latent EP>1 arms died in the #7421 CUBIN loader bug, so latent
EP scaling is essentially unmeasured.

**Report both tok/s and arch-aware MFU on every leg, and state the FLOPs/token you used.** Latent
changes the denominator, so an MFU-only readout is not interpretable. This is the single most
important reporting requirement in this brief.

## The port

The standalone implementation is compact and transfers directly. In `MoEMLP.__call__`
(`experiments/grug/moe/model.py:465`), the expert call is:

```python
moe_out = self.expert_mlp(x_flat, selected_experts.astype(jnp.int32), combine_weights, ...)
```

Dispatch happens *inside* `expert_mlp`, so projecting `x_flat` from D to L before that call and back
from L to D after it halves the all-to-all payload automatically. That is the whole mechanism.

From `50fa034cd`, mirror: config fields `moe_latent_dim: int | None` / `moe_latent_norm: bool` with
their validation (positive, `< hidden_dim`, `hidden_dim % moe_latent_dim == 0`, norm requires dim);
down `(d, L)` and up `(L, d)` projections, replicated `P(None, None)`; optional `RMSNorm` on the
latent; `MoEExpertMlp.init(hidden_dim=<L>)`. The router and shared expert stay at D — do not project
the router input. QB routing, the drop metric, and spill all live outside the projected region and
should need no change; confirm that rather than assuming it.

Follow the repo's config conventions: this is a model-architecture parameter, so put it on
`GrugModelConfig` and thread it through `launch_cw_scale.py` the way `SCALE_INTERMEDIATE` is threaded,
rather than reading an env var inside the layer.

## Measurement plan

Hero shape and reference command: `submit_d5.sh` in `.worktrees/ep25-d5-d6144` and that worktree's
`AGENT_LOG.md`. The 24.594% reference leg is
`/mwittmann/ep25d5-d6144-e128-bf16-120-0726-1140-v3` — d6144 · 4-of-128 · 48L · EP64 · one GB200 rack
· batch 1024 · seq 4096 · QB-on · cf1.0 · host offload on · **default BFC allocator at the default
0.75 fraction** · sliding window 2048 · 120 steps.

1. **Correctness before throughput.** A small-config leg confirming latent trains and that drops,
   QB betas, and spill still behave. Loss curve sanity, not just "it ran".
2. **Param-preserving latent at the hero shape** — the arm that matters: latent L3072, e256, I3072,
   top-4, so routed params match 4-of-128 · i3072 while dispatch width halves. Same 120-step
   protocol as the reference leg so the comparison is like-for-like.
3. **Profile it.** The prediction is specific: exposed collective time should drop roughly in half.
   Confirm the mechanism, do not just report the endpoint. If tok/s improves and exposed collective
   time did *not* fall, the win came from somewhere else and the thesis is wrong.
4. **Drops.** `SCALE_REPORT_DROPS=1`, true tail statistics, never a single step. The compliance bar
   is <3%.

## Operating rules (unchanged from prior rounds)

- Keep an append-only `AGENT_LOG.md` in this worktree and **commit locally**. Do **not** push, and do
  **not** comment on or open GitHub issues or PRs — that is the coordinator's job.
- Do not stop, restart, or bounce an Iris cluster. Do not mutate jobs you did not submit.
- Memory-fraction inversion, learned the hard way: an **XLA** OOM says raise the fraction, an **NCCL**
  OOM says lower it. NCCL buffers live outside the XLA arena. Do not raise the fraction above the
  0.75 default to buy a couple of GiB — that is what starved NCCL at 0.90. Prefer
  `SCALE_OFFLOAD_OPT_STATE=1`.
- `iris job logs` truncates at 1000 lines by default, which biases any still-trending tail statistic
  in the direction of the trend. Pass `--max-lines`.
- A side effect inside a rematerialized scan body defeats remat: a bare `jax.debug.print` touching no
  tensor cost 1.41x compiled temp memory. Instrument through the metrics path or a small probe
  config, never with prints at 48 layers.
- Compare drop fractions only at the same fraction of the LR schedule, prefer a tail window over any
  single step, and state run length beside every drop figure. The LR schedule is defined over
  `num_train_steps`, so step N sits at a different schedule position in runs of different length.
- Operational failures are not scientific negatives. The #7421 CUBIN loader fault that killed the
  prior latent EP arms is intermittent and unrelated to latent — retry rather than concluding.
- Report findings-so-far and a confidence number when the coordinator asks. Flag falsification
  early; a clean negative on the byte thesis is a real result and worth reporting fast.
