# AGENT_LOG — ep25-d6 (latent MoE on top of the EP25 stack)

Worktree: `/home/marin/projects/marin/.worktrees/ep25-d6-latent`, branch `agent/ep25-d6-latent`,
base `agent/ep25-d1-adjoint` @ f53f781ce (custom scatter-add adjoint + gather dispatch + drops
metric + same-step spill + capacity knob). d1's own log is preserved in its own worktree; this file
is ep25-d6's log from here down.

Mission: port latent MoE from the standalone harness into the grug training path, then measure it at
the hero shape (d6144, EP64, one GB200 rack) against d5's 24.594% reference leg
(`/mwittmann/ep25d5-d6144-e128-bf16-120-0726-1140-v3`).

Thesis under test: EP25 established the step is collective-volume-bound and that reducing collective
BYTES is the only remaining lever. fp8 validated the byte thesis but lost because quantization
compute cost more than the bytes saved. Latent MoE halves the dispatched activation width
structurally, with no quantization compute — the byte thesis without the mechanism that killed fp8.

## Check-in 1 — port landed, and the arithmetic of the comparison is fixed before any GPU time

Commit a6fa47f6b. The port is exactly the standalone mechanism (`50fa034cd`) moved into
`experiments/grug/moe/model.py`: `moe_latent_dim` / `moe_latent_norm` on `GrugModelConfig` with the
same validation, replicated `[D, L]` down and `[L, D]` up projections on `MoEMLP`, an optional
`RMSNorm` on the latent, and `MoEExpertMlp.init(hidden_dim=L)`. Dispatch happens INSIDE
`expert_mlp`, so projecting before that call is the whole mechanism.

CONFIRMED RATHER THAN ASSUMED, as the brief asked: the router, QB balancing (`_compute_qb_beta`),
the drop metric and spill all read `x_flat` / `router_logits` at hidden width, upstream of the
projection, and needed no change. The launcher threads `SCALE_MOE_LATENT_DIM` /
`SCALE_MOE_LATENT_NORM` through `build_scale_model` the way `SCALE_INTERMEDIATE` is threaded, not
through an env read inside the layer. 8 unit tests pass on CPU, including a real MoE forward.

### The pair, resolved through `build_scale_model()` and the SHIPPING `_compute_flops`

| | reference (d5) | latent arm |
|---|---|---|
| experts x top-k | 128 x 4 | 256 x 4 |
| expert MLP width | 6144 | **3072** (latent) |
| intermediate | 3072 | 3072 |
| routed params | 347.892 B | **347.892 B** (exactly preserved) |
| latent projections | — | 1.812 B |
| total params | 359.472 B | 361.321 B |
| active params excl. embed | 20.875 B | 17.289 B |
| **analytic FLOPs/token** | **48.186 G** | **41.014 G** (-14.9%) |
| per-(sender,expert) bucket mean at cf1.0 | 2048 | **1024** |

Routed-parameter preservation is exact, not approximate: `128 x 3 x 6144 x 3072 == 256 x 3 x 3072 x 3072`.

VALIDATION THAT THE DENOMINATOR IS THE RIGHT ONE: the reference leg's own log line reports
`total_gflops 72,758,818,573.66` over `total_tokens 503,316,480` = 144.56 GFLOP/token = 3 x 48.186 G,
which is my computed dense figure to five significant figures. So my latent denominator is produced
by the same code path that produced the number I am comparing against.

### PRE-REGISTERED breakeven, stated before the leg runs

MFU = tok/s x 3 x FLOPs/token / (64 x 2.5e15). At the reference p50 of 24.594% the reference tok/s is
272,166. For the latent arm to MATCH 24.594% arch-aware MFU it needs **319,779 tok/s, i.e. +17.5%
tok/s**. Anything less is a tok/s win and an MFU loss, which is precisely why an MFU-only readout
would be uninterpretable here.

Reference points for that +17.5% bar:
- The prior standalone latent result measured **+16.0% tok/s at -0.28pp MFU** — just under breakeven.
- But that was measured at **EP4**, where the all-to-all spans 4 GPUs. This leg is **EP64**, where
  the collective is 16x wider and where EP25 measured the exposure that motivates the whole
  direction. If the byte thesis is right, the tok/s gain should EXCEED the standalone +16%.
- That is the falsifiable claim: **latent's tok/s advantage should grow with EP width.** If it lands
  at or below +16% at EP64, the mechanism is not collective bytes and the thesis is wrong.

### Memory, projected from d5's measured decomposition

d5 measured the e128 temp arena at 90.64 GiB (fp32 grad accumulators 20.25 + bf16 residual stack
36.0 + working set ~34) with a ~49 GiB resident set, needing host offload to fit at the default 0.75
fraction. For the latent arm: expert params and their gradients are unchanged (4 local experts of
3 x 3072 x 3072 = the same 20.25 GiB as 2 local experts of 3 x 6144 x 3072), the residual stack is
token-scaled and unchanged at 36.0, and every MoE dispatch buffer halves in width. The a2a send
buffer is `assignments_per_shard x width`, and `assignments_per_shard = 262,144` regardless of expert
count, so it goes 3.2 GiB -> 1.6 GiB. Latent should therefore fit wherever the reference fits.
Same memory configuration as the reference leg: host offload on, default BFC allocator, default 0.75
fraction, no fraction bump (that is the knob that starved NCCL at 0.90).

Jobs in flight, both mine, EP4 4-GPU 4-layer 40-step smokes that mirror the rack pair's routing
regime (the latent arm doubles experts, so its bucket mean halves exactly as it does at the rack):
- `/mwittmann/ep25d6-smoke-latent-0726-1355`
- `/mwittmann/ep25d6-smoke-dense-0726-1356`

Confidence: 9/10 that the port is correct (unit-tested, and the FLOPs path cross-validates against
the reference leg's own emitted number); 5/10 that latent clears the +17.5% arch-aware-MFU breakeven
at EP64; ~8/10 that it clears the standalone's +16% tok/s.
