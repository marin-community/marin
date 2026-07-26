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

## Check-in 2 — brief amendment folded in: the PRIMARY arm is now matched-work, and the amendment's own criterion needs one correction

Coordinator amendment received and written into `D6_BRIEF.md`. It is right, and it changes the
headline: the prior latent result's +16.0% tok/s is fully accounted for by a 15.2% work cut, so it
never tested the wire at all. Timing note for the record: I had already submitted the two EP4
4-GPU correctness smokes before it arrived; no rack time had been spent, so nothing was wasted.

### The three arms, all resolved through `build_scale_model()` and the shipping `_compute_flops`

| arm | E | I | latent | FLOPs/token | routed params | a2a send buffer | bucket mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| **dense** (the reference) | 128 | 3072 | — | **48.186 G** | 347.892 B | 3.000 GiB | 2048 |
| **matched-work** (PRIMARY) | 128 | 6144 | 3072 | **51.810 G** (+7.52%) | 347.892 B | **1.500 GiB** | 2048 |
| **param-preserving** (secondary) | 256 | 3072 | 3072 | **41.014 G** (-14.9%) | 347.892 B | **1.500 GiB** | 1024 |

The matched-work arm holds THREE things fixed that the e256 arm does not: routed parameters, routed
expert FLOPs (`3*L*I` per expert unchanged), and the per-(sender,expert) bucket mean. That third one
matters more than it looks: this session established that heavy-drop runs read HIGHER MFU, because
dropped assignments gather a zero pad row. The e256 arm halves the bucket mean, so it changes the
drop regime at the same time as the wire. The matched-work arm does not. It is the right primary.

### CORRECTION to the amendment's falsification criterion: "higher MFU" is satisfied by the NULL too

The amendment says a real wire win must show up as higher tok/s AND higher arch-aware MFU. The
second half does not discriminate on this arm, and the reason is in the table above: the
matched-work arm is matched on *routed* work but the two latent projections ADD work —
`4*d*L` per layer, i.e. **+7.52% analytic FLOPs/token**, not zero. Those projections are large,
well-shaped dense GEMMs that run well above the step-average efficiency, so they raise the analytic
numerator faster than they raise step time, and MFU rises even if the wire buys nothing.

Quantified, so it can be checked rather than asserted. Added projection work per token is
`3 x 4 x 6144 x 3072 x 48 = 1.0872e10` FLOP, i.e. `7.126e14` FLOP per device per step at 65,536
tokens/device. At an achieved efficiency `e` of the 2.5e15 per-GPU peak that costs `0.285/e` seconds.
Take the reference step as `T = 15.408 s` (see the discrepancy note below) and EP25's measured 12.9%
exposed-collective share, so `X_ref = 1.99 s` and compute `C_ref = 13.42 s`:

| | e = 0.246 (step average, pessimistic) | e = 0.50 (realistic for these GEMMs) |
|---|---|---|
| added projection time | 1.158 s | 0.570 s |
| **NULL** (wire buys nothing) | tok/s **-7.0%**, MFU **+0.00pp** | tok/s **-3.6%**, MFU **+0.91pp** |
| **THESIS** (exposed collective halves) | tok/s **-1.0%**, MFU **+1.58pp** | tok/s **+2.8%**, MFU **+2.60pp** |

So the null already predicts MFU flat-to-up-0.9pp. The discriminating readouts are:
1. **tok/s**, which flips sign between the hypotheses (null -3.6..-7.0%, thesis -1.0..+2.8%), and
2. **the profile**, which measures both terms directly rather than assuming either.

PRE-REGISTERED, before any rack leg: I will read the verdict off the profile pair, and use tok/s as
the endpoint check. Exposed collective time on the matched-work arm should fall to roughly half the
dense arm's. If collective bytes are confirmed halved (they are, by construction: 3.000 -> 1.500 GiB
of send buffer) and exposed collective time does NOT fall, the thesis is falsified regardless of
what MFU does, and I will report that as the result.

### A discrepancy in the reference leg's own numbers, flagged rather than papered over

d5's steady-tail row reads `p50 MFU 24.594% | 276,413 tok/s | 15.174 s/step`. Those three cannot all
be true: MFU is exactly `tokens_per_step x 3 x FLOPs_per_token / (duration x peak)`, so 15.174 s
implies 276,414 tok/s and **24.97%**, while 24.594% implies 272,210 tok/s and 15.408 s/step. The
step-119 log line is internally consistent (270,378 tok/s / 15.513 s / 24.428%), so the arithmetic is
sound and it is the summary row that mixes windows. I use `24.594% <-> 272,210 tok/s <-> 15.408 s`
above, and I am running my own dense leg so the comparison rests on a matched pair I measured rather
than on that row.

### Named confounds on the matched-work arm

- `I` doubles from 3072 to 6144, so the expert GEMM changes shape ([rows, 3072] x [3072, 12288]
  instead of [rows, 6144] x [6144, 6144]) at identical FLOPs. Matched *work* is not automatically
  matched *achieved efficiency*; the profile's per-kernel occupancy will show whether it is.
- The expert intermediate buffer scales with `2I`, so it doubles (3.0 -> 6.0 GiB) while the two a2a
  buffers halve (3.0 -> 1.5 GiB each). Net memory should be near-neutral, but the prior work OOM'd
  this arm at a 181.34 GiB plan on the standalone harness at replica axis 1 with no offload. Here it
  gets host offload and the default 0.75 fraction. If it still does not fit I will record the plan
  number rather than substitute the e256 arm.
- Quality is not measured. Every routed token now passes a rank-3072 bottleneck shared across all
  experts and each expert sees half the input width. 120 steps cannot settle that.

Jobs: added `smoke-matched` / `rack-matched` to `submit_d6.sh` and submitted
`/mwittmann/ep25d6-smoke-matched-0726-1425` (EP4, 64 experts, I6144, L3072 — routed work, routed
params and bucket mean all identical to `smoke-dense`).

Confidence: 9/10 on the port; 9/10 that the amendment's MFU criterion needs the correction above
(it is arithmetic, not judgement); 4/10 that the matched-work arm shows a real wire win at EP64.

## Check-in 3 — all three smokes clean, and the EP4 smoke pair turns out to be a WIRE-NULL CONTROL

All three EP4 4-GPU 4-layer 40-step smokes completed 40/40 with descending loss. Deliverable 1
(correctness) is answered: latent trains, drops and QB behave, and the arms are configured as
intended (hparams confirm `intermediate_dim 6144 / moe_latent_dim 3072 / num_experts 64` on the
matched arm). Window steps 10-39.

| smoke arm | GFLOP/token | MFU p50 | tok/s p50 | step p50 | drops @39 | loss @39 |
|---|--:|--:|--:|--:|--:|--:|
| dense (64 x i3072) | 16.371 | 9.215 | 56,286 | 1.1646 s | 0.148 | 6.732 |
| matched-work (64 x i6144, L3072) | 17.277 | **10.098** | **58,448** | 1.1221 s | 0.146 | 6.786 |
| param-preserving (128 x i3072, L3072) | 14.569 | 8.486 | 58,246 | 1.1262 s | 0.179 | 6.790 |

Two things fall out, one expected and one useful.

EXPECTED: the drop series confirm the regime claim. Matched-work tracks dense almost exactly
(0.146 vs 0.148 at step 39, and the whole series within ~0.02), while the param-preserving arm runs
consistently heavier (0.179) because its bucket mean halved. So comparisons against the
param-preserving arm are cross-drop-regime and comparisons against the matched-work arm are not.

USEFUL, AND IT CHANGES HOW I WILL READ THE RACK LEG: the matched-work smoke is +3.84% tok/s AND
+0.88pp MFU over dense **at EP4, where the all-to-all is intra-node NVLink and the wire is nearly
irrelevant**. It did 5.54% more analytic work in 3.65% less time — a ~9.5% swing in work-per-second
that the wire cannot explain at this scale. The available explanations are that the two latent
projections and the reshaped expert GEMM ([rows, 3072] x [3072, 12288] instead of
[rows, 6144] x [6144, 6144]) simply run at higher efficiency than the kernels they replace.

That makes the EP4 pair an empirical handle on exactly the confound my check-in-2 model had to
parameterize by hand as "efficiency e". **The EP4 smoke is a wire-null control.** The rack test
becomes a difference-in-differences rather than a single comparison:

    wire contribution ~= (matched-work advantage at EP64) - (matched-work advantage at EP4)

If the rack leg also lands near +3.8% tok/s, the wire added nothing and the thesis is falsified
despite a positive-looking headline. If it lands materially above, the excess is the wire.
Caveat, and it is a real one: at 4 layers the lm_head and embedding are a much larger share of the
step than at 48, and the batch is 64x smaller, so the EP4 number is a qualitative control on the
sign and rough size of the efficiency effect, not a quantitative subtraction. The profile remains
the decisive measurement; this control is what keeps the endpoint number honest if the profile is
ambiguous.

Rack legs in flight, both mine, both with a 3-step profiler window at step 20 (excluded from the
90-119 steady tail, so the headline is uncontaminated) so each yields both a throughput series and
an xspace for the mechanism comparison:
- `/mwittmann/ep25d6-d6144-e128-dense-120-0726-1440` — matched control, byte-for-byte d5's
  reference configuration plus the profiler window.
- `/mwittmann/ep25d6-d6144-e128-matched-i6144-L3072-120-0726-1455` — PRIMARY arm.

Also carried d4's `xplane_overlap.py` / `xplane_op_detail.py` onto this branch; they read the xprof
dump straight from S3 and run as an iris CPU job, which is how the exposed-collective number gets
measured without S3 credentials in this sandbox.

Confidence: 9/10 the port is correct (three clean smokes); 3/10 that the wire shows a measurable
win at EP64 once the EP4 efficiency control is subtracted — down from 4/10, because the smoke shows
the efficiency confound is real and comparable in size to the wire effect I am looking for.
