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

## Check-in 4 — the dense baseline's mechanism number, MEASURED: 4.13 s exposed collective per 3 steps

`/mwittmann/ep25d6-d6144-e128-dense-120-0726-1440` is the matched control — d5's reference
configuration plus a 3-step profiler window at step 20 (steady tail 90-119 is untouched). hparams
confirm the shape (`intermediate_dim 3072 / moe_latent_dim null / num_experts 128 / 48 layers`) and
the emitted denominator is 144.5588 GFLOP/token = 3 x 48.186 G, i.e. exactly what check-in 1
predicted. Early window (steps 0-29, drop-heavy): MFU p50 27.150%, 300,496 tok/s, 13.964 s/step —
which reproduces d5's early-window 27.04% and is the first cross-session confirmation that this
configuration is stable.

Ran d4's `xplane_overlap` against the uploaded xprof dump as an iris CPU job (the sandbox has no S3
credentials). Steps 20-22, one host, all four GPUs:

| GPU | trace span | collective total | concurrent | **exposed** | exposed % of span |
|---|--:|--:|--:|--:|--:|
| 0 | 42,198 ms | 12,287 | 8,161 | **4,126** | 9.8% |
| 1 | 42,198 | 12,005 | 7,621 | **4,384** | 10.4% |
| 2 | 42,198 | 12,980 | 8,152 | **4,828** | 11.4% |
| 3 | 42,198 | 12,543 | 7,788 | **4,755** | 11.3% |

Compute stream busy 38,700 ms of the 42,198 ms span on GPU:0 = 91.7%, so 3,498 ms idle against
4,126 ms exposed collective. **Exposed collective time again almost exactly fills compute idle**,
which is the EP25 finding reproduced at the hero shape rather than at the d5120 proxy — the premise
of this whole direction now has a measurement at the shape it is being applied to.

THE NUMBER THE PRIMARY ARM HAS TO MOVE: 4.13-4.83 ms per GPU per 3 steps, i.e. **~1.4-1.6 s of every
14.1 s step**. Halving the wire should take roughly 0.7-0.8 s per step out of that if the thesis is
right. Note this is a smaller share than my check-in-2 model assumed (I used EP25's 12.9% from the
proxy; the hero shape measures 9.8-11.4%), so the thesis-side predictions in that table are
optimistic by about 20% and I am re-deriving them against this measurement rather than the proxy's:

    predicted saving if exposed collective halves: ~0.69 s/step (GPU:0) to ~0.80 s/step (GPU:2)
    added latent-projection cost at e=0.5:         ~0.57 s/step
    => the two nearly CANCEL. The matched-work arm's expected tok/s change is within a couple of
       percent of zero EITHER WAY, and MFU rises ~+1 to +2pp mostly because the denominator moved.

That is a sharper statement of check-in 2's correction and it makes the profile, not the endpoint,
the deciding evidence. I will compare exposed collective time directly between the two profiles.

Legs in flight: matched-work (`...-matched-i6144-L3072-120-0726-1455`, restarted once after 11
`SIGTERM caught` = a preemption per d5's triage class 1, not a code fault) and the secondary
param-preserving arm (`...-e256-latent3072-120-0726-1530`), both with the same profiler window.

## Check-in 5 — an implementation cost the standalone work never accounted for: the projections are REPLICATED

Noticed while waiting on the rack legs, and it belongs in the writeup because it is a property of the
mechanism, not of my port. Following `50fa034cd`, the latent projections are initialised with
`P(None, None)` — fully replicated, like the router. At the hero shape that is
`48 layers x 2 x 6144 x 3072 = 1.812 B` parameters **on every GPU**:

    fp32 params    6.75 GiB resident per GPU
    MuonH momentum 6.75 GiB (host, since SCALE_OFFLOAD_OPT_STATE=1 is on)
    fp32 gradient  6.75 GiB in the temp arena

against a reference resident set of ~29 GiB with offload. Everything else in this model is sharded:
expert weights over `expert`, attention over `Pfsdp = ("data", "expert")`. So latent MoE buys a
halved wire and pays for it partly in replicated state that grows with `num_layers x d x L`.

This was invisible in the standalone harness because nothing there was memory-limited, and it is a
plausible contributor to the 181.34 GiB plan that OOM'd the matched-work arm in the prior work. It is
also FIXABLE and I am deliberately not fixing it in this leg, so the measurement stays a faithful
port: sharding the projections over the expert axis the way the attention projections are sharded
would trade the replication for one small per-layer all-gather, and sharing a single projection pair
across all layers would remove it entirely at some quality cost. Both are follow-ups, and the second
is interesting on its own terms because a shared bottleneck across layers is a different architecture.

I am recording it now, before the results, so it cannot be read as a post-hoc explanation of whatever
the legs return.

## Check-in 6 — a cluster-wide instability window at 21:28-21:40Z hit both later legs; classification says it is not my code

Triaged with d5's recipe. Neither latent leg has reached step 0 yet; both were killed mid-compile.

| leg | event | class |
|---|---|---|
| matched-work | 21:10Z, 11 x `SIGTERM caught` | 1 — preemption |
| matched-work | 21:28Z, 9 x `another task died`, no primary | 1/2 — gang abort |
| matched-work | 21:40Z, 26 x `another task died`, 2 x SIGTERM (tasks 0 and 15) | 1 — leader evicted |
| param-preserving | 21:29Z, 14 x `another task died` | 1/2 — gang abort |

Zero hits on `failed to allocate`, `RESOURCE_EXHAUSTED`, `ncclAlltoAll` and `Cuda failure` across all
of them, so this is NOT a memory result and specifically NOT the OOM the prior work saw on the
matched-work arm. The important observation is the CLUSTERING: two independent jobs with different
configurations died inside a twelve-minute window, while the dense leg — which had already cleared
compile before that window — ran through it untouched. That is a cluster event, not a property of
either latent configuration. Per the standing policy, operational friction closes nothing.

Both jobs are auto-restarting. What makes this more than routine friction is the interaction with
compile time: a d6144/48L compile is ~20 minutes on preemptible workers, so an eviction costs the
whole compile and a run can be evicted repeatedly without ever reaching step 0 — the matched-work
leg has now spent three compiles and produced no steps. Added `SCALE_PREEMPTIBLE` to the launcher
(default unchanged) so the non-preemptible pool is one env away if the retries keep losing; I have
not used it yet, because switching pools changes placement and I would rather not add a variable to
the primary arm while a plain retry might land.

## Check-in 7 — the DENSE CONTROL COMPLETED and reproduces d5's reference leg cleanly

`/mwittmann/ep25d6-d6144-e128-dense-120-0726-1440` — SUCCEEDED, 120/120 steps.
d6144, 128 experts, top-4, 48 layers, EP64 on one GB200 rack, batch 1024, seq 4096, QB-on, cf1.0,
custom adjoint + gather dispatch, drops on, host offload on, default BFC allocator at the default
0.75 fraction, sliding window 2048, plus a 3-step profiler window at step 20.

**Analytic FLOPs/token: 48.186 G (emitted 144.5588 GFLOP/token = 3 x that).**

| window | MFU p10 | **MFU p50** | MFU p90 | sd | tok/s p50 | step p50 |
|---|--:|--:|--:|--:|--:|--:|
| steady tail, 90-119 | 24.719 | **24.842** | 24.978 | 0.103 | 274,954 | 15.266 s |
| early, 10-24 | 26.891 | 27.177 | 27.308 | 0.156 | 300,803 | 13.944 s |

Drops: 0.169@0 -> 0.870@12 -> 0.590@36 -> 0.296@48 -> 0.150@96 -> **0.091@119** (120-step run, so
step 119 is end-of-anneal). Loss 10.31@2 -> **5.654@119**, descending throughout.

Against d5's reference leg (24.594% / 276,413 tok/s / 15.174 s / drops 0.089@119 / loss 5.59) this
is a reproduction to within +0.25pp MFU and 0.002 on the drop fraction, from a different session on
a different rack allocation. Two consequences:
1. The EP25 hero-shape result replicates. That was not guaranteed and it is worth stating on its own.
2. My control is INTERNALLY CONSISTENT where d5's summary row was not: 274,954 tok/s x 144.5588e9 /
   1.6e17 = 24.84%, exactly the reported p50. Every comparison below is against my own leg, so the
   check-in-2 discrepancy does not propagate into any conclusion.

MECHANISM BASELINE, from the profiler window (steps 20-22, one host, four GPUs):
exposed collective **4,126 / 4,384 / 4,828 / 4,755 ms** per 3 steps = **1.38-1.61 s of every step**,
against compute-stream idle of 3,498 ms per 3 steps on GPU:0. Collective total 12.0-13.0 s per 3
steps, of which ~62-68% is already concurrent with non-collective work.

### The primary arm lost three compiles to the cluster; resubmitted

`...-matched-i6144-L3072-120-0726-1455` exhausted its retries without reaching step 0 (21:10
preemption, 21:28 and 21:40 gang aborts — the 21:28-21:40 window that also hit the param-preserving
leg). The `Metadata mismatch` line in its failure text is a benign `levanter.store.cache` warning
present in healthy runs too, not the cause. Resubmitted unchanged as
`/mwittmann/ep25d6-d6144-e128-matched-i6144-L3072-120-0726-1600-v2`. Deliberately still on the
default preemptible pool: the dense leg completed there, so the evidence says the window was
transient, and I would rather not introduce a placement variable into the primary arm.

## Check-in 8 — DECOMPOSING the exposed collective time, and it lowers the ceiling on what latent can win

The headline exposure number (4,126 ms per 3 steps on GPU:0) is not all a2a. Breaking it down by
kernel from the same profile — occupancy x (1 - overlap%) per op, which sums to 4,078 ms against the
union-measured 4,126 ms, so the decomposition is essentially complete:

| collective | stream | occupancy | overlap | **exposed** | what latent does to it |
|---|---|--:|--:|--:|---|
| `SendRecv` (the expert a2a) | #159 async | 5,620 ms | 85.1% | **838 ms** | halves the bytes |
| `SendRecv` (the expert a2a) | #50 **inline on compute** | 1,266 ms | **0.0%** | **1,266 ms** | halves the bytes |
| `AllGather_RING_LL` (FSDP weights) | #159 | 1,930 | 70.7% | 566 | nothing |
| `ReduceScatter_Sum_bf16` (FSDP grads) | #159 | 1,668 | 56.3% | 729 | nothing |
| `AllReduce_Sum_f32` (QB beta + norms) | #159 | 1,596 | 57.4% | 680 | nothing |

**Only 2,104 ms of the 4,126 ms of exposure — 51% — is the expert all-to-all at all.** The other
half is FSDP weight/gradient movement and the QB all-reduce, and latent MoE does not touch a byte of
it. So the ceiling on this mechanism is HALF of half:

    best case if a2a exposure scales exactly with payload bytes:
      saving = 2,104 ms / 2 = 1,052 ms per 3 steps = 351 ms per step
    against an added latent-projection cost of ~285 ms/step (e = 1.0) to ~570 ms/step (e = 0.5)

REVISED PRE-REGISTRATION for the matched-work arm, and it is now a prediction of a small LOSS on
tok/s rather than the coin-flip of check-in 2. Taking the dense control's own steady-tail step of
15.266 s:

| scenario | Δstep | tok/s | arch-aware MFU |
|---|--:|--:|--:|
| a2a exposure halves, projections at e = 0.5 | +219 ms | **-1.4%** | 26.34% (+1.50pp) |
| a2a exposure halves, projections at e = 1.0 | -66 ms | **+0.4%** | 26.79% (+1.95pp) |
| a2a exposure unchanged (pure null), e = 0.5 | +570 ms | **-3.6%** | 25.75% (+0.91pp) |

The three scenarios are separated by only ~4% in tok/s and ~1pp in MFU, which is more than the
run-to-run spread (my dense tail sd is 0.103pp) but not by a lot. **This is why the endpoint number
cannot carry the verdict and the profile has to.** The clean test is the direct one: measure
`SendRecv` exposure on the matched-work arm's profile and compare it to 2,104 ms. If it lands near
1,050 ms the mechanism works and the economics are simply unfavourable; if it lands near 2,104 ms
the mechanism itself does not transfer.

INDEPENDENT FINDING, worth more than the latent question and available to any arm: **1,266 ms per 3
steps of `SendRecv` runs INLINE on the compute stream at 0.0% overlap** — 422 ms of every 15.3 s
step, 2.8% of the step, completely serialized. That is 31% of all exposed collective time in one
kernel placement. Every other collective on this stack is on the async stream #159 and is 56-85%
hidden. Whatever forces that copy of the a2a onto the compute stream is worth a look on its own; it
is a scheduling fix rather than a bytes fix, and this effort has established that scheduling fixes
are the cheap kind. I am not chasing it in this direction, but it should not be lost.

### Attribution of the inline a2a (xplane_op_detail, dense control, GPU:0, steps 20-22)

Every `SendRecv` event on this stack is a MoE dispatch/combine all-to-all — 12 distinct
`all_to_all.N.1` HLO ops, 144 events each (48 layers x 3 steps), all under
`Block/MoEMLP/MoEExpertMlp/moe_mlp/shard_map/{dispatch,combine}/all_to_all`. So the whole 6,886 ms of
`SendRecv` occupancy per 3 steps is payload latent halves, with nothing else mixed in.

Exactly THREE of the twelve are scheduled on the compute stream instead of the async collective
stream, and those three are the fully-serialized 1,266 ms:

    Stream #50 (compute)  all_to_all.40.1  bwd dispatch  439.18 ms
    Stream #50 (compute)  all_to_all.46.1  bwd combine   423.72 ms
    Stream #50 (compute)  all_to_all.56.1  fwd dispatch  402.75 ms

The other nine sit on stream #159 and are 85% hidden. Consistent across all four GPUs to within 1%,
so it is a scheduling decision, not jitter. This sharpens the independent finding in check-in 8: the
question is not "why is the a2a exposed" but "why did XLA put three of the twelve a2a instances
inline". Fixing those three placements would recover ~422 ms/step — comparable to the entire
best-case win from halving the payload — for no bytes and no arithmetic.

## Check-in 9 — the latent legs keep dying where the dense leg did not, and there is a concrete mechanism

Tally so far on the rack: dense 1 submission, 1 success. Latent 3 submissions, 0 successes, 6 deaths.
That asymmetry is now large enough that "cluster transient" is no longer the leading explanation.

Classification of every latent death (d5's recipe, full warning streams):

    matched-work v1: 21:10 (11 x SIGTERM = preemption) | 21:28 gang | 21:40 gang (+2 SIGTERM on
                     tasks 0 and 15, i.e. the leader) -> retries exhausted, job failed
    param-preserving: 21:29 gang | 21:59 gang | 22:03 gang
    matched-work v2: 21:48 gang, then quiet (compiling)

`failed to allocate` = 0, `RESOURCE_EXHAUSTED` = 0, `ncclAlltoAll` = 0, `Cuda failure` = 0 in ALL of
them, and at 21:59 all sixteen tasks logged "another task died" — so no task is the primary, which
means the primary left no log at all. Per d5's triage that is class 2: a kernel SIGKILL, i.e. what a
container host-memory OOM looks like from inside the container.

THE MECHANISM, and it follows directly from check-in 5's finding rather than being invented to fit:
the latent arms carry 1.812 B REPLICATED projection parameters per GPU, and with
`SCALE_OFFLOAD_OPT_STATE=1` their MuonH momentum is parked in PINNED HOST memory alongside the
expert momentum. Per GPU that is ~20.25 GiB (expert momentum) + ~6.75 GiB (projection momentum)
versus the dense arm's ~20.25 GiB alone. With `SCALE_PROCESSES_PER_TASK=1` one container holds all
four GPUs' worth:

    dense  arm: ~81 GiB pinned per node
    latent arms: ~108 GiB pinned per node          against the launcher's ram="256g" default

plus JAX host allocations, the loader prefetch and the process itself. A gb200-4x node has 960 GB, so
the container is capped far below the hardware. The timing fits too: the 21:59 death landed ~30
minutes in, at the compile-to-execution transition, which is exactly when the optimizer state is
first materialised on the host.

TEST, deliberately set up as an A/B rather than a blanket change:
- `/mwittmann/ep25d6-d6144-e256-latent3072-120-0726-1615-v2` — resubmitted with **SCALE_RAM=600g**
  (the knob added to the launcher this session; nothing else changed). Stopped the crashlooping
  0726-1530 first rather than let it keep taking the rack.
- `/mwittmann/ep25d6-d6144-e128-matched-i6144-L3072-120-0726-1600-v2` — LEFT ON THE 256g DEFAULT and
  still compiling. If it dies at its own compile-to-execution transition while the 600g leg lives,
  that is the hypothesis confirmed on a matched pair rather than on a single-arm change.

Recording d5's caution alongside: they raised 256g -> 600g on a different arm and it changed nothing,
which is why their triage recipe words class 2 as "suspect, then test". This is the test.

---

# STANDALONE RESULTS — ep25-d6, written to survive without the narrative above

Question: does latent MoE — projecting the MoE input from `d` to `L` before dispatch and back after
combine, so the expert-parallel all-to-all carries half-width rows — win at the hero shape (d6144,
top-4, 48 layers, EP64, one GB200 rack) on top of the EP25 stack? The thesis being tested is that the
step is collective-BYTES-bound, which fp8 validated but could not exploit.

## R1. The port, and the arithmetic every reader needs

Latent MoE is now a first-class option in the grug training path (`moe_latent_dim` /
`moe_latent_norm` on `GrugModelConfig`, `SCALE_MOE_LATENT_DIM` / `SCALE_MOE_LATENT_NORM` through
`launch_cw_scale`). The router, QB balancing, the drop metric and spill all sit upstream of the
projection and needed no change — verified, not assumed. `_compute_flops` replaces the routed-expert
term with the latent-width GLU plus the two projections, so **arch-aware MFU is correct on both arms**.

ANALYTIC FLOPs/TOKEN, the number that makes MFU interpretable, resolved through the shipping code:

| arm | E | I | latent | **FLOPs/token** | routed params | a2a send buffer | bucket mean |
|---|--:|--:|--:|--:|--:|--:|--:|
| dense (reference) | 128 | 3072 | — | **48.186 G** | 347.892 B | 3.000 GiB | 2048 |
| matched-work (primary) | 128 | 6144 | 3072 | **51.810 G** (+7.52%) | 347.892 B | 1.500 GiB | 2048 |
| param-preserving (secondary) | 256 | 3072 | 3072 | **41.014 G** (-14.9%) | 347.892 B | 1.500 GiB | 1024 |

Latent moves the denominator in BOTH directions depending on how the arm is constructed, which is
why tok/s and MFU must always be reported together. The matched-work arm is the honest wire test: it
holds routed parameters, routed expert FLOPs and the drop regime fixed and changes only the wire —
but it is NOT work-neutral, because the two projections add 7.52% analytic FLOPs/token.

## R2. The dense control, and a clean reproduction of the EP25 hero-shape result

`/mwittmann/ep25d6-d6144-e128-dense-120-0726-1440`, 120/120 steps, 48.186 GFLOP/token:

    steady tail 90-119: MFU p50 24.842% (p10 24.719 / p90 24.978, sd 0.103), 274,954 tok/s, 15.266 s
    early window 10-24: MFU p50 27.177%, 300,803 tok/s, 13.944 s
    drops 0.169@0 -> 0.870@12 -> 0.296@48 -> 0.091@119 (120-step run: step 119 is end-of-anneal)
    loss 10.31@2 -> 5.654@119

d5's reference leg read 24.594% / 276,413 tok/s / drops 0.089@119 / loss 5.59. This reproduces it to
+0.25pp MFU and 0.002 drop fraction from an independent session and rack allocation. It is also
internally consistent where d5's summary row was not (274,954 x 144.5588e9 / 1.6e17 = 24.84% exactly),
so all comparisons here rest on a self-consistent control.

## R3. THE MECHANISM MEASUREMENT, and it lowers the ceiling on this direction before any latent leg reports

From the dense control's own profiler window (steps 20-22, `xplane_overlap`, one host, 4 GPUs):

    exposed collective 4,126 / 4,384 / 4,828 / 4,755 ms per 3 steps = 1.38-1.61 s of every 15.3 s step
    compute stream busy 38,700 of 42,198 ms (91.7%), i.e. 3,498 ms idle against 4,126 ms exposed

Exposed collective time again almost exactly fills compute idle — EP25's premise, now measured at the
hero shape instead of the d5120 proxy. But the decomposition is the finding:

| collective | occupancy | overlap | exposed | latent's effect |
|---|--:|--:|--:|---|
| `SendRecv` async (expert a2a) | 5,620 ms | 85.1% | 838 ms | halves the bytes |
| `SendRecv` INLINE on the compute stream | 1,266 ms | **0.0%** | 1,266 ms | halves the bytes |
| `AllGather` (FSDP weights) | 1,930 | 70.7% | 566 | none |
| `ReduceScatter` (FSDP grads) | 1,668 | 56.3% | 729 | none |
| `AllReduce f32` (QB beta, norms) | 1,596 | 57.4% | 680 | none |

**Only 51% of exposed collective time is the expert all-to-all.** The rest is FSDP and QB traffic that
latent does not touch. So halving the dispatch payload can recover at most ~351 ms of a 15.3 s step
(2.3%) even if a2a exposure scales perfectly with bytes — against an added latent-projection cost of
285-570 ms/step. **The mechanism's best case is roughly the size of its own overhead.** That is a
quantitative ceiling on the direction, derived from the baseline alone, and it holds regardless of how
the latent legs land.

## R4. An independent finding worth more than the latent question

`xplane_op_detail` attributes every `SendRecv` event to `MoEExpertMlp/moe_mlp/shard_map/{dispatch,
combine}/all_to_all` — 12 distinct HLO `all_to_all` ops, 144 events each (48 layers x 3 steps).
**Exactly three of the twelve are scheduled on the compute stream and are 0.0% overlapped:**

    all_to_all.40.1  bwd dispatch  439.18 ms      all_to_all.46.1  bwd combine  423.72 ms
    all_to_all.56.1  fwd dispatch  402.75 ms      (per 3 steps, consistent across all 4 GPUs to 1%)

The other nine are on the async stream and 85% hidden. Those three placements cost **422 ms of every
step — 31% of all exposed collective time**, which is larger than the best case of the entire latent
mechanism, and they cost no bytes and no arithmetic to fix. This effort has repeatedly found that
scheduling fixes are the cheap kind; this is a concrete, localized one.

## R5. Latent MoE adds 1.8 B REPLICATED parameters per GPU, and that is not free

Following the standalone implementation, the projections are `P(None, None)` — replicated, like the
router, while everything else in the model is sharded. At the hero shape that is
`48 x 2 x 6144 x 3072 = 1.812 B` parameters on every GPU: 6.75 GiB of fp32 params, 6.75 GiB of MuonH
momentum, 6.75 GiB of gradient. Invisible in the standalone harness, which was not memory-limited.
Two fixes exist and neither was applied here, to keep the port faithful: shard the projections over
the expert axis (trades replication for a small per-layer all-gather), or share one projection pair
across all layers (removes it entirely, at a quality cost, and is a genuinely different architecture).

## R6. Operational: the latent arms are harder to land than the dense arm, with a mechanism

Rack tally: dense 1 submission / 1 success; latent 3 submissions / 0 successes / 6 deaths. Every
latent death classifies as preemption or a gang abort with **no primary task** and zero hits on
`failed to allocate`, `RESOURCE_EXHAUSTED`, `ncclAlltoAll` or `Cuda failure` — d5's class 2, i.e. a
kernel SIGKILL, the signature of a container host-memory OOM. The mechanism follows from R5: with
`SCALE_OFFLOAD_OPT_STATE=1` the replicated projection momentum is pinned host memory, so a latent
node needs ~108 GiB against the dense arm's ~81 GiB, versus a launcher default of `ram="256g"` on a
node with 960 GB. Under test as an A/B (one leg at 600g, one left at 256g).
Added `SCALE_RAM` and `SCALE_PREEMPTIBLE` to the launcher so neither needs a code edit again.

## Check-in 10 — the host-memory hypothesis is FALSIFIED, and the real signature is the incarnation abort

The A/B set up in check-in 9 returned, and it returned against me. The 600g leg
(`...-e256-latent3072-120-0726-1615-v2`) died at 22:15-22:16 in exactly the same way as the 256g legs:
9 x `another task died`, zero `SIGTERM caught`, zero HBM OOM. **Raising the host-memory request from
256g to 600g changed nothing.** Recording that as a failed prediction of mine, the same way d5
recorded theirs on the identical hypothesis — which is now two independent falsifications of the
class-2 host-memory branch on this cluster, and the triage recipe should probably be reworded from
"suspect, then test" to "usually not it".

The primary error, found by reading the full (non-warning) stream instead of the warning stream:

    F0726 22:13:21 client.h:80] Terminating process because the JAX distributed service detected
    fatal errors ... absl::Status: ALREADY_EXISTS: Aborted connect attempt as there is a request
    from a newer incarnation. RPC: CoordinationService/RegisterTask

That is not a memory failure at all. A task restarted, re-registered with a NEW incarnation, and the
peers still holding the old one aborted — the cluster-wide incarnation-mismatch class this session has
been chasing. d5 predicted exactly this mechanism from preemption evidence ("an evicted task that
comes back carries a NEW incarnation, and a peer still holding the old one reports exactly the
'different incarnation' gang-abort"); this is that prediction confirmed with the error text.

Which makes eviction the root and COMPILE TIME the exposure. My dense leg was submitted at 21:01 and
was executing steps by ~21:20; it then ran through the 21:28-21:40 window untouched. Every latent leg
has been caught mid-compile, and a ~20-minute compile that restarts from scratch on every eviction
can lose indefinitely. The latent-versus-dense asymmetry in check-in 9 is better explained by WHEN
each job was in its lifecycle than by anything about latent — which also means check-in 9's
"latent legs are harder to land" framing was over-read from the arm labels, and I am withdrawing it.

ACTION: stopped the crashlooping matched v2 and resubmitted the PRIMARY arm as
`/mwittmann/ep25d6-d6144-e128-matched-i6144-L3072-120-0726-1630-v3` with **`SCALE_PREEMPTIBLE=0`**
(the launcher knob added earlier today) plus SCALE_RAM=600g. It scheduled, so non-preemptible GB200
capacity exists. This is the direct test of the eviction root cause: if v3 clears compile and steps
where four preemptible attempts did not, the mechanism is settled and the fix is a one-line launcher
change for anyone running long compiles on this cluster.

## Check-in 11 — FALSIFICATION, flagged early as the brief asks. The wire mechanism WORKS and latent still loses.

The primary arm cleared compile on the non-preemptible pool (that fix worked: first latent leg to
reach step 0 after four preemptible attempts failed) and is running. Its emitted denominator is
155.4304 GFLOP/token = 3 x 51.810 G, exactly the figure check-in 2 predicted, so the MFU is
arch-aware and comparable.

### The endpoint, matched early window (steps 10-24), against my own dense control

| arm | FLOPs/token | MFU p50 | tok/s p50 | step p50 |
|---|--:|--:|--:|--:|
| dense | 48.186 G | **27.177%** | 300,803 | 13.944 s |
| matched-work latent | 51.810 G | **27.155%** | **279,529** | 15.005 s |

**-7.1% tok/s, -0.02pp MFU.** The matched-work arm did +7.52% more analytic work in +7.6% more time.
That is the PURE NULL, hit almost exactly: check-in 8 predicted "a2a exposure unchanged, projections
at step-average efficiency -> tok/s -7.0%, MFU +0.00pp". Measured: -7.1% and -0.02pp.

### But the profile says the mechanism is NOT what failed. It worked.

Matched-work profile vs dense profile, same 3-step window at step 20, GPU:0, occupancy in ms per 3
steps with exposed = occupancy x (1 - overlap):

| collective | dense occ | dense exposed | matched occ | matched exposed |
|---|--:|--:|--:|--:|
| `SendRecv` async (expert a2a) | 5,620 | 838 | 3,911 | **70** |
| `SendRecv` inline on compute | 1,266 | 1,266 | 898 | 898 |
| **a2a subtotal** | **6,886** | **2,104** | **4,809 (-30%)** | **969 (-54%)** |
| `AllGather` | 1,930 | 566 | 2,491 | 1,124 |
| `ReduceScatter bf16` | 1,668 | 729 | 1,660 | 0 |
| `AllReduce f32` | 1,596 | 680 | 3,051 | 833 |
| `AllReduce bf16` | 168 | 14 | **1,170** | 724 |
| **non-a2a subtotal** | **5,363** | **1,989** | **8,371 (+56%)** | **2,680 (+35%)** |
| **TOTAL** | **12,249** | **4,126** | **13,180 (+8%)** | **3,728 (-9.6%)** |

Read the a2a row first: **halving the dispatch width cut a2a exposure by 54%, from 2,104 ms to 969 ms
per 3 steps — within 8% of the 1,052 ms my check-in-8 prediction named.** The collective-bytes
mechanism is confirmed, quantitatively, at the hero shape. This is not a null result about the
mechanism.

Now read the rest of the table. **Latent MoE pays for the halved all-to-all with new collectives of
its own.** `AllReduce bf16` goes 168 -> 1,170 ms (7x), `AllReduce f32` 1,596 -> 3,051 (+91%),
`AllGather` 1,930 -> 2,491 (+29%). Total collective occupancy goes UP 8% despite the a2a falling 30%.
The cause is the replicated projections flagged in check-in 5: 1.812 B parameters that live on every
GPU need their gradients ALL-REDUCED across the batch axes each step, and MuonH's Newton-Schulz runs
on them replicated as well. So net exposed collective time falls only 9.6% (-133 ms/step) instead of
the -378 ms/step the a2a alone delivered — 61% of the win is eaten before it reaches the step.

And then the step time goes the other way anyway: compute-stream busy 38,700 -> 42,893 ms (+10.8%)
against +7.52% analytic work, i.e. +1,398 ms/step of compute against -133 ms/step of collective.

### The verdict, stated plainly

**The byte thesis survives; latent MoE as a way to exploit it does not.** Halving dispatched
activation width does exactly what EP25 predicted it would to the wire. It loses anyway, and for a
reason that is specific to this mechanism rather than to the thesis: the projections that buy the
narrower wire cost more in added compute and added parameter-gradient traffic than the wire saves.
That is structurally the same failure as fp8 — pay compute to save bytes, and the compute wins — but
arrived at through a completely different mechanism, which makes it a second independent data point
on the same economics rather than a repeat.

WHAT WOULD CHANGE THE ANSWER, and it is now well-motivated rather than speculative: the added
collective traffic is an artifact of REPLICATING the projections, not of latent MoE as an idea. A
projection shared across all 48 layers would be 48x smaller in both gradient traffic and Newton-Schulz
work while preserving the entire wire saving. That is the experiment this result argues for, and it
is a different architecture rather than a tuning knob.

Confidence: 8/10 on the falsification of latent-MoE-as-throughput-win at this shape (the profile and
the endpoint agree, and the endpoint hit a pre-registered null prediction to 0.1%); 9/10 that the
wire mechanism itself works as advertised (54% exposure cut, measured); pending the steady tail
(90-119) and the secondary arm to finalize.
