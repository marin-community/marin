# mfu30-stack: stacking and schedule (agent C)

Branch `research/mcwitt/mfu30-stack` (worktree `~/projects/marin.mfu30-stack`) on the campaign base
`03738a82a8`. Parent logbook: `.agents/logbooks/mfu30.md`. Tools: `autoresearch/loop-260930-mfu30/stack/`.

## Layout (2026-09-30)

Linear history; an arm runs whichever commit its checkout is on.

| commit | content | switch |
|---|---|---|
| `314b638a71` | B: invert routing permutations with a unique-index scatter (`d1ccdd9959`) | always on (bitwise-exact) |
| `bb4653525b` | B: chain ragged a2a cotangents across expert chunks (`c67ee1f965`) | always on (bitwise-exact) |
| `a9762fca82` | B: leave transport buffers unfilled (`e612b34244`; only a gate-script conflict) | always on (bitwise-exact) |
| `2458135edd` | C: fused RMSNorm + GatedNorm kernels | `--gated-norm-implementation pallas_gpu` (default off) |
| `e9d7d6034d` | PR #9481 `429bfcb9d0` pipelined expert chunks, ported onto B's loop | code (no switch) |
| `a9056f1adf` | PR #9481 `6982f881f2` mirror transpose parameters, ported onto B's wrappers | code |
| `ebde49980b` | PR #9481 `d527dd9822` re-gather attention weights in the backward | code |
| `d68a7b5a0e` | PR #9481 `f00fb8330c` gather MLP weights before routing (ragged only) | code |
| `cbb41c3310` | PR #9481 `d666a6729b` router QB statistics after the routed MLP | code |
| `46edad1a0f` | PR #9481 `2a9cb19306` `pgle_profile.py` + README | tool |
| later | arm/PGLE/scoring tools, this logbook | - |

Env switches (arm.sh `--xla`): A `--xla_gpu_enable_host_memory_offloading=true`; C
`--xla_gpu_enable_triton_gemm=false`; PGLE `--xla_gpu_pgle_profile_file_or_directory_path=<pbtxt>`.
The stack without #9481 is `2458135edd`.

### #9481 against B's transport

The three model commits and the PGLE tool applied cleanly on top of B and C. The two transport commits
conflict with B's `c67ee1f965`/`e612b34244`, which rewrote the chunk loop and the `_ragged_a2a` custom VJP:

- Pipelining: B's loop forwards the sorted buffer from chunk to chunk (`_ragged_a2a_add_forwarding`) and
  ties each dispatch to `(chunk_source, returned)`. The port plans all chunks first, dispatches chunk 0
  unbarriered, and inside chunk c dispatches c+1 behind `optimization_barrier((forwarded source, x_dispatch))`,
  then barriers `(out_dispatch, next_x_dispatch)` before chunk c's return, as in #9481. The forwarded
  buffer semantics are unchanged: chunk c+1 reads the buffer chunk c forwarded.
- Mirror parameters: B's `_reverse_ragged_a2a` exchanged offsets with two all-to-alls. The port passes the
  mirror chunk's parameters (`return_params` for a dispatch, `dispatch_params` for a return) as
  `transpose_params` through `_ragged_a2a_add` and `_ragged_a2a_add_forwarding` and drops the exchange.
  #9481's test that the two parameter sets are each other's transpose is included.
- CPU: `lib/levanter/tests/grug/test_grugformer_moe.py` 75 passed (9 GPU-only skipped); hero model tests
  71 passed; gated norm kernel tests 7 passed.
- GPU: B's gate (`autoresearch/loop-260930-mfu30/b/gpu_gate.sh`) on the stack at `46edad1a0f`, job
  `m30c-stackgate-02` (GB200x4, hero env: cuda_async, fraction 0.75, ragged flags, overlap limit 1; the
  first attempt without that env failed NCCL's symmetric registration, as B's first gate did). All six
  routing cases (drops, padding, one-hot, hero shard shapes, up to 358,395 drops) are bitwise equal to main
  for out, drops, dx, d_weights, dW13 and dW2. QuACK row contract passes. 3-layer rematted scan: losses
  equal; median step main 84.4 ms, B's A+B 82.2 ms, stack 77.0 ms (+9.6% vs main; B measured +6.1% for
  A+B+C alone in its own run). pytest: 80 passed, the same 3 GPU-only failures as main (f32 into QuACK).

B's SonicMoE backward (`992a33a975`, `5a3df1f500`, `ea3e4d6194`) replaces the chunk loop with one custom
VJP (`_routed_experts`) and its own barriers. The two #9481 transport ports do not carry over; the ideas
do: the backward can take `return_params`/`dispatch_params` from its chunk residuals instead of
`_reverse_ragged_a2a`'s offset exchange, and the forward can pipeline its chunks. B owns that code.

Overlap with A: #9481's attention re-gather (`ebde49980b`) targets the same symptom as A's flag (XLA's
post-schedule remat turning attention weight gathers into exposed synchronous all-gathers). With the flag
the XLA remat may disappear; the JAX-level re-gather then only changes where the gathers sit.

## PGLE runbook

The profile matches HLO instruction names, so it is built from a trace of the exact program it schedules
(same commit, same XLA flags other than the PGLE flag):

1. Trace arm: `arm.sh <run> <port> --xla "<final flags>" --trace -- <final CLI switches>` (profiles
   180021-180023; the rest of the window still scores the no-PGLE program).
2. Build: `pgle_build.sh <run>` downloads the rank-0 host's xplane from
   `marin-cw:hero-checkpoints/tmp/ttl=30d/xprof/<run>/plugins/profile/`, writes the plain profile
   `experiments/grug/moe_hero_ep/pgle/<run>.pbtxt` (~3.8k instruction costs; ~1 s on a CPU) and the
   D2H-patched `<run>-d2h.pbtxt` (A's `a/pgle_patch_d2h.py` on `tfop_dump.py` rows of the same xplane: every
   end-of-step optimizer-state D2H copy-start >= 1 ms gets the batch's serialized total x 1.1, because the
   GPU LHS models async memcpys with unlimited concurrency), force-adds and commits both, and prints both
   flags. On the Sep 24 trace the patch moves seven copy-starts (3 x ~69 ms, 2 x ~17 ms, 2 x 2.5 ms) to
   272.6 ms each. Validate in the PGLE run's trace: copy-start.97/98/99 inside the expert Newton-Schulz
   phase and no ~0.26 s tail. Watch (A, t21): forward carry D2H fully waited under PGLE (0.19 s/step), and
   the first momentum H2D hoisted to the start of the backward (+10 GiB through the backward peak).
3. Scored rerun from that commit: same flags plus the printed PGLE flag. XLA's default
   `--xla_gpu_pgle_accuracy_checker=PGLE_STRICTNESS_LEVEL_WARN` logs instructions missing from the
   profile; grep the task logs for them to confirm the profile matched.
4. Loss check against a same-code control: one PGLE variant in #8317 drifted +1e-3, one-signed.

Scoring: `score.py <run> ...` (median MFU and duration over 180011-180059, profiled steps excluded; peak
memory; drop fraction; pointwise loss difference against `mhep-ctx4k-s0-20260930`).

## Plan update (orchestrator, 2026-09-30)

B folds #9481's mirror parameters and pipelined chunks into its SonicMoE `_routed_experts` (return-path
barrier `out_c, _ = barrier((out_c, next_x_dispatch))`) and will ask C to review. Once that gates, the stack
replaces `e9d7d6034d` and `a9056f1adf` with B's commits. B's SonicMoE (D) is only correct and repeatable at
collective overlap limit 1; B forces limit 1 for ragged runs, and the hero offload already forces it.

C's loss/lm_head item (job `m30c-ce-01`, hero CE at [65536, 6144] x 128256, fanout on GB200x4): production
tiles 331-335 ms fwd+bwd; best larger vocab tiles 320 ms (backward v 8192 or 16384), i.e. <= 0.012 s/step.
Closed.

## Rebuild on B's D + mirror + E (2026-09-30)

The stack moved to B's lineage. Old head `d4c62b5a5c` (my hand ports of #9481's transport commits on B's
A+B+C) is superseded; the branch was force-pushed.

- `research/mcwitt/mfu30-stack` @ `8eff7b8ec4` = B `12643e682c` (A+B+C, D expert-side router gradient with
  the routed output saved, #9481's mirror parameters folded into `_routed_experts`, E dswiglu epilogue,
  sequential chunks) + merge of the campaign branch + C's fused gated norm + #9481's attention re-gather,
  MLP-weight prefetch, QB-after-MLP and `pgle_profile.py` + tools. `lib/levanter/src/levanter/grug/_moe/`
  is identical to B's `12643e682c`.
- `research/mcwitt/mfu30-stack-pipelined` @ `7393a9ae26` (worktree `~/projects/marin.mfu30-stack-pipe`) =
  the same + B's pipelined forward `64909b24d0`.
- #9481's model commits against D: the MLP-weight prefetch ties the gathered latent/shared weights to
  `mlp_in` with a forward-only barrier; D's saved routed output is downstream of the expert MLP, so the
  recompute still needs `mlp_in` and still sees the prefetch. The QB barrier ties `s_minus_alpha` to the
  whole MoE output; QB statistics are not differentiated, so the recompute drops the barrier and does not
  pull back the down GEMM or the return that D removed. The attention re-gather is an inner checkpoint
  and independent of the MoE policy.
- CPU: hero model tests 72 passed; MoE + gated norm tests 82 passed, 10 skipped (GPU-only).
- GPU gates (GB200x4, hero env): `m30c-stackgate-03` (sequential), `m30c-stackgate-pipe-01` (pipelined):
  B's module gate plus `stack/model_smoke.py`, a 4-layer EP4 hero model with the ragged backend, carry
  offload (saving the routed output), #9481's model commits, and the fused norm on and off.

### Gates and trace arms (2026-09-30 14:36 PT)

Both lineages gate on GB200x4 with the hero env (`m30c-stackgate-03` sequential, `m30c-stackgate-pipe-01`
pipelined): routing gate 0 failures over 6 cases (out, drops, dx, dW13, dW2 bitwise equal to main; d_weights
at ~1 bf16 ulp, max 0.4%), QuACK contract passes, pytest 88 passed with the same 3 GPU-only failures as main.
3-layer rematted scan (median step): main 339.0 / 340.6 ms, D 292.5 / 293.6, sequential stack module 289.9,
pipelined stack module 296.0. The scan has no shared experts, so it cannot show the pipelining's benefit.
Model smoke (4-layer EP4 hero model, ragged backend, carry offload saving the routed output, #9481 model
commits): finite loss and gradients with the norm modules and with the fused norm; loss 7.657223 vs
7.657212 (sequential) and 7.657224 vs 7.657212 (pipelined); temp 0.28 vs 0.26 GiB. The largest per-leaf
relative gradient difference, fused vs modules, is ~6%, in the routed-MoE leaves (router, experts, latent
projections). A bf16 CPU reproduction localizes it there; in f32 every leaf agrees within 1e-4. It is top-k
routing flipping on near-ties after a rounding-level change in `mlp_in`, not a kernel error, so any
forward-rounding change (the fused norm, the Triton-GEMM flag) moves the first-step loss at ~1e-6-1e-5
relative and cannot reproduce 1.261413 exactly.

Trace arms (program: stack HEAD + `--xla_gpu_enable_host_memory_offloading=true
--xla_gpu_enable_triton_gemm=false` + `--gated-norm-implementation pallas_gpu`, remat VLOG
`TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=hlo_rematerialization=1`, seed 0, 180000-180060, profiled
180021-180023, no --timeout), pipelined first:
- `m30c-stackpipe-trace-01` (`/mwittmann/m30c-stackpipe-trace-01-coord`, port 33302) from
  `research/mcwitt/mfu30-stack-pipelined` @ `7393a9ae26`.
- `m30c-stackseq-trace-01` (`/mwittmann/m30c-stackseq-trace-01-coord`, port 33303) from
  `research/mcwitt/mfu30-stack` @ `4ee7986fb4` (code identical to `8eff7b8ec4`).

### Short conv (B) and the final program (2026-09-30)

B's Triton short conv (`d466f12f3f..5d64137a67`, handoff patch `sconv_on_stack.patch` @ `9303d20b97`) is on
both branches behind `--sconv-implementation triton_gpu`, default off (the default stays the Pallas
kernel), so the queued trace arms' commits keep their meaning: `research/mcwitt/mfu30-stack` @ `cc55c78f45`,
`research/mcwitt/mfu30-stack-pipelined` @ `a7cc657e28`; the two differ only in `ep_ragged_all_to_all.py`.
CPU: short-conv + gated norm tests 48 passed (12 GPU-only skipped), hero tests 72 passed; lint clean.

Endgame order (orchestrator): (1) `m30c-stackpipe-trace-01` vs `m30c-stackseq-trace-01` pick the D lineage;
(2) final program = winner + `--sconv-implementation triton_gpu` (+ A's carry-prefetch flag if its paired arm
wins), with its own trace arm, since PGLE matches instruction names; (3) `pgle_build.sh` from that trace;
(4) scored PGLE run + a repeat. Final flags: `--xla_gpu_enable_host_memory_offloading=true
--xla_gpu_enable_triton_gemm=false` [+ A's carry-prefetch flag] [+ the PGLE flag in step 4]; CLI
`--gated-norm-implementation pallas_gpu --sconv-implementation triton_gpu`. B reports the attention backward
is run-to-run nondeterministic (~1e-5 rel-rms), so same-code runs are not bitwise; the fidelity reference is
the rounding-perturbation band from `m30c-grnflag-01`.

### Trace arms resubmitted with A's memory settings (2026-09-30 18:40 PT)

The `-01` pair was cancelled before starting: D's ~18 GiB saved routed output would push XLA's remat limit
((pool - persistent) x slop = 87.66 GiB at fraction 0.75 / slop 85) into cutting D's gain. A's pair for D
arms: `XLA_PYTHON_CLIENT_MEM_FRACTION=0.78`, `--xla_gpu_memory_limit_slop_factor=105` (remat limit ~114 GiB).
#9481's attention re-gather is now a switch, `--regather-attention-weights` (default off; A found it
redundant with the host-offloading flag). This pair keeps it on and folds in the Triton short conv, so the
trace is the final-program candidate for PGLE:

- `m30c-stackpipe-trace-02` (`/mwittmann/m30c-stackpipe-trace-02-coord`, port 33304) from
  `research/mcwitt/mfu30-stack-pipelined` @ `fcb44f6920`.
- `m30c-stackseq-trace-02` (`/mwittmann/m30c-stackseq-trace-02-coord`, port 33305) from
  `research/mcwitt/mfu30-stack` @ `7e9e2aa52c`.

Both: `--xla "--xla_gpu_enable_host_memory_offloading=true --xla_gpu_memory_limit_slop_factor=105
--xla_gpu_enable_triton_gemm=false"`, env `XLA_PYTHON_CLIENT_MEM_FRACTION=0.78 TF_CPP_MIN_LOG_LEVEL=0
TF_CPP_VMODULE=hlo_rematerialization=1`, CLI `--gated-norm-implementation pallas_gpu --sconv-implementation
triton_gpu --regather-attention-weights`, seed 0, 180000-180060, profiled 180021-180023. Checks per arm:
"Rematerialized N instructions" N <= ~10; "Peak memory for main" <= ~110 GiB (else slop 110);
memory/limit_gib ~143.76; memory/peak_gib < ~139; loss against the grnflag-01 band.

### -02 cancelled after the grnflag-01 regression; -03 without the flag and the fused norm (2026-09-30 18:50 PT)

`m30c-grnflag-01` regressed (below), so the orchestrator cancelled the -02 pair. Resubmitted without
`--xla_gpu_enable_triton_gemm=false` and without `--gated-norm-implementation pallas_gpu`, everything else
unchanged, pipelined first:

- `m30c-stackpipe-trace-03` (`/mwittmann/m30c-stackpipe-trace-03-coord`, port 33306) from
  `research/mcwitt/mfu30-stack-pipelined` @ `fcb44f6920`.
- `m30c-stackseq-trace-03` (`/mwittmann/m30c-stackseq-trace-03-coord`, port 33307) from
  `research/mcwitt/mfu30-stack` @ `a1f483afed`.

Both: `--xla "--xla_gpu_enable_host_memory_offloading=true --xla_gpu_memory_limit_slop_factor=105"`, env
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.78 TF_CPP_MIN_LOG_LEVEL=0 TF_CPP_VMODULE=hlo_rematerialization=1`, CLI
`--sconv-implementation triton_gpu --regather-attention-weights`, seed 0, 180000-180060, profiled
180021-180023. Extra trace checks (new, see below): `stack/copy_schedule.py` (copy-start.44 after the backward
loop) and `stack/carry_stall.py` (forward carry D2H exposed < ~10 ms/step).

### grnflag-01 diagnosis (2026-09-30)

Score: 28.023 MFU (-0.238 vs `mhep-ctx4k-s0` 28.261), 14.007 s (+0.118), peak 117.70 GiB (+14.6), drop
fraction 1.8e-4 (control 1.9e-4); loss vs control first step -9.5e-7, max |d| 3.0e-4, mean -1.9e-5 over 60
steps (rounding band). Two mechanisms, neither the kernels' own cost:

1. Memory: the Triton-GEMM flag reorders the optimizer phase and hoists the first momentum H2D to the top of
   the step. At main level the flag turns 120 Muon Newton-Schulz GEMMs from Triton fusions into cuBLAS calls
   and regroups their elementwise fusions. The LHS runs on T-shirt costs (A: the SOL estimator rejects
   ragged all-to-all): Triton fusion 1, cuBLAS call 1000, a while loop 1 regardless of its body. The expert
   momentum update now directly follows the backward, so its H2D (`copy-start.44`, f32[48,6,3072,3072], 10.1
   GiB) heads the serialized H2D chain. Nothing above it in the step claims the copy resource, so it floats to
   position 1 of main, before the forward loop. On main and hmo-02 the chain's head is an s32[] scalar, and
   `copy-start.44` sits after the backward (main position 1345). The fused norm only changes the scan bodies.
   The main-level LHS sees those as 1-unit whiles with unchanged resource sets, and its memory limit does not
   bind, so the kernel cannot cause the reorder. Effect: the copy runs in 0-55 ms of the step, but its
   buffer is live through the backward peak. The arena grows 68.5 -> 82.1 GiB (live peak 49.4 -> 55.2 GiB,
   now in the backward body; the rest is fragmentation). Without H-A4, remat then clones more in the
   backward: 0.868 s/step of XLA remat kernels vs 0.633 on the main baseline trace, 113 vs 103 distinct
   clones.
2. Time: forward carry D2H stall, 0.165 s/step. Each layer's compute stream idles ~3.3 ms because the first
   weight dynamic-slice fusions it needs (async D2D) are queued behind the 4.4 ms carry D2H on the same
   memcpy stream. XLA's `ExecutionStreamAssignment` hands async compute ops to 4 streams round-robin over the
   module in post-order, so the collision depends on how many async ops come first. Both components change
   the forward body's async slices: the fused norm adds a [6144] norm-scale slice, and the flag removes the
   [6144,48] and [6144,384] projection-weight slices. So this is a lottery any program change re-draws. #9481's
   PGLE trace t21 shows the same signature (0.192 s/step). A's M30A-004 "PGLE exposes the forward carry
   D2H" is this collision, not PGLE's latency model.

Rough budget: +0.165 stall, up to +0.235 extra remat compute (part of it under collectives), -0.085
kernel+flag (block bench) -> same sign and order as the measured +0.118 s.

No single-component arm now. Flag only: re-creates (1), since the main-level reorder is all the flag's
doing, for <= 0.055 s (block bench). Kernel only: no main-level change and -1.5 GiB temp, predicted -0.06
s/step, but it re-draws (2). It is worth one add-on arm on the final program after the lineage pick,
gated on `carry_stall.py`. The larger lever is (2): a lost draw costs 0.165-0.19 s/step (~0.35-0.4 MFU), and
the PGLE endgame re-draws it. A deterministic fix would pin the carry D2H to its own stream or keep the
weight slices off the memcpy streams.

### Decisions after the diagnosis (orchestrator, 2026-09-30 19:15 PT)

- `--xla_gpu_enable_triton_gemm=false` is out of the final program.
- The fused norm is a post-lineage add-on arm. It is kept only if its trace passes `carry_stall.py` and
  `copy_schedule.py`.
- The deterministic stream-collision fix goes to A.
- Rack queue: `m30-ctl-s0-r2` (running since 02:07Z) -> `m30b-sonic-02` -> `m30c-stackpipe-trace-03` ->
  `m30c-stackseq-trace-03`.
- Every landed trace (including sonic-02 and ctl-r2 when profiled) gets `stack/trace_checks.sh <run>`. It
  pulls the xplane and runs the carry-stall, copy-schedule, exposed-memcpy and XLA-remat checks.

### Same-code repeat and the endgame plan (orchestrator, 2026-09-30 19:40 PT)

`m30-ctl-s0-r2` (same code as `mhep-ctx4k-s0`, not profiled) scored 28.232 MFU on my score.py (28.235 on the
orchestrator's), 13.903 s/step, peak 103.09 GiB. Against s0: first-step loss identical, max |dloss| 3.3e-4,
late mean -3.9e-5, 35/89 steps positive. This is the fidelity band for every arm. grnflag-01 (max 3.0e-4) sits
inside it.

Plan:
1. Score both -03 arms, run `trace_checks.sh` on each, and pick the lineage.
2. The final program runs on the production wheel: the winner at 0.78/105 with H-A4. Its trace feeds
   `pgle_build.sh` (plain, and A's D2H patch as an option). Then a scored PGLE run, and repeats for the
   loss check.
3. Add-ons each get a paired comparison against the final program, never bundled blind: the short-conv
   switch, A's carry-prefetch flag, and the fused norm alone.

A's stream-collision fix (custom PJRT wheel) waits on the user's approval and is not part of any arm. Until
then `carry_stall.py` is the gate.

Open point: the -03 arms also carry `--sconv-implementation triton_gpu` and `--regather-attention-weights`.
So the -03 trace is the final program's trace only if both flags stay in the final program. Otherwise the
PGLE build needs a new trace arm of the final program.

### B's forward-order backward; final-program pair and ownership (2026-09-30 20:40 PT)

B's forward-order backward (D's backward chunks run c0 then c1, three lines in `_routed_experts_bwd`, gated
bitwise by B in `m30b-reorder-gate-01`) is applied, lib change only, to `research/mcwitt/mfu30-stack` @
`d10320ca2c` (from B's `547bf2ad20`) and to `research/mcwitt/mfu30-stack-pipelined` @ `0c30e9dd9a` (from
`2cc470d88f`). GB200x4 model smoke passes on both (`m30c-reorder-smoke-seq-01`, `-pipe-01`: finite loss and
gradients, MODEL_SMOKE_OK). The queued -03 arms predate it.

The permission system blocked my cherry-pick of A's custom-wheel plumbing (`cf5bc74409`). The user then
approved the wheel directly with the orchestrator, who owns every arm that installs it:
`research/mcwitt/mfu30-final-seq` @ `d4234c88e7` and `research/mcwitt/mfu30-final-pipe` @ `743b5826d2` (my
branch tips + `cf5bc74409`). F1 arms `m30-f1-pipe-01` (port 33010) and `m30-f1-seq-01` (port 33011) are
queued behind the -03 arms. Each runs the full stack + forward order + custom wheel
`jax_cuda13_pjrt-0.11.1+marin.283d5b6d98cd` with `XLA_GPU_HOST_TRANSFER_STREAMS=1`, the short-conv and
re-gather switches, H-A4 at 0.78/105, remat VLOG, traced 180021-180023.

My part:
- Score and check the -03 arms, then pick the lineage. The orchestrator then cancels the losing F1 and
  submits the F0 pair (same wheel, env off).
- Run `trace_checks.sh` on the F1 traces.
- Build the plain and D2H-patched PGLE profiles from the winning F1 trace and hand over the profile commit.

Note that F1 against -03 measures the wheel, the transfer streams and the forward order together.

### m30c-stackpipe-trace-03 (2026-09-30 21:30 PT)

Score: 29.768 MFU and 13.186 s/step, +1.507 against s0 and -0.017 against `m30b-sonic-02` (29.783, 13.179).
Peak 123.20 GiB. Loss max |d| 6.8e-4 against s0 and 4.4e-4 against sonic-02, mean +6.8e-5.

Trace checks:
- Carry stall: 146.9 ms/step exposed (3.0 ms per copy), so it lost the stream draw. sonic-02 shows 2.8.
- `copy-start.44` sits after the backward.
- Zero XLA remat.

Profiled step 180022 took 14.33 s, so the trace's mean span (13.59 s) overstates the scored step.

Against sonic-02, compute busy is 10.80 vs 11.19 s, but exposed collectives are 1.86 vs 1.34 s:
- ragged all-to-all 0.97 vs 0.74;
- backward weight all-gathers (`remat_carry` scope) 0.43 vs 0.11.

Forward body order: in sonic-02 the shared-expert GEMMs sit between the ragged all-to-all starts and dones.
In stackpipe-03 every shared-expert GEMM comes after the last all-to-all done, so the transports run bare.

B's attribution (M30B-025), relayed by the orchestrator:
- compute -0.39 s;
- exposed ragged all-to-all +0.21 s from pipelining;
- carry stall +0.15 s;
- re-gather about neutral.

Orchestrator decision: sequential lineage, no re-gather. F1 pipe/seq-01 are cancelled. Queued from
`research/mcwitt/mfu30-final-seq` @ `d4234c88e7` (short conv on, H-A4 at 0.78/105, no re-gather):
- `m30-f1-seq-02`: custom wheel, `XLA_GPU_HOST_TRANSFER_STREAMS=1`, traced;
- `m30-f0-seq-01`: same wheel, env off, traced.

stackseq-03's trace settles whether pipelining or #9481's model commits moved the shared-expert GEMMs after
the MoE.
