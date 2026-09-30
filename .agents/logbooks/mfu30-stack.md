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
   `marin-cw:hero-checkpoints/tmp/ttl=30d/xprof/<run>/plugins/profile/`, writes
   `experiments/grug/moe_hero_ep/pgle/<run>.pbtxt` (~3.8k instruction costs; ~1 s on a CPU), force-adds and
   commits it, and prints the flag.
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
