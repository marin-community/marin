# D-2 build report

## TL;DR

Branch `agent/deri-d2-build` now contains the padded-Muon outbound reshard, the
manual-PGLE artifacts, the `75c517148` no-merge EP Newton–Schulz path, its
multi-axis inbound two-hop reshard, and an EP-capable `SCALE_MUON_SYRK` path.
The three code checkpoints are:

| Commit | Result |
|---|---|
| `5c031c31b` | Cherry-pick of `497423bc6`; padded stacks return directly to the parameter sharding. |
| `ef31e3257` | Manual-PGLE profile and conversion helper copied from `62b026409`. |
| `888fff904` | Decision-1 reconciliation: port the `75c517148` EP layout and inbound reshard while preserving SYRK. |

No GPU or cluster job was submitted. The composed stack has not run end to end.

## What landed

`497423bc6` now passes the parameter sharding into the padded-stack helper at
`lib/levanter/src/levanter/optim/grugmuon.py:234-244`. After the local
Newton–Schulz vmap, the result is resharded to that layout before slicing at
`lib/levanter/src/levanter/optim/grugmuon.py:673-682`. The regression test checks
that no fully replicated padded reshard remains and that the original shape and
sharding are restored at `lib/levanter/tests/test_grugmuon.py:122-154`.

The copied profile is
`experiments/grug/moe/pgle/ep64-qb-adjoint-prefetch.pb`, blob
`1b61567a014f5c7277e0e58168502d06d06713ed`, 297,128 bytes. The converter reads
all xplane dumps, extracts per-xspace FDO profiles, aggregates them at p90, and
emits gzip+base64 markers at `experiments/grug/moe/pgle_convert.py:23-55`.
Ruff removed three duplicated blank lines from the copied Python file; its
behavior is otherwise the `62b026409` version.

The normal EP 4D path no longer merges `(L, E)`. It keeps E on `"expert"`,
selects layer-sharding axes only from the non-expert axes, and restores
`orig_4d_spec` at `lib/levanter/src/levanter/optim/grugmuon.py:445-498`. The
non-EP path still uses the original merged 3D route at
`lib/levanter/src/levanter/optim/grugmuon.py:500-538`.

## Audit of derisking-plan.md section 2.2

### Verified

- The `grugmuon.py` diff claim is exact. `git diff --numstat f53f781ce
  origin/research/rav/7201-ep64-muon-pad` reports `20 3`;
  `git show --numstat 497423bc6` reports `18 1`. The commit has two production
  hunks. The only residual is a `GrugMuonConfig` routing-docstring rewrite
  (`+2/-2`). The cherry-pick also carries its separate 36-line test, so the full
  commit is `+54/-1` across two files.
- The two production hunks landed in the intended symbols, not duplicated
  context. The caller hunk is inside `_grug_scale_with_muon` at
  `lib/levanter/src/levanter/optim/grugmuon.py:234-244`; the callee hunk is
  inside `_newtonschulz_padded_stack_sharded` at
  `lib/levanter/src/levanter/optim/grugmuon.py:632-684`.
- The preimage really did reshard the padded result to
  `P(None, None, None)` before slicing. The current outbound branch instead
  uses `target_sharding` at `lib/levanter/src/levanter/optim/grugmuon.py:674-682`.
- C2 was already present on the starting branch. `git merge-base --is-ancestor
  fe21ea495 f53f781ce` succeeds. The starting implementation was the rav
  transpose/merge variant; `54bbe3d23` itself is not an ancestor because this
  lineage is a replay.
- The profile was captured before padded Muon. `497423bc6` is not an ancestor
  of `62b026409`. The profile-add commit `3a73ca1b5` names the
  `ep25d4-pgle-capture-30-v1` 16-host capture and steps 8-11. Padded Muon changes
  the instruction set, so this profile is evidence and a fallback artifact, not
  a qualified profile for the composed build.
- The manual-PGLE caveat is supported by the rav handoff: only 217 of 535
  instructions matched, and the run reached 23.051%, 0.235pp below v153
  ([snapshot lines 329-331](https://github.com/marin-community/marin/blob/1a88e1f3be7c26673664922e409234226952a068/.agents/projects/7201-ep64-drop3/HANDOFF.md#L329-L331)).
- `overlap_limit=4` requires no repository code. It remains a submission-time
  XLA flag.

### Falsified or narrowed

- “Missing exactly two things” is too narrow once section 4.1 is applied. The
  starting branch had padded-Muon and PGLE gaps, but its rav C2 implementation
  also retained the guarded merge and lacked `75c517148`'s inbound padded-stack
  reshard. The D-4-capable build therefore needed the Decision-1 reconciliation
  in `888fff904`.
- “The PGLE files exist on no other branch” is false literally in this clone.
  Before this port, both `agent/ep25-d4-pipelined` and its descendant
  `agent/deri-d5-census` contained `62b026409`; the d4 branch also has
  `origin/agent/ep25-d4-pipelined`. The files were absent from the D-2 starting
  branch, which is the operationally relevant part of the claim.

## Audit of derisking-plan.md section 4.1

### Evidence ordering

The recommendation to take `75c517148` is supported. The recorded 64-GPU probe
explicitly names that commit and both mechanisms: 20.22% / 14.84 s with 208
warnings became 22.02% / 13.63 s with zero warnings
([#7279 comment 5012284704](https://github.com/marin-community/marin/issues/7279#issuecomment-5012284704)).
The `fe21ea495` commit body records 17.823% over 30 steps while bundling expert
sharding, fixed-capacity all-to-all, environment forwarding, and CUTLASS
installation. It is not an isolated A/B of the rav transpose mechanism.

The implementation comparison is also correct:

- `75c517148` has no guarded fallback under EP. Its EP branch tests only
  `expert_axis_size > 1`, keeps E sharded, and returns through
  `orig_4d_spec`. The port retains those properties at
  `lib/levanter/src/levanter/optim/grugmuon.py:445-498`.
- The rav post-image engaged only when `"expert" in best_axes`. Its transpose
  exit had sharding `P(None, "expert", None, None)`, rather than restoring the
  D/I axes in `orig_4d_spec`.
- `75c517148` contains the two-hop padded inbound reshard. The rav version did
  not.
- `git show --numstat 75c517148 -- grugmuon.py` is exactly `+54/-11`.
- `git merge-base 75c517148 54bbe3d23` is
  `696eb370dbf5b3f41b43d485817e26348ed00642`.

The two layouts are mathematically value-equivalent: both apply Newton–Schulz
independently to each `(D, last)` matrix. At one-rack L48/E256/EP64, the rav
merge assigns `(48 * 256) / 64 = 192` matrices per device; the no-merge path
assigns `48 * (256 / 64) = 192`. They are not bitwise interchangeable under
all CPU lowering choices; see Risks.

### Complementarity and run provenance

The complementary-hunk claim is exact in the composed function. The inbound
two-hop is immediately before the vmap at
`lib/levanter/src/levanter/optim/grugmuon.py:667-673`; the `497423bc6` outbound
reshard is immediately after it at
`lib/levanter/src/levanter/optim/grugmuon.py:674-682`. The edits do not overlap.

The recorded +1.78pp screen did not contain the inbound fix. Its published
snapshot is `1a88e1f3b`
([#7201 comment 5088824573](https://github.com/marin-community/marin/issues/7201#issuecomment-5088824573));
`497423bc6` is an ancestor of that snapshot and `75c517148` is not. No ref in
this clone contained both commits before this build. This verifies the claim
for the recorded run and available refs; a universal claim that no unrecorded
workspace ever ran the pair is not independently provable.

At the recorded one-rack mesh, only `"expert"` has size greater than one. The
second inbound hop is guarded by `len(batch_axis) > 1` at
`lib/levanter/src/levanter/optim/grugmuon.py:670-672`, so it is dormant there.
The multi-axis regression uses a `(replica_dcn=2, expert=4)` abstract mesh and
asserts the single-axis then tuple-axis sequence at
`lib/levanter/tests/test_grugmuon.py:157-187`. Taking the fix is still warranted
for D-4, where the mesh gains another size-greater-than-one axis.

### SYRK decision

This tree is on the rav replay lineage and already had
`SCALE_MUON_SYRK`. Before reconciliation, its normal merged route placed a
batched SYRK `shard_map` between the forward and inverse transforms. Declaring
SYRK FSDP-only would silently change the EP recipe that produced
`fe21ea495`'s 17.823% result.

The EP branch therefore preserves SYRK. Each shard-local `(L, E, D, last)` tile
is flattened to `(L*E, D, last)`, passed to `_newtonschulz_batched_syrk`, and
reshaped back inside `shard_map` at
`lib/levanter/src/levanter/optim/grugmuon.py:480-494`. The double-vmap remains
the non-SYRK branch at line 496. The FSDP route retains its original SYRK
branch at `lib/levanter/src/levanter/optim/grugmuon.py:525-535`.

## Verification

- `./infra/pre-commit.py --fix --files` passes for
  `grugmuon.py`, `test_grugmuon.py`, the profile, and `pgle_convert.py`.
- Literal `uv run pyrefly` exits 2 both before and after the changes because the
  worktree `.venv` has no `pyrefly` executable. Running the locked checker as
  `uv run --with 'pyrefly>=1.0.0,<1.1.0' pyrefly check` reports
  `0 errors (411 suppressed, 531 warnings not shown)` before and after.
- The requested broad pytest command fails during collection before applying
  its `-k` filter because the shared environment lacks `tensorboardX`
  (`lib/levanter/tests/test_tensorboard.py:8`).
- Collecting only the two files containing Muon tests runs the relevant suite.
  Eight tests pass. One starting-branch failure remains:
  `test_grug_muon_mask_routes_stacked_expert_weights_to_muon` expects
  `router -> muon`, while `f53f781ce` routes any path containing `"router"` to
  AdamW at `lib/levanter/src/levanter/optim/grugmuon.py:149-150`. This work did
  not change either line.
- The three directly affected structural tests pass: the 4D EP path has no
  `(L*E, D, last)` reshape, the padded outbound path does not fully replicate,
  and the multi-axis inbound path performs the two reshards in order
  (`lib/levanter/tests/test_grugmuon.py:95-187`).
- With `SCALE_MUON_SYRK=1`, a CPU abstract trace reaches the new EP
  `local_syrk` branch and then stops importing the unavailable optional
  `cutlass` package. This verifies dispatch but not QuACK compilation or GPU
  numerics.

## Risks relative to the plan

1. `75c517148` and `497423bc6` have still not run together on GPU. The CPU
   jaxpr tests verify layout structure, not absence of XLA involuntary-remat
   warnings in a real compiled GB200 step.
2. A forced eight-device CPU probe did not reproduce the source commit's
   stronger “bit-exact” wording. Against an unsharded per-matrix reference, the
   no-merge path differed by at most `9.765625e-4` with zero NS iterations and
   `6.005859e-2` after two iterations. The tiny BF16 probe shapes were
   `L4/E16/D8/I4` and `L4/E16/D4/I2`, respectively.
   Different reduction trees can explain the difference, and the recorded
   `75c517148` run reported loss parity, but the composed GPU update still needs
   qualification. No tolerance was changed.
3. The checked-in PGLE profile predates padded Muon and the Decision-1 layout.
   Its instruction names and costs cannot be assumed to match the composed
   executable. The d4 profile was valid for its own capture, but it is stale for
   this build.
4. Manual PGLE has weaker EP evidence than the plan's FSDP analogy suggests.
   The ECHO manual profile matched 217/535 instructions and was 0.235pp below
   auto-PGLE. A fresh profile may still fail to cover enough of the new
   executable to help.
5. The SYRK EP branch cannot compile in this CPU environment because CUTLASS is
   intentionally optional and Blackwell-only. Its `shard_map` wiring is
   structurally covered but unexecuted.
6. The repository's default type-check and broad-test launchers are not
   reproducible from this worktree without adding the missing development
   executables. The underlying type check is clean, and all D-2-caused tests
   pass.

## Required before submitting D-2

1. Capture a new manual PGLE profile from this exact commit and D-2
   configuration. Audit the profile's matched/missing instruction counts,
   especially all-to-all, reduce-scatter, all-reduce, and Muon operations.
2. Run a GB200 compile/HLO smoke for both `SCALE_MUON_SYRK=0` and `1`. Confirm
   the 4D EP path has no `(L,E)->LE` merge, the padded inbound path has two
   reshards on a multi-axis mesh, the outbound path never materializes
   `P(None,None,None)`, and there are zero involuntary-remat warnings.
3. Run a small GPU numerical comparison of the composed Muon update against the
   existing path before spending a rack draw. Check all expert and non-expert
   stack orientations and record max/mean absolute differences plus loss
   parity.
4. Submit from one immutable SHA with the standing protocol: d5120,
   8-of-256, EP64, one rack, batch 1024, sequence 4096, 350 steps, QB on,
   `cf=1.0625`, spill `m=3`, custom adjoint, padded Muon, fresh PGLE, and three
   sequential placement draws.
5. Set and record
   `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`,
   `--xla_gpu_experimental_ragged_all_to_all_use_barrier_with_nccl=false`, and
   `--xla_gpu_experimental_parallel_collective_overlap_limit=4`. Verify the
   logged capacity factor and every feature flag before accepting step 0.
6. Pre-register the approximately 22.5% prediction and falsification threshold.
   Report tok/s with locked 2.5-PFLOP/s MFU, tail-window drops at matched LR
   position, run length, and placement distribution. Inspect loss and drop
   trajectories even when Iris reports success.
7. Use the one-rack D-2 result as the denominator before submitting D-4 on this
   identical build. The 17.8% bundle needs re-running because the rav C2 layout
   has been replaced.
