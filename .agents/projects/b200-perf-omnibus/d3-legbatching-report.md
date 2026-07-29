# D-3 leg-batching recovery and evidence review

Effort: high

Stop rule: stop searching when the exact submitted source bundle is recovered and
authenticated, all reachable Git objects/reflogs/stashes and the cited issue history
have been checked, and the remaining performance question can be expressed as a
matched, falsifiable experiment.

## TL;DR

The original positive implementation is not lost. The exact source submitted for
`/rav/ep64-batched-expert-stability-120-v1-20260724-2353` remains downloadable as Iris
bundle `0483b2f207323fb3cd79ec326b7592546aabb0812ef8c058be95bd6c8049cd43`.
The same mechanism was committed the next morning in
[`98737aecf`](https://github.com/marin-community/marin/commit/98737aecfa5cd05b9bffe09c96754c96d7177f06),
although that snapshot contains much more than leg batching and is not the byte-exact
job bundle. The plan's claim that this patch was never committed is wrong.

The apparent performance contradiction does not compare the same change:

- The positive implementation keeps the already-packed one-dispatch/one-combine
  collective schedule and replaces only the per-local-expert GEMM loop with batched
  matrix multiplications.
- The reconstruction changes both GEMM batching and collective scheduling. Its `G=2`
  treatment executes two dispatch and two combine all-to-alls and concatenates the
  groups. Its control executes one dispatch and one combine per local expert.

The matched `G=2` result is credible evidence that the reconstruction regresses:
22.66% control versus 19.00% treatment, with non-overlapping p10/p90 bands. It is not
evidence that the original compute-only batching regresses. Conversely, the 25.39%
positive run is not causal evidence for a win: it was QB-off, used an unmatched
baseline from another build, omitted exact matched drop accounting, and was reported
with a different FLOPs denominator. It also did not contain the custom adjoint that
the issue comment said it stacked with.

Therefore 25.39% is not an achievable planning number. The record supports neither
sign for the original compute-only change. Settling that sign requires a new matched
rack A/B, but only after the treatment passes the EP64 compile/runtime/memory gate
that the reconstructed `G=4` path failed.

## What was recovered

### Exact submitted bundle

Iris records identify the positive job's bundle as
`0483b2f207323fb3cd79ec326b7592546aabb0812ef8c058be95bd6c8049cd43`. I downloaded it
through the read-only Iris controller path and verified that the ZIP's SHA-256 is the
bundle ID. The bundle contains the submitted
`lib/levanter/src/levanter/models/grug/ep_ragged_all_to_all.py`.

The recovered source defines `SCALE_A2A_BATCH_EXPERT_GEMMS`. It requires
`SCALE_A2A_PACK_DISPATCH=1` and `SCALE_A2A_PACK_COMBINE=1`. Packing already combines
the local-expert traffic into one dispatch and one combine all-to-all, both using
expert axis 1. The batching flag only changes the expert compute from a Python loop
over local experts to batched `jnp.matmul` calls. This is the implementation needed
to interpret the positive run.

The job declares base commit `fe21ea495` and dirty-tree hash `f28603c8a`. Its
configuration is:

- d5120, 48 layers, 256 experts, top-8 routing
- routed intermediate size 1280 and shared intermediate size 5120
- global batch 1024, sequence length 4096, sliding window 2048
- EP64 on 16 four-GPU GB200 nodes
- QB off, capacity factor 1.0, packed dispatch and combine on
- gather dispatch, custom distributed combine, Sonic collective derivatives,
  slot unpermute, no barrier, MuonH SYRK, scan, and full recomputation
- 120 configured steps

The job later reached training step 119 and attempted the step-120 checkpoint. The
Iris parent is recorded as killed rather than cleanly succeeded, so this was not a
clean terminal success. The 25.39% figure itself came from a step-59 harvest with
57 samples; later logs show the training continued.

### Later Git snapshot

Commit `98737aecfa5cd05b9bffe09c96754c96d7177f06`, titled
`[grug] Snapshot 27% EP64 research path`, contains the same compute-only batching
mechanism and is reachable from `origin/research/rav/7201-ep64-drop3` and related
tags. The commit is broad: 10 files and 1,932 insertions/66 deletions. It is not a
safe cherry-pick for a PR-sized leg-batching change, and its MoE source is not
byte-identical to the submitted bundle because the snapshot includes later
refactoring and other research changes.

No recovered code was copied into this branch. D-3 is a derisking item, the mechanism
already has a reachable Git object, and the exact submitted source is identified by a
content-addressed Iris bundle. Shipping the entangled snapshot or extracting a feature
whose sign remains unresolved would turn this finding into an unapproved feature.

## Search record

I checked:

- all reachable and unreachable objects with `git fsck --full --no-reflogs
  --unreachable`
- reflogs across all worktrees with `git reflog --all`
- the repository stash list, which is empty
- Marin worktrees under `.worktrees/` and `.claude/worktrees/`
- `.agents/logbooks/`, branch `AGENT_LOG.md` files, and the omnibus project records
- Iris job configs, durable logs, and source bundle metadata
- the cited GitHub issue bodies and comments using structured `gh` output

The object search found the committed snapshot, the reconstruction commits, and the
result-harvest commits. It did not find a second original implementation. The exact
Iris bundle closes the source-recovery question without relying on commit-message
claims.

## Verification

- The downloaded positive ZIP hashes to its Iris bundle ID. I read the recovered flag
  guard, dispatch/combine path, tensor shapes, and both compute branches in the
  submitted source rather than relying on the later snapshot.
- Iris job configs verified the environment, source provenance, model shape, EP
  degree, and bundle for the positive, reconstruction-control, `G=2`, and failed
  `G=4` jobs. Durable logs supplied the run lengths, bands, drops, losses, and analytic
  FLOPs denominators quoted below.
- Git diffs of `98737aecf`, `65e3ca50d`, and `0789a8482` verified that the original and
  reconstructed flags change different collective schedules.
- `./infra/pre-commit.py --all-files --fix` passed, including its Pyrefly and Markdown
  checks. `uv run pyrefly check` reported zero errors.
- Default `uv run pytest` selected 1,273 tests without overriding the repository
  marker expression. It finished with 1,252 passed, 17 skipped, 5 xfailed, and one
  failure in
  `test_grug_base_run_emits_expected_metrics_with_json_tracker`. A focused rerun
  failed identically: explicit CPU sharding differs between two operands to
  `jnp.concatenate` in `experiments/grug/base/model.py:227`. This branch does not
  change that code or test. I did not fix the unrelated baseline failure in this
  report-only task.

## Why the measurements disagree

| Dimension | Positive observation | Matched reconstruction |
| --- | --- | --- |
| Treatment | Packed transport plus compute-only batched GEMMs | `G=2` grouped batched experts, which also changes the collective schedule |
| Baseline | 24.04% custom-adjoint run from another build | 22.66% flag-off control from the same reconstruction build |
| Result | 25.39%, reported as +1.35pp | 19.00%, -3.66pp |
| QB | Off | On |
| Drops | No matched exact drop comparison; QB-off fixed-A2A runs were known to collapse | Matched: 8.78% control, 9.23% treatment at step 119 |
| Shape | d5120/L48/E256/top-8/i1280/shared5120, B1024/S4096/SW2048 | Same |
| EP | 64 | 64 |
| Run length | 120 configured; 25.39 harvested at step 59 from 57 samples; training reached step 119 | 120 configured; 119 MFU samples in each completed arm |
| Collective schedule | One packed dispatch and one packed combine in both flag states | Control loops per expert; `G=2` treatment uses two dispatches and two combines |
| Analytic denominator | 32.405B FLOPs/token | 34.430B FLOPs/token |
| Draws | One unmatched placement | One matched control/treatment placement |

The denominator difference is 6.25%, so the absolute MFU values across the two builds
should not be compared directly. The positive run's claimed within-family +1.35pp is
also not a valid A/B because the 24.04% baseline was a separate custom-adjoint build.
The positive job's environment does not set `SCALE_A2A_CUSTOM_ADJOINT`; this disproves
the contemporaneous claim that the two changes were independently measured and
stacked. That claim appears in
[#7279 comment 5080435482](https://github.com/marin-community/marin/issues/7279#issuecomment-5080435482).

The QB difference is not cosmetic. The same research thread records QB-off fixed-A2A
runs with 85–89% early routing collapse and large oscillation, while QB-on capacity
factor 1.0 held near 6% in a comparable run
([#7201 comment 5080459722](https://github.com/marin-community/marin/issues/7201#issuecomment-5080459722)).
The exact positive job did not produce a matched flag-off drop comparison, so those
figures describe the regime, not a measured drop rate for this exact treatment.

The reconstruction's bit-exact CPU comparison is still important. It showed zero
maximum absolute difference against its loop and identical drops. That rules out a
numerical-semantics explanation for its slowdown. It points to lowering, collective
schedule, memory pressure, or other build interactions. It does not make the original
and reconstructed performance paths equivalent because their collective schedules
differ.

## Judgment

I believe the -3.66pp measurement for the `G=2` reconstruction. It has a matched
control, equal run length, QB on, exact drops, identical analytic denominator, and
non-overlapping p10/p90 bands. It rejects that reconstruction as a performance change.

I do not believe 25.39% establishes that the original implementation is faster. The
run is real and its code is recovered, but the causal comparison is not: there is no
same-build packed-control arm, the claimed baseline used a different feature/build,
QB was off, drops were not matched, and only one placement was observed. The different
analytic denominator further prevents direct comparison with the reconstruction.

The record cannot choose a performance sign for the original compute-only mechanism.
It is possible for the recovered implementation to outperform its packed-loop control
while the reconstruction's `G=2` schedule regresses. It is also possible that the
positive observation was placement or build-stack noise. Until a matched test resolves
that, 25.39% must remain excluded from achievable-number planning.

## Experiment required to settle the original mechanism

No rack job was submitted for D-3.

### Gate before the A/B

First port only the recovered compute-only switch onto the composed D-2 build. The
control and treatment must have identical packed dispatch/combine operations,
collective counts, axes, and shapes; only the local-expert GEMM lowering may differ.
Before rack use:

1. Check values and gradients against the packed loop at the repository's MoE
   tolerances, with exact agreement in routed/drop counts.
2. Inspect lowered HLO to assert equal all-to-all count and layout in both arms and a
   batched expert GEMM only in treatment.
3. Run an EP64 target-shape gate. Treatment must compile, stay within the rack's HBM
   budget with at least 5% headroom, and produce 10 consecutive post-warmup steps
   without a worker restart, gang abort, incarnation mismatch, or allocator failure.

Failure of any gate falsifies deployability and ends the experiment. This condition is
necessary because the reconstruction's full `G=4` batching compiled for roughly
23 minutes and then lost a worker/wedged before producing a step. The logs did not
prove an OOM, so the cause remains “runtime or memory failure,” not “confirmed OOM.”

### Matched arms

Use one composed D-2 commit and change one flag:

- Control: packed dispatch/combine with the per-local-expert GEMM loop.
- Treatment: the same packed transport with
  `SCALE_A2A_BATCH_EXPERT_GEMMS=1`.

Both arms use d5120/L48/E256/top-8/i1280/shared5120, batch 1024, sequence 4096,
sliding window 2048, EP64, QB on, capacity factor 1.0625 with spill multiplier 3,
exact drop accounting, and the same D-2 custom-adjoint, padded-Muon, PGLE, overlap,
allocator, barrier, and rematerialization settings. Record tokens/s as the primary
metric and compute MFU afterward with one fixed denominator.

Run 350 steps and evaluate the last 100. Use three paired placement draws per arm,
keeping each control/treatment pair on the same allocation where possible and
alternating order `AB`, `BA`, `AB`. Record compile time, peak HBM, step time,
tokens/s, loss, and exact drop fraction.

### Pre-registered decision

The recovered claim corresponds to roughly a 5% throughput effect. Before running,
register:

- Confirmed: all three paired treatment deltas are positive, their median tail-100
  tokens/s gain is at least 4%, and the paired 95% bootstrap interval excludes zero.
- Falsified as the claimed win: the median paired gain is below 2%, any two paired
  deltas are non-positive, or treatment fails the runtime/memory gate.
- Inconclusive: a 2–4% median gain, a confidence interval spanning zero, or a loss/drop
  divergence that prevents a fidelity-matched comparison.

Require treatment's median tail drop fraction to stay within 0.2 percentage points
absolute of control and reject any arm with a materially divergent loss trajectory.
These thresholds leave an explicit inconclusive band instead of converting placement
noise into a win.

## Diff size, omissions, and uncertainty

The functional code diff is 0 lines. The assignment is report-only and `sequence.md`
does not give D-3 a code-size estimate, so there is no functional estimate to compare
against. This report is the only repository change.

I deliberately omitted:

- the 1,932-line entangled snapshot, because it is not a PR-sized extraction
- the reconstructed batching code, because it implements a different collective
  schedule and already has a matched negative result
- any new feature or compatibility path, because the performance sign is unresolved
- cluster execution, because D-3 prohibits submitting jobs

Known uncertainties:

- The exact positive run has no matched packed-loop control and no exact paired drop
  measurement.
- The `G=4` failure has no conclusive OOM signature; runtime and memory remain the
  competing explanations.
- The two branches use different analytic-FLOPs accounting. The observed logged
  denominators are authoritative for this report; earlier omnibus prose gives other
  absolute denominator values.
- Commit `98737aecf` preserves the mechanism but not the byte-exact submitted MoE
  source. The content-addressed Iris bundle is the authoritative source for the run.
- `derisking.md` and one part of `evidence.md` say the positive patch was never
  committed. `README.md` and another part of `evidence.md` say it was committed.
  Git and Iris support the latter.

The current omnibus documents already classify the idea as blocked or disputed, but
their contradictory provenance text should be corrected when the series is assembled.
No document should cite 25.39% as an achievable composed result.
