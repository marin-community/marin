# Command-buffer capture stabilization

Shuttle accepts command-buffer performance only after the exact timed launch
topology reaches a measured callback plateau. Static custom-call sites determine
which callbacks are expected. They do not predict how many command-buffer
instances XLA records for runtime streams or execution slots.

## Static capture sites

`CaptureSiteManifest` is derived from the final optimized HLO for one executable.
Each manifest entry records:

- the final-HLO custom-call target;
- the generated handler counter that instruments it;
- the number of target occurrences in the module.

Several targets may share one handler counter. Their occurrences are summed to
derive logical handler calls per executable execution. This supports a complete
composition with many generated target families as well as a two-call component.
The compiler rejects a registered target absent from final HLO, an executable
using another executable's manifest, or one target mapped inconsistently across
variants.

An uninstrumented reference uses an explicit empty manifest. A callback observed
while that variant runs is rejected.

## Topology-matched stabilization

The acceptance policy is fixed before device allocation:

```text
maximum stabilization rounds:       8
required consecutive quiet rounds:  2
timed callback allowance:            0
```

One stabilization round traverses every declared counterbalanced order once.
Each variant uses the same iteration burst, synchronization boundary, argument
liveness, and ordering used by one timed sample. Handler counts are checkpointed
around every variant. A round is quiet only when every delta is zero.

Timing begins after two consecutive complete quiet rounds. Failure to reach this
plateau within eight rounds rejects the result. Every handler present in the
static manifest must also have a positive count before timing.

The timed phase retains per-variant checkpoints. Any callback after the plateau
rejects the result as `steady_state_recapture`; the harness does not discard a
sample or allow one callback per order. Raw timing distributions and callback
checkpoints are serialized with a pending assessment before the final gate runs.

The acceptance predicate is:

```text
manifest valid
and every manifest handler observed
and two consecutive quiet rounds reached within eight rounds
and callback checkpoints continuous and internally consistent
and no callback attributed to an uninstrumented variant
and every timed callback delta is zero
and workload correctness and determinism pass
```

Repeated callback deltas equal to statically derived logical calls in two
consecutive stabilization rounds are classified as `per_logical_call_fallback`.
A finite set of initial recordings is allowed to settle and is not confused with
fallback.

## Why the policy uses measured stabilization

OpenXLA records a `CommandBufferThunk` on its first execution on a given GPU
stream and replays it thereafter. A synchronized one-call warmup may therefore
touch fewer runtime instances than a timed burst that enqueues many executions
before blocking. Final HLO supplies the static call sites, but the number of
stream-specific recordings is a runtime property.

TLTC-XLA-068 observed this distinction: synchronized warmups did not establish a
complete plateau for a 1,000-enqueue burst. The first timed generated phase added
two callbacks per handler and every later phase added zero. That run remains
rejected under its precommitted one-callback-per-order policy. It did not execute
the stabilization protocol described here and cannot be reclassified.

Reference: [OpenXLA, From HLO to Thunks](https://openxla.org/xla/hlo_to_thunks).
