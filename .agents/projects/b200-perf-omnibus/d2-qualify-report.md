# D-2 qualification report

## TL;DR

Numerical qualification is blocked before the comparison can run. The current
ARM64 `iris-task:latest` image exits with `exec format error` in Iris's
`stage-workdir` init container, and that init container ignores the per-job
`--task-image` override. No numerical result was observed. Per the fail-stop
protocol, the compile smokes, fresh PGLE capture, and all D-2 rack draws were not
run. D-2 is not qualified and the Phase D composition question remains
unanswered.

## Immutable build

The starting handoff is `0b305d520` on `agent/deri-d2-build`. The qualification
probe was committed at `0e9bfb9f1`; its explicit-sharding metric reduction was
fixed at `fcbf431b0`. No immutable D-2 submission SHA was established because
qualification stopped before rack submission.

## Pre-registration

The additive ledger predicts approximately **22.5% MFU**: 20.7% for C2's best
compliant point plus 1.78pp for padded Muon, less any gain that fails to transfer
to this architecture and composed layout.

The prediction is falsified if the median steady-tail p50 MFU across the three
placement draws is below **21.5%** at matched drop fraction and LR position, or
if the loss trajectory is unstable. A result from 20.7% through 21.5% is partial
transfer, not composition at the predicted scale.

## Numerical qualification

The GPU comparison will use FP32 parameter/update arrays, matching the D-2
`params=float32,compute=bfloat16,output=bfloat16` policy; Newton-Schulz casts
internally to BF16. It will use the D-2 matrix dimensions 5120 and 1280 and cover
expert gate/up and down orientations plus tall, wide, and square non-expert
stacks on a four-GB200 `data=2, expert=2` mesh.

The pre-registered structural-versus-numerical gate is:

- all values are finite;
- relative L2 difference is at most `2e-3` at every tested NS depth;
- cosine similarity is at least `0.99999`;
- relative-L2 difference at NS5 is at most twice the NS2 value;
- per-step relative divergence of the paired synthetic loss trajectories is at
  most `1e-4`.

These are qualification criteria, not changes to any repository test tolerance.
No criterion will be changed after reading the result.

Results: no max/mean absolute differences or loss trajectory were produced.

The first direct production attempt failed before startup on ARM64 GB200 node
`s4bk6j84`: `stage-workdir` used
`iris-task@sha256:cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425`
and exited with `exec /usr/local/bin/python: exec format error`. One retry
reached four GB200s using a working image and exposed a qualification-harness
bug: `jnp.vdot` tried to flatten an explicitly sharded 4D array without an
output sharding. Replacing the flattening dot/norm operations with
shape-preserving reductions passed on an explicit four-device CPU mesh.

Every corrected-bundle attempt then returned to `s4bk6j84` and failed before
bundle fetch with the same bad ARM64 image. `--task-image` changed the main
container but not `stage-workdir`. `--zone 136` did not change placement. A
forced `imagePullPolicy: Always` helper confirmed that the registry's current
ARM64 `latest` manifest resolves to the bad digest. A final pod-only image patch
was rejected by Kueue because it no longer matched the admitted Workload
template; the stranded Iris job was stopped.

Verdict: no structural-versus-numerical determination is possible. No Muon
comparison ran after the harness fix. Qualification can resume when
`iris-task:latest` has a working ARM64 manifest or the Kubernetes backend uses
the per-task image for `stage-workdir`.

## GB200 compile and HLO smoke

| `SCALE_MUON_SYRK` | 4D `(L,E)->LE` merges | padded inbound reshards | replicated padded outbound | involuntary-remat warnings |
|---:|---:|---|---:|---:|
| 0 | not run | not run | not run | not run |
| 1 | not run | not run | not run | not run |

The numerical gate did not pass, so no compile job was submitted.

## Fresh PGLE capture

Not run. The numerical gate did not pass, so no capture job was submitted and
no PGLE on/off decision was made.

## D-2 placement draws

The denominator is fixed at 2.5 PFLOP/s per GB200. Every MFU will be reported
with tok/s, run length, and a drop fraction from the same LR-schedule position.

| draw | job | steady-tail tok/s | steady-tail p50 MFU | matched-LR drop fraction | loss trajectory |
|---:|---|---:|---:|---:|---|
| 1 | not submitted | — | — | — | — |
| 2 | not submitted | — | — | — | — |
| 3 | not submitted | — | — | — | — |

## Do the Phase D gains compose?

Unknown. The 22.5% prediction and 21.5% falsification threshold were committed
before any GPU result was seen, but no D-2 placement result exists. The Phase D
gains cannot be called composed or falsified from this qualification attempt.
