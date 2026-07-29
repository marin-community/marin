# D-2 qualification report

## TL;DR

The composed Muon update passed the numerical gate on four GB200s. All 15
realistic-shape point comparisons were exactly equal to the pre-reconciliation
path, and the largest relative loss divergence across 25 paired synthetic
steps was `1.63e-7`.

The compile/HLO gate then failed. Both `SCALE_MUON_SYRK=0` and `1` preserved the
intended structure and executed finite results, but each emitted one XLA
involuntary-full-rematerialization warning in the padded non-expert path. The
gate requires zero. Per the fail-stop protocol, fresh PGLE capture and all D-2
rack draws were not run. D-2 is not qualified, and the Phase D composition
question remains unanswered.

## Immutable build

The starting handoff is `0b305d520` on `agent/deri-d2-build`. The qualification
probe was committed at `0e9bfb9f1`; its explicit-sharding metric reduction was
fixed at `fcbf431b0`. The successful numerical and compile bundles used
`8779abc42`. No immutable D-2 submission SHA was established because
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

The retry after `fcbf431b0` first failed before startup on node `s4bk6j84`.
`stage-workdir` used
`iris-task@sha256:cfe4e8dd08f6d43076ade21a2a018ef7c1616356e46960f0a1ebc66434bb3425`
and exited with `exec /usr/local/bin/python: exec format error`. This is
node-local or intermittent, not a registry-wide blocker: other GB200 legs were
running concurrently, and running and failed pods have overlapping image
digests. The next identical attempt landed on `sjxsxs64` with
`iris-task@sha256:29ec7e8d4702faa36b0006ee34fd084e3e634a541e0736475446051bba091524`
and completed.

The successful job was
`/mwittmann/d2-muon-num-syrk1-r7-0728-1648`. It used JAX 0.10.1 on a
`data=2, expert=2` explicit mesh, FP32 parameter and update arrays, BF16
Newton-Schulz internals, D-2 matrix dimensions 5120 and 1280, and SYRK enabled.

| orientation | NS depths | max abs | mean abs | relative L2 | exact fraction |
|---|---|---:|---:|---:|---:|
| expert gate/up | 0, 2, 5 | 0 | 0 | 0 | 1.0 |
| expert down | 0, 2, 5 | 0 | 0 | 0 | 1.0 |
| non-expert tall | 0, 2, 5 | 0 | 0 | 0 | 1.0 |
| non-expert wide | 0, 2, 5 | 0 | 0 | 0 | 1.0 |
| non-expert square | 0, 2, 5 | 0 | 0 | 0 | 1.0 |

All values were finite. Cosine similarities ranged from
`0.9999998807907104` to `1.0000001192092896`. All 25 paired update comparisons
were exactly equal. The largest relative loss divergence was
`1.627454638974007e-7` at expert-down step 3; the other 24 steps were zero.

Verdict: all five structural-versus-numerical criteria pass. The composed and
pre-reconciliation Muon paths are numerically equivalent at the tested
realistic GPU shapes. There is no structural divergence signal.

## GB200 compile and HLO smoke

| `SCALE_MUON_SYRK` | 4D `(L,E)->LE` merges | padded inbound reshards | replicated padded outbound | involuntary-remat warnings |
|---:|---:|---|---:|---:|
| 0 | 0 | `P('data',None,None)` then `P(('data','expert'),None,None)` | 0 | **1** |
| 1 | 0 | `P('data',None,None)` then `P(('data','expert'),None,None)` | 0 | **1** |

The jobs were `/mwittmann/d2-muon-compile-syrk0-0728-1655` and
`/mwittmann/d2-muon-compile-syrk1-0728-1657`, both on `sdxsxs64`. Iris marked
both succeeded. The non-SYRK expert and padded non-expert outputs were finite
and restored `P(None,'expert','data','model')` and
`P(None,'data','model')`, respectively. The previously unexecuted EP SYRK
branch also compiled on Blackwell, produced finite outputs, and restored the
same shardings.

Both complete logs contain one `spmd_partitioner.cc:668` involuntary full
rematerialization warning. The warning occurs in the padded non-expert
`jit(current)/vmap()/convert_element_type` path while changing from
`{devices=[4,1,1]<=[4]}` to
`{devices=[1,2,1,2]<=[4] last_tile_dim_replicate}`.

Verdict: the structural checks and first Blackwell SYRK compile pass, but the
zero-warning gate fails for both settings. The isolated Muon smoke failed
before a full rematerialized training-step smoke was justified. Qualification
stopped here.

## Fresh PGLE capture

Not run. The compile/HLO gate failed, so no capture job was submitted and no
PGLE on/off decision was made.

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
before any GPU result was seen. The numerical gate passed, but the compile gate
failed before any D-2 placement result existed. The Phase D gains cannot be
called composed or falsified from this qualification attempt.
