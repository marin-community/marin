# D-2 qualification report

## TL;DR

Qualification is in progress. No D-2 rack draw has been submitted.

## Immutable build

The starting handoff is `0b305d520` on `agent/deri-d2-build`. The immutable
qualification and submission SHA will be recorded after the qualification probe
is added and validated.

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

Results: pending.

## GB200 compile and HLO smoke

| `SCALE_MUON_SYRK` | 4D `(L,E)->LE` merges | padded inbound reshards | replicated padded outbound | involuntary-remat warnings |
|---:|---:|---|---:|---:|
| 0 | pending | pending | pending | pending |
| 1 | pending | pending | pending | pending |

## Fresh PGLE capture

Coverage audit and PGLE on/off decision: pending.

## D-2 placement draws

The denominator is fixed at 2.5 PFLOP/s per GB200. Every MFU will be reported
with tok/s, run length, and a drop fraction from the same LR-schedule position.

| draw | job | steady-tail tok/s | steady-tail p50 MFU | matched-LR drop fraction | loss trajectory |
|---:|---|---:|---:|---:|---|
| 1 | pending | pending | pending | pending | pending |
| 2 | pending | pending | pending | pending | pending |
| 3 | pending | pending | pending | pending | pending |

## Do the Phase D gains compose?

Pending all qualification gates and three terminal draws.
