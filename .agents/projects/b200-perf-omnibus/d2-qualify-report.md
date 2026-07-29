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

## Decision-1 warning A/B — 2026-07-28

The warning is a regression relative to the pre-reconciliation
`f53f781ce` baseline. Both baseline settings emitted zero
`spmd_partitioner.cc:668` involuntary-full-rematerialization warnings; both
composed settings emitted one. The pre-registered zero-warning gate is therefore
not mis-scoped for this four-GPU mesh.

The baseline ran from detached harness commit `9f46c3ca0`, whose parent is
`f53f781ce`. The only added file is the qualification harness. It selects the
harness's frozen `f53f781ce` reference functions and asserts the baseline
structure; `lib/levanter/src/levanter/optim/grugmuon.py` is byte-identical to
`f53f781ce`. Both arms otherwise used the same probe cases, `data=2, expert=2`
mesh, four GB200s, JAX 0.10.1, allocator, XLA flags, resources, and SYRK
settings.

| arm | `SCALE_MUON_SYRK` | job | warnings | offending path |
|---|---:|---|---:|---|
| composed `8779abc42` | 0 | `/mwittmann/d2-muon-compile-syrk0-0728-1655` | 1 | `jit(current)/vmap()/convert_element_type` |
| composed `8779abc42` | 1 | `/mwittmann/d2-muon-compile-syrk1-0728-1657` | 1 | `jit(current)/vmap()/convert_element_type` |
| baseline `f53f781ce` | 0 | `/mwittmann/d2-muon-baseline-f53-syrk0-0728-1709` | 0 | none |
| baseline `f53f781ce` | 1 | `/mwittmann/d2-muon-baseline-f53-syrk1-0728-1710` | 0 | none |

The two composed warnings are identical. XLA changes
`%convert_element_type.21`, shape `f32[1,5120,1280]`, from
`{devices=[4,1,1]<=[4]}` to
`{devices=[1,2,1,2]<=[4] last_tile_dim_replicate}`. Both complete baseline logs
contain no offending path because their counts are zero. All four jobs
succeeded and produced finite expert and padded non-expert outputs.

The count A/B establishes a post-`f53f781ce` regression, but the warned
transition does not point to any of the three `888fff904` mechanisms. The
no-merge layout affects the expert case, while this warning is in
`nonexpert_tall`. SYRK threading is excluded because the warning is identical
with SYRK disabled and enabled. The inbound two-hop ends at the warning's
four-way leading-axis source sharding. Its second hop would instead move from a
two-way leading-axis sharding to `{devices=[4,1,1]<=[4]}`.

The warning is on the direct padded outbound reshard from the four-way
leading-axis layout to the parameter layout. That reshard was added by
`5c031c31b` before the Decision-1 reconciliation and is still blamed to that
commit at `grugmuon.py:681`; `888fff904` changed the inbound path at lines
670-672. The baseline first reshards the padded result to
`P(None,None,None)` and then restores the parameter layout outside the helper,
and emitted zero warnings. No code was changed in this A/B.

Recommendation: do not start fresh PGLE capture or D-2 rack draws. Keep the
existing four-GPU zero-warning criterion unchanged: both SYRK settings must
return to zero before rack work proceeds. The first real EP64 full-rack compile
must still satisfy the original zero-warning check before any placement result
is accepted. Re-scoping the toy-mesh gate to “no more than baseline” would not
change it because the measured baseline is zero.

## Production-mesh discriminator — 2026-07-28

The compile warning is mesh-scoped. The leading-axis-only
`data=1, expert=4` mesh emitted zero warnings at both SYRK settings, while the
same-session `data=2, expert=2` positive controls reproduced one warning each.
This confirms the pre-registered mesh hypothesis at four GPUs and removes the
`data=2, expert=2` warning as a blocker for D-2. It does not waive the real
EP64 compile gate.

The discriminator used harness commit `12ce482bb`. The harness added an
explicit mesh selector and derived the structural expectation from that mesh;
the probe cases, realistic 5120/1280 dimensions, compile path, four-GB200
resource request, JAX 0.10.1 environment, allocator, and XLA flags remained
unchanged. All four completed jobs produced finite expert and
`nonexpert_tall` outputs.

| mesh | `SCALE_MUON_SYRK` | job | warnings |
|---|---:|---|---:|
| `data=1, expert=4` | 0 | `/mwittmann/d2-muon-mesh-d1e4-syrk0-0728-2120` | 0 |
| `data=1, expert=4` | 1 | `/mwittmann/d2-muon-mesh-d1e4-syrk1-0728-2120` | 0 |
| `data=2, expert=2` | 0 | `/mwittmann/d2-muon-mesh-d2e2-syrk0-0728-2120` | 1 |
| `data=2, expert=2` | 1 | `/mwittmann/d2-muon-mesh-d2e2-syrk1-direct-r1-0728-2127` | 1 |

The first three jobs used the default-priority federated route through `marin`
to `cw-us-east-08a`. The fourth federated submission,
`/mwittmann/d2-muon-mesh-d2e2-syrk1-0728-2120`, remained pending at `Queued for
peer cw-us-east-08a to report free capacity`. It was stopped without running,
then resubmitted through the shared protocol's direct-cluster fallback at
default priority. No shared infrastructure or other user's job was changed,
and no `stage-workdir` failure occurred.

On `data=1, expert=4`, the padded jaxpr changed from
`P('expert',None,None)` directly to the parameter spec
`P(None,'data','model')`. XLA compiled that leading-axis all-gather without an
involuntary-full-rematerialization warning. On `data=2, expert=2`, both controls
reproduced the earlier `f32[1,5120,1280]` warning from
`{devices=[4,1,1]<=[4]}` to
`{devices=[1,2,1,2]<=[4] last_tile_dim_replicate}`.

This result does not establish that `497423bc6` avoids physical replication at
the real D-2 mesh. When `data` and `model` both have size one,
`P(None,'data','model')` is physically replicated across the expert devices
even though the jaxpr contains no literal `P(None,None,None)` reshard. The zero
warning establishes only that XLA can lower the direct leading-axis all-gather
without its involuntary-remat fallback. Peak memory and materialized-shape
comparisons were not run because the warning vanished, so the composed build
has not independently reproduced `497423bc6`'s recorded +1.78pp gain or proven
a replication-memory reduction.

Do not implement the outbound two-hop fix from this result. D-2 rack draws may
proceed only after the exact composed build passes a full-rack compile on the
real `replica_dcn=1, data=1, expert=64, model=1` mesh at both
`SCALE_MUON_SYRK=0` and `1`. That gate requires complete-log counts of zero
`spmd_partitioner.cc:668` warnings, finite realistic D-2 outputs, no expert
`(L,E)->LE` merge, the single `P('expert',None,None)` padded inbound reshard,
and restoration to the real parameter shardings. Any warning on that mesh
blocks D-2 and triggers the outbound two-hop fix and its before/after
verification. Until this full-rack compile passes, do not capture fresh PGLE or
start a D-2 placement draw.
