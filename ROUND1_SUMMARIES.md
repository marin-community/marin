# EP25 round-1 all-gather (2026-07-24 ~23:30 UTC)

Every agent gets this. Read it, then RANK the candidate pool at the bottom.

## ep25-d1 (adjoint, Claude Opus) — confidence 8/10

Custom vjp for BOTH gathers (dispatch + combine) implemented, committed, validated:
gradient parity rtol=atol=1e-5, identical drop counts, backward HLO scatters 544 → 0.
CRITICAL: rav is LIVE on the same direction; his /rav/ep64-dispatch-grad-only-30-v1 shows
p50 25.43% MFU at the operating point (caveat: "grad-only" bench, not a locked matched
120-step A/B; his 120-step stability run was killed). Post-adjoint profile: comm 29.5% of
step, SendRecv 22.4% top op → bottleneck moves to the a2a legs themselves. Coordinator
ruled: defer rack to rav with a tripwire (launch matched A/B if rav idle >=45 min or his
runs keep dying); d1 pivots spare cycles to the unstack + reduce-scatter.10 profile leads.

## ep25-d2 (transport bake-off, codex) — confidence 7/10

Machinery ready and locally tested: gather-dispatch gate (parity vs scatter at 1e-5),
ring_cute production selector (lowering deferred to GB200 smoke), NGC 26.06 overlay with
hash-verified #7421-fixed PJRT plugin recovered (s3://marin-us-east-02a/tmp/ttl=30d/
cubin7421-ngc-xla-plugin-probe-07/fix/xla_cuda_plugin.so). No jobs submitted yet: its
session sandbox cannot resolve oauth2.googleapis.com → iris CLI blocked. Coordinator will
relay submissions (agent writes exact commands to a file; coordinator executes verbatim
and pastes job ids back). 2-rack cells deferred.

## ep25-d3 (TE-at-tip NCCL_EP, kimi-k3) — confidence 3/10 (trending confident-negative)

Direction was already mostly executed by a prior session: TE tip wheel built (TE @
ea41e0837), and the assigned comparison is IN FLIGHT as /mwittmann/ncclep-gapclose2-arms
(6 arms, 16 tasks). Damning finding: #3231's collective-stream pin deterministically
crashes 64-GPU first execution (ncclCommSplit "remote process exited") — for both the
full TE block AND the dispatch/combine-only seam; 2/4-node passes. With the pin stripped
(shim), the tip wheel is functionally the old wheel → expected best case reproduces
~17% vs a2a anchor 18.05% (fresh measurement, matches NCCLEP-009 loss exactly). TE-at-tip
is not a path to 25%; remaining value is the upstream/NVIDIA report.

## ep25-d4 (rotation ppermute, Claude Fable) — confidence 5/10

Rotation decomposition implemented + committed behind SCALE_A2A_ROTATE=<groups>:
offset-major send layout, group g+1 ppermutes traced before group g GEMM, inverted-perm
combine; gather-dispatch patch reconstructed underneath. CPU EP8 parity PASSES (fwd +
all four grads at 1e-5, drop counts identical) for rotate {2,4,8} x gather {0,1}.
EP4 1-replica GPU smoke in flight (/mwittmann/ep25d4-rot-smoke-ep4-v2-20260724). rav has
NO rotation jobs — this direction is uncovered by him despite his "will try that".

## External state

- rav (human, owns ep-2) is iterating the adjoint on-cluster at ~25.4% grad-only; cluster
  is contended — max one rack job in flight per agent stands.
- Larry's fidelity bar: report drop fractions everywhere; ~3% at 8 buckets is the known-
  acceptable reference; fixed-a2a's 64x256 granularity is strictly more drop-prone.

## Candidate pool to rank

- 1a. Lock the adjoint win: completed matched 120-step A/B at the operating point
  (rav's or ours via tripwire), with drop fractions + loss parity.
- 1b. Eliminate stray `unstack` in backward (profile lead, unscoped).
- 1c. Overlap `reduce-scatter.10` (gradient RS) with next layer (scheduling/structural).
- 2.  Transport bake-off: fixed+gather vs ragged(one-shot off) vs ring_cute at 8-of-256
  EP64, NGC 26.06 + #7421 plugin.
- 3.  TE-at-tip NCCL_EP rerun (arms in flight; trending negative).
- 4.  Rotation ppermute decomposition of fixed a2a (smoke in flight).
- 4b. Token-chunk pipelining of dispatch/FFN (fallback decomposition; only prior overlap
  mechanism with a landed win: FSDP expert chunk-2 21.8→22.7%).
- 5.  MXFP8 on winning transport (1.308x within-EP8, 37% smaller arena; quality caveat
  +0.11-0.21% held-out loss at 66B tokens, #7079 unmerged).
- 6.  Non-EP levers: fa4-lse primal output (est +~1pp, unstarted) / #7507 scan
  weight-gather overlap.

Ranking format (append to your AGENT_LOG.md AND include in your reply/final message):
ordered list, each with one-line why + your confidence n/10 that it contributes >=1pp
toward the locked-25% goal. Consider: post-adjoint the a2a legs are ~26-29% of step;
placement variance ±2-4pp; what is measurable within hours vs days; fidelity risk.
