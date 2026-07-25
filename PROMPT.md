You are agent **ep25-d2** in a four-agent fan-out coordinated from a parent session.
Your worktree (work ONLY here): /home/marin/projects/marin/.worktrees/ep25-d2-bakeoff
(branch agent/ep25-d2-bakeoff, based on rav/ep-2 @ fe21ea495).

FIRST: read `EP25_BRIEF.md` in this directory — it has the goal (>=25% MFU), the exact
baseline iris submission, cluster gotchas, logging/commit rules, and the check-in format
(append to AGENT_LOG.md at least every 15 minutes with findings + Confidence n/10).

## Your direction: matched transport bake-off at the production-candidate shape

Direction (2) from the ranked comment (read it: `gh api
repos/marin-community/marin/issues/comments/5074952738 --jq .body`). The fixed-a2a
transport was adopted without a matched ragged control, and ring_cute — which won every
backend ladder at e64/e128 — has NEVER been run at e256/top-8/EP64. Your job is the
decision matrix that settles the transport choice:

Arms (all at MuonH d5120 · 8-of-256 · 48L · EP64 · batch 1024 · seq 4096, one rack):
1. fixed-a2a + gather dispatch (reconstruct the uncommitted gather patch from comment
   5073017396 — it modifies lib/levanter/src/levanter/grug/_moe/ep_ragged_all_to_all.py,
   gated on SCALE_A2A_GATHER_DISPATCH=1). Baseline reference: 20.558% p50.
2. ragged all_to_all with `--xla_gpu_unsupported_use_ragged_all_to_all_one_shot_kernel=false`
   (i.e. SCALE_A2A_FIXED unset, one-shot kernel disabled via XLA_FLAGS).
3. ring_cute (EP4 and/or EP8 sub-axis variants as the code supports) at this shape.

Toolchain: NGC JAX 26.06 with the #7421 fix (openxla/xla 4c1b00509e64). Prior NGC 26.06
launch machinery exists on branch research/codex/7421-ngc-7279 (local worktree
/home/marin/projects/marin/.worktrees/issue-7421-ngc-7279 — read-only reference). If NGC
26.06 setup stalls you >45 min, fall back to the stock baseline toolchain for a
same-toolchain 3-way comparison and say so in the log — a matched comparison on one
toolchain beats an unmatched one on two.

Protocol:
- 120-step runs, SCALE_DISABLE_CHECKPOINT=1, json_logger; p50 MFU at the 2.5 PF/s
  denominator; record dropped-token counts per arm.
- Placement variance is ±2–4pp: at least 2 draws per arm before concluding; run arms
  back-to-back interleaved rather than all-of-A-then-all-of-B.
- ONE rack-scale job in flight at a time (three peer agents share the cluster). 2-rack
  arms are DEFERRED until the coordinator approves capacity — note them as pending.
- Smoke each arm at 1 replica (4 GPUs) before the rack run.

Also weigh and report the overlap ceiling, not just current speed: the pipelined
decomposition (direction 4, being probed by a peer agent) is only implementable on the
fixed-capacity layout — a ragged win at parity is weaker than it looks.

Deliverable per round: a table (arm × draw → p50/p10/p90 MFU, tok/s, drop count) plus
your ranking of transports and Confidence n/10, appended to AGENT_LOG.md and committed
locally. NEVER push or write to GitHub.
