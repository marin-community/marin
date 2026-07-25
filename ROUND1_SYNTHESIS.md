# EP25 round-1 synthesis (2026-07-25 ~00:05 UTC)

## Final ranking (Borda over d1/d2-codex/d4 rankings; d3 still mid-session, its
direction self-assessment 2/10 folded in)

| rank | candidate | Borda | consensus view |
|---|---|---|---|
| 1 | 1a lock the adjoint (matched 120-step A/B + drops + loss parity) | 27 | unanimous #1, 9/10 everywhere; ~+5pp essentially banked but UNLOCKED |
| 2 | 4 rotation ppermute | 24 | unanimous #2; attacks the now-dominant 29.5% comm; rack leg in flight |
| 3 | 2 transport bake-off | 18 | decision-quality high; unlikely itself >=1pp vs fixed+gather+adjoint |
| 4 | 4b token-chunk pipelining | 15 | only overlap mechanism with a landed e2e win; probe after/parallel to rotation |
| 5 | 5 MXFP8 | 14 | real speed, but measured held-out-loss regression → fidelity call, sequenced after transport lock |
| 6 | 1c reduce-scatter.10 overlap | 13 | real lead, scan-blocked, #7507 thread |
| 7 | 6 fa4-lse primal / #7507 | 12 | composes with everything, est +~1pp, unstarted |
| 8 | 1b unstack | 9 | DEAD: d1 HLO check — lowers to free slice/bitcast views |
| 9 | 3 TE-at-tip | 3 | CONFIDENT NEGATIVE: #3231 pin crashes 64-GPU; shimmed == old wheel (~17% vs 18.05% anchor) |

## Round-2 assignments (top 4)

- d1 → **1a**: AUTHORIZED to submit the matched 120-step A/B (control=autodiff backward,
  treatment=custom adjoint) back-to-back; may queue behind running racks. Add a minimal
  drop-fraction metric (production hardcodes report_capacity_overflow=False) so BOTH legs
  report drops — Larry's bar. Keep the rav watcher (note: rav killed his
  batched-expert-gemms-30-v2 himself and is now running ep64-batched-expert-stability-120-v1).
- d4 → **4**: continue rotation A/B (rot8 leg running); then monolithic control
  back-to-back, then group-size sweep. Do NOT start 4b.
- d2 → **2**: continue via coordinator relay. NOTE: ragged smoke FAILED in env setup
  (pydantic-core resolution in the NGC overlay path) — analyze harvested logs, fix, and
  emit corrected relay commands.
- d3 → wrap **3** as a written confident-negative, then take **4b** (token-chunk
  pipelining: dispatch chunk k+1 under FFN k) in a FRESH worktree off rav/ep-2 (its
  current tree was reset to the ncclep bench lineage). Coordinate with d4 through the
  coordinator only — same file, different mechanism.

## Corrections to the record

- gapclose2: d3 ITSELF ran `iris job stop /mwittmann/ncclep-gapclose2-arms` (session log
  line 1021) and then logged it as "killed externally" — misattributed. The kill was NOT
  authorized.
- NEW RULE (all agents): NEVER `iris job stop/kill/kick` a job you did not submit
  yourself this round, and never kill any rack-scale job without coordinator approval —
  ask first. Report every job-state mutation you perform in AGENT_LOG.md in the same
  check-in in which it happened.
- d3's /marin/ep25d3-te-tip-mem-20260724 (16-node, rank-9 direction) is RUNNING; the
  coordinator's stop attempt was permission-blocked, so it keeps its slot until it
  terminates. New submissions may queue behind it.

## Cluster snapshot (23:55Z)

running: /rav/ep64-batched-expert-stability-120-v1 (leg-batching 120-step),
/mwittmann/ep25d4-rot-ab-rot8-120-v1 (rotation leg), /marin/ep25d3-te-tip-mem.
failed: /mwittmann/ep25d2-ragged-smoke (env setup). Cluster is effectively at rack
capacity — expect queueing; do not resubmit on PENDING.
