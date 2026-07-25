You are agent **ep25-d2** in the EP25 multi-agent research loop, round 6. Work ONLY in
this worktree: /home/marin/projects/marin/.worktrees/ep25-d2-bakeoff (your round-1-5
transport-bake-off work is committed here; this is a NEW direction).

FIRST read `ROUND6_BRIEF.md` here — goal (>=25% p50 MFU at the d5120 8-of-256 EP64
operating point WITH <3% drops), the new all-QB-on measurement protocol, reference
numbers, and fleet rules. Also skim AGENT_LOG.md for your own prior context.

## R6-4: MXFP8 expert GEMMs at the operating point

Compute is ~70% of the step post-adjoint; MXFP8 grouped kernels are the biggest
unmeasured compute lever. Prior art to mine (read-only; branches are local):
- Issue #7282 (`gh api repos/marin-community/marin/issues/7282 --jq .body` and its
  comments): grouped MXFP8 CuTeDSL kernel hit 2.2 PF/s green via cutlass_call on
  Blackwell; dense mxfp8 was shelved (per-tensor wins on sm100); measured 1.308x
  within-EP8 with a 37% smaller memory arena.
- Local branches `research/mcwitt/7282-uniform-mxfp8` and `research/mcwitt/7282-*`
  (access via `git log`/`git show`/`git checkout <branch> -- <path>` from THIS worktree;
  do not read other worktree directories). PR #7079 (fp8 loss-val machinery) is unmerged.
- Quality caveat, flagged not resolved this round: #7271 measured +0.11-0.21% held-out
  loss at 66B tokens. Your deliverable is SPEED at the operating point + short-horizon
  fidelity (drops parity + 120-step loss delta), with the long-horizon quality gate
  stated as the adoption condition.

Plan:
1. Extract the grouped MXFP8 GEMM path from the 7282 branches and integrate it into the
   grug standalone MoE expert GEMMs (w13/w2 ragged/grouped calls in
   lib/levanter/src/levanter/grug/_moe/) behind an env gate (e.g. SCALE_MOE_MXFP8=1),
   on top of gather-dispatch + custom adjoint (cherry-pick from agent/ep25-d1-adjoint:
   45ce02d20 c9e30f848 4fbc89152 2d4a87395 — resolve conflicts with your NGC work or
   start a clean branch off rav/ep-2 and cherry-pick, your call; `git add -f` for new
   files under lib/).
2. Kernel-level numerics: mxfp8 vs bf16 GEMM error within the known-good envelope from
   7282; drop counts identical (quantization must sit after routing).
3. EP4 smoke on GB200 (stock toolchain first; your NGC overlay only if the kernel needs
   a newer stack — note which).
4. Matched rack pair, QB-on cf1.0, 120 steps, drops on: bf16 GEMMs control vs MXFP8
   GEMMs. Reference band for the control: 22.595/22.002.
5. Report: p50 A/B, drop parity, 120-step loss delta, arena/memory observation (does the
   37% arena shrink reproduce — it matters for cf1.15 headroom), and the projected
   compliant-config MFU if MXFP8 stacks with cf1.15.

Cluster submissions: if your sandbox still cannot reach iris (DNS), write exact commands
to EP25_D2_RELAY_COMMANDS.md (fresh job names, ep25d2-mxfp8-*) and STOP — the coordinator
executes and harvests into relay-results/ as before. Commits: if the index lock still
blocks you, list exactly what to commit in AGENT_LOG.md.

Rules: AGENT_LOG.md check-ins every ~15 min with Confidence n/10; commit locally (or via
coordinator); NEVER push or write to GitHub; no job mutations on jobs you didn't submit;
resubmit setup flakes with -vN. If kernel integration stalls >3 focused attempts, deliver
a scoping assessment (what's missing, est effort) instead of burning hours.
