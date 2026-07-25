Coordinator: both your rack arms are done and the bake-off is decided.

Results:
1. ragged one-shot-off (ep25d2-rack-ragged-120): ran 120 steps clean, throughput/mean_mfu
   ~12.38% (vs fixed+adjoint 24.04% p50), final loss 5.708, moe/drop_fraction 0.433 at
   end-of-run (receiver-side capacity also drops under QB-off router collapse). Harvest in
   relay-results/ep25d2-rack-ragged-120-20260725.{log,summary}.
2. ring_cute EP64 (ep25d2-rack-ring-ep64-120): FAILED with OOM allocating 141.79GiB in
   jit_train_step at the operating point — ring cannot fit e256/EP64 as implemented.

Fleet context: QB-on is confirmed as the fidelity lever (d1: 3.4x drop cut at ~zero cost;
d4 120-step QB-on leg: drops 0.90 peak -> 0.083 at step 119, still above the 3% bar; QB-on
costs ~1.44pp, giving 22.60% honest production-config MFU).

Your job now: write the FINAL transport verdict in AGENT_LOG.md — the decision table
(fixed+gather+adjoint 24.04 QB-off / 22.60 QB-on; ragged ~12.4; ring OOM-141.79GiB),
caveats (single draw, mean vs p50, QB-off arms, ring OOM possibly recoverable with
remat/memory work but that is a project, not tuning), and your final answer to the
direction-2 question: does transport choice leave >=1pp on the table? Measured answer
appears to be no — fixed wins decisively at this shape; the one-shot-kernel-disabled
ragged control that #7279 asked for is now on record. Also state whether any further arm
is worth a rack (my prior: no; 2-rack cells moot). Commit nothing — tell me what to
commit as before. Do NOT submit more jobs.
