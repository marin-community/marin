## TL;DR

Complete the 280-policy Delphi TPP40 two-phase swarm across East5 and Europe. Twenty-seven East5 policies are complete. The frozen candidate assignment gives East5 126 policies, including two resumable phase-boundary checkpoints, and Europe 127 fresh policies. Production launch remains gated on one paired cross-accelerator trajectory with absolute BPB differences at most 0.002 for phase-boundary Uncheatable, endpoint Uncheatable, and endpoint Table 9.

## Description

The original East5-only parent made slow progress because v5p-8 capacity was scarce. The multiregion plan uses region-local datasets, validation caches, state, checkpoints, and evaluations. It does not transfer training corpora across regions. East5 keeps the 27 completed and two resumable lineages; Europe receives disjoint fresh policies and trains on v6e-8.

The assignment covers all 280 run orders exactly once. A content-addressed path audit resolved all 153 East-assigned lineages: 27 completed rows have successful executor status and final artifacts, rows 27 and 29 have permanent phase-boundary checkpoints, and 124 rows are fresh. Europe production remains blocked until run order 2 passes the paired numerical and idempotence gate.

## Hypothesis or Goal

Complete the preregistered TPP40 panel without changing its policies or training configuration, while accepting only cross-accelerator differences that remain within the frozen 0.002 BPB paired threshold.

## Status

The Europe bridge trajectory for run order 2 is training. The production assignment and commands have passed locality and dry-run checks but are provisional until the bridge gate passes. After acceptance, the assignment will be re-materialized and required to retain byte-identical and semantic hashes before the East5 and Europe parents are submitted.

## Links

* Logbook: https://github.com/marin-community/marin/blob/calvin/swarm-olmo3-regmix-test/.agents/logbooks/delphi-tpp40-multiregion.md
* Fieldbook: `exp_01kz3nq7y7mp3a51kz26cvv4tr`
* W&B Report: pending
* Important updates: pending

## Decision Log

* 2026-08-30: Use one paired trajectory, run order 2, as the production bridge gate. East5 evaluation capacity may delay the paired result but does not authorize an unpaired launch.
* 2026-08-30: Keep all datasets and mutable artifacts region-local. Do not use Storage Transfer Service or copy training corpora across regions.
* 2026-08-30: Freeze a disjoint 126-row East5 and 127-row Europe assignment after accounting for 27 completed East5 rows.

## Conclusion

Pending the run-order-2 bridge gate and production launch.
