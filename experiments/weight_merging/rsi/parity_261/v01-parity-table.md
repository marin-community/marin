# Task261 v0.1 multi-host parity

One reserved v4-32, four hosts / 16 chips, BF16 TP16/PP1; all four tasks succeeded and resources released. Server context 73728, effective evaluator cap 73664, concurrency 16, Evalchemy c1d5c62c and task249 campaign YAMLs/client. Verified 7 global layers / 19 local layers and four KV groups.

| Task | v4-32 | Single-host TPU | H100 | n | Combined z vs H100 | Verdict | Eval seconds | Generated tokens | Stop / length |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| math500 | 0.9580 | 0.9620 | 0.9620 | 500 | -0.32 | PARITY | 1286.1 | 1,039,461 | 500 / 0 |
| gsm8k-0shot | 0.9303 | 0.9295 | 0.9140 | 1319 | +1.56 | PARITY | 358.5 | 305,849 | 1319 / 0 |
| boolq | 0.8339 | 0.8336 | 0.8320 | 3270 | +0.21 | PARITY | 142.0 | 6,540 | 0 / 6540 |
| piqa | 0.8210 | 0.8205 | 0.8205 | 1838 | +0.04 | PARITY | 59.5 | 3,676 | 131 / 3545 |
| humanevalplus | 0.8902 | 0.8841 | 0.8720 | 164 | +0.51 | PARITY | 391.1 | 237,458 | 164 / 0 |
| aime24 | 0.7300 | 0.7267 | 0.7300 | 30 | +0.00 | PARITY | 11577.2 | 3,617,491 | 291 / 9 |

All task outcomes succeeded with full coverage. AIME is 10 repeats of 30 questions (300 responses), not 300 independent questions; the z calculation uses question-level binomial SEM and the supplied H100 SEM. Its across-repeat SEM is 0.0110. Single-host values are task249 references (some rounded).

BoolQ/PIQA use loglikelihood: their one generated token per choice is discarded from scoring; length finishes are expected, not truncated answers. Prometheus deltas exclude warmup. Eval seconds include harness/scoring overhead; throughput is not pure decode throughput.

Total leader worker wall: 15020.2 s. Post-load HBM across 16 chips: 13,582,974,976–13,583,270,912 bytes; device limit [33014407168] bytes. This is a post-load snapshot, not peak runtime memory.

Artifacts: gs://marin-us-central2/scratch/karan/task261/v01-r1/; each task has results.json, settings.json, process.json, metrics-before/after.txt and FineStore data/. Job /karan/rsi261-v01-v4-32-r1. The v4 tiling remains a job-local patch; this does not establish stock upstream v4 support.
