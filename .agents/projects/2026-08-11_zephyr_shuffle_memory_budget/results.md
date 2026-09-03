# Zephyr shuffle memory calibration: thread-aware GCP validation

## TL;DR

The additive, thread-aware budget model passed the operation-fit matrix, the
held-out interpolation matrix, the PR #7941 incident shape, the 66-run wide
reducer, a 40 GiB repetition, and the full Tier-2 and Nemotron ferries on GCP.
I recommend promoting these constants:

| Parameter | Value |
|---|---:|
| `FIXED_OVERHEAD_WRITE_BYTES` | 320 MiB |
| `R_WRITE` | 4.9 |
| `FIXED_OVERHEAD_READ_BYTES` | 80 MiB |
| `R_READ_MAX` | 8.0 |
| `R_READ_PAYLOAD` | 2.4 |
| `READ_ROW_OVERHEAD_BYTES` | 5.5 KiB |
| `R_READ_BUFFERED_INPUT_PAYLOAD` | 3.2 |
| `R_READ_THREAD_SHARD` | 1.1 |
| `R_READ_SPILL_PAYLOAD` | 1.1 |
| `SAFETY_FRACTION_WRITE` | 0.90 |
| `SAFETY_FRACTION_READ` | 0.90 |
| `STREAMING_CHUNK_SIZE_ROWS` | 10,000 |
| `MIN_MERGE_FAN_IN` | 2 |

The incident-shaped treatment reduced cgroup peak memory from 15.280 GiB on
the legacy path to 6.479 GiB, a 57.6% reduction. The 66-input wide reducer
selected a direct merge and reduced cgroup peak memory from 41.187 GiB to
31.295 GiB while reducing elapsed time from 513.45 seconds to 7.05 seconds.

The fresh Europe West 4 Tier-2 ferry completed end to end in 52 minutes,
including all 171 semantic-verifier reducers and final tokenization. All 20
Finelog stage rows reached `END`; the Iris root had no failures or preemptions.
The fresh Nemotron ferry also completed end to end. Its 20 Finelog stage rows
across ten executions reached `END`, including the 1,382-way semantic verifier,
consolidation, and tokenization. The initial fuzzy shuffle set the run's peak at
9.349 GB; the semantic verifier peaked at 3.040 GB. The matched GCP
main/treatment arms completed the initial fuzzy shuffle and three full
connected-components repetitions. Both then reached the same fourth repetition
and stopped only at the configured six-hour root timeout, with no child-task
failures or preemptions.

- Prior protocol: https://loom.oa.dev/s/s1wvhffi/artifacts/calibration-plan
- Prior zero-intercept results: https://loom.oa.dev/s/s1wvhffi/artifacts/calibration-results
- Prior additive-model findings: https://loom.oa.dev/s/s1wvhffi/artifacts/calibration-results-overhead-model
- Session: https://loom.oa.dev/s/wr1n8bmk
- PR: https://github.com/marin-community/marin/pull/8204

## Model and constant derivation

The write budget is:

```text
growth_budget = 0.90 * task_memory - measured_baseline_RSS - 320 MiB
flush_threshold = growth_budget / 4.9
```

All coefficient selection and acceptance results in this report come from Iris
cluster jobs. Local runs were limited to harness smoke tests and code
correctness and were excluded from every fit and performance comparison.

The baseline is sampled in the shard process after Polars is initialized.
Ordinary least squares over 16 Iris cells and 80 measured repetitions gave a
281.8 MiB intercept, a 4.352 slope, and `R²=0.9962`. Rounding the intercept to
320 MiB left a maximum required slope of 4.808; `R_WRITE=4.9` covered every
measurement. Observed growth used 80.6–98.3% of the prediction.

The read model plans a merge from the measured baseline, row width, number of
chunks, total shard payload, and Polars thread count:

```text
active_rows = min(fan_in * 10,000, shard_payload / average_row_bytes)
batch_bytes = active_rows * average_row_bytes

expanded_batch = min(
    8.0 * batch_bytes,
    2.4 * batch_bytes + 5.5 KiB * active_rows,
)

active_input_payload = min(shard_payload, fan_in * mean_chunk_payload)
buffered_input_payload = max(0, active_input_payload - batch_bytes)

predicted_growth = 80 MiB
                 + expanded_batch
                 + 3.2 * buffered_input_payload
                 + 1.1 * min(threads, fan_in) * mean_chunk_payload
                 + 1.1 * shard_payload  # when the plan spills
```

The planner chooses the largest fan-in whose predicted peak fits under 90% of
task memory. A direct merge is the same calculation with `fan_in` equal to the
chunk count. Below the fitted envelope, the planner saturates at a two-way
merge so small shuffles can still make progress; those cases cannot claim the
90% bound.

The decomposition came from a 36-cell sufficient-shard Iris matrix spanning
row width, streaming batch size, and 1, 2, 8, and 30 Polars threads. The fit
reached `R²=0.970`; its diagnostic coefficients were a 62.91 MiB intercept,
0.630 on active streaming bytes, 1.058 on total reducer payload, and 2.574 on
active-thread shard bytes. Those correlated terms were reparameterized jointly
into the piecewise row-expansion term and the rounded 80 MiB, 2.4, 5.5 KiB,
1.1-thread, and 1.1-spill production envelope above. They are not independent
per-term fits, so acceptance uses the prediction of the full sum. Eight
predeclared interpolation cells at an unseen 2,000-byte row width used
85.9–122.9% of their predictions; all 24 individual repetitions stayed within
the 75–125% gate. The production envelope was then run through the full Zephyr
spill path, insufficient-shard wide merge, task-memory matrix, incident shape,
and ferries.

The 0.90 safety fractions come from the earlier 2, 4, 8, 16, and 60 GiB cgroup
matrix and retain the predeclared 5% task-memory reserve. End-to-end repetitions
at 0.85, 0.90, and 0.95 all completed; 0.95 was excluded because it crossed the
reserve ceiling.

The first 64-shard incident rerun exposed the remaining omitted variable. A
direct 191-way merge of a 4.41 GiB reducer was predicted below 3 GiB but every
shard exhausted three 16 GiB OOM retries. Treating the OOM limit as censored
data gives a coefficient lower bound of 3.18 on input payload held outside the
streaming batch; this is rounded to 3.2. Subtracting the streaming batch avoids
charging the same bytes twice and preserves the successful 66-way direct merge
where the streaming batch already covers the full payload. The repaired plan
predicts a 16.08 GiB direct peak, rejects it against the 14.4 GiB ceiling, and
selects fan-in 111 with a 14.39 GiB prediction. All 64 reducers completed,
peaking at 13,609,631,744 bytes (12.68 GiB), 87.8% of predicted growth. The run
then completed ten connected-components iterations and its final reduction.
The Iris root succeeded; all 25 Finelog stage rows across 12 executions have
`status=END`.

## Held-out and end-to-end results

| Validation | Result | Peak memory |
|---|---|---:|
| PR #7941 shape, 16 GiB | Passed three treatment repetitions; legacy control crossed the 95% headroom gate | 6.479 GiB treatment cgroup; 15.280 GiB control |
| 64-shard incident rerun, 191 inputs per reducer | Direct treatment failed all 64 reducers after three OOM retries; repaired fan-in-111 treatment completed the full fuzzy-dedup pipeline | 12.68 GiB repaired reduce; direct treatment exceeded 16 GiB |
| 66-run, 356,637-byte rows | Passed direct merge; forced fan-in 2 is the main-path comparison | 31.295 GiB direct cgroup; 41.187 GiB forced spill |
| 40 GiB, 30-thread repetition | Passed all stages with zero task failures | 17.591 GB |
| Tier-2 skewed ferry | Fresh Europe West 4 run passed every pipeline step, including the 171-way semantic verifier and tokenization | 10.446 GB verifier peak; 6.327 GB normalize peak |
| Nemotron ferry | Passed all ten executions and 20 stages over 1,000 pinned files, including the 1,382-way verifier and tokenization | 9.349 GB initial fuzzy peak; 3.040 GB verifier peak |
| Matched main/treatment GCS sample, 4,096-way fuzzy shuffle | Both arms completed all three initial fuzzy stages with identical item counts and matching map/intermediate bytes; treatment used 1.38% less CPU and 1.06% less summed stage elapsed | 7.257 GiB treatment vs 7.218 GiB main overall maximum; final reduce/write 6.970 vs 7.034 GiB |

The 40 GiB repetition processed 1.6 million 16 KiB items. Its map/scatter stage
finished in 1,244.86 seconds with a 17,591,128,064-byte peak. Reduce/fold peaked
at 14,909,251,584 bytes. The root had no failures or restarts; its worker pool
recovered 11 infrastructure preemptions.

### Improvement over main

The PR #7941 workload is the primary memory comparison because it reproduces
the unsafe first-flush shape. The treatment reduced cgroup peak by 57.6% and
kept the same 49 million input rows. At 4 GiB, the legacy control reached the
exact cgroup limit; the treatment peaked at 1.821 GB.

The wide reducer measures the read decision. The model selected one direct
66-way merge. Compared with the forced fan-in-2 path, process peak fell 23.0%,
cgroup peak fell 24.0%, and elapsed time fell 98.6% (72.9x).

In the matched 16 GiB production-shape exact-dedup stage, both arms processed
103,717,756 records on six workers. Main and treatment reduce/scatter peaks
were 8,531,943,424 and 8,530,538,496 bytes. CPU time was 540.68 and 537.78
seconds. This cell is neutral: the peak changed by -0.02% and CPU by -0.54%.
The branch does not add a regression when the existing plan already fits.

The 64-shard incident rerun is a separate production-shape comparison. Main's
external merge completed in 914.38 seconds with 67,440.59 CPU-seconds and a
10.89 GiB peak. The repaired model completed in 903.47 seconds with 66,699.21
CPU-seconds and a 12.68 GiB peak. That is 1.2% less elapsed time, 1.1% less CPU,
and 16.4% more peak memory than main, while remaining 1.72 GiB below the
planner's 90% ceiling. The preceding direct treatment exhausted three OOM
retries on every reducer, so the added payload term converts a failed plan into
a bounded one without a throughput penalty.

Both full incident roots succeeded with 12 executions and 25 `END` stage rows;
their item counts match on every stage. Both reached the workload's configured
ten-iteration connected-components cap and emitted the same deterministic-
but-incomplete warning, which is independent of shuffle memory planning.

The fresh matched GCS comparison rebased both arms onto main commit
`6190f2012c58e5e9452a6a2c51f610d155d94d68` and used 64 workers, 2 CPU and
16 GiB per worker, four concurrent source pipelines, and 4,096 logical fuzzy
partitions. Across the initial fuzzy shuffle, treatment used 305,598.11 CPU-
seconds versus 309,872.06 on main (-1.38%) and 5,909.10 seconds of summed
stage elapsed versus 5,972.26 (-1.06%). The final reduce/write stage improved
by 2.22% CPU, 1.50% elapsed, and 0.92% peak RSS. The maximum across all three
stages occurred in map/scatter and was 7.257 GiB on treatment versus 7.218 GiB
on main (+0.54%). Both arms processed 2,696,641,688 map rows and
2,697,374,630 reduce/scatter rows with byte-for-byte matching input and
intermediate byte counts. Final file bytes differed by 8 KiB across 4,096
Parquet outputs; row counts were identical.

The first 4,096-way connected-components pass also completed in both arms.
Its map/scatter rows matched exactly at 210,424,410 items and 57,235,439,520
bytes. Treatment used 91,072.02 CPU-seconds in reduce/map versus 93,067.06
on main (-2.14%), and peak RSS was 5,346,770,944 versus 5,339,361,280 bytes
(+0.14%). Treatment elapsed was 3,090.12 seconds versus 2,912.57 (+6.10%);
one treatment reducer hit and recovered from a GCS `429 SlowDown`, so this
elapsed difference is classified as storage noise rather than a code effect.

The second connected-components pass again matched exactly at 210,424,410
map/scatter items and 57,235,439,520 bytes. Treatment reduce/map CPU was
90,513.57 seconds versus 92,759.63 on main (-2.42%), and peak RSS was
5,335,912,448 versus 5,364,031,488 bytes (-0.52%). Its 2,933.49-second elapsed
time was 5.07% above main's 2,792.02 seconds while both arms shared the same
GCS workload. Across the first two repetitions, the stable signal is lower
treatment CPU with neutral peak memory; elapsed time remains sensitive to
storage service variance.

The third connected-components pass was free of storage errors and matched
the same 210,424,410 map/scatter items and 57,235,439,520 bytes. Treatment
reduce/map CPU was 90,280.03 seconds versus 92,652.30 on main (-2.56%), peak
RSS was 5,337,690,112 versus 5,355,048,960 bytes (-0.32%), and elapsed time
was 2,806.11 versus 2,836.71 seconds (-1.08%). This clean repetition confirms
that the treatment's lower CPU and neutral memory persist when elapsed time is
not distorted by a reported GCS throttle.

## Ferry results

Tier-2 read 98 pinned Parquet files from
`gs://marin-eu-west4/raw/datakit-tier2-skewed-v2-de656ef`. The fresh run
completed normalize, MinHash, initial fuzzy dedup, three connected-components
passes, convergence, the 171-way semantic verifier, consolidation, and final
tokenization in 52 minutes. Its Iris root succeeded with zero failures and
preemptions; all 20 Finelog stage rows across ten executions reached `END`.
The initial fuzzy execution processed 268,588,944 map records and 269,552,905
reduce/scatter records, peaking at 4,625,055,744 bytes. The verifier completed
all 171 reducers and peaked at 10,445,750,272 bytes; final tokenization peaked
at 5,709,742,080 bytes.

Nemotron read 1,000 pinned files from
`gs://marin-eu-west4/raw/nemotro-cc-eeb783`. The fresh run processed the pinned
sample in two cache-preserving roots. The first root completed normalization,
MinHash, the initial fuzzy shuffle, three connected-components iterations, and
the convergence check. Its 14 Finelog rows all reached `END`. The fuzzy stages
processed 7.162 billion map items / 1.318 TB and 7.318 billion reduce items /
1.346 TB, peaking at 9,349,263,360, 9,288,990,720, and 8,826,277,888 bytes.

The semantic verifier requires every memory-store actor to be admitted before
it can process a shard. A 64-worker verifier admitted 41 actors and left 23
pending, so that root was stopped rather than spending another timeout without
work. The final ferry configuration uses 40 verifier workers while leaving the
logical 1,000-way fuzzy partitioning unchanged. A recovery root reused the
completed TTL outputs, admitted all 40 verifier workers, and finished the
1,382-way verifier, consolidation, and tokenization. Its six Finelog rows also
reached `END`; the verifier's three stages peaked at 935,014,400,
3,039,895,552, and 2,261,794,816 bytes. Consolidation peaked at 1,133,711,360
bytes and tokenization at 2,304,126,976 bytes. The recovery root succeeded in
46 minutes with no failures or preemptions.

GCS throttling produced 95 first-attempt shard retries in the third
connected-components iteration and 149 in the verifier. Every retry recovered
on its second attempt; neither execution had a second- or third-attempt
failure. These are storage-service observations, not memory-model failures.

## Storage and workload isolation

The ferries read pinned GCS sources in the same region as their jobs. The
Tier-2 source check verifies the 98-file manifest and refuses to download or
replace it; Nemotron's source check is also read-only. Each source-check
`StepSpec` writes its runner metadata to a one-day TTL path. Raw locations are
passed only as explicit read inputs to the check and normalizer, and a focused
regression test asserts that no ferry step output is under either raw prefix.
All pipeline outputs, status, spill files, and diagnostics use unique one-day
or seven-day TTL prefixes. The jobs did not write to raw, artifact, alias,
manifest, ferry-history, or stable-status paths.

The jobs used the shared Iris worker identity, which still has project-wide
object-admin capability. The protection here is the executable step graph and
audited destinations, not per-job IAM isolation.

## GCP protocol corrections and caveats

The first matched A/B submission carried a CoreWeave-shaped worker request of
16 CPU and 160 GiB per worker. GCP's available CPU scale group exposes 2 CPU /
16 GiB slices, so no measured task scheduled. Both roots were stopped and
replaced with matched six-worker, 2 CPU / 16 GiB arms. No observation from the
unscheduled pair is included.

The matched run started both arms together. Treatment briefly had six workers
while control had three during the first large tokenization step. Stage CPU
and peak-memory telemetry remains valid; elapsed comparisons from intervals
with different active worker counts are excluded.

The older Tier-2 and Nemotron attempts below remain useful negative records,
but are superseded for the final gate by fresh runs using GCS in Europe West 4,
64-worker shuffle pools, and a 40-worker Nemotron verifier pool. Their storage
and actor-readiness failures are excluded from the memory acceptance data.

## Job ledger

| Workload | Iris root | Terminal result |
|---|---|---|
| 40 GiB repetition | `/loom/zephyr-mem-r6-40g-r1-20260818-wr1n8bmk` | Succeeded |
| Tier-2 ferry | `/loom/zephyr-mem-r6-tier2-ferry-20260819-wr1n8bmk` and `-retry2` verifier | Shuffle passed; verifier infrastructure failure |
| Nemotron throttled retry | `/loom/zephyr-mem-r6-nemotron-ferry-20260818-wr1n8bmk-retry2` | Shuffle and convergence passed; verifier readiness timeout |
| 64-shard incident main | `/app/zephyr-mem-r7-euw4-control-20260819-wr1n8bmk-r3` | Succeeded; 12 executions / 25 `END` rows |
| 64-shard incident repaired treatment | `/app/zephyr-mem-r9-euw4-treatment-incident-20260820-wr1n8bmk` | Succeeded; 12 executions / 25 `END` rows |
| Matched main | `/app/zephyr-mem-r10-euw4-control-20260820-wr1n8bmk` | Initial fuzzy plus three CC repetitions passed; timed out during CC4 at 6h |
| Matched treatment | `/app/zephyr-mem-r10-euw4-treatment-20260820-wr1n8bmk` | Initial fuzzy plus three CC repetitions passed; timed out during CC4 at 6h |
| Fresh Tier-2 ferry | `/app/zephyr-mem-r11-euw4-tier2-20260820-wr1n8bmk` | Succeeded in 52m; 10 executions / 20 `END` rows |
| Fresh Nemotron through convergence | `/app/zephyr-mem-r12-euw4-nemotron-20260820-wr1n8bmk` | 7 executions / 14 `END` rows; intentionally stopped after verifier admitted only 41/64 actors |
| Fresh Nemotron verifier through tokenization | `/app/zephyr-mem-r13-euw4-nemotron-resume-20260820-wr1n8bmk` | Succeeded in 46m; 3 executions / 6 `END` rows; no failures or preemptions |

Iris marks successful Zephyr coordinators as `killed` after StepRunner tears
them down. Root state and Finelog rows with `status=END` are the success
criteria used here.

## Decision

Promote the constants at the top of this report. The final additive,
thread-aware model passed every predeclared memory gate, converted the
incident-shaped direct-merge OOM into a bounded fan-in-111 merge, improved the
PR #7941 workload's cgroup peak by 57.6%, and completed both fresh ferries on
GCP. The matched main/treatment jobs show neutral peak memory and lower CPU on
the production-shaped sample, so the protection does not impose a measurable
steady-state regression.

No additional coefficient is justified by the remaining variance. The
outliers that delayed the fresh ferries were explicit GCS `429 SlowDown`
responses and recovered retries; their workers stayed alive and their stages
ended successfully. The only operational adjustment is the Nemotron verifier's
40-worker pool, which fits the all-actors-ready protocol within the available
Europe West 4 capacity without changing logical partitioning or the shuffle
budget model.
