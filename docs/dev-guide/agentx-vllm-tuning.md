# Tune AgentX throughput with vLLM

Use this guide for Mochi (Hero) on B200-class hardware and Snowball on H100s.
Start with a reproducible baseline. Find what limits throughput, then test the
smallest change that could explain it. Use output tokens/s and output tokens/s
per allocated GPU as the default objective. Before running experiments, agree
on the task's objective, correctness requirements, latency limit, compute budget
and when to stop.

This guide has not been exercised on a live tuning task.
Follow [use-iris](https://github.com/marin-community/marin/blob/main/.agents/skills/use-iris/SKILL.md)
for job operations and [research engineering](research-engineering.md) for
experiments. Work within the task's authority and coordinate costly shared
resources. Keep compilation and CUDA graphs enabled under the
[serving policy](https://github.com/marin-community/marin/blob/main/experiments/evaluation/serve/models/README.md#vllm-eager-execution).

## Fix the comparison

Read the target recipe and existing evidence. Agree on which settings may
change. For each result, record:

| Part | Record |
| --- | --- |
| Model | Checkpoint/export revision and hashes, tokenizer, chat template, weight and KV dtypes, context limit and model overrides |
| Workload | Harness version/SHA, scenario, corpus revision/hash, prepared-data cache identity, seed, warmup, duration, delays, cache busting, routing and client settings |
| Runtime | Marin and vLLM SHAs, wheel/image digests, Torch/CUDA/driver versions, resolved command and environment, actual backend and graph mode |
| Allocation | Provider, GPU SKU, host count, GPU UUIDs/count, topology/NVLink domain, TP/DP/EP/PP layout, CPU allocation, job and attempt IDs |

TP, DP, EP and PP mean tensor, data, expert and pipeline parallelism.
`C128` means 128 concurrent replay sessions across the benchmark.
The prepared-data cache stores traces reconstructed for a particular tokenizer
and configuration. Record its identity separately from the server's KV cache.

Historical Hero measurements used **GB200 nodes**. Record the actual device
and fabric before applying those results to a B200 server. Count every allocated
GPU once, including all replicas. Check the processor's GPU count against the
scheduler allocation.

Keep workload, timing and error policy fixed while comparing server changes.
Change one setting first, such as concurrency, batch budget, parallelism,
cache policy or backend. Treat changed precision, context overrides or corpus
as a separate comparison with declared correctness requirements. Preserve prompt
and output lengths, all sessions, trace delays and validity thresholds.

The pinned [scenario source][scenario] requires streaming, `ignore_eos`,
recorded replay timing, full inputs and first-turn-prefix cache busting.
Its minimum duration is 900 seconds. The `agentx` preset defaults to 3,600
seconds, ten warmup requests per lane and seed 42. The older
`inferencex-agentx-mvp` preset has different defaults. Check the resolved
configuration and environment overrides. Follow the [standalone procedure][standalone]
for CLI usage. Start a fresh server at each concurrency point and use a fresh
output directory. Preserve native warmup and session routing without manually
flushing the cache between phases.

## Establish a trustworthy baseline

Check startup, model loading, the resolved runtime and one representative
request. Restore prepared workload data before allocating GPUs when the runner
supports it. Set job deadlines that allow for preparation, compilation,
count-driven warmup, profiling and artifact retention. A one-hour profile can
take much longer than one hour to complete.

| Run | What it can establish | Limit |
| --- | --- | --- |
| Quick diagnosis | Startup, a suspected failure, a small numerical check or trace | Short runs using `--unsafe-override` keep their native invalid status |
| Calibration | A useful load range, repeat variation and a promising candidate | A valid 900-second window cannot establish a full-hour ranking |
| Final confirmation | The task's conclusion at the agreed duration and workload | Repeat the baseline and candidate near the chosen operating point |

Check `submission_valid` and its reasons, exit status, profiling errors,
context overflows, metric coverage and unfinished/cancelled requests. Report
warmup failures separately. Native validity does not establish model correctness
or an acceptable error rate. Generic errors do not directly set validity,
though an error threshold can abort the run. Keep invalid runs for diagnosis
and exclude them from confirmed comparisons. See the [validity FAQ][validity].

Keep raw exports and the unchanged processor's output. InferenceX
[excludes warmup/error records][records] and computes throughput over the
[time from the first retained request starting to the last finishing][metrics].
Output tokens/s differs from total tokens/s, which includes inputs. Its
[per-GPU result][aggregation] uses topology metadata; single-node DP replicas
need explicit accounting. Correct mismatched GPU metadata passed to the
processor. Native deadlines can [cancel in-flight work][phase], changing which
requests finish at different loads and window lengths. Report cancellation
shares alongside errors and throughput.

## Choose the next experiment

Write one hypothesis, its predicted observation and a result that would reject
it. Use logs and existing metrics first. For cumulative counters, preserve
worker identity and handle resets before calculating rates; use
[query-finelog](https://github.com/marin-community/marin/blob/main/.agents/skills/query-finelog/SKILL.md).
A counter warning alone does not prove a restart. Check client CPU, throttling
and network measurements before blaming a flat curve on the GPU.

Choose a check that fits the evidence:

| Observation | Useful next check |
| --- | --- |
| Throughput rises with load, then flattens or falls | Measure below, near and above the bend. Repeat close points; inspect queueing, cache reuse, cancellations and latency. |
| KV pressure or preemptions rise, prefix reuse falls | Check cache capacity and reuse. Test one memory/layout or load change. Account for the cost of changing TP. |
| Prefill and decode compete | Test a small `max_num_batched_tokens` range at fixed load. Compare throughput, time to first token (TTFT) and inter-token latency (ITL). |
| GPU gaps or collectives dominate a trace | Check graph replay, rank skew, routing and topology. Recount allocated GPUs if parallelism changes. |
| One kernel dominates relevant requests | Reproduce its shapes/dtypes, check correctness, then microbenchmark it. Confirm the gain in untraced AgentX. |

The official [optimization guide][optimization] explains preemption and the
batch-budget tradeoff. Test its suggestions against these models. Holding a
2,048-token batch budget fixed during a TP comparison does not make it optimal.

Capture a short trace when metrics cannot distinguish the causes. Use
[vLLM profiling][profiling] for Nsight Systems or PyTorch Profiler. Marin's
`profile-training` targets JAX/Levanter. Preserve representative graph replay
and inspect all relevant ranks. Measure throughput in a separate untraced run
to exclude profiling overhead. For kernel work, follow the vLLM
[microbenchmark guide][microbench] and Marin's
[performance workflow](https://github.com/marin-community/marin/blob/main/.agents/skills/add-pallas-kernel/docs/performance-workflow.md).
A faster kernel still needs an AgentX comparison.

## Protect correctness

Check correctness of the changed code before spending on a long run. Compare
identical token histories against an independent reference with declared
tolerances. For cache changes, test cached and uncached requests, chunk
boundaries, fork reuse and session isolation. Verify actual graph replay and
the selected backend/collective; a successful launch may use a fallback.

The [native attention and cache checks][gqa] show why matching the previous
implementation still needs an independent reference. Snowball FP8 matched the
previous implementation but failed some BF16 Levanter reference checks.
Snowball EAGLE3 speculative decoding passed sampled correctness checks; those
checks do not establish an AgentX speedup. Hero's cache capacity gains
also left long-context quality unmeasured. Read the linked numerical criteria
and results before choosing precision, caching or speculation changes.

## Decide and hand off

Repeat promising points against the same baseline until the gain is useful
relative to observed variation. Test interactions only when evidence warrants
them. Prefer the smallest load near the repeatable throughput maximum that
meets the task's constraints. Keep all points, including failures and slower
results, with latency and error counts.

Stop when the agreed target is supported, bounded tests reject the promising
alternatives, or the remaining gap needs a user decision about cost or semantics.
Report the chosen settings, gain and spread, correctness scope, limiting
evidence and next useful experiment. Follow [experiment records](experiment-records.md)
for commands, software/model/workload identities, raw artifacts, failed attempts
and allocation/time costs. Separate confirmed observations, explanations still
to test and unresolved questions.

## Lessons from prior runs

Snowball's [ten retained H100 measurements](data/agentx-snowball-h100-calibration.csv)
show why a load sweep can stop. On eight H100s, three full-hour C128 runs
averaged 134.95 output tokens/s/GPU; three C256 runs averaged 110.55. TTFT p90
rose from 7.7–7.9 to 197–207 seconds. This supported stopping that doubling
sweep for that recipe. The CSV preserves original values, submission links and
raw archive URIs/hashes after Actions artifacts expire. The recipe used BF16,
TP1/DP8/EP8/PP1, 262,144-token context, an 8,192-token batch budget, vLLM
`01911be3`, harness `e4bb2f83` and processor `80dff120`. Its
[runner PR #10645](https://github.com/marin-community/marin/pull/10645) was draft
on 2026-11-13 UTC.

The [Hero handoff][hero] shows the limits of short calibration. Its 900-second
GB200 windows suggest TP/load candidates but leave full-hour rankings
unconfirmed. At TP1/C256, peak KV use reached 97.8% and deadline cancellations
reached 32.4%. The handoff retains exact rates, allocations, builds and recipes,
including the NCCL fallback used after TP2 reduction failures whose cause
remains unresolved. Both TP2 and TP6 used draft vLLM
[#119](https://github.com/marin-community/vllm/pull/119) stacked on draft
[#110](https://github.com/marin-community/vllm/pull/110) at authoring. Check
current release/adoption status, or recover the measured build, before planning
confirmation. Separate microbenchmarks timed artificial collective work;
those results do not establish AgentX throughput.

[scenario]: https://github.com/SemiAnalysisAI/agentx-harness/blob/60a094f256b42f55fde1efb3d45672e1d9a44566/src/aiperf/common/scenario/inferencex_agentx_mvp.py
[standalone]: https://github.com/SemiAnalysisAI/InferenceX/blob/8f29d8dc12329bb30a0300cd1c31f4e4cc2a19e4/inferencex-e2e/docs/agentx-standalone.md
[validity]: https://github.com/SemiAnalysisAI/agentx-harness/blob/60a094f256b42f55fde1efb3d45672e1d9a44566/docs/benchmark-modes/semianalysis-agentx-faq.md#7-reading-the-results-metrics-validity-and-submission-requirements
[records]: https://github.com/SemiAnalysisAI/InferenceX/blob/8f29d8dc12329bb30a0300cd1c31f4e4cc2a19e4/inferencex-e2e/infx/results/agentic/artifacts.py
[metrics]: https://github.com/SemiAnalysisAI/InferenceX/blob/8f29d8dc12329bb30a0300cd1c31f4e4cc2a19e4/inferencex-e2e/infx/results/agentic/request_metrics.py
[aggregation]: https://github.com/SemiAnalysisAI/InferenceX/blob/8f29d8dc12329bb30a0300cd1c31f4e4cc2a19e4/inferencex-e2e/infx/results/agentic/__init__.py
[phase]: https://github.com/SemiAnalysisAI/agentx-harness/blob/60a094f256b42f55fde1efb3d45672e1d9a44566/src/aiperf/timing/phase/runner.py
[optimization]: https://docs.vllm.ai/en/latest/configuration/optimization/
[profiling]: https://docs.vllm.ai/en/latest/contributing/profiling/
[microbench]: https://github.com/vllm-project/vllm/blob/bbbd0a02c9d1bec965ee818af29a1319baefa69c/.agents/skills/kernel-microbenchmark/SKILL.md
[gqa]: https://github.com/marin-community/vllm/pull/110#issuecomment-6686676052
[hero]: https://echo.oa.dev/wiki/945
