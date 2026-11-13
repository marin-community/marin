# Tune AgentX throughput with vLLM

Use this guide for Mochi (Hero) on B200-class hardware and Snowball on H100s.
Start from a reproducible baseline, find the limiting work, and test the
smallest change that could explain it. The default objective here is output
tokens/s and output tokens/s per allocated GPU, subject to correctness and the
task's constraints. Agree on a different objective or latency limit when the
task needs one.

This guide has source and editorial checks only. Exercising it on a live tuning
task is deferred. Job operations follow [use-iris](https://github.com/marin-community/marin/blob/main/.agents/skills/use-iris/SKILL.md);
iterative experiments follow [research engineering](research-engineering.md).
Honor current task authority and shared-resource coordination before costly
work. This guide grants no additional authority to launch jobs or change model
semantics. Keep compilation and CUDA graphs enabled under the
[serving policy](https://github.com/marin-community/marin/blob/main/experiments/evaluation/serve/models/README.md#vllm-eager-execution).

## Fix the comparison

Read the target recipe and retained evidence before proposing a sweep. Record
the objective, baseline, allowed axes, compute budget and stopping condition.
Bind each result to:

| Identity | Record |
| --- | --- |
| Model | Checkpoint/export revision and hashes, tokenizer and chat template, weight and KV dtypes, context limit and any model overrides |
| Workload | Harness version/SHA, scenario, corpus revision/hash, reconstruction-cache identity, seed, warmup, duration, delays, cache busting, routing and client settings |
| Runtime | Marin and vLLM SHAs, wheel/image digests, Torch/CUDA/driver versions, resolved server command and environment, actual backend and graph mode |
| Allocation | Provider, actual GPU SKU, host count, GPU UUIDs/count, topology/NVLink domain, TP/DP/EP/PP layout, CPU allocation, job and attempt IDs |

TP, DP, EP and PP mean tensor, data, expert and pipeline parallelism.
`C128` means 128 concurrent replay sessions across the benchmark, rather than
128 per replica. A reconstruction-cache receipt identifies the prepared trace
data and tokenizer/configuration used to build it; it is separate from the
server's KV cache.

B200 results from the historical Hero work below used **GB200 nodes**. Record
the actual device and fabric; do not assume the same placement or result on a
B200 server. Count every allocated GPU once, including all replicas. TP size
alone is not the denominator. Check the processor's reported GPU count against
the scheduler allocation.

Keep the workload, timing and error policy fixed while comparing server
changes. Concurrency, batch budget, parallelism, cache policy, backend or
precision can each be an explicit axis. Label changed precision, context
overrides or a different corpus as a new comparison with its own correctness
contract. Do not obtain a win by truncating prompts, shortening outputs,
dropping sessions, removing trace delays or relaxing validity thresholds.

The pinned [scenario source][scenario] requires streaming, `ignore_eos`,
recorded replay timing, full input lengths and first-turn-prefix cache busting.
Its minimum duration is 900 seconds. The `agentx` preset defaults to 3,600
seconds, ten warmup requests per lane and seed 42. The older
`inferencex-agentx-mvp` preset has different defaults. Inspect the resolved
configuration, including environment overrides. Follow the
[standalone procedure][standalone] for exact CLI usage, a fresh server at each
concurrency point and fresh output directories. Preserve native warmup and
session routing; do not manually flush the cache between phases.

## Establish a baseline you can trust

First check startup, model loading, the resolved runtime and one representative
request. Restore prepared workload data before allocating GPUs when the runner
supports it. Allow outer deadlines for preparation, compilation, count-driven
warmup, profiling and artifact retention. A one-hour profile can take much
longer than one hour end to end.

Use three levels of evidence:

| Level | Supports | Limits |
| --- | --- | --- |
| Quick diagnosis | Startup, a suspected failure, a small numerical check or trace | Short runs using `--unsafe-override` retain native invalid status |
| Calibration | A load bracket, repeat variation and a promising candidate | A valid 900-second window does not establish a full-hour ranking |
| Final confirmation | The task's conclusion at its agreed duration and workload | Repeat the baseline and candidate near the selected operating point |

Inspect `submission_valid` and its reasons, process exit status, profiling
errors, context overflows, metric coverage and unfinished/cancelled requests.
Report warmup failures separately. Native validity alone does not establish
model correctness or an acceptable error rate; generic errors are not a direct
validity input. See the [validity implementation and FAQ][validity]. Keep invalid
runs for diagnosis and exclude them from the confirmed comparison.

Preserve raw exports and the unchanged processor's output. InferenceX
[filters warmup/error records][records] and computes throughput from retained
request tokens over their [first-start to last-end span][metrics]. Output
tokens/s differs from total tokens/s, which includes input tokens. Its
[per-GPU result][aggregation] divides by the GPU count derived from topology
metadata. Verify that metadata instead of silently editing the metric.
Native deadline handling can [cancel in-flight work][phase]; completed-request
mixtures can therefore change with concurrency and window length. Report
cancellation shares alongside errors and throughput.

## Choose the next experiment

Write one hypothesis with a predicted observation and a result that would
reject it. Use logs and existing metrics first. For cumulative server counters,
preserve worker identity and handle resets before deriving rates; use
[query-finelog](https://github.com/marin-community/marin/blob/main/.agents/skills/query-finelog/SKILL.md). A counter warning
alone does not prove a server restart. Check client CPU, throttling and network
scope before attributing a flat curve to the GPU.

Choose the row supported by the current evidence; this is not a list of required
experiments.

| Observation | Useful next check |
| --- | --- |
| Throughput rises with load, then flattens or falls | Bracket concurrency below/near/above the bend. Repeat the close points and inspect queueing, cache reuse, cancellations and latency. |
| KV pressure or preemptions rise, prefix reuse falls | Check actual cache capacity and native reuse. Test one memory/layout or load change; reducing batch pressure and changing TP have different costs. |
| Prefill and decode compete | Sweep a small `max_num_batched_tokens` range at fixed load. Compare throughput, time to first token (TTFT) and inter-token latency (ITL). |
| GPU gaps or collectives dominate a trace | Check graph replay, rank skew, routing and topology before changing kernels or parallelism. Normalize any allocation change. |
| One kernel dominates relevant requests | Reproduce its real shapes/dtypes, verify parity, then microbenchmark it. Confirm any win in untraced AgentX. |

The official [optimization guide][optimization] explains preemption and the
batch-budget tradeoff. Treat its suggested settings as hypotheses for these
models. A historical 2,048-token recipe is not an optimum merely because it was
held fixed during a TP comparison.

Capture a short representative trace when metrics cannot distinguish the
causes. Use [vLLM profiling][profiling] for Nsight Systems or PyTorch Profiler;
Marin's `profile-training` workflow targets JAX/Levanter. Keep graph replay
representative and inspect all relevant ranks. Profiling adds overhead, so
measure benchmark throughput in a separate untraced run. For kernel work, use
the vLLM [microbenchmark guide][microbench] and Marin's
[performance workflow](https://github.com/marin-community/marin/blob/main/.agents/skills/add-pallas-kernel/docs/performance-workflow.md)
for controlled comparisons and stopping criteria. Kernel timing supports a
mechanism; it does not establish an AgentX speedup.

## Protect correctness while tuning

Qualify the changed path before spending on a long run. Compare identical
token histories against an independent reference with declared tolerances.
Exercise cached and uncached requests, chunk boundaries, fork reuse and session
isolation when caching changes. Verify actual graph replay and the selected
collective/backend, since a successful launch can exercise a fallback.

The [native GQA qualification][gqa] illustrates why these checks matter.
Reducing duplicated Hero KV-cache heads required preserving its full ShortConv
history. Cache capacity gains did not establish long-context quality. Snowball
FP8 matched the previous implementation, but 10/58 H100 numerical observations
exceeded the declared 0.075 probability-error bound against BF16 Levanter.
Implementation parity alone did not establish quantization quality. Snowball
EAGLE3 speculative decoding passed that sampled screen; five accepted draft
tokens out of nine did not establish throughput benefit. Precision and
speculation remain possible axes when the task permits them and their
quality/performance evidence supports them.

## Use the measured lessons within their limits

Snowball's historical BF16 calibration used one host with eight H100s,
TP1/DP8/EP8/PP1, 262,144-token context and an 8,192-token batch budget. Three
full-hour C128 runs averaged **134.95 output tokens/s/GPU**; three C256 runs
averaged **110.55**. C256's TTFT p90 was roughly 197–207 seconds, versus 7.7–7.9
at C128. The repeated decline justified ending that doubling sweep. See the
collection artifacts for C128
([first](https://github.com/marin-community/marin/actions/runs/40696525912),
[second](https://github.com/marin-community/marin/actions/runs/40728980006),
[third](https://github.com/marin-community/marin/actions/runs/40743123733))
and C256
([first](https://github.com/marin-community/marin/actions/runs/40712952732),
[second](https://github.com/marin-community/marin/actions/runs/40757943142),
[third](https://github.com/marin-community/marin/actions/runs/40773289445)),
and [runner PR #10645](https://github.com/marin-community/marin/pull/10645).
These used vLLM `01911be3`, harness `e4bb2f83` and processor `80dff120`.
The operating point is specific to that recipe and revision. The 5%-of-best
elbow heuristic was a local choice, not an AgentX rule.

The [Hero calibration handoff][hero] records BF16 vLLM `ef7f5a0`, runner
`d669e9e4`, the full 393-session corpus and a fixed 2,048-token batch budget.
In 900-second windows, repeated output rates per allocated GB200 were 18.63 at
TP1/DP16/EP16 C128 (16 GPUs), 37.12 at TP2/DP8/EP16 C128 (16 GPUs), and 45.12
at TP6/DP2/EP12 C64 (12 GPUs). These suggest candidates for confirmation;
separate windows do not make a full-hour run. TP1/C256's single higher result
had 97.8% peak cache use and 32.4% deadline cancellations. Full-hour rates and
rankings remain unconfirmed. TP2 needed a qualified NCCL fallback; its root
cause remained unresolved. Use the handoff's portable recipe bundle to recover
the exact collective settings. Artificial collective timings did not predict
AgentX performance. Recheck current code before carrying that policy forward.

## Decide and hand off

Repeat promising points against the same baseline until the difference is
useful relative to observed variation. Start with one-axis tests; investigate
interactions only when evidence warrants them. Prefer the smallest load near
the repeatable throughput maximum that meets the task's constraints. Preserve
all measured points, failures and dominated results, with latency and error
counts. A sampled maximum is not a universal optimum.

Stop when the agreed target is supported, a bounded test rejects the promising
alternatives, or the remaining gap needs a user choice about cost or semantics.
State the chosen configuration, measured gain and spread, correctness scope,
limiting evidence and next useful experiment. Link exact commands, identities,
raw artifacts, failed attempts and allocation/time costs using
[experiment records](experiment-records.md). Keep confirmed observations,
causal hypotheses and unresolved questions distinct.

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
