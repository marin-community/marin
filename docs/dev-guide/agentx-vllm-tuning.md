# Tune AgentX throughput with vLLM

Use this guide for Mochi (Hero) on B200s and Snowball on H100s. For RL rollouts
and evaluations, prioritize throughput with acceptable interactivity. Choose
near the bend where extra concurrency buys little throughput and sharply
worsens latency. Show nearby measured tradeoffs; ask when the choice remains
unclear. Use output tokens/s at fixed allocation and output tokens/s per
allocated GPU across allocations. AgentX is a proxy; confirm application gains
on rollout/evaluation traffic.

Live validation of this guide is deferred. Follow [use-iris][iris] and current
resource-coordination rules for execution. Keep compilation and CUDA graphs
enabled under the [serving policy][serving].

## Read before choosing an experiment

Search Marin evidence through [use-echo][echo], then upstream docs, recipes,
release notes, PRs/issues and reviews. Start with [vLLM][optimization] and
[SGLang tuning][sglang-tuning]. Check revisions, model/GPU/topology support,
workloads, caveats and merge/release/fork-adoption status. Treat discussion and
other-model wins as hypotheses; verify relevant code and tests.

Use existing agent workflows when they fit:

| Need | Upstream guidance |
| --- | --- |
| AgentX phases, metrics and unproductive runs | InferenceX [AgentX diagnostics][agentx-debug] |
| Model-forward or kernel timing | vLLM [model][model-bench] and [kernel microbenchmarks][microbench] |
| Kernel, fusion and communication analysis | SGLang [profiler analysis][sglang-profile] |
| Cache-state consistency | SGLang [KL checks][sglang-kl] |
| A large parameter search | Harness [search recipes][search-recipes] and [adaptive search][search] |

Check support and defaults: synthetic weights, disabled compilation or altered
warmup may invalidate the comparison. Upstream launch/cleanup examples do not
override Marin's rules or authorize changes to shared processes.

## Freeze the baseline and measurement

Find the primary run through the task's benchmark issue or dashboard; ask for
its location if missing. Use the latest run passing declared correctness and
measurement criteria and the best comparable run. Pin both when they differ.
Use their measured
interactivity as the starting reference for acceptable tradeoffs.
If neither is comparable, report the gap and establish a baseline within the task budget.

Record model/export/tokenizer revisions, template, dtypes and context overrides;
corpus/hash, prepared-data cache, harness/scenario, seed, warmup and duration;
software SHAs, wheel/image digests, command/environment and actual backend/graph
mode; GPU SKU/UUIDs, hosts/topology, CPU resources, parallelism and attempt IDs.
Prepared data and server KV are separate caches. Verify InferenceX's GPU count
against all allocated GPUs, including replicas.

Keep comparison settings fixed except the declared change; record new run and
attempt IDs for every execution. Preserve all replay sessions, input/output lengths,
trace delays, cache busting and error/validity rules.
The pinned [scenario][scenario] requires streaming and `ignore_eos`, with a
900-second minimum. The `agentx` preset defaults to a one-hour profile, ten
warmup requests per lane and seed 42. Check resolved overrides. Use a fresh
server and output directory per point, preserving AgentX warmup and session-routing settings;
see the [standalone procedure][standalone]. Budget preparation, compilation,
warmup and retention time as well as profiling.

AgentX reports metrics and throughput/interactivity curves. The compute ladder
below is our strategy; [search objectives are caller-chosen][search]. There is
no universal AgentX score here. Keep output and total-token throughput separate.
InferenceX [excludes warmup/error records][records] and [times from the first
retained request starting to the last finishing][metrics]. Check `submission_valid`
and its reasons, errors, context overflows, metric coverage and deadline cancellations separately.
Keep warmup failures separate. Valid replay does not prove model correctness.
Short windows change which requests finish and can change configuration rankings.
Use the [versioned metric definitions][metric-definitions] for interactivity.

## Form and test a bottleneck hypothesis

Write: observation → suspected cause → one change → prediction → rejection
condition. Distinguish client/router limits, CPU gaps, KV pressure/recomputation,
compute/memory work and communication. Correlate request, queue, cache and device
trends across workers; handle counter resets with [query-finelog][finelog].
High GPU utilization alone does not identify the bottleneck.

Use a short [vLLM trace][profiling] when metrics cannot separate causes. Inspect
representative prefill/decode, cache reuse and all relevant ranks. Follow time
on the execution path that determines completion; summed kernel time can count
overlap twice. SGLang's [overlap heuristics][overlap] help rank suspects but do
not prove that dependencies permit overlap. Use a controlled change or ablation
to test the explanation. Measure throughput separately without profiler overhead.

## Spend compute as evidence improves

Rank attempts by plausible end-to-end gain, supporting evidence and total
GPU-hours. Label estimates. A tiny or overlapped kernel offers little headroom.
Prefer cheap checks that can reject expensive ideas early.

| Stage | Spend on | Continue when |
| --- | --- | --- |
| Inspect | Existing sources, metrics and a rough gain/cost estimate | A relevant, testable explanation remains |
| Isolate | Correctness checks, a representative trace or microbenchmark | The changed path works and the predicted effect appears |
| Calibrate | A small load/configuration bracket and repeats | Gains exceed repeat variation with useful interactivity |
| Confirm | Repeated baseline/candidate runs under the full target protocol | Correctness, validity and the chosen operating point hold |

Skip stages already supported by evidence. Use representative shapes, cache
state, backend and topology. Before expanding, estimate cost including startup
and warmup, and state what result would justify it. Sub-minimum runs need
`--unsafe-override`; preserve `submission_valid: false` and exclude them from
confirmed comparisons. A valid 900-second calibration cannot establish a
full-hour result. Stop unproductive owned work within task authority and retain
its evidence. Expand promising candidates, then test interactions.

## Check correctness before claiming a win

Separate benchmark validity, numerical/state correctness, task quality and RL
compatibility. Declare acceptance criteria before comparing. Check changed
operators against independent references. For cache/state changes, test
cold/warm histories, chunk boundaries, forks, isolation and real graph/parallel
execution. SGLang's [KL checks][sglang-kl] cover state consistency; task accuracy
alone can miss corruption.

For RL, identify the actual trainer, loss and log-probability convention.
Compare baseline and candidate rollout log-probabilities on identical generated
tokens with the trainer at the same weight version, tokenization, masks,
temperature and filtering. Include different batch shapes, long histories,
cache states and MoE routing. Use [fixed-token replay][token-replay] or the
existing [mismatch probe][mismatch-probe] when their model/backend contracts fit.
Require finite log-probabilities at every response position included in the
training loss. Compare generation-time rollout scores with trainer and engine rescoring.
Report per-token error tails, probability ratios, clipping/rejection rates and
effective sample size (the information retained after weighting);
separate numerical error from stale-policy effects.

Keep a trusted numerical reference. [Batch invariance][batch-invariance] can
help but does not guarantee trainer parity. [Importance/rejection correction][rollout-correction]
and [quantization-aware training][qat] are options to test with the trainer,
with their bias, variance and compute costs. Do not assume FP8 or small average
error is RL-safe. An inference win needs a bounded learning/stability comparison
before an RL claim; report learning progress per GPU-hour alongside throughput.
Until then, label it inference-qualified with RL validation pending.

Distinguish a candidate regression from an existing baseline discrepancy.
The [Snowball checks][gqa] found FP8 matching the old implementation but failing
BF16-reference checks. Preserve that failure; never silently widen tolerances
or treat baseline agreement as proof of RL compatibility.

## Keep runnable experiments without merging

For an iterative tuning campaign, follow [research engineering](research-engineering.md).
Commit and push experimental code, recipes, analysis and a notebook to a
Marin-derived research branch in `marin-community/marin-logbooks`; use its
[current archive workflow][logbooks]. No merge into Marin or logbooks `main` is
needed to preserve the experiment. Keep large raw outputs in durable artifact
storage with URIs/hashes. Link exact source commits, attempts and artifacts.

Record hypothesis, prediction, correctness, measurement, cost, interpretation
and next decision, including negative results. Push supporting commits before
milestone issue updates. Preserve the research branch; promote selected changes
through ordinary PRs. Bounded one-off checks use the existing issue/report under
[experiment records](experiment-records.md).

## Decide and report

Show the throughput/interactivity curve and recommend a high-throughput point
before the latency cliff. Report repeated rates/spread, all-GPU normalization,
latency/tails, cancellations, quality limits and remaining uncertainty.
Stop when the choice is supported, promising alternatives are rejected, or
further work needs a user decision about cost, quality or interactivity.

Snowball's [ten H100 measurements](data/agentx-snowball-h100-calibration.csv)
include three full-hour runs at 128 concurrent replay sessions averaging 134.95
output tokens/s/GPU and three at 256 averaging 110.55. The 90th-percentile time
to first token rose from 7.7–7.9 to 197–207 seconds.
The [Hero handoff][hero] covers 900-second GB200 calibration, draft parallelism
builds and an unresolved reduction failure at tensor-parallel size two.
Its full-hour rankings and long-context quality remain unconfirmed.
Verify actual B200/GB200 hardware, topology and current software support before
reusing either recipe.

[iris]: https://github.com/marin-community/marin/blob/main/.agents/skills/use-iris/SKILL.md
[serving]: https://github.com/marin-community/marin/blob/main/experiments/evaluation/serve/models/README.md#vllm-eager-execution
[optimization]: https://docs.vllm.ai/en/latest/configuration/optimization/
[sglang-tuning]: https://github.com/sgl-project/sglang/blob/bd4783409967b186ecb95956a6ab11256ce6d296/docs/docs/advanced_features/hyperparameter_tuning.mdx
[agentx-debug]: https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/.agents/skills/debug-agentx-runs/SKILL.md
[model-bench]: https://github.com/vllm-project/vllm/blob/55aa0cab78dedbb018ad4f4ea649f2d3d5e1c2a8/.agents/skills/model-microbenchmark/SKILL.md
[microbench]: https://github.com/vllm-project/vllm/blob/55aa0cab78dedbb018ad4f4ea649f2d3d5e1c2a8/.agents/skills/kernel-microbenchmark/SKILL.md
[sglang-profile]: https://github.com/sgl-project/sglang/blob/bd4783409967b186ecb95956a6ab11256ce6d296/.agents/skills/llm-torch-profiler-analysis/SKILL.md
[sglang-kl]: https://github.com/sgl-project/sglang/blob/bd4783409967b186ecb95956a6ab11256ce6d296/.agents/skills/kl-consistency-test/SKILL.md
[search-recipes]: https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/sweeping/search-recipes.md
[search]: https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/sweeping/bayesian-optimization.md
[scenario]: https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/src/aiperf/common/scenario/inferencex_agentx_mvp.py
[standalone]: https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/inferencex-e2e/docs/agentx-standalone.md
[records]: https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/inferencex-e2e/infx/results/agentic/artifacts.py
[metrics]: https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/inferencex-e2e/infx/results/agentic/request_metrics.py
[metric-definitions]: https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/metrics-reference.md#e2e-normalized-interactivity
[finelog]: https://github.com/marin-community/marin/blob/main/.agents/skills/query-finelog/SKILL.md
[profiling]: https://docs.vllm.ai/en/latest/contributing/profiling/
[overlap]: https://github.com/sgl-project/sglang/blob/bd4783409967b186ecb95956a6ab11256ce6d296/.agents/skills/llm-torch-profiler-analysis/references/heuristics.md
[gqa]: https://github.com/marin-community/vllm/pull/110#issuecomment-6686676052
[logbooks]: https://github.com/marin-community/marin-logbooks/blob/main/WORKFLOW.md
[hero]: https://echo.oa.dev/wiki/945
[echo]: https://github.com/marin-community/marin/blob/main/.agents/skills/use-echo/SKILL.md
[token-replay]: https://github.com/vllm-project/vllm/blob/55aa0cab78dedbb018ad4f4ea649f2d3d5e1c2a8/docs/serving/online_serving/trace_replay.md
[mismatch-probe]: https://github.com/marin-community/marin/blob/80004cc0654537cb39f386539f91cd28db653e4e/vendor/skyrl/skyrl-train/skyrl_train/config/mismatch_probe.py
[batch-invariance]: https://docs.vllm.ai/en/latest/features/batch_invariance/
[rollout-correction]: https://verl.readthedocs.io/en/latest/algo/rollout_corr.html
[qat]: https://verl.readthedocs.io/en/latest/low_precision/qat.html
