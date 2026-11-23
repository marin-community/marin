# Measurement and correctness references

Use [AgentX's scenario](https://github.com/SemiAnalysisAI/agentx-harness/blob/18015a2d77a6077bee9750c6929a889d52c7df54/docs/benchmark-modes/semianalysis-agentx-faq.md)
and [InferenceX's standalone protocol](https://github.com/SemiAnalysisAI/InferenceX/blob/d40220b7bc455a6f146f202ea08b64597a9b27f4/inferencex-e2e/docs/agentx-standalone.md)
to interpret the runner's pinned recipe. Check the actual installed versions.
The official 256K corpus excludes individual requests whose input plus output
exceeds 256,000 tokens. Preserve its identity and replay timing.

Concurrency counts resident session-tree lanes, including pauses. Subagents can
produce more in-flight requests than the lane count. Native warmup is count
driven. A benchmark duration flag does not include server startup and warmup.
The runner requires an explicit Probe (180 s), Screen (900 s) or Confirm (3,600 s).
Probe uses the unsafe override. Screen meets the pinned scenario's 900-second
minimum without that override; it still cannot produce an accepted recommendation.
Only a valid Confirm with complete required checks qualifies. Shortened Confirm
runs and unsafe overrides remain diagnostic. AgentX has no campaign or ladder score.

The warmup counts are additional requests per lane after native cache warmup.
Each invocation runs one window on a fresh server. One window provides no measured
repeatability. Upstream multi-run confidence reports can summarize repeats, but
native multi-run reuses vLLM and drops additional warmup after the first by default.
Use separate runner invocations for fresh-server repeats. Confidence estimates
do not establish that one configuration beats another.

Completed output throughput uses retained successful profiling requests and the
upstream measurement span. Preserve request lengths, errors, partial work and
deadline cancellations when interpreting a result. Normalize to all physical
GPUs, including replicas. Keep p90 time to first token and p90 completion time
separate. Do not infer a latency requirement from the benchmark's native validity.

For optimization ideas, start with
[vLLM performance guidance](https://docs.vllm.ai/en/latest/configuration/optimization/),
[vLLM profiling](https://docs.vllm.ai/en/latest/contributing/profiling/),
[SGLang tuning](https://docs.sglang.ai/advanced_features/hyperparameter_tuning.html)
and their repository skills. Inspect relevant code and current discussions;
settings from another architecture or release may not transfer.

## Numerical evidence and memory claims

Treat benchmark validity, numerical/state checks, task quality and RL learning
as separate claims. Run existing checks with their original checkpoint scope
and tolerances. For example, Marin's
[Snowball endpoint check](../../../../lib/marin/src/marin/testing/inference/snowball_backend_parity_jobs.py)
uses representative step-42150 goldens, rank-pinned recomputation and a scoped
vLLM bound. It does not qualify the public v0.1.1 preset automatically.
Earlier [native GQA evidence](https://github.com/marin-community/vllm/pull/110)
also retains FP8 failures against BF16 references. Do not turn matching old/new
inference paths into proof of alignment with the trainer.

Verify quantization with actual tensor metadata. FP8 kernels can pack weight
bytes into INT32 words, so inspect the consumer format and packing contract.
Method names establish method selection. This
[Hero example](https://github.com/marin-community/marin-logbooks/blob/6acea60db03ab0a2c782e10dc9972c579866ddf7/logbooks/57-inference-perfmax/analysis/hero-fp8-01a1ec55/hero_weight_metadata.py)
uses a native worker extension to check weight storage and separate KV/state
views without reading tensor values.

For a numerical failure, align raw token scores with the golden and a passing
BF16 control. Check token IDs and score positions. See whether the limit is
crossed by an observed probability gap or only by the top-k bound.
Exercise new checks against retained responses before loading another model.
For quantization, verify loaded weight storage and record activation and KV
precision separately. For a memory claim, report stored parameter bytes or usable
KV capacity with topology and throughput for both configurations. Label load-time
memory separately.

For an RL claim, use evidence for the actual trainer and rollout path: the same
weights, token IDs, masks and sampling transformations. Means alone can hide
tails in log-probability differences, importance ratios and clipping. Any new
mismatch metric, correction method or learning experiment is a separate task.
Keep existing qualification gaps visible while measuring serving performance.
