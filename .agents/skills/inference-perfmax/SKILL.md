---
name: inference-perfmax
description: Improve serving throughput and GPU efficiency for RL rollouts and evaluations. Use for inference tuning campaigns, AgentX comparisons, latency tradeoffs and reusable experiment evidence.
---

# Improve inference throughput

Use the manual [Knotmeter runner](../../../experiments/evaluation/knotmeter/README.md)
to replay AgentX's agent sessions.
Start from its Snowball/H100 or Hero/GB200 YAML preset. Hero and Mochi name the
same model. Resolve the actual checkpoint and runtime; the presets use BF16.

## Agree on the campaign

Read the dispatch request and retained findings, then resolve defaults and overrides:

```sh
uv run python -m experiments.evaluation.knotmeter.cli plan \
  --profile snowball-h100 --campaign CAMPAIGN --stage probe --output /tmp/perfmax-plan.json
```

Present the model and tokenizer revisions, workload, hardware, runtime, latest
qualified and best comparable baselines, objective, allowed changes, required
checks and resource budget. Show consequential unknowns together. Record the
agreed scope and baseline with the research notebook and resolved trial plans.
Treat a complete dispatch as agreement; use its delegated choices without asking again.

Prioritize completed output tokens/s/GPU for RL rollouts and evaluations. Seek
high throughput before latency rises sharply for little additional throughput.
Prefer lower latency among similar throughput results. Agree on any numerical
decision criteria against the baseline; there is no universal latency cap.

After agreement, choose hypotheses, code changes, loads, repeats, parallelism
and stopping autonomously. Consult for a material scope or cost change, changed
correctness criteria, or a consequential decision that is hard to reverse.

## Spend compute on evidence

Consult upstream vLLM and SGLang guidance, recent relevant features and PR or
issue discussions. Read the source and check the versions before applying an
idea. The [measurement reference](references/measurement.md) links useful starting
points. Reuse prior evidence when its model, settings and runtime still apply.
Read prior winning recipes beyond the maintained preset. Verify which settings
the installed source consumes. Explain missing optimizations before calling a
configuration the best applicable baseline.

Identify a plausible bottleneck, predict what a change should affect and seek
evidence that could reject the hypothesis. Separate a measured cause from a
correlation. Use profiles and controlled changes when they can resolve that gap.
For an unexplained serving bottleneck or custom model/kernel work, use the
[vLLM diagnostic reference](references/vllm-diagnostics.md). Choose tools for
the hypothesis and record what each test establishes. Confirm gains with
unprofiled serving measurements.
When prefix reuse is poor, compare AgentX's reported actual and theoretical hit rates.
Check KV use and routing across data-parallel engines before blaming cache capacity.

Start with applicable correctness checks and Probe. Select each stage explicitly;
one invocation runs its checks and that stage, then stops. Advance only when the
result supports the hypothesis and the likely gain warrants more GPU time:

| Stage | Measured time | Additional warmup/lane | Load | Native override |
| --- | ---: | ---: | --- | --- |
| Probe | 180 s | 1 | min(C, 8) | unsafe; diagnostic |
| Screen | 900 s | 10 | C | none |
| Confirm | 3,600 s | 10 | selected C | none |

C is the target `--session-concurrency`. Start near the best known throughput
bend. Use the full pinned corpus at every stage. Override measured seconds,
warmup or effective load with `--duration`, `--warmup-requests-per-lane` or
`--concurrency`; inspect requested and effective settings in the plan. A shorter
Confirm cannot qualify. Probe and Screen cannot produce accepted recommendations.
There is no AgentX ladder score. Use matching evidence to choose experiments;
partial checks remain incomplete. This runner reruns the full endpoint checker
for every trial. Earlier passes do not replace those checks.

Coordinate shared GPU capacity through your team's process and read
[use-iris](../use-iris/SKILL.md) before GPU work. Stay within the agreed
concurrent ceilings across trials and retries. Let the runner's CPU coordinator
prepare uncached data before GPU allocation. Release idle resources and count setup, compilation, warmup, drain
and failed allocations. Prioritize GPU efficiency over elapsed time.
Keep `run` attached to track allocation cost. Detached runs may lose timestamps
after Pod cleanup. Retain known usage and label estimates and unknowns;
incomplete cost alone does not block a recommendation.

## Measure and return the result

Use the runner's shared recipe and upstream public-dataset selector names.
Preserve AgentX replay, warmup, cache busting and validity. Each attempt starts
a fresh server. Startup, checks, warmup and drain fall outside measured time.

Use the preset's [shared checks](../../../experiments/evaluation/inference_correctness/README.md)
against the actual settings. Reuse its immutable goldens across runtime changes.
For numerical failures, quantization or memory claims, use the
[numerical and memory reference](references/measurement.md#numerical-evidence-and-memory-claims).
The runner checks before AgentX and resets caches before native warmup. Retain
failed observations; missing required correctness evidence and failed checks block recommendations.
They stop traffic unless `--measure-failed-checks` explicitly permits diagnostics.
Use the agreed policy, including its provisional status. Change bounds only with
recorded evidence and delegated authority. These checks do not qualify RL learning
or trainer probability ratios; new investigations need their own scope.

Preserve native exports, actual settings, checks, failures, repeats, Iris
identities and allocation cost under the configured `MARIN_PREFIX` campaign.
Use the [research-engineering workflow](../research-engineering/SKILL.md) for
experimental code, the notebook and negative findings. Promote selected reusable
changes into draft contribution PRs; keep the research branch.

Report stage, native/protocol validity, numerical outcomes, latency, request
errors, evidence links and total allocated GPU cost separately. One window has
no measured repeatability. Default to one run; repeat promising Confirm trials
when noise could change the decision. One valid Confirm may be recommended with
repeatability labeled unmeasured. Use separate runner trials for fresh servers
and compare their rows. The runner does not expose native multi-run or confidence
aggregation. Native multi-run reuses the server and skips additional warmup
after the first by default.

The terminal report and `summary.md` must match. Select a valid Confirm point
with complete required checks explicitly to write `recommended.yaml` with its settings and
exact runtime. Explain why it wins against the baseline and label deferred RL
qualification. Use [manage-pr](../manage-pr/SKILL.md) for publication; preserve the user's merge boundary.

`report --recommend TRIAL_ID --baseline TRIAL_ID` takes IDs such as
`20261120T093456Z-5d738d2cdf86`. Status and recovery take full `--trial-uri` paths.
Compare runtime changes after both sides qualify under matching controls; a
numerical pass from an older runtime does not qualify the changed runtime.
