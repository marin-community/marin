# PPO/TIS mismatch diagnostics and optional score centering

Training–inference mismatch (TIM) is a difference between the trainer's and
sampler's token probabilities. Ordinary PPO with tokenwise truncated importance
sampling (TIS) logs this mismatch and, on supported configurations, collects
periodic hypothetical score-centering probes. Score centering (SC) remains off
by default. A probe measures a possible scalar correction without applying it.

## Read the normal training metrics

Start with counts, then compare mismatch at known policy ages. Empty buckets
omit their means; a missing mean does not establish agreement.

| Metric family | What to check |
| --- | --- |
| `policy/mismatch/pooled/` | `selected_tokens`, `finite_tokens`, `missing_behavior`, `log_ratio_abs_mean`, `log_ratio_abs_p99`, `frac_outside_0_5_2`. The log ratio is old trainer minus behavior sampler, before this update. |
| `policy/mismatch/staleness0/` | The same comparison for tokens stamped with this learner snapshot's policy version. Disagreement at matching weights can indicate engine mismatch. |
| `policy/mismatch/staleness1/` through `staleness8+/` | Older generating versions. Disagreement also includes weight drift. The intermediate age buckets are 1, 2, 3 and 4–7. |
| `policy/mismatch/unknown/` | Tokens without a generating-policy stamp. A rollout lease can bound their age, but cannot establish matching weights. |
| `async/staleness_tokens_*` | Token-age and stamp coverage. Check `stamped` before interpreting fresh or stale subsets. Colocated generation can have no token stamps. |
| `policy/correction/truncated_fraction` | Fraction of loss tokens whose uncapped TIS weight exceeded the configured cap. This differs from PPO clipping during optimization. |
| `policy/ppo_clip_ratio*`, `policy/log_ratio_*` | PPO clipping and current-versus-old trainer probability changes during optimization. These describe a different comparison from trainer-versus-sampler TIM. |

Each mismatch age bucket, including `pooled` and `unknown`, also reports
`pos_first256/`, `pos_last256/` and `response_ge2048/` counts and absolute mean
log ratios. Window boundaries count all response positions, including positions
masked out of the loss. First and last windows overlap on short responses.
The long-response subset contains responses with at least 2,048 tokens; inspect
its token count before comparing means. These summaries use loss tokens with
finite probabilities and preserve their selected/finite counts separately.

## Adjust periodic probes

These settings are already the defaults for supported ordinary TIS runs:

```yaml
skyrl:
  trainer:
    algorithm:
      score_centering_probe:
        interval: 10
        topk: 128
        tokens_per_response: 32
```

A positive interval spaces capture requests by published policy steps. `0`
disables probes. `topk` and `tokens_per_response` accept positive integers;
the qualified settings are K128 and 32 positions. Changing K changes the
approximation and payload size; the measurements here do not qualify other K.
The scheduler selects at most one eligible prompt group at startup and after
each interval. It exports at most the configured position limit per response,
uses finalized advantages, then removes the evidence before the
optimizer update. This sampling does not change the ordinary TIS objective.
There is no automatic cadence controller.

At assembly, selection spans evenly from the first to the last loss-eligible
response position. A one-position sample takes the first eligible position.
Tool or other masked positions are excluded. Later trainer masks can remove
selected positions before the summary.

The bound limits export and trainer scoring. vLLM still collects a head at
every generated position in the selected group. A head request can also cause
other requests in the same decode batch to compute top K. Admission can reject
the selected group, so a scheduled request need not become an observation.

Metrics under `score_centering_probe/` report:

| Metrics | Meaning |
| --- | --- |
| `available`, `interval`, `observed`, `sampled_tokens`, `coverage` | Support, configured cadence and evidence actually consumed. Coverage divides sampled positions by this batch's loss tokens. One prompt group is not an unbiased population sample. |
| `steps_since_previous`, `seconds_since_previous` | Spacing between consumed observations. Step spacing also advances on intervening batches; seconds are emitted when another observation arrives. |
| `sampled_response_length_mean`, `sampled_response_length_max`, `first256_tokens`, `last256_tokens`, `long_response_tokens`, `long_responses` | Length and position coverage. The length mean weights each sampled token's response length; long means at least 2,048 response tokens. |
| `fresh_source_tokens`, `stale_source_tokens`, `unknown_source_tokens`, `long_response/{fresh,stale,unknown}_tokens` | Coverage by actual generating-policy stamp, including long responses. Means for empty source subsets are omitted. |
| `hypothetical_loss_*`, `ordinary_loss_abs_mean` | The additive SC correction, with actual advantages. `hypothetical_loss_abs_mean` averages its absolute per-token value over valid sampled positions. The ordinary value averages the absolute TIS-weighted PPO loss on those same positions. Both use the pre-update snapshot and unweighted sample means. |
| `unit_advantage_loss_*`, `zero_advantage_fraction` | A separate calculation for flat-reward groups. Unit advantages are never applied to training. |
| `*_retained_mass_mean`, `*_omitted_mass_mean`, `tail_floor_fraction` | Probability mass inside/outside K128 and use of the 1e-6 tail floor. FP32 subtraction affects very small omitted masses. |
| `old_behavior_log_ratio_*`, `sampled_tis_clipped_fraction`, source-age means | Chosen-token mismatch, TIS clipping and generating-weight age within the sample. |

Current and old trainer probabilities coincide at this snapshot. A fresh probe
can expose engine disagreement; a stale probe also includes weight drift. It
does not measure current/old divergence in later minibatches. Scalar correction
size, retained mass and zero-advantage probes do not establish parameter-gradient
error or learning benefit.

Probes require regular PPO/TIS, local vLLM, and unpacked Megatron with tensor,
context and sequence parallel sizes one. Other ordinary configurations retain
their behavior and report probe unavailability. Data and pipeline parallelism
are supported. Continuous capture requires `score_centering_topk: 128` and
matching `generator.sampling_params.logprobs: 128`, as in the SC example below.
`score_centering_enabled` controls whether its correction is applied. With
continuous capture configured, periodic probes report unavailability. Set
`score_centering_diagnostic_tokens` to a positive value for continuous-capture
diagnostics; it bounds the first valid tokens per microbatch. Their
`score_centering/sampled_*` values average microbatch samples across ranks,
rather than pooling all tokens.

## Read capture cost

Normal logs expose logical payload sizes and host stage durations. The
`score_centering_capture/` fields cover admitted requests that requested heads:

| Fields | Scope |
| --- | --- |
| `requests`, `probe_requests` | Capturing prompt-group requests, and the subset used for periodic probes. Rejected groups are absent from these consumed-batch totals. |
| `captured_tokens`, `captured_head_bytes` | Full generated heads before probe selection: NumPy ID and FP32 score array bytes. K128 uses 1,024 numeric bytes per captured token. |
| `exported_head_bytes` | Numeric heads and probe positions after selection, before rollout storage. This excludes Python objects and storage protocol overhead. |
| `collection_seconds_sum`, `assembly_seconds_sum` | Sums of request generation/grading and trajectory-assembly durations. Collection includes ordinary generation work. The later rollout-payload store write is outside these stages. |
| `learner_input_bytes`, `learner_returned_head_bytes` | Logical collated input tensors, including padding and source/position arrays, and returned old-trainer head scores. These are not measured network traffic. |

Existing `timing/score_centering_collation`,
`timing/score_centering_probe_collation` and
`timing/score_centering_probe_summary` separate driver work. The forward and
training timers include their normal device work. No extra GPU synchronization
is added for these counters or timers.

Request stages can overlap one another and training. Their sums cannot be
subtracted from a cycle to obtain added training time. Compare matched full
cycles to estimate overhead. To reduce probe work, increase `interval`; use
`observed` and actual observation spacing to check that useful samples still
arrive. Check long and fresh/stale coverage before reducing frequency further.

## Enable SC

Merge these settings into the launch document's `skyrl` recipe:

```yaml
skyrl:
  trainer:
    strategy: megatron
    use_sample_packing: false
    algorithm:
      policy_loss_type: regular
      off_policy_correction: tis
      score_centering_topk: 128
      score_centering_enabled: true
      score_centering_diagnostic_tokens: 0
    policy:
      sequence_parallel_size: 1
      megatron_config:
        tensor_model_parallel_size: 1
        context_parallel_size: 1
  generator:
    backend: vllm
    run_engines_locally: true
    sampling_params:
      logprobs: 128
      temperature: 1.0
      top_p: 1.0
      top_k: -1
      min_p: 0.0
```

The `tis` preset caps importance weights at two. A custom correction must
contain exactly one token truncation rule. SC rejects distillation, auxiliary
FTPO, other policy losses, remote inference, packed samples and
probability-changing sampling filters. Its TP/CP/SP1 layout guard applies even
when capture is enabled only for a control. SC uses PPO's policy-data weights
and whole-step loss reduction. With `token_mean`, it divides the weighted loss
sum by the weighted token count across microbatches and data-parallel ranks;
pipeline scaling preserves this mean. Ordinary configurations keep their existing
support, including CP2.

For a capture-only control, keep K128 and set `score_centering_enabled: false`.
Ordinary TIS uses `score_centering_topk: 0`, `score_centering_enabled: false`
and `generator.sampling_params.logprobs: 0`. Disable periodic probes as well
when constructing a control with no head collection.

The head contains the sampler's natural top K IDs and full-vocabulary
probabilities. Megatron scores those same IDs at the same causal positions
before the update and during optimization. SC already integrates TIS; applying
the sampled action's TIS weight again changes its gradient. Outside the head,
old and behavior probabilities are modeled as proportional to the current
policy, preserving measured omitted mass. See the
[paper](https://arxiv.org/html/2609.20807v1) and
[design note](https://marina.oa.dev/echo/wiki/608).

## Runtime and evidence

The runtime lives under `vendor/skyrl`, with its independent lock and isolated
executable. Marin libraries are editable dependencies from the same checkout;
Iris uses the standard Marin workspace bundle. See
[SkyRL runtime](../dev-guide/skyrl.md) and [RL launching](rl-launching.md).
The vLLM pin remains `39e62869693c`, with no dependency upgrade or patch.

The request adapter packs owned NumPy arrays before Ray transport. Public chat
responses keep their schema. K128 requests 129 candidates because the display
can include a sampled token outside the natural head. Selection orders by
probability, then token ID; it preserves the display limit and -9999 score
floor. Abort continuations preserve exact token alignment and policy stamps.
Dense response alignment reuses contiguous arrays; masked positions retain
explicit padding.

The [retained qualification](https://github.com/yonromai/marin/tree/3b9f0183badb2c20d4c9bd4468489a442f9222e2/experiments/post_training/score_centering_integration)
preserves earlier configurations, precision checks and costs. Its Qwen check
matched numeric and chat captures at temperatures 1.0/0.7 and lengths 128/4096;
independent causal references differed by less than 2e-6. Those references apply
to the unchanged SC calculation and numeric representation.

Earlier serialized Snowball controls measured SC cycle wall time 9.27% above
ordinary TIS after replacing chat objects with arrays. They used one allocation,
one warmup and two measured cycles per arm. Separate asynchronous arms had
different response lengths, so their throughput comparison does not isolate
overhead. These earlier measurements do not qualify
the current runtime or establish a 1% probe-overhead ceiling.

The earlier [Countdown reference](https://github.com/yonromai/marin/blob/a745f814811e84481301bd61db965c1fdc4e87e3/experiments/post_training/score_centering_overnight/results/reference_results.json)
and [SkyRL reproduction](https://github.com/yonromai/marin/blob/a745f814811e84481301bd61db965c1fdc4e87e3/experiments/post_training/score_centering_overnight/results/skyrl_faithful_v4/skyrl_faithful_results.json)
completed three noisy seed pairs and one no-noise pair per implementation.
Their mean SC differences were positive, with wide paired intervals crossing
zero. Evaluation used the first 64 shuffled training prompts after 299 updates;
each run completed 300 updates. The learning campaign is stopped. Unfinished
Qwen 8,192-token training-window and Snowball learning comparisons remain
incomplete. [SkyRL #996](https://github.com/marin-community/MarinSkyRL/pull/996)
remains research. No Snowball learning benefit is established.

## Grader dependency

[Marin #10394](https://github.com/marin-community/marin/pull/10394) keeps the
scalar-grader repair separate. It rejects a nonscalar matrix-root candidate
before SymPy's scalar finiteness check. A checkout without that source fix can
still raise `TypeError` while grading a rollout.

Editable source dependencies make the repair available to a worker when it is
present in the bundled checkout. A package release or dependency-floor update
is unnecessary. The current Qwen check verified the guard in its installed
worker source using an explicit overlay of #10394. That overlay is separate
from this feature; deployment still needs the repair in its checkout.
