# Optional PPO/TIS score centering

Score centering (SC) adds a loss term to regular PPO with tokenwise truncated
importance sampling (TIS). SC and continuous head capture are off by default.
Supported ordinary TIS runs collect occasional hypothetical probes before users
opt into SC. The [design](https://echo.oa.dev/wiki/608) explains the correction.

This ports [archived SkyRL #968](https://github.com/marin-community/MarinSkyRL/pull/968)
through its reconciled source `f66a6fdccc5d1b9e8aa1c687e24d641a067b165d`.
That revision merges SkyRL main `82396a73` and adapts its published task-harness
API; the archived PR head remains `ef1adaf25`.
The active runtime remains under `vendor/skyrl`, with its independent lock,
isolated executable and verified Iris source bundle. See
[launching RL experiments](rl-launching.md) and
[SkyRL migration](../dev-guide/skyrl-migration.md) for packaging and provenance.

## Enable SC

Merge these settings into the launch document's `skyrl` recipe. Use local vLLM
and unpacked Megatron with tensor, context and sequence parallel sizes one.
Data and pipeline parallelism retain the existing whole-step normalization.

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

The `tis` preset caps the importance weight at two. A custom correction must
contain exactly one token truncation rule. SC rejects distillation, auxiliary
FTPO, other policy losses, remote inference and probability-changing sampling
filters. Ordinary configurations retain their existing support, including CP2.

For a capture-only control, keep `score_centering_topk: 128` and set
`score_centering_enabled: false`. This still captures heads and rescores them.
Ordinary TIS uses `score_centering_topk: 0`, `score_centering_enabled: false`
and `generator.sampling_params.logprobs: 0`. Set
`trainer.algorithm.score_centering_probe.interval: 0` to disable periodic probes.

The head contains the sampler's natural top K IDs and their full-vocabulary
probabilities. Megatron scores those same IDs at the same causal positions
before the update and in the training forward. The correction already
integrates TIS; applying the sampled action's TIS weight to it again changes
the gradient. Outside the head, old and behavior probabilities are modeled as
proportional to the current policy, preserving their measured omitted mass.

## Numeric transport

The pinned vLLM `39e62869693c` public `generate` interface supplies flat
logprobs. A request-scoped adapter packs owned NumPy arrays before they cross
Ray. The existing chat service renders prompts and parses messages and tools;
public chat responses keep their schema. K128 retains the 129-candidate
request because the chat display can include a sampled token outside the head.
The 128 head entries are selected by probability, then token ID, from the
displayed candidates; the sampled action is not forced into the head. The
adapter preserves the chat display limit and -9999 score floor. Abort
continuations retain exact token alignment, heads and source policy versions.
This adds no vLLM patch or dependency upgrade.

## Interpret probes

`trainer.algorithm.score_centering_probe` defaults to `interval: 10`,
`topk: 128`, `tokens_per_response: 32`. It selects at most one eligible prompt
group at startup and after each ten policy steps. It exports at most 32 valid
positions per response, uses finalized advantages, then discards probe tensors
before optimization. SC-off updates keep their ordinary inputs.

The 32-position bound limits export and rescoring. vLLM still collects a head
at every generated position in the selected group. One head request can also
make other requests in the same decode batch compute top K. An interval does
not guarantee an observation: normal admission can reject the selected group.

Metrics under `score_centering_probe/` report:

| Metrics | Meaning |
| --- | --- |
| `available`, `observed`, `sampled_tokens`, `coverage`, `seconds_since_previous` | Supported configuration and evidence actually consumed. Coverage divides sampled positions by this batch's loss tokens. One prompt group is not an unbiased population sample. |
| `hypothetical_loss_*`, `ordinary_loss_abs_mean` | Would-be scalar correction at the pre-update snapshot, with actual advantages. |
| `unit_advantage_loss_*`, `zero_advantage_fraction` | A separate signal for flat-reward groups. The unit-advantage value is not applied to training. |
| `*_retained_mass_mean`, `*_omitted_mass_mean`, `tail_floor_fraction` | Mass inside/outside K128 and use of the 1e-6 tail floor. Near unit head mass, FP32 subtraction can affect small scalar corrections. |
| `old_behavior_log_ratio_*`, `sampled_tis_clipped_fraction`, source-age metrics | Chosen-token mismatch and generating-weight age. Unstamped tokens remain unknown. |

Current and old trainer probabilities coincide at this snapshot. Fresh-source
disagreement measures engine mismatch at matching weights; stale-source
disagreement mixes engine mismatch and weight drift. Probes do not measure
later minibatches' current/old divergence. Scalar correction size and retained
mass do not establish parameter-gradient error or learning benefit.

Probes require ordinary regular PPO/TIS, local vLLM and unpacked Megatron at
TP/CP/SP1. Other configurations report unavailability. Continuous capture uses
`score_centering_diagnostic_tokens` instead: a positive value bounds the first
valid tokens per microbatch. Its `score_centering/sampled_*` metrics average
microbatch samples across ranks, rather than pooling tokens.

## Evidence and limits

The [retained qualification](https://github.com/yonromai/marin/tree/3b9f0183badb2c20d4c9bd4468489a442f9222e2/experiments/post_training/score_centering_integration)
preserves configurations, revisions, full reports, precision checks and costs.
Its 35-update Qwen check matched numeric captures to chat-logprob objects exactly at temperatures
1.0/0.7 and lengths 128/4096. Independent causal references differed by less
than 2e-6. These results apply to the unchanged SC kernel and numeric transport;
they do not qualify the migrated launcher or its published task harness.

Replacing chat-logprob objects with numeric arrays in matched Snowball controls
reduced SC cycles from 88.381 to 62.248 seconds (29.57%). SC overhead relative to
ordinary TIS fell from 53.00% to 9.27%. Each arm had one allocation,
one warmup and two measured cycles. The asynchronous arms each applied 67
updates, but SC response lengths differed (716 versus 1883 tokens). Their
3007 versus 6864 response tokens/s do not isolate a causal throughput gain.
Sparse probe observations do not establish the earlier aspirational 1%
overhead ceiling. No Snowball learning benefit is established.

The earlier [Countdown reference](https://github.com/yonromai/marin/blob/a745f814811e84481301bd61db965c1fdc4e87e3/experiments/post_training/score_centering_overnight/results/reference_results.json)
and [faithful SkyRL reproduction](https://github.com/yonromai/marin/blob/a745f814811e84481301bd61db965c1fdc4e87e3/experiments/post_training/score_centering_overnight/results/skyrl_faithful_v4/skyrl_faithful_results.json)
completed three noisy seed pairs and one no-noise control pair per implementation.
Both showed positive mean SC differences at the declared endpoint, with wide
paired intervals crossing zero. Evaluation uses the first 64 shuffled training
prompts after 299 updates; each run completes 300 updates. This successful
reproduction is separate from Snowball transfer. The learning campaign is
stopped; unfinished Qwen 8,192-token training-window and Snowball learning
comparisons remain incomplete. The
[experiment study (#996)](https://github.com/marin-community/MarinSkyRL/pull/996)
remains research.

## Grader release dependency

[Marin #10394](https://github.com/marin-community/marin/pull/10394) keeps the
scalar-grader repair separate. The nested lock selects Marin package family
`0.2.202.dev39324959133`. As of 21 October 2026, that Verifyit wheel and the newest
published `0.2.215.dev39586101863` wheel lack the repair. Ordinary retained replies
keep their grades, but a matrix-root candidate can still raise `TypeError`.
This draft is not integration-ready for that failure mode.

After the fix merges and a compatible coupled release is published, raise the
nested public dependency floors, update the exact Rigging override, refresh the
independent lock, and inspect the code installed in both launcher and worker
environments. Repeat grader regressions, retained-reply replay and wheel/runtime
acceptance. Marin's root source and a local `PYTHONPATH` injection do not supply
a deployable repair to the nested runtime.
