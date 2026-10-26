# PPO/TIS mismatch and optional score centering

Training–inference mismatch (TIM) compares the trainer's and sampler's token
probabilities. Ordinary PPO with tokenwise truncated importance sampling (TIS)
logs TIM and collects periodic hypothetical score-centering probes on supported
configurations. Score centering (SC) is off by default. Probes report a scalar
correction without applying it to training.

## Read TIM

Check counts before comparing means. Empty buckets omit means; absence does
not establish agreement.

| Metrics | Interpretation |
| --- | --- |
| `policy/mismatch/pooled/` | Old trainer minus behavior sampler log probabilities, before this update. Check `selected_tokens`, `finite_tokens`, `missing_behavior`, `log_ratio_abs_mean`, `log_ratio_abs_p99` and `frac_outside_0_5_2`. |
| `policy/mismatch/staleness0/` | Tokens stamped with this learner snapshot's version. Disagreement at matching weights can indicate engine mismatch. |
| `policy/mismatch/staleness1/` through `staleness8+/` | Older generating versions; disagreement includes weight drift. Age buckets are 1, 2, 3, 4–7 and 8+. |
| `policy/mismatch/unknown/` | No generating-policy stamp. A rollout lease bounds age but does not establish matching weights. Check `async/staleness_tokens_stamped` for stamp coverage. |
| `policy/correction/truncated_fraction` | Loss tokens whose uncapped TIS weight exceeded the cap. |
| `policy/ppo_clip_ratio*`, `policy/log_ratio_*` | PPO clipping and current-versus-old trainer changes during optimization. |

Each mismatch bucket also reports `pos_first256/`, `pos_last256/` and
`response_ge2048/` counts and absolute mean log ratios. Positions include masked
response tokens. First and last windows overlap on short responses. Inspect
long-response counts before interpreting their means. Selected and finite counts
remain separate.

## Adjust probes

The default is one eligible prompt group at startup and after every 10 published
policy steps, with 128 candidates and at most 32 positions per response. A policy
step numbers a learner batch, not an optimizer microbatch. Change only the
frequency; `0` disables probes:

```yaml
skyrl:
  trainer:
    algorithm:
      score_centering_probe:
        interval: 10
```

Selection spans evenly from the first to last loss-eligible response position.
Later trainer masks can remove selected positions. The trainer uses finalized
advantages and removes probe evidence before optimization. Probes leave the
ordinary TIS objective and optimizer inputs unchanged.

The position bound limits export and trainer scoring. vLLM still collects heads
at every generated position in the selected group, and a head request can cause
other requests in the decode batch to compute top K. Admission can reject the
selected group. Scheduled requests therefore need not become observations.
There is no automatic cadence controller.

| `score_centering_probe/` fields | Meaning |
| --- | --- |
| `available`, `interval`, `observed`, `sampled_tokens`, `coverage` | Support, requested frequency and consumed evidence. Coverage divides sampled positions by batch loss tokens. One prompt group is not an unbiased population sample. |
| `steps_since_previous`, `seconds_since_previous` | Spacing between consumed observations. Step spacing advances on intervening batches; seconds appear when another observation arrives. |
| `sampled_response_length_mean`, `sampled_response_length_max`, `first256_tokens`, `last256_tokens`, `long_response_tokens`, `long_responses` | Position and length coverage. The mean weights each sampled token's response length; long means at least 2,048 response tokens. |
| `fresh_source_tokens`, `stale_source_tokens`, `unknown_source_tokens`, `long_response/{fresh,stale,unknown}_tokens` | Coverage by actual generating-policy stamp. Empty source subsets omit means. |
| `hypothetical_loss_*`, `ordinary_loss_abs_mean` | Additive SC correction with actual advantages and absolute TIS/PPO loss on the same positions, as unweighted sample means. |
| `unit_advantage_loss_*`, `zero_advantage_fraction` | Companion calculation for flat-reward groups. Unit advantages are never applied to training. |
| `*_omitted_mass_mean`, `tail_floor_fraction` | Mass outside K128 and use of the 1e-6 tail floor. |
| `old_behavior_log_ratio_*`, `sampled_tis_clipped_fraction`, source-age means | Chosen-token mismatch, TIS truncation and generating-policy age within the sample. |

Current and old trainer probabilities coincide at this snapshot. A stale probe
also includes weight drift; it does not measure later current/old divergence.
Scalar correction size does not establish parameter-gradient error or learning
benefit. Small omitted masses are sensitive to FP32 subtraction, even just above
the tail floor. Inspect omitted mass as well as floor usage.

Probes require regular PPO/TIS, local vLLM and Megatron with tensor parallelism
one and sequence parallelism disabled. CP=1 supports padded or packed rows;
CP>1 requires packing. Data, pipeline and expert parallelism retain their
ordinary layout constraints. Other configurations report probe
unavailability. Probes are unavailable when SC is enabled.

## Read capture cost

Logs expose logical payload sizes and host stage durations:

| `score_centering_capture/` fields | Scope |
| --- | --- |
| `requests`, `probe_requests` | Admitted capturing prompt groups and their probe subset. Rejected groups are absent. |
| `captured_tokens`, `captured_head_bytes` | Heads before probe selection. K128 uses 1,024 numeric bytes per captured token. |
| `exported_head_bytes` | Numeric heads and positions after selection, before storage; excludes Python and storage protocol overhead. |
| `collection_seconds_sum`, `assembly_seconds_sum` | Request generation/grading and trajectory assembly. Collection includes ordinary generation; the later payload-store write is outside these stages. |
| `learner_input_bytes`, `learner_returned_head_bytes` | Collated tensors, including padding and position/source arrays, and returned old-trainer head scores. These are not measured network traffic. |

`timing/score_centering_collation`, `timing/score_centering_probe_collation` and
`timing/score_centering_probe_summary` separate driver work. Stage sums can overlap
training and cannot establish added cycle time. No extra device synchronization
is added. Increase `interval` when capture is expensive, then check observation
spacing and fresh/stale, length and position coverage.

## Enable SC

Merge these settings into the launch document's `skyrl` recipe:

Megatron supports CP=1 with either padded rows or sample packing. Context
parallelism requires sample packing. Tensor parallelism must be one. The
Megatron provider enables sequence parallelism only with tensor parallelism;
independent sequence parallelism is outside this support. Data, pipeline and
expert parallelism retain their ordinary layout constraints.

```yaml
skyrl:
  trainer:
    strategy: megatron
    use_sample_packing: true
    algorithm:
      policy_loss_type: regular
      off_policy_correction: tis
      score_centering_enabled: true
    policy:
      sequence_parallel_size: 1
      megatron_config:
        tensor_model_parallel_size: 1
        context_parallel_size: 5
  generator:
    backend: vllm
    run_engines_locally: true
```

The flag selects K128 capture and trainer rescoring automatically. The `tis`
preset caps importance weights at two; a custom correction must contain exactly
one token truncation rule. SC rejects distillation, auxiliary FTPO, other policy
losses, remote inference and probability-changing sampling
filters. Temperature-only sampling is supported. Disabling the flag returns to
ordinary training; set the probe interval to zero for a control without heads.

The head contains the sampler's natural top 128 token IDs and their
probabilities normalized over the full vocabulary. Megatron scores those IDs at
the same causal positions before and during optimization. Outside the head, old and behavior probabilities are
modeled as proportional to the current policy, preserving measured omitted mass.
SC subtracts the sampler's expected clipped, TIS-weighted score contribution.
The correction's probability weights and clipping decisions are detached. It
already integrates TIS and must not receive the sampled action's TIS weight
again. See the
[paper](https://arxiv.org/html/2609.20807v1) and
[design note](https://marina.oa.dev/echo/wiki/608).

SC uses PPO's policy-data weights and whole-step loss reduction. With
`token_mean`, it divides the weighted sum by the weighted token count across
microbatches and data-parallel ranks. CP ranks reconstruct the same token loss
while retaining gradients only for their local logits. Megatron's CP gradient
reduction and pipeline scaling preserve the whole-step mean. Only differences
between requested candidate logits and the chosen-token logit cross CP ranks;
vocabulary logits stay local.

## Packed CP5 qualification

The [2026-10-26 qualification][cp-qualification] used Grug 67B-A2B Datakit SFT
on 80 H100 learners: packed CP5, TP1, PP2, EP8 and DP8, with a colocated frozen
reference and eight separate inference H100s. Each data-parallel replica replayed
102,400 non-padding input tokens, including 49,152 response tokens, sharded
across its five CP ranks. All 80 learners completed scoring and backpropagation
with nonzero local gradient norms. Replays used balanced expert routes; live
updates used vLLM's recorded routes.

The comparison held tokens, weights and RNG state fixed, with one warmup and
five repeats per arm. Each time adds the slowest learner's input-evidence copy
to the slowest learner's scoring/training window. That window includes old-policy
scoring, a returned-score host round trip, and training forward/backward. It
excludes reference scoring, optimizer updates and common gradient clearing.
Intervals are 95% Student-t intervals; extra time uses paired repeats against
TIS. These stage measurements do not estimate a full production update.

| Arm | Measured stages, seconds | Extra time, 95% interval | Peak allocated GiB |
| --- | --- | --- | --- |
| TIS, heads off | 5.235 ± 0.019 | baseline | 45.44 |
| SC, K128 | 5.298 ± 0.051 | +1.20% [−0.06%, 2.45%] | 45.60 |
| TIS with probes | 5.267 ± 0.061 | +0.61% [−0.65%, 1.86%] | 45.44 |

Memory is the maximum PyTorch tensor allocation across all 80 learners.
Reserved memory was 56.69 GiB in every arm because the allocator cache was
retained. These counters do not measure physical GPU headroom. Both overhead
intervals include zero. SC copied 72 MiB of input evidence per learner
in 22.0 ms and returned a 24 MiB head per response; the host round trip
averaged 554 ms, including chosen scores and pipeline transport.

The controlled probe arm sampled up to 32 positions in every replayed response,
which is denser than the live one-group schedule. After masking, 248 positions
per repeat spanned every CP response shard. Probe input copies averaged
0.225 ms, returned heads used 16 KiB per response, and CPU summaries averaged
1.45 ms. The table includes the summaries.

The live workload used eight prompts with four responses each, compared with
512 prompts and 16 responses in the reference recipe. Inference used eight
GPUs compared with 48. Six updates per arm exercised ordinary RLOO-N, PPO/TIS,
entropy, frozen-reference KL and optimizer updates; the first update was warmup.
The short CatCount workloads and response lengths differed between arms. The
record includes end-to-end timings and capture payloads; their differences
cannot establish causal overhead or learning benefit.

Live probes requested interval 1, leaving the runtime default of 10 unchanged.
They consumed 512 positions after warmup, across three of five measured
updates: 0.165% of 310,435 loss tokens. Of those positions, 192 came from
responses of at least 2,048 tokens; sampled responses reached 11,892 tokens.
Capture produced 41.35 MiB before selection and exported 514 KiB. Collated
learner inputs used 3.04 MiB and returned heads used 1.50 MiB, including padding.
These are logical payload sizes. Scheduled groups can be rejected or arrive in
a later update, so interval 1 does not guarantee an observation in every update.

## Runtime and evidence

SkyRL uses its independent lock, editable Marin dependencies and the standard
Iris workspace bundle. See [SkyRL runtime](../dev-guide/skyrl.md). vLLM remains
pinned to `39e62869693c`, without a patch. The request adapter transports owned
NumPy arrays while public chat responses retain their schema. K128 requests
129 candidates to preserve the natural head when the sampled token lies outside
it. Selection preserves the display limit, probability/token-ID order and -9999
floor. Abort continuations preserve exact token alignment and policy stamps.

The [dated Snowball report][snowball-report] retains cost measurements and full
experimental records. Those measurements qualify their pinned source, not later
code changes. That report's async probes sampled no responses of at least 2,048
tokens; the CP5 qualification above includes live observations from longer
responses. No Snowball learning benefit is established. Continuous capture without SC, training-time microbatch
diagnostics and variable head/position sizes were retired from the runtime;
matched research controls remain at the report's immutable revisions.

The scalar-grader repair [#10394](https://github.com/marin-community/marin/pull/10394)
is merged into the integrated base. Its nonscalar guard precedes SymPy's scalar
finiteness check. Editable dependencies carry that source in the workspace bundle.

[snowball-report]: https://github.com/yonromai/marin/blob/14dae4c5e21179fd1b664e97ae6104b21837d6cd/docs/reports/score-centering-tim.md

[cp-qualification]: https://wandb.ai/marin-community/score-centering-cp-packing/artifacts/qualification/packed-cp5-qualification-01a16b97/v0
