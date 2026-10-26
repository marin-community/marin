# PPO/TIS mismatch and optional score centering

Training–inference mismatch (TIM) compares the trainer's and sampler's token
probabilities. Ordinary PPO with tokenwise truncated importance sampling (TIS)
logs TIM. Score centering (SC) is off by default and can be enabled for supported
configurations.

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

## Read capture cost

Logs expose logical payload sizes and host stage durations:

| `score_centering_capture/` fields | Scope |
| --- | --- |
| `requests` | Admitted capturing prompt groups. Rejected groups are absent. |
| `captured_tokens`, `captured_head_bytes` | Captured behavior heads. K128 uses 1,024 numeric bytes per captured token. |
| `exported_head_bytes` | Numeric heads before storage; excludes Python and storage protocol overhead. |
| `collection_seconds_sum`, `assembly_seconds_sum` | Request generation/grading and trajectory assembly. Collection includes ordinary generation; the later payload-store write is outside these stages. |
| `learner_input_bytes`, `learner_returned_head_bytes` | Collated tensors, including padding, and returned old-trainer head scores. These are not measured network traffic. |

`timing/score_centering_collation` measures driver collation. Stage sums can
overlap training and cannot establish added cycle time. No extra device
synchronization is added.

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
ordinary training without candidate-head capture.

The head contains the sampler's natural top K IDs and full-vocabulary
probabilities. Megatron scores those IDs at the same causal positions before
and during optimization. Outside the head, old and behavior probabilities are
modeled as proportional to the current policy, preserving measured omitted mass.
SC subtracts the sampler's expected clipped, TIS-weighted score contribution.
Its coefficient is detached. It already integrates TIS and must not receive the
sampled action's TIS weight again. See the
[paper](https://arxiv.org/html/2609.20807v1) and
[design note](https://marina.oa.dev/echo/wiki/608).

SC uses PPO's policy-data weights and whole-step loss reduction. With
`token_mean`, it divides the weighted sum by the weighted token count across
microbatches and data-parallel ranks. CP ranks reconstruct the same token loss
while retaining gradients only for their local logits. Megatron's CP gradient
reduction and pipeline scaling preserve the whole-step mean. Only requested
candidate-score differences cross CP ranks; vocabulary logits stay local.

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
code changes. No Snowball learning benefit is established. Research controls
remain at the report's immutable revisions.

The scalar-grader repair [#10394](https://github.com/marin-community/marin/pull/10394)
is merged into the integrated base. Its nonscalar guard precedes SymPy's scalar
finiteness check. Editable dependencies carry that source in the workspace bundle.

[snowball-report]: https://github.com/yonromai/marin/blob/14dae4c5e21179fd1b664e97ae6104b21837d6cd/docs/reports/score-centering-tim.md
