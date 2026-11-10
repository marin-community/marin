# PPO/TIS score centering

Score centering (SC) adds a correction to ordinary PPO with tokenwise truncated
importance sampling (TIS). SC is off by default. Training–inference mismatch
diagnostics use chosen-token probabilities independently of SC.

## Enable SC

Merge these settings into the launch document's `skyrl` recipe:

```yaml
skyrl:
  trainer:
    strategy: megatron
    use_sample_packing: true
    algorithm:
      policy_loss_type: regular
      off_policy_correction: tis
      score_centering_enabled: true
      score_centering_top_k: 128
    policy:
      sequence_parallel_size: 1
      megatron_config:
        tensor_model_parallel_size: 1
        context_parallel_size: 5
  generator:
    backend: vllm
    run_engines_locally: true
```

CP=1 supports padded or packed rows. CP>1 requires packing. Tensor and sequence
parallel sizes must be one; data, pipeline and expert parallelism retain their
ordinary layout constraints. SC rejects distillation, auxiliary FTPO, other
policy losses and remote inference. Behavior-logprob training requires
full-distribution, temperature-only sampling; probability-changing filters are
rejected at launch. The `tis` preset caps importance weights at two. A custom
correction must contain exactly one token truncation rule.

`score_centering_top_k` sets the positive capture and rescoring width, independently
of sampling's `top_k`. Its default is 128. K32 reduces numeric head bytes by 75%
but omits more probability mass. Neither width guarantees an accurate correction
on a broad distribution. Disable `score_centering_enabled` to return to ordinary
training without candidate heads.

## Correction and alignment

The head contains the sampler's natural top K token IDs and their probabilities
normalized over the full vocabulary. Behavior is the generating policy; old is
the learner before optimization; current is the learner being optimized. Megatron
scores the same IDs at the same causal positions before and during optimization.
Outside the head, old and behavior probabilities are modeled as proportional to
the current policy, preserving their measured omitted masses.

Here, score means the gradient of the current-policy token log probability.

SC subtracts the sampler's expected clipped, TIS-weighted score contribution.
Its probability coefficients and clip decisions are detached; gradients flow
through current-policy log probabilities. The correction already integrates TIS
and must not receive the sampled action's TIS weight again. See the
[paper](https://arxiv.org/html/2609.20807v1) and
[design note](https://marina.oa.dev/echo/wiki/608).

The correction uses PPO's policy-data weights and whole-step loss reduction.
With `token_mean`, it divides by the weighted token count across microbatches
and data-parallel ranks. CP ranks reconstruct the token loss while retaining
gradients only for their local logits. Megatron's CP gradient reduction and
pipeline scaling preserve the whole-step mean. Candidate/chosen logit differences
cross CP ranks; vocabulary logits stay local.

The compact chat adapter transports owned NumPy arrays through existing vLLM
APIs. K128 requests 129 candidates to preserve the natural head when the sampled
token lies outside it. Token log probabilities retain ordinary chat's −9999 floor. Candidates sort by decreasing
probability, then token ID. Aborted continuations retain response-token
alignment and policy stamps. Public chat responses keep their schema.

## Evidence

The [Grug cost study](https://github.com/marin-community/marin-logbooks/issues/13)
and [packed CP5 qualification](https://wandb.ai/marin-community/score-centering-cp-packing/artifacts/qualification/packed-cp5-qualification-01a16b97/v0)
retain measurements and their methods. The
[dated Snowball report](https://github.com/yonromai/marin/blob/14dae4c5e21179fd1b664e97ae6104b21837d6cd/docs/reports/score-centering-tim.md)
contains earlier experiments. Cost results qualify their pinned instrumented
source and workload. Current training omits capture-stage timers and payload-byte
accounting. No Snowball learning benefit is established. See [SkyRL runtime](../dev-guide/skyrl.md)
for the independent lock, editable dependencies and Iris workspace bundle.
vLLM remains pinned to `39e62869693c`.
