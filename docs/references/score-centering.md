# PPO/TIS score centering and periodic probes

Score centering (SC) adds a correction to ordinary PPO with tokenwise truncated
importance sampling (TIS). SC is off by default. Training–inference mismatch
diagnostics use chosen-token probabilities independently of SC.
Ordinary PPO/TIS also collects periodic hypothetical SC probes on supported
configurations. Probes leave the training objective and optimizer inputs unchanged.

## Adjust probes

The default marks the next dispatched prompt group at startup and after every
ten published policy steps. A group contains the answers to one prompt. Capture
uses K128 and at most 32 loss-eligible positions per response. A policy step
numbers a learner batch. Set interval `0` to disable
probes; K is shared with SC:

```yaml
skyrl:
  trainer:
    algorithm:
      score_centering_top_k: 128
      score_centering_probe:
        interval: 10
```

Positions are evenly spaced over the list of loss-eligible tokens, including
the first and last. Later trainer masks can remove selected evidence. Normal
batch admission and reward-selection checks can reject the group, so scheduled
requests need not become observations. The position bound limits
export and trainer scoring. vLLM collects heads at every generated position in
the selected group, and a head request can make neighboring requests compute
top K. Sampled-token-only outputs discard unused candidates before CPU token
formatting. No automatic cadence controller is installed.

Probes use SC's PPO/TIS, sampling and Megatron restrictions below. They are
unavailable when SC is enabled. The trainer uses finalized advantages, summarizes
an isolated head tensor, then removes probe tensors before optimization.

| `score_centering_probe/` fields | Interpretation |
| --- | --- |
| `available`, `interval`, `observed`, `sampled_tokens`, `coverage` | Support, requested cadence and consumed evidence. Coverage divides sampled positions by batch loss tokens; one prompt group is not an unbiased population sample. |
| `steps_since_previous`, `seconds_since_previous` | Spacing between consumed observations. Steps advance on intervening batches; seconds appear at the next observation. |
| `sampled_response_length_{mean,max}`, `first256_tokens`, `last256_tokens`, `long_response_tokens`, `long_responses` | Position counts pool selected tokens across responses. The mean weights each sampled token's response length. Long means at least 2,048 response tokens. |
| `{fresh,stale,unknown}_source_tokens`, `long_response/{fresh,stale,unknown}_tokens`, source-age summaries | Source age is learner step minus generating-policy stamp. Fresh is zero; stale is positive; negative stamps are unknown. Empty subsets omit means. |
| `hypothetical_loss_*`, `ordinary_loss_abs_mean`, `unit_advantage_loss_*`, `zero_advantage_fraction` | Unweighted sample means of additive SC and ordinary PPO/TIS losses. Unit-advantage companions explain flat-reward groups and are never applied to training. |
| `logit_correction_l1_mean`, `ordinary_logit_gradient_l1_mean`, `logit_correction_l1_ratio`, `unit_advantage_logit_correction_l1_mean` | Hypothetical output-gradient L1 norms, including the modeled tail. The ratio divides sums over the same positions and is omitted for a zero denominator. |
| `old_behavior_log_ratio_*`, `sampled_tis_clipped_fraction`, `*_omitted_mass_mean`, `tail_floor_fraction` | Log(old chosen-token probability / behavior probability), the fraction above log(TIS cap), omitted current/behavior mass and use of the 1e-6 tail floor. |

Probes score before the optimizer update, so current and old learner policies
coincide. Fresh-source disagreement measures sampler/learner numerical mismatch;
stale sources also include weight drift. Output-gradient sensitivity does not establish
parameter-gradient error or learning benefit. Small omitted masses are sensitive
to FP32 subtraction, even just above the floor; inspect mass and floor use together.
Increase the interval when capture is costly, then check consumed observation
spacing and fresh/stale, position and length coverage.

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
training; set the probe interval to zero for a control without candidate heads.

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

The [periodic Grug cost study](https://github.com/marin-community/marin-logbooks/issues/29)
and [packed CP5 qualification](https://wandb.ai/marin-community/score-centering-cp-packing/artifacts/qualification/packed-cp5-qualification-01a16b97/v0)
retain measurements and their methods. The
[dated Snowball report](https://github.com/yonromai/marin/blob/14dae4c5e21179fd1b664e97ae6104b21837d6cd/docs/reports/score-centering-tim.md)
contains earlier experiments. Cost results qualify their pinned instrumented
source and workload. Current training omits capture-stage timers and payload-byte
accounting. No Snowball learning benefit is established. See [SkyRL runtime](../dev-guide/skyrl.md)
for the independent lock, editable dependencies and Iris workspace bundle.
vLLM remains pinned to `39e62869693c`.
