# Score centering in fully asynchronous RL

Study started 2026-09-19. This document records the exact experiment contract and results as
they become available. The question is whether score centering lets the learner use older sampled
tokens without giving up completed-answer quality, and whether that extra tolerance saves time or
GPU work.

## Implementation and frozen inputs

- MarinSkyRL branch `goal/score-centering-01a0bb6f`, pinned at
  `218e492ed63f39bdce87d9f411f5b2bb01ef0deb` by the launcher. Qualification runs r10-r16
  used earlier commit `e3186d29f29bfccddc37bca5c9940231b878761b`; the newer commit only
  adds tail-mass metrics. The correction applies to the regular clipped PPO loss
  with truncated importance sampling (TIS). It uses behavior top-k token probabilities captured by
  the serving engine, plus current and stored-old trainer probabilities for the same token IDs.
  It leaves the sampled-token PPO/TIS term intact and adds a detached, per-token control variate.
  `score_centering_topk=0` disables it.
- Marin launcher branch `goal/score-centering-launcher-01a0bb6f`. Qwen uses one eight-H100
  Megatron learner node and one eight-H100 rollout node with eight independent one-GPU vLLM
  engines. Snowball retains its four learner nodes plus one rollout node.
- The first Qwen qualification uses model artifact
  `models/curriculum-rl-qwen3-0.6b@2026.08.29` and pool artifact
  `documents/curriculum-rl-pool@2026.08.29.1`. The latter has 10,427 training rows and 756
  validation rows, both `reward_spec` and `reward_model`, and a consistent ground truth in each.
  The earlier `2026.08.29` pool lacks `reward_model`, so its AIME rows cannot run in this SkyRL
  environment. The `.1` revision also changed AIME prompt endings to `Answer: <answer>`; later
  matched arms must keep that revision and their evaluation prompt membership fixed.
- The Qwen smoke config has two optimizer updates, 32 prompts per update, four samples per
  prompt, 32 generation workers, an eight-group buffer, and a 1,024-token response cap. It
  disables evaluation to isolate the training path. Qwen's default preset evaluates 256 held-out
  prompts every five updates, with one greedy response per prompt. TIS is enabled with cap 2.0.
  The qualification correction width is 32; a width of 128 and capture-only controls are still
  needed before the main comparison. All work uses Iris `interactive` priority.

For one sampled token, let `q` be the behavior policy that sampled it, `o` the stored trainer
policy at the start of the optimizer update, and `p` the trainer policy being differentiated.
The sampled regular-PPO/TIS score coefficient is `A * min(o/q, cap) * (p/o)` while PPO's
directional clip is inactive, and zero while it is active. The correction adds the behavior
expectation of this coefficient times `log p`, with the coefficient detached. This cancels the
expected sampled score gradient for a constant advantage. With no active clipping or TIS cap,
the coefficient reduces to `p/q`, and exact full-vocabulary centering has zero gradient by the
score identity. The implemented top-k correction uses exact behavior probabilities for the
captured head and models each policy's tail as proportional to `p` while preserving its tail
mass. Every ratio, clip decision, and tail coefficient in the correction is detached; normal
loss masking and distributed token normalization apply to the combined loss.

## Qualification ledger

These are integration probes, not evidence that the correction improves learning. Canceled
failures count toward the campaign's total task cost.

| Probe | Reached GPUs? | Finding |
| --- | --- | --- |
| r1 | No | Role bundles exceeded the declared two-node topology. |
| r2 | Yes | An overly broad exact-evidence preflight rejected the supported single-turn gym path. |
| r3 | Yes | The default W&B entity rejected the campaign credential. |
| r4 | Yes | The credential's login name was not a writable W&B entity. |
| r5 | Yes | vLLM rejected external DP8 for the non-MoE Qwen model. |
| r6 | No | A second topology check still derived eight rollout nodes from eight engines. |
| r7 | Yes | The 2026.08.29 pool lacked AIME's required `reward_model` field. |
| r8 | Yes | The fully async plain-chat HTTP client omitted behavior top-k from request and response. |
| r9 | Yes | Top-k reached Megatron, but selected-logprob gathering assumed padded response positions survived left-padding compaction. |
| r10 | Yes | Two Qwen optimizer updates completed with captured top-k 32, score centering, and exact sampled-token alignment. The terminal HF export is a separate child job. |
| r11 | Yes | Top-k 128 score-centering cost and stability control; running. |
| r12 | Yes | Top-k 32 capture with score centering disabled; running. |
| r13 | Yes | TIS with sampled-token logprobs but no top-k capture; running. |
| r14 | Yes | Full-cap default schedule, TIS plus top-k 32 capture, eight-update age calibration; running. |
| r15 | Yes | Full-cap 128-worker, age-limit-eight schedule, TIS plus top-k 32 capture, eight-update age calibration; running. |
| r16 | Pending | Top-k 8 score-centering cost and approximation-width control; submitted. |

The r8 retained trajectory archive confirms sampled-token logprobs and exact engine token IDs
reached generation without alignment alerts. It does not contain top-k evidence; the learner
rejected its first batch before an optimizer update. The r8 run is
[tva86i2y](https://wandb.ai/marin-community/marin-async-rl/runs/tva86i2y). Earlier W&B runs
and exact Iris job IDs will be listed with cost accounting.

The r10 [Qwen smoke run](https://wandb.ai/marin-community/marin-async-rl/runs/ms580xuq)
trained at token-weighted consumed ages 0 and 1. Its two batches had 116,181 and 122,486
consumed response tokens, 100% sampled-token ID/logprob alignment, no alignment alerts, and
nonzero mean absolute score-centering terms of 1.40e-5 and 1.28e-5 per token. Mean absolute
log(trainer/behavior) was 0.0138 and 0.0160. These small mismatches qualify the mechanics but
do not yet test meaningful older-policy tolerance. The 1,024-token smoke response cap caused
69.5% and 82.8% length stops, so smoke reward cannot be used as the quality comparison. Its
first two measured training cycles took 75.3 and 91.1 seconds, of which weight sync took 24.7
and 30.8 seconds. The cost controls use the same smoke cap and geometry, with [top-k 128 plus
SC](https://wandb.ai/marin-community/marin-async-rl/runs/1pld6uwv) and [top-k 32 capture
only](https://wandb.ai/marin-community/marin-async-rl/runs/iqmow1j2) as separate runs.

## Comparison contract

The first causal comparison will hold the model, pool, evaluator, optimizer, topology, seed,
generation workers, buffer, weight publication cadence, TIS cap, and top-k capture width fixed.
It will compare TIS with TIS plus score centering at near-fresh and demonstrably older consumed
token ages. A matched plain PPO arm checks the current launcher incumbent. We will measure
token-weighted age from policy-version spans at optimizer consumption, policy mismatch, rejected
groups and tokens, and correction size. An age limit is an exposure setting, not a measured age.

The primary quality endpoint is a correct answer with an accepted stop reason (`complete`,
`end_turn`, `eos`, or `stop`). Raw reward, completion and length-stop fractions, answer lengths,
and response dumps remain separate. Quality curves will use optimizer updates, consumed tokens,
elapsed training time, and full task GPU-hours as distinct axes. Question resampling within one
training seed does not measure between-seed uncertainty.

The logged `completed_stop_score_contribution` is a signed reward contribution, not the binary
completed-correct fraction: AIME assigns -1 to an incorrect answer. We will compute the primary
fraction from each dumped response's score and stop reason, after checking that the frozen run
does not reshape correctness rewards. We will also hash the held-out prompts and ground truths to
confirm the evaluated questions match across arms.

The Qwen screen decides which comparison merits a matched Snowball follow-up. A scheduling
change, such as allowing more age or changing the worker pool, will be reported as a separate
configuration comparison. No quality-loss margin or target score has been selected, so this study
will not claim non-inferiority or time-to-target until one is fixed before confirmation.
