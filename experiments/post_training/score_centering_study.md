# Score centering in fully asynchronous RL

Study started 2026-09-19. This document records the exact experiment contract and results as
they become available. The question is whether score centering lets the learner use older sampled
tokens without giving up completed-answer quality, and whether that extra tolerance saves time or
GPU work.

## Implementation and frozen inputs

- MarinSkyRL branch `goal/score-centering-01a0bb6f`, pinned at
  `a7b51d31d7ed44157219b5852f49ffd69de4038b` by the launcher. Qualification runs r10-r16
  used earlier commit `e3186d29f29bfccddc37bca5c9940231b878761b`; r17 uses
  `c7b4ac4bdd9c57de18f9b01809f082227e2dae06`. The newer commits add tail-mass metrics,
  explicit W&B finish, and an optional delayed weight-publication cadence. The correction applies to the
  regular clipped PPO loss
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
  disables evaluation to isolate the training path. Qwen's default preset evaluates all 756 held-out
  prompts in batches of 256 every five updates, with one greedy response per prompt. TIS is enabled with cap 2.0.
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
| r11 | Yes | Top-k 128 score-centering cost control; two updates trained, export pending. |
| r12 | Yes | Top-k 32 capture with score centering disabled; succeeded. |
| r13 | Yes | TIS with sampled-token logprobs but no top-k capture; succeeded. |
| r14 | Yes | Full-cap default schedule, TIS plus top-k 32 capture, eight-update age calibration; running. |
| r15 | Yes | Full-cap 128-worker, age-limit-eight schedule, TIS plus top-k 32 capture, eight-update age calibration; running. |
| r16 | Yes | Top-k 8 score-centering cost control; two updates trained, export pending. |
| r17 | Yes | Explicit W&B finish worked: the primary run reports `finished` and retains step 2. Tail-mass mean telemetry became NaN on padded rows; correction stayed finite. |
| r18 | Pending | Tests masked tail-mass telemetry and every-two-step weight publication on four updates. |

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

Iris reports r10, r12 and r13 successful, with clean checkpoints and exports, but W&B marks
their runs crashed and retains only the first training step. The terminal step is present in
Iris `WANDB_MIRROR` logs. These runs used an older source pin that relied on process teardown to
finish W&B; `c7b4ac4b` explicitly finishes the primary run after fully async trainer teardown.
The Iris mirror and saved evaluation dumps remain the durable measurement sources for the
earlier runs. The r17 top-k 8 smoke verifies the W&B finish fix.

The r17 [W&B run](https://wandb.ai/marin-community/marin-async-rl/runs/ajbwdxcd) reports
`finished` with both optimizer steps after explicit trainer shutdown. Its new tail-mass means are
NaN because padded selected-logprob sentinels entered the telemetry reduction; score-centering
losses and gradients remained finite. Commit `a7b51d31` masks those positions, with a CPU
regression test, and r18 tests it on GPUs.

The first smoke cycle gives a useful collection-cost control at almost equal consumed-token
counts (about 115,000–116,000). TIS without top-k took 21.3 seconds and returned 0.09 MB per
response; top-k 8 with SC took 37.1 seconds and 1.01 MB; top-k 32 capture without SC took
77.0 seconds and 3.49 MB; top-k 32 with SC took 75.3 seconds and 3.50 MB; top-k 128 with SC
took 261.6 seconds and 13.48 MB. These concurrent short runs suggest top-k collection and
transport dominate the correction's incremental learner cost. They do not isolate cluster
contention or predict Snowball throughput. The top-k 8 correction has not yet been qualified
for its omitted tail probability.

The full-cap [age-limit-four calibration](https://wandb.ai/marin-community/marin-async-rl/runs/g0iq70y0)
used 64 generation workers. Its token-weighted mean consumed age rose from 0 to 2.72 by update
5 and stayed below age four through update 8, with no stale-group rejection. The
[age-limit-eight calibration](https://wandb.ai/marin-community/marin-async-rl/runs/b61qmnym)
used 128 workers and consumed all update-5 tokens at age four. It took 510 seconds for its first
training cycle, versus 279 seconds for the 64-worker schedule, while using fewer response tokens
in that first batch. Both used TIS and top-k 32 capture without SC and scored 93/756 completed
correct at update 5. This establishes age separation but is a scheduling comparison, not an SC
effect estimate.

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

The two full-cap calibration runs' step-0 dumps contain 256 GSM8K and 500 Math500 rows each.
Their sorted prompt plus ground-truth SHA-256 is the same,
`448615d2489352d13e1c4e994bfe458485d7503c1076ddf0aa608fd6c637048d`. The age-limit-four
run had 78/756 completed correct (66 GSM8K, 12 Math500); the age-limit-eight run had 76/756
(68 GSM8K, 8 Math500). The accepted stop was `stop` for all completed responses; length stops
were 228 and 236. This step-0 variation occurred before any optimizer update despite the same
model artifact and held-out membership, and should not be mistaken for a training effect.

The Qwen screen decides which comparison merits a matched Snowball follow-up. A scheduling
change, such as allowing more age or changing the worker pool, will be reported as a separate
configuration comparison. No quality-loss margin or target score has been selected, so this study
will not claim non-inferiority or time-to-target until one is fixed before confirmation.

`analyze_score_centering.py` reads every dumped evaluation response and the durable Iris
`WANDB_MIRROR` lines. It writes separate CSV files for completion-aware quality and per-update
age, mismatch, consumed tokens, cycle time, and nominal GPU-hours across inclusive step cycles.
The step-cycle cost includes in-run evaluation and checkpointing but excludes setup, terminal
export, and failed attempts; those require Iris task durations in the total-cost table. The
script checks that prompt and ground-truth membership
match at every evaluation step and across the compared runs.
