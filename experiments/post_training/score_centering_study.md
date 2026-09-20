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
  The main comparison width is 32; width-8 and width-128 cost controls and capture-only controls
  have completed. All work uses Iris `interactive` priority.

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
| r11 | Yes | Top-k 128 score-centering cost control; two updates trained. Iris later left the second GPU child pending after deleting its pod, so the idle parent was canceled without a terminal export. |
| r12 | Yes | Top-k 32 capture with score centering disabled; succeeded. |
| r13 | Yes | TIS with sampled-token logprobs but no top-k capture; succeeded. |
| r14 | Yes | Full-cap default schedule, TIS plus top-k 32 capture, eight-update age calibration; succeeded with terminal export. |
| r15 | Yes | Full-cap 128-worker, age-limit-eight schedule, TIS plus top-k 32 capture, eight-update age calibration; succeeded with terminal export. |
| r16 | Yes | Top-k 8 score-centering cost control; two updates trained, export pending. |
| r17 | Yes | Explicit W&B finish worked: the primary run reports `finished` and retains step 2. Tail-mass mean telemetry became NaN on padded rows; correction stayed finite. |
| r18 | Yes | Four updates tested finite masked tail-mass telemetry and every-two-step weight publication. |

Iris's `task_attempts` records include every started accelerator attempt, including retries and
killed probes. Summing `(finished_at_ms - started_at_ms) * 8 / 3,600,000` for each eight-H100
child task gives this complete cost ledger through r18. It counts startup, training, in-run
evaluation, terminal export, and failed attempts. CPU-only coordinators contribute no GPU-hours.

| Runs | H100 GPU-hours | Detail |
| --- | ---: | --- |
| r1, r6 | 0.00 | Failed before an accelerator child started. |
| r2–r5, r7–r9 | 9.00 | Failed integration probes, including their retries. |
| r10–r13, r16–r18 | 20.22 | Successful smoke/diagnostic training, plus r11's trained but canceled width-128 control. |
| r14–r15 | 16.31 | Two eight-update full-cap age calibrations. |
| **r1–r18 total** | **45.53** | All completed GPU attempts, regardless of outcome. |

The per-run values, in run order r2–r5 and r7–r18, are 1.244, 0.886, 0.874, 0.954, 1.870,
1.500, 1.673, 3.049, 3.467, 2.936, 2.600, 7.415, 8.892, 2.709, 2.641, and 2.822 GPU-hours.
The raw read-only Iris query was
`SELECT task_id,attempt_id,state,started_at_ms,finished_at_ms FROM task_attempts WHERE task_id LIKE '/romain/score-centering-qwen-%01a0bb6f%' AND task_id LIKE '%/users-%'`.
Only attempts with both timestamps entered the completed ledger; r19–r23 were still running at
this accounting checkpoint.

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
contention or predict Snowball throughput. The top-k 8 tail approximation needs a separate
full-vocabulary error measurement before it could replace k32 in a quality run; that measurement
follows here.

`measure_score_centering_tail.py` now compares the implemented proportional-tail coefficient
with an exact full-vocabulary score gradient at the start of a one-pass PPO update (`p=o`). It
uses full Qwen3-0.6B distributions at six positions on each of the first four held-out GSM8K
and Math500 prompts, with the `.08.29` model as behavior `q` and r14's update-8 exported model
as current `p`. On these 48 deterministic contexts, behavior's mean omitted probability was
0.00233 at k8, 0.000297 at k32, and 0.0000748 at k128. The actual checkpoint pair had mean
behavior-weighted absolute log ratio 0.00370 and no behavior mass above the configured TIS cap
2.0. The exact correction gradient and k8/k32/k128 approximation error were zero to numerical
precision in that case. At a diagnostic cap of 1.05, k32's mean L1 gradient error was
4.24e-7, or 0.17% of the mean exact correction L1 norm. This two-checkpoint pair is an offline
test, not the exact behavior/current pair from any one consumed training token.

The training GPU's trainer-versus-behavior absolute log ratio was about 0.016 at measured age
five, so the same script also makes a **synthetic sensitivity check**: it perturbs the real Qwen
full-vocabulary vectors to behavior-weighted absolute log ratios of 0.016 and 0.05, using a
fixed seed per context. At magnitude 0.016 and cap 2.0, k32's mean L1 error was 0.000133,
0.94% of the mean exact correction norm; k8 was 1.46% and k128 0.68%. At cap 1.05, k32's
corresponding error was 0.46%. These are ratios of aggregate means, not per-context maxima.
The synthetic perturbations match only one mismatch statistic and cannot stand in for actual
vLLM/Megatron distribution differences or later generation positions. The output CSV is
`/tmp/score-centering-qwen-tail-error-01a0bb6f.csv`; rerun the script with the exact S3
model and evaluation paths to reproduce it.

The full-cap [age-limit-four calibration](https://wandb.ai/marin-community/marin-async-rl/runs/g0iq70y0)
used 64 generation workers. Its token-weighted mean consumed age rose from 0 to 2.72 by update
5 and stayed below age four through update 8, with no stale-group rejection. The
[age-limit-eight calibration](https://wandb.ai/marin-community/marin-async-rl/runs/b61qmnym)
used 128 workers and consumed all update-5 tokens at age four. It took 510 seconds for its first
training cycle, versus 279 seconds for the 64-worker schedule, while using fewer response tokens
in that first batch. Both used TIS and top-k 32 capture without SC and scored 93/756 completed
correct at update 5. At update 8, the age-limit-four run consumed tokens at mean age 2.70 and
scored 85/756 completed correct; the age-limit-eight run consumed tokens at mean age 5.0 and
scored 88/756. Both fell below their update-5 scores, which is why the comparison needs a longer
quality curve. The nominal eight-update step cycles used 3.96 and 5.36 GPU-hours respectively
across 16 allocated H100s, including their in-run evaluations but excluding setup and terminal
export. Iris child-task durations give total accelerator occupancy of 7.42 and 8.89 GPU-hours
respectively, including setup and each eight-GPU terminal export; the parent wall times were
30:55 and 36:38. This establishes age separation but is a scheduling comparison, not an SC
effect estimate.

The resumable checkpoint's `data_consumption_state.pt` identifies consumed prompt UIDs. At
update 5, each calibration had consumed 160 unique prompts, of which 152 were shared
(Jaccard 0.905); the different schedules changed exposure to eight prompts per arm even with
the same seed. `analyze_score_centering_exposure.py` compares such checkpoints at equal steps.
The update-8 terminal checkpoints had already advanced the tracker to a new epoch and cleared
its UID set, so they cannot support an update-8 membership comparison; the script rejects that
empty-set case.

The r18 cadence probe used top-k 8 and SC with a two-update publication interval. Its four
updates published weights after updates 2 and 4; `timing/sync_weights` was zero after updates 1
and 3. Behavior, stored-old, and current top-k tail-mass means were finite, around 0.006–0.008.
Its consumed-token mean age was 0, 1, 2, and 1.53 across the four updates. It qualifies the
implementation of delayed publication; the 1,024-token smoke cap leaves quality uninterpretable.
Iris reports the parent and three accelerator children successful; its total accelerator occupancy
was 2.82 GPU-hours across two eight-GPU training tasks and one eight-GPU export task.

The present one-pass Qwen preset offers little opportunity for SC to change the gradient.
`policy_mini_batch_size=train_batch_size` and `update_epochs_per_batch=1` mean the differentiated
policy `p` equals the stored-old policy `o` during each update. The calibration runs report a
zero `policy/log_ratio_abs_mean`, confirming this on the GPU. Where TIS is uncapped and PPO is
unclipped, the exact score coefficient is `q * (o/q) * (p/o) = p`; its full-vocabulary score
expectation and the implemented head-plus-tail correction have zero gradient. The measured TIS
cap fraction stayed at or below 0.000189 through the eight-update calibrations, including mean
consumed age five. A null effect under this preset would show that the correction is largely
inactive here; it would not establish that SC cannot help when caps or clips are active.
The reported absolute correction *loss value* can still be nonzero from finite-precision
head/tail arithmetic; it is not evidence of a material correction gradient.

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

The first 40-update screen uses seed 17, an evaluation every ten updates plus step zero and
terminal evaluation, the frozen `.08.29.1` pool and `.08.29` model, and MarinSkyRL
`a7b51d31`. The Iris parent bundles came from Marin commit `3abefce4f8`. All four TIS arms
capture behavior top-k 32, including the SC-disabled controls.
The older schedule uses 128 generation workers, a 32-group buffer, age limit eight, and weight
publication after each update. The near-fresh schedule uses 32 workers, a 16-group buffer, and
age limit zero. The five jobs are:

| Job | Objective | Schedule |
| --- | --- | --- |
| [r19](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-tis-seed17-01a0bb6f-r19) | TIS, SC off | Older |
| [r20](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-sc-seed17-01a0bb6f-r20) | TIS plus SC32 | Older |
| [r21](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-fresh-tis-seed17-01a0bb6f-r21) | TIS, SC off | Near-fresh |
| [r22](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-fresh-sc-seed17-01a0bb6f-r22) | TIS plus SC32 | Near-fresh |
| [r23](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-ppo-seed17-01a0bb6f-r23) | Plain PPO, TIS off | Older |

The async engine's response sampling and GPU scheduling remain nondeterministic, even with a
fixed training seed. The step-zero correct counts in the first four runs differ; comparisons
must include each run's starting point and between-seed uncertainty.

At update 10, the two older-schedule TIS arms each consumed 320 unique prompt UIDs, with exact
set overlap, and both had mean consumed-token age 5.0. Completed correct was 103/756 for TIS
alone (87/756 at step zero) and 83/756 for TIS plus SC32 (79/756 at step zero). Their TIS cap
fractions were near zero through update 10. These are one-seed interim values, with visible
step-zero evaluator variation; they do not establish a quality effect. The near-fresh TIS and
plain-PPO arms also consumed the same first 320 prompt UIDs, despite different realized ages.
At step zero, the older TIS and SC arms used the same 756 held-out prompts and the same starting
model artifact. Their two greedy response dumps agreed on the first eight response characters
for 755 prompts, but only 88 full responses matched exactly and 674 scores agreed. The median
common response prefix was 375 characters. Small generation differences can branch into long,
different solutions; the exact cause of the divergence has not been isolated. A single greedy
pass has visible run-to-run outcome variation here.

The next saved evaluations show 137/756 completed correct for older TIS and 142/756 for older
TIS plus SC32 at update 20; near-fresh TIS had 120/756, and near-fresh SC32 had 102/756 at
update 10 on its retry. Plain PPO had 129/756 at update 20 and 180/756 at update 30. The
older TIS pair's apparent difference changed sign between updates 10 and 20. All five arms
have the same held-out membership hash, but these one-seed curves remain descriptive.

The near-fresh SC arm's first GPU attempt reached update ten, then its checkpoint hit an S3
`OSError` (`errno 16`, "Please reduce your request rate") at 00:03 UTC on September 20. Iris
retried the child, and its second attempt restarted at update zero; the evaluation dump at step
zero was replaced. Attempt number therefore matters when reading the training curve, and total
GPU cost includes both attempts. The plain-PPO arm appeared stalled during its update-30
rank-zero multipart checkpoint upload: part 38 of a 7.15 GB object was the last logged
completion at 00:08:35 UTC, with no logged progress for more than ten minutes. A later S3 object
inspection found the full rank-zero shard with a 00:18:45 UTC modification time, before the
working preempt. A preempt request against the federated Iris controller returned success
without changing the child. A preempt request to the `cw-rno2a` controller at 00:21 UTC stopped
rank zero and atomically restarted its sibling. Both child tasks entered attempt one. The
restart selected the complete step-30 checkpoint, then rank zero was OOM-killed while loading
it. A second restore attempt was OOM-killed at the same point. The peer job was canceled to
avoid further repeated GPU use. The Qwen child memory request was raised from 128 to 256 GB for
a continuation using the same artifact address and settings; recovery still needs verification.
These infrastructure interruptions and their GPU attempts belong in cost and provenance.

A second older-schedule pair launched on September 20 with TIS cap 1.05, the same seed and
schedule, and behavior top-k 32 in both arms. This deliberately activates more capped tokens
than the cap-2 screen; it is a distinct objective comparison. The Iris parents are
[r24](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-tis-cap105-seed17-01a0bb6f-r24)
and [r25](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-sc-cap105-seed17-01a0bb6f-r25).
Their children use `iris-interactive` GPU pods. The cap fraction and outcomes are pending.

`analyze_score_centering.py` reads every dumped evaluation response and the durable Iris
`WANDB_MIRROR` lines. It writes separate CSV files for completion-aware quality and per-update
age, mismatch, consumed tokens, cycle time, and nominal GPU-hours across inclusive step cycles.
When Iris retries a GPU child, a repeated optimizer step uses the later attempt's mirror, and
the metrics CSV records the selected attempt number.
The step-cycle cost includes in-run evaluation and checkpointing but excludes setup, terminal
export, and failed attempts; those require Iris task durations in the total-cost table. The
script checks that prompt and ground-truth membership
match at every evaluation step and across the compared runs.
