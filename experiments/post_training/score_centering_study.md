# Score centering in fully asynchronous RL

Study started 2026-09-19. This document records the exact experiment contract and results as
they become available. The question is whether score centering lets the learner use older sampled
tokens without giving up completed-answer quality, and whether that extra tolerance saves time or
GPU work.

## Result and recommendation

The correction runs end to end, but this study does not establish that it makes older
rollouts cheaper at comparable answer quality. At TIS cap 1.05 and top-k 32, three
matched 40-update older-policy seed pairs favored centering by +36.5, +43.5, and
+9.0 completed answers out of 756 when averaging the two evaluations at the same
final weights. The saved final evaluation alone favored it in two of three seeds.
Their average two-pass difference was +29.7 answers; an illustrative 95% Student-t
interval across the three training seeds is -15.6 to +75.0, before uncertainty
about this small-seed interval's assumptions. This is a promising quality signal,
not a reliable improvement estimate.

In the near-fresh seed-17 check, centering led by 20 answers across the two final
evaluations but started 15 ahead at step zero. The deliberately delayed top-k-eight
pair consumed tokens at mean age about 8.5 rather than 4.7 updates. Centering led
its matched control by 22 answers across the two final evaluations, but both
delayed arms scored below their every-update-publication counterparts. Delaying
publication shortened median training cycles, yet total time and H100 work through
evaluation remained close to the regular top-k-eight pair. Different generation
timing and repeated-evaluator variation limit these one-seed schedule comparisons.

Keep `score_centering_topk=0` as the default. The implementation can support a
controlled follow-up where active TIS and older rollouts are expected, but these
runs do not justify routine adoption or a claim of recovered useful staleness.
Top-k-eight behavior capture cut measured Qwen cost by more than half versus
top-k 32 at the same schedule; its quality effect was inconclusive. The Snowball
pair validated two Megatron optimizer updates with active TIS and finite
correction, but its SC job did not finish terminal export after preemption and
restore failures, so this report makes no Snowball quality claim.

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

The [resolved run configurations](configs/score_centering/README.md) preserve the exact
materialized Hydra arguments and artifact references for the main Qwen arms and Snowball
smokes. All of these runs pin MarinSkyRL `a7b51d31`. The Marin launcher source bundles were:

| Jobs | Marin commit | Relevant source change |
| --- | --- | --- |
| r19–r23 | `3abefce4f8` | Frozen first 40-update screen |
| r26 | `00d3ce0442` | Same PPO artifact and settings; 256 GB host-memory request for restore |
| r24–r25 | `0674c9300d` | First cap-1.05 pair, 128 GB host-memory request |
| r27–r28 | `d3a120425c` | Second seed, 256 GB host-memory request |
| r29 | `437f37d6d9` | Top-k-one cost control, 256 GB request |
| r30–r31 | `5c43d7bcfb` | Third cap-1.05 seed |
| r32–r33 | `6fbf47d3b8` | Cap-1.05 near-fresh pair; same learner settings as r21–r22 except TIS cap |
| r34–r35 | `16596f4a02` | Cap-1.05 older pair with top-k eight capture and optional SC8 |
| r36–r37 | `09b623bfb8` | Cap-1.05 top-k-eight pair with delayed weight publication |
| Snowball smokes | `6f66ee6c22` | Megatron MoE two-update pair |

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
| r16 | Yes | Top-k 8 score-centering cost control; two updates trained, terminal export not confirmed. |
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
and exact Iris job IDs are retained in the linked Iris parents and artifact records below.

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
vLLM/Megatron distribution differences or later generation positions. The
[width-8/32/128 results](results/score_centering_qwen_tail_error_k8_32_128.csv) use r14's
step-zero saved responses; rerun the script against its S3 evaluation root with
`--topk 8 --topk 32 --topk 128` to reproduce them. A rerun against
`qwen-default-set-b2fc0319/2026.09.19.3/exports/dumped_evals/global_step_0_evals`
matched the checked-in CSV byte for byte.

A second run of the same full-vocabulary script added widths one and four and used the first
four response continuations per suite from r19's step-zero evaluation. The eight prompts and
sampled positions match the earlier check, but the generated continuations differ, so their
answer-position distributions differ. At synthetic behavior-weighted mean absolute log ratio
0.016 and TIS cap 1.05, mean L1 gradient error divided by mean exact correction L1 norm was
11.29% at k1, 1.96% at k4, 0.75% at k8, 0.41% at k32, and 0.29% at k128. Mean behavior
tail mass was 0.120, 0.0192, 0.00497, 0.000424, and 0.000116 respectively. On this proxy,
k1 is too coarse for score centering despite being an inexpensive TIS-only collection path;
k8 merits an end-to-end cost and quality check. Its error is still an offline average, not a
guarantee about every training token. The
[width-1/4/8/32/128 results](results/score_centering_qwen_tail_error_k1_4_8_32_128.csv)
use `qwen-default-set-a8761bbb/2026.09.19.6/exports/dumped_evals/global_step_0_evals`
as the evaluation root. Numeric CSV values are rounded to 12 significant digits for the
repository size gate; the summary above was computed before rounding.

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

Each TIS-versus-centering pair held the model, pool, evaluator, optimizer, topology, seed,
generation workers, buffer, weight publication cadence, TIS cap, and top-k capture width fixed.
Separate near-fresh, older, and delayed-publication schedules tested exposure. The cap-2
screen included plain PPO as a current launcher incumbent, though its different behavior
capture means its time and cost comparison is descriptive. The metrics record token-weighted
age from policy-version spans at optimizer consumption, behavior-versus-trainer mismatch,
rejected groups, and correction size. An age limit is an exposure setting, not a measured age.

The primary quality endpoint is a correct answer with an accepted stop reason (`complete`,
`end_turn`, `eos`, or `stop`). Raw reward, completion and length-stop fractions, answer lengths,
and response dumps remain separate. Quality curves use optimizer updates, consumed tokens,
elapsed training time, and full task GPU-hours as distinct axes. Question resampling within one
training seed does not measure between-seed uncertainty.

The logged `completed_stop_score_contribution` is a signed reward contribution, not the binary
completed-correct fraction: Math500 assigns -1 to an incorrect answer. The primary fraction comes
from each dumped response's score and stop reason, after checking that the frozen run does not
reshape correctness rewards. The held-out prompt and ground-truth hash confirms matching
question membership across arms.

The two full-cap calibration runs' step-0 dumps contain 256 GSM8K and 500 Math500 rows each.
Their sorted prompt plus ground-truth SHA-256 is the same,
`448615d2489352d13e1c4e994bfe458485d7503c1076ddf0aa608fd6c637048d`. The age-limit-four
run had 78/756 completed correct (66 GSM8K, 12 Math500); the age-limit-eight run had 76/756
(68 GSM8K, 8 Math500). The accepted stop was `stop` for all completed responses; length stops
were 228 and 236. This step-0 variation occurred before any optimizer update despite the same
model artifact and held-out membership, and should not be mistaken for a training effect.

The Qwen screen led to a two-update Snowball smoke rather than a longer quality comparison.
The schedule and capture changes are reported separately from the correction comparison.
No quality-loss margin or target score was selected, so this study makes no non-inferiority
or time-to-target claim.

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
completion at 00:08:35 UTC, with no logged multipart progress for more than ten minutes.
The full rank-zero shard was actually committed to S3 at 00:18:45 UTC. Iris mirrors show
update 30 completed at 00:19:34 and updates 31–36 by 00:21:04; later log lines report updates
37–38. A preempt request against the federated Iris controller returned success without
changing the child. A direct request to the `cw-rno2a` controller at 00:21 UTC stopped rank
zero and atomically restarted its sibling. This direct preempt interrupted a worker that had
already resumed training. The decision was based on incomplete log visibility and added
avoidable cost; only the step-30 checkpoint was durable, so the later updates were repeated.
Both child tasks entered attempt one. The restart selected that complete checkpoint, then
rank zero was OOM-killed while loading it. A second restore attempt was OOM-killed at the same
point. The peer job was canceled to avoid further repeated GPU use. The Qwen child memory
request was raised from 128 to 256 GB for a continuation with the same artifact address and
settings. The full ledger includes the slow upload, manual interruption, OOM retries, and
continuation.

The continuation [r26](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-ppo-resume-seed17-01a0bb6f-r26)
selected that same step-30 checkpoint and loaded model and optimizer state. Its learner pod
reached 176.9 GB peak cgroup memory during restore, which explains why the 128 GB request failed.
It restored 32 buffered groups, trained updates 31–40, saved the step-40 checkpoint, and dumped
the terminal 756-response evaluation. During shutdown, a trajectory-retention publication
reported a 120-second storage timeout, while the training driver exited with code zero. The
separate terminal model export completed at `exports/global_step_40/policy/model.safetensors`
(1.503 GB), and the Iris parent succeeded. The combined r23+r26 ledger has nine accelerator
attempts, including the export, and 15.366 reserved H100-hours in total.

A second older-schedule pair launched on September 20 with TIS cap 1.05, the same seed and
schedule, and behavior top-k 32 in both arms. This deliberately activates more capped tokens
than the cap-2 screen; it is a distinct objective comparison. The Iris parents are
[r24](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-tis-cap105-seed17-01a0bb6f-r24)
and [r25](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-sc-cap105-seed17-01a0bb6f-r25).
Their children use `iris-interactive` GPU pods.
Their step-zero completed-correct counts were 73/756 for TIS and 79/756 for TIS plus SC32.
At the first learner update, 4.57% and 4.61% of sampled tokens respectively hit the TIS cap,
compared with near-zero cap fractions in the cap-2 screen. The SC arm logged 0.00315 mean
absolute correction loss value. This confirms that the new cap changes the active objective,
but the loss value alone does not quantify the correction gradient or a quality effect.
At update two the cap fractions remained 4.97% and 4.88%, with mean consumed-token age one.
By update eight, both arms consumed tokens at mean age near five, and about 5.6% of sampled
tokens hit the cap. Their step-10 checkpoint UID sets each contained 320 consumed prompts,
with 318 in common (Jaccard 0.988). The first post-training evaluation was 112/756 completed
correct for TIS and 98/756 for TIS plus SC32, versus step-zero counts 73 and 79. A single
early difference does not establish a quality effect or stability.
At update 20, TIS had 128/756 completed correct and TIS plus SC32 had 158/756. Relative to
their own step-zero counts, those arms gained 55 and 79 correct answers. The apparent
SC difference thus changed sign between updates 10 and 20. Their update-20 checkpoints each
recorded 640 consumed prompt UIDs and shared 639 (Jaccard 0.997). Through their first 24–25
updates, both arms consumed tokens at mean age about 4.4 and capped about 5.3% of tokens.
At update 30, their completed-correct counts were 177 and 190. At the prespecified update-40
endpoint, TIS had 248/756 completed correct and TIS plus SC32 had 283/756. Relative to each
arm's own step-zero count, the gains were 175 and 204, a baseline-adjusted difference of 29
answers in this seed. The pair used 24.20 and 23.58 reserved H100-hours, respectively, through
the terminal evaluation, with 1.51 and 1.47 hours since their first GPU tasks; full job costs,
including export, were 25.26 and 24.54 H100-hours. Across all 40 updates their
token-weighted mean consumed ages were both 4.72, and 5.33% and 5.39% of tokens hit the TIS
cap. The difference can still reflect asynchronous sampling and evaluator variation; two
further seed comparisons follow.
The terminal consumed-prompt trackers contain exactly the same 1,280 unique prompt UIDs in
both seed-17 arms (Jaccard 1.0). The top-k-one control r29 consumed that same UID set despite
its faster collection path. Prompt membership therefore does not explain their endpoint
differences, though sampled completions, truncation, and optimizer timing still can.
A second matched seed-18 pair, [r27](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-tis-cap105-seed18-01a0bb6f-r27)
and [r28](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-sc-cap105-seed18-01a0bb6f-r28),
uses the same settings and 256 GB host request. Its step-zero completed-correct counts are
88/756 and 79/756; both have the same held-out membership as seed 17. At update 10, the
counts were 95 and 117, with each arm consuming 320 prompt UIDs and 318 in common. Their
update-20 counts were 130 and 148, and update-30 counts were 173 and 196. At the prespecified
update-40 endpoint, TIS had 245/756 completed correct and TIS plus SC32 had 289/756. Their
step-zero-adjusted gains were 157 and 210, a paired difference of 53 answers. Mean consumed
ages across all 40 updates were 4.70 and 4.71, with 5.38% and 5.50% of tokens hitting the
TIS cap. Both arms consumed about 13.2–13.4 million loss tokens. Through terminal evaluation,
the TIS and SC arms used 24.80 and 24.41 H100-hours, respectively; full jobs, including export,
used 25.64 and 25.50 H100-hours. These two final saved comparisons favor SC; the third pair
below reverses that result. Their terminal
consumed-prompt trackers also match exactly within the seed-18 pair: 1,280 unique UIDs in each.
The two seed-18 arms share only 161 of those UIDs with the seed-17 set, as expected from a
different shuffled training seed. The [terminal exposure comparisons](results/score_centering_terminal_exposure_seed17_18.csv)
include both matched pairs and the top-k-one control. New launches were held
when cluster use rose to 504/512 H100s with zero queued workloads at 00:59 UTC.
A third matched seed-19 pair started after capacity returned to 284/512 H100s with no queued
workloads: [TIS r30](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-tis-cap105-seed19-01a0bb6f-r30)
and [TIS plus SC32 r31](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-sc-cap105-seed19-01a0bb6f-r31).
Both use the same cap-1.05 configuration, 40-update endpoint, frozen model and pool, and
`iris-interactive` accelerator pods. Their step-zero completed-correct counts were 80 and
75. At updates ten and twenty, TIS had 88 and 124 completed correct, while SC32 had 96 and
147. At update thirty, counts were 191 and 196. The final saved update-40 dumps scored TIS
304/756 and SC32 293/756, a raw SC difference of -11 and step-zero-adjusted difference of -6.
Their token-weighted consumed ages were 4.68 and 4.69; 5.44% and 5.49% of sampled tokens hit
the TIS cap. Both arms consumed about 13.2–13.3 million loss tokens. Median inclusive cycles
were 89.71 and 83.71 seconds. Through the final evaluation they used 25.20 and 24.49 reserved
H100-hours, with 1.58 and 1.53 elapsed hours. Their
[terminal prompt UID sets](results/score_centering_terminal_exposure_seed19.csv) also match
exactly: 1,280 unique UIDs in each arm. Both exports succeeded; full-run costs were
26.10 and 25.40 H100-hours.

SkyRL evaluates update 40 twice at the same weights. The first scheduled score remains in
the train mirror; the finalization score is saved with the response dump. The
[repeat-evaluation ledger](results/score_centering_terminal_repeat_evals.csv) checks each
reconstructed final count against the saved responses. All counts below are completed correct
out of the same 756 held-out questions.

| Training seed | Step zero TIS / SC32 | Scheduled update 40 TIS / SC32 | Saved final update 40 TIS / SC32 | Two-pass mean SC32 minus TIS |
| --- | ---: | ---: | ---: | ---: |
| 17 | 73 / 79 | 275 / 313 | 248 / 283 | +36.5 |
| 18 | 88 / 79 | 233 / 276 | 245 / 289 | +43.5 |
| 19 | 80 / 75 | 283 / 312 | 304 / 293 | +9.0 |

The saved final pass favors SC32 in two seeds and TIS in one. Averaging the two same-checkpoint
passes favors SC32 in all three, but the second pass changes a single arm by as many as 30
correct answers among the earlier completed runs and 21 in seed 19. These are only three
training seeds and two evaluator passes per final checkpoint. They show an exploratory quality
signal, not a stable improvement or a benefit caused specifically by policy age. Held-out-
question resampling would measure a different uncertainty than training-run or evaluator
variation.

![Three older-policy Qwen seed pairs at cap 1.05, plotted against updates, consumed tokens, elapsed time, and H100-hours](figures/score_centering_cap105_older_seeds.svg)

A near-fresh cap-1.05 pair, [TIS r32](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-fresh-tis-cap105-seed17-01a0bb6f-r32)
and [TIS plus SC32 r33](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-fresh-sc-cap105-seed17-01a0bb6f-r33),
uses the same seed, model, pool, active TIS cap, top-k capture, 40-update horizon, and
evaluation contract as the older seed-17 pair. It sets maximum consumed age zero, 32
generation workers, and a 16-group buffer, matching the cap-2 near-fresh screen. Their GPU
children were admitted at `iris-interactive` priority. This pair tests whether an
apparent SC advantage persists without aged rollouts while keeping the objective fixed
within the pair.
Their step-zero completed-correct counts were 73/756 for TIS and 88/756 for SC32. At update
ten, the counts were 99 and 101; at update 20 they were 130 and 131; and at update 30 they
were 186 and 207. The scheduled update-40 evaluations scored 275 and 299, and the saved
final evaluations scored 284 and 300. The two-pass mean SC lead was 20 answers, or five
after subtracting the 15-answer step-zero lead. Through final evaluation the TIS and SC
arms used 22.70 and 23.35 reserved H100-hours and 1.419 and 1.460 elapsed hours.
Their terminal exports succeeded; full-run costs were 23.49 and 24.10 H100-hours.
The arms consumed exactly the same 1,280 prompt UIDs as each other and as the older seed-17 arms;
the [terminal exposure ledger](results/score_centering_terminal_exposure_seed17_schedules.csv)
contains every cross-schedule comparison. Across all 40 updates, their token-weighted
consumed ages were zero, their TIS cap fractions were 5.21% and 5.23%, and their
behavior-versus-trainer mean absolute log ratios were 0.01497 and 0.01510. Across the
first nine updates,
both arms consumed tokens at measured age zero, while 5.28% and 5.29% of sampled tokens still
hit the active TIS cap. Their behavior-versus-trainer mean absolute log ratios were 0.01532
and 0.01536, essentially the same as 0.01539 and 0.01545 in the older seed-17 arms' first
nine updates despite mean token age 3.66–3.68 there. Across all 40 older updates the mismatch
stayed near 0.0153. Thus the cap also acts on vLLM-versus-learner mismatch at age zero, and
the existing age separation does not establish a comparably large distribution-mismatch
separation. The near-fresh comparison therefore cannot isolate an age-specific SC benefit.

![Near-fresh and older Qwen cap-1.05 seed-17 comparisons against work and cost](figures/score_centering_cap105_fresh_vs_older.svg)

A narrower behavior-capture pair, [TIS r34](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-tis-cap105-topk8-seed17-01a0bb6f-r34)
and [TIS plus SC8 r35](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-sc-cap105-topk8-seed17-01a0bb6f-r35),
keeps the older seed-17 cap-1.05 schedule and changes behavior-logprob capture from top-k 32
to top-k eight in both arms. The SC arm uses the matching correction width eight. It isolates
SC within the narrow-capture pair and measures whether the cheaper capture path preserves the
older pair's quality signal. All four accelerator tasks were admitted at `iris-interactive`
priority. At updates zero, ten, twenty, thirty, and forty, TIS completed-correct counts were
70, 96, 132, 192, and **292**; SC8 counts were 81, 98, 136, 189, and **279**. Both arms consumed
the [same 1,280 prompt UIDs](results/score_centering_terminal_exposure_topk8.csv), also
identical to the seed-17 top-k-32 pair. Yet evaluation variation changes this comparison's
sign at the same final weights: the scheduled step-40 evaluations scored TIS/SC8 **278/292**,
while the finalization evaluations saved to the response dumps scored **292/279**. The final
SC8 difference is -13 raw and -24 after subtracting its 11-answer step-zero lead; the
scheduled difference was +14 raw. This one pair therefore does not identify a quality effect
from a narrower correction. Across its first seven unique learner updates, both arms consumed
tokens at mean age about 3.3 and capped about 5.3% of tokens, with no TIS-skipped batch. Median
inclusive cycles were 41.9 and 39.7 seconds, and bridge responses averaged 2.59 and 2.62 MB.
Across all 40 updates, token-weighted ages were 4.72 in both arms, active-cap fractions were
5.43% and 5.39%, and consumed loss tokens were 13.11 and 13.15 million. Median inclusive
cycles were 37.56 and 39.52 seconds, compared with 86.13 and 81.02 seconds for the seed-17
top-k-32 pair. Through the final evaluation, the top-k-eight arms took 0.687 and 0.698 elapsed
hours and 10.99 and 11.16 reserved H100-hours. The top-k-32 arms took 1.512 and 1.474 hours
and 24.20 and 23.58 H100-hours. Thus narrower behavior capture more than halved measured
time and GPU work in this schedule; its quality comparison remains sensitive to evaluation
variation. Both exports succeeded; full-run costs were 11.77 and 12.03 H100-hours.

![Top-k-eight and top-k-32 Qwen cap-1.05 seed-17 comparisons against work and cost](figures/score_centering_cap105_topk_width.svg)

The cap-1.05 top-k-eight delayed-publication pair, [TIS r36](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fusers-romain-checkpoints-async-rl-qwen-default-set-7a08fd68-2026.09.20.5-ecc74beeec58)
and [TIS plus SC8 r37](https://iris-cw-rno2a.oa.dev/#/job/%2Fromain%2Fusers-romain-checkpoints-async-rl-qwen-default-set-c776578e-2026.09.20.5-6e92378f4f0b),
started on September 20 with the same seed, data, 40-update endpoint, 16 H100s per arm, and
`iris-interactive` GPU pods. The [resolved configurations](configs/score_centering/README.md)
differ within the pair only in the SC switch. Relative to r34/r35, the maximum admitted
version age rises from eight to 16 updates, and inference weights are published every ten
updates instead of every update; evaluations at updates ten, twenty, thirty, and forty also
require the latest weights. The purpose is to measure whether the observed learner-versus-
behavior log-probability mismatch rises with deliberate publication delay and whether SC helps
under that larger drift. Initial remote coordinator attempts failed before reaching GPUs
because those pods lacked S3 credentials; the authenticated local launcher submitted the
linked GPU jobs directly.

Both arms reached 40 optimizer updates and saved two terminal evaluations. Their step-zero
completed-correct counts were 72 for TIS and 74 for SC8. At updates ten, twenty, and thirty,
the counts were 106/98, 117/133, and 159/185. The scheduled update-40 pass scored 233/244;
the saved final pass scored 231/264. Thus the raw SC lead was 11 and 33 answers across the
two passes, or 22 on average; subtracting the two-answer step-zero lead gives 20. All eight
seed-17 schedule and capture arms consumed exactly the same 1,280 prompt UIDs. The
[response analysis](results/score_centering_qwen_cap105_evals.csv), [per-update metrics](results/score_centering_qwen_cap105_metrics.csv),
[task costs](results/score_centering_qwen_cap105_cost.csv), and
[terminal repeat ledger](results/score_centering_terminal_repeat_evals.csv) contain the
individual observations.

The delayed TIS and SC8 arms consumed tokens at token-weighted mean ages 8.47 and 8.50,
versus 4.72 in both every-update-publication top-k-eight arms. Their behavior-versus-trainer
mean absolute log ratios were 0.01561 and 0.01575, while the every-update controls were
0.01636 and 0.01546. These small and inconsistent differences do not establish a
substantially larger distribution mismatch despite the larger version age. Active TIS caps
covered 5.55% and 5.62% of delayed tokens, versus 5.43% and 5.39% in the controls; neither
schedule rejected stale groups. Median inclusive training cycles fell from 37.56/39.52
seconds in the every-update top-k-eight pair to 22.50/26.30 seconds with delayed
publication. Through final evaluation, however, elapsed time was 0.671/0.690 hours and
reserved work 10.73/11.04 H100-hours, close to 0.687/0.698 hours and 10.99/11.16 H100-hours
with every-update publication. Setup, evaluations, and checkpoints consume part of the job
independent of median training-cycle speed. Both delayed arms finished below their
every-update-publication counterparts at the saved endpoint (231 versus 292 for TIS,
264 versus 279 for SC8), so this one-seed schedule change did not establish a useful
quality-per-cost gain. The delayed SC job's second GPU pod was deleted during teardown
after the terminal response dump was saved. Iris left that task pending; the idle parent
was canceled after confirming the final response and step-40 checkpoint were durable.
The delayed TIS arm completed terminal export and cost 11.45 H100-hours in total. The
delayed SC arm used 11.34 H100-hours through cancellation, without a terminal export;
those full-job totals are not equal deliverables.

![Every-update and delayed weight-publication Qwen top-k-eight comparisons](figures/score_centering_cap105_delayed_publication.svg)

`analyze_score_centering.py` reads every dumped evaluation response and the durable Iris
`WANDB_MIRROR` lines. It writes separate CSV files for completion-aware quality and per-update
age, mismatch, consumed tokens, cycle time, and nominal GPU-hours across inclusive step cycles.
When Iris retries a GPU child, a repeated optimizer step uses the later attempt's mirror.
Repeated `--iris-log` inputs in job order also let the PPO analysis select r26's updates
31–40 over r23's interrupted post-checkpoint updates. Identical mirrored lines are
deduplicated. The metrics CSV records both the selected job index and attempt number.
The step-cycle cost includes in-run evaluation and checkpointing but excludes setup, terminal
export, and failed attempts; those require Iris task durations in the total-cost table. The
script checks that prompt and ground-truth membership
match at every evaluation step and across the compared runs.

The evaluation CSV also records the UTC write time of each aggregate dump. The separate
`analyze_score_centering_cost.py` joins those times to a read-only Iris `task_attempts` CSV,
using the exact parent job prefix for each run. For the PPO arm it joins both r23 and r26.
It computes elapsed time from the first GPU task start and reserved H100-hours through each
evaluation. It includes retries and failed attempts, while excluding a later export from an
earlier evaluation's cost. Full-run H100-hours appear only after all matched attempts finish.
The raw Iris query and the eight-GPU-per-child assumption are stated above. This GPU task clock
does not include time spent queued before the first accelerator task started.

At `max_steps`, SkyRL runs both the scheduled step-end evaluation and a finalization evaluation
at the same weights. The latter overwrites the same step-40 response paths. Terminal quality
and cost in this report use the **final saved** response dump after finalization; an interim
read during the first write is not a terminal endpoint. For example, r28's scheduled evaluation
dump at 02:16:21 UTC had 276/756 completed correct, while its finalization dump at 02:17:05
UTC had 289/756, with no intervening optimizer update. This same-checkpoint variation is a
direct reason to avoid interpreting small single-run score differences as training effects.
For this frozen pool, GSM8K rewards are zero or one and Math500 rewards are minus one or one.
`analyze_score_centering_terminal_repeats.py` recovers each suite's completed-correct count
from its completed-stop fraction and signed reward contribution, then checks the final mirror
against the saved responses. Across 18 completed arms, the two evaluations at fixed weights
differed by 14.6 correct answers on average in absolute value, with a maximum difference of 30.
The [paired-evaluation CSV](results/score_centering_terminal_repeat_evals.csv) preserves each
count. The cap-1.05 comparisons show how the choice of terminal pass can change an inference:

| Schedule and seed | Scheduled TIS / SC | Final TIS / SC | SC minus TIS, scheduled / final |
| --- | ---: | ---: | ---: |
| Older, top-k 32, seed 17 | 275 / 313 | 248 / 283 | +38 / +35 |
| Older, top-k 32, seed 18 | 233 / 276 | 245 / 289 | +43 / +44 |
| Older, top-k 32, seed 19 | 283 / 312 | 304 / 293 | +29 / -11 |
| Near-fresh, top-k 32, seed 17 | 275 / 299 | 284 / 300 | +24 / +16 |
| Older, top-k eight, seed 17 | 278 / 292 | 292 / 279 | +14 / -13 |
| Delayed publication, top-k eight, seed 17 | 233 / 244 | 231 / 264 | +11 / +33 |

By contrast, the near-fresh cap-2 pair changed from 289 / 278 in the scheduled pass to
262 / 293 in finalization, reversing the apparent sign. These are repeated evaluator passes,
not independent training seeds. They narrow one source of ambiguity but do not establish a
staleness-specific benefit.

At the cap-2 screen's update-30 checkpoint, the comparable points were:

| Arm | Completed correct / 756 | Hours since first GPU task | Reserved H100-hours through evaluation |
| --- | ---: | ---: | ---: |
| Older TIS, cap 2 (r19) | 181 | 1.16 | 18.60 |
| Older TIS plus SC32, cap 2 (r20) | 160 | 1.19 | 19.09 |
| Near-fresh TIS, cap 2 (r21) | 184 | 1.09 | 17.43 |
| Plain PPO (r23 + r26) | 182 | 0.88 | 13.19 |

The plain-PPO step-30 evaluation was rewritten after its failed restore attempts, so this
reported point includes their elapsed time and GPU cost. It later reached 301/756 completed
correct at update 40 after 0.96 hours and 14.52 reserved H100-hours from the first GPU task.
All five cap-2 arms have update-40 evaluation dumps. Their results are:

| Arm | Completed correct / 756 | Consumed loss tokens (M) | Hours to evaluation | H100-hours to evaluation | Full H100-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| Older TIS, cap 2 (r19) | 250 | 13.150 | 1.49 | 23.87 | 24.22 |
| Older TIS plus SC32, cap 2 (r20) | 267 | 13.342 | 1.53 | 24.45 | 25.32 |
| Near-fresh TIS, cap 2 (r21) | 262 | 13.301 | 1.45 | 23.21 | 24.03 |
| Near-fresh TIS plus SC32, cap 2 (r22) | 293 | 13.125 | 1.92 | 30.65 | 31.82 |
| Plain PPO (r23 + r26) | 301 | 12.700 | 0.96 | 14.52 | 15.37 |

The r22 ledger includes its failed first attempt and successful retry. The quality counts
come from the saved response dumps, with the same 756-prompt membership hash in every arm.
The figure below plots the same primary measure against updates, consumed loss tokens, elapsed
GPU-task time, and reserved H100-hours. The underlying [evaluation](results/score_centering_qwen_cap2_evals.csv),
[training metric](results/score_centering_qwen_cap2_metrics.csv), and
[cost](results/score_centering_qwen_cap2_cost.csv) CSVs are checked in beside the analysis code.

![Completed-correct Qwen quality against updates, consumed tokens, elapsed time, and reserved GPU-hours](figures/score_centering_cap2_qwen.svg)

The two older TIS arms had mean consumed-token age 4.62 across updates, versus zero for the
near-fresh arms. Their TIS cap fractions averaged about 0.001% and never exceeded 0.011%; the
SC arm's absolute correction loss value averaged about 3e-5. At update 30, older TIS,
near-fresh TIS, and plain PPO had consumed exactly the same 960 prompt UIDs. Older SC shared
959 of its 960 prompts with them (Jaccard 0.998). Thus prompt membership barely differs, but
the generation and optimizer paths still vary across asynchronous runs. The cap-2 TIS control
and SC arm changed order repeatedly along the quality curve; the final 15-answer spread is
not evidence of a meaningful correction gradient under this nearly uncapped objective. The
near-fresh SC arm's terminal 31-answer lead over its near-fresh TIS control is subject to the
same limitation and to its extra retry.

The older-schedule captured TIS arms returned roughly 8.9–9.0 MB per inference-bridge response
and had median inclusive update cycles of 84–91 seconds. Plain PPO returned about 0.21 MB and had a
15.5-second median cycle in its original and resumed training segments. This contrasts the
current TIS-plus-top-k collection path with plain PPO. A narrower top-k-one TIS control,
[r29](https://iris.oa.dev/#/job/%2Fromain%2Fscore-centering-qwen-age8-tis-topk1-seed17-01a0bb6f-r29),
was launched with the same seed, age schedule, TIS cap 2, and 40-update budget as r19. It
changes the requested behavior-logprob width from 32 to one; a later launcher commit also
raised the pod's host-memory request from 128 to 256 GB after a separate checkpoint-restore
failure. Its first nine updates had median inclusive cycles of 20.66 seconds, compared with
84.26 seconds in r19, and mean inference-bridge response sizes of 0.720 MB versus 9.378 MB.
Mean consumed-token age was 3.32 versus 3.29 in those same early updates. Both arms had zero
TIS skipped fraction and zero batches without sampled-token logprobs. Their step-zero
completed-correct counts were both 87/756. At updates 10, 20, 30, and 40, r29 scored 99,
135, 209, and 279/756, versus r19's 103, 137, 181, and 250/756. The top-k-one run reached
its terminal evaluation in 0.63 elapsed GPU-task hours and 10.07 reserved H100-hours, versus
1.49 hours and 23.87 H100-hours for top-k 32. It consumed 13.12 million loss tokens at mean
age 4.78, versus 13.15 million at mean age 4.62 for r19. The changed capture width also changes
asynchronous generation speed and therefore policy-age exposure; these quality counts do not
isolate the effect of width on learning. The near-2.4-fold cost reduction through evaluation,
plus the 13-fold smaller bridge responses, identify wide behavior-logprob capture as a major
cost in this implementation. Both jobs succeeded; including terminal export, r29 used 12.23
reserved H100-hours versus r19's 24.22. The much longer r29 export time narrows the full-job
cost ratio compared with the ratio through evaluation.
`plot_score_centering.py` draws the completed-correct curve
against updates, consumed loss tokens, elapsed task time, and reserved H100-hours from the
three analysis CSVs. These descriptive comparisons do not identify a score-centering quality
effect or prove a plain-PPO advantage across training seeds.

The supported Snowball Megatron recipe permits a separate MoE integration check with the same
SC implementation. A matched two-update TIS and TIS-plus-SC32 smoke pair started in
`cw-us-east-02a` on September 20, each using five eight-H100 nodes at `iris-interactive`
priority, the adopted Snowball SFT export `2026.08.30`, pool `2026.08.29.1`, seed 17,
TIS cap 1.05, top-k 32 behavior capture, and age limit eight. Their Iris parents are
[TIS](https://iris-cw-us-east-02a.oa.dev/#/job/%2Fromain%2Fusers-romain-checkpoints-async-rl-snowball-smoke-set-399c37f1-2026.09.20.1-918d8daf4d8c)
and [TIS plus SC32](https://iris-cw-us-east-02a.oa.dev/#/job/%2Fromain%2Fusers-romain-checkpoints-async-rl-snowball-smoke-set-81e8344e-2026.09.20.1-29e2743294fb).
The [per-update metrics](results/score_centering_snowball_smoke_metrics.csv) preserve the four
W&B history rows, and the [task-attempt cost ledger](results/score_centering_snowball_smoke_cost.csv)
preserves Iris start and finish times for all five nodes and every attempt.
Both arms completed two Megatron updates. The TIS cap was active for 8.98% and 8.92% of sampled
tokens in the control's two updates, and 8.87% and 9.03% in the SC arm; mean absolute
trainer-versus-behavior log ratios stayed near 0.034–0.035. Neither arm skipped TIS or lacked
sampled-token logprobs. The SC arm's mean absolute correction loss values were 0.00253 and
0.00248, and its behavior top-k-32 tail masses averaged 0.00293 and 0.00284. All four raw
gradient norms were finite. The first inclusive update took 426 seconds for TIS and 443 seconds
for TIS plus SC32; the second took 673 and 653 seconds, respectively. The second policy-training
substage took 19.75 and 20.12 seconds. Different sampled lengths and concurrent storage work
prevent a clean incremental-cost estimate from two batches. The 1,024-token smoke response cap
caused 66–72% length stops, so these runs are for wiring, not answer-quality comparison. The
control job succeeded and used 24.02 reserved H100-hours. The SC job completed both optimizer
updates and a durable global-step-two checkpoint before Iris system-preempted its rank-zero pod
(`PodDeleted`); its four siblings were coscheduled for restart. The first restore attempt failed
while downloading policy weights from S3 with `[Errno 16] Please reduce your request rate`.
The next attempt loaded trainer and dataloader state and reached the Megatron optimizer restore,
then ran out of GPU memory while allocating 26 MiB with about 7 MiB free on an H100. We stopped
further retries after that repeatable restore failure. The SC job is therefore a successful
two-update integration check with a failed terminal restore/export, not a completed training job;
its three attempts consumed 50.77 reserved H100-hours. The older curriculum Snowball launcher
uses FSDP2; this fully async experiment launcher uses Megatron and does not require an FSDP2
port.
