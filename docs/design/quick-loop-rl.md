# Levanter quick-loop RL

Draft for review. Accelerator status last checked November 20, 2026 at 9:17 p.m. PST.

Move the learner in the Snowball inventory quick loop from Megatron to
Levanter with SkyRL and vLLM in place. The inventory quick loop trains the
67B-A2B Snowball checkpoint on verifiable tasks from the Snowball 100k task
selection. Once that combination passes the full-model smoke and the matched
comparison, qualify Levanter's Snowball engine as the rollout backend. Advance
native inference in parallel when the learner work is waiting on capacity or
results. Use a typed binary protocol for internal rollout and learner traffic.

## Status

| Milestone | Evidence | Next gate |
| --- | --- | --- |
| Megatron learner with vLLM | The inventory quick loop has historical 25-update runs. | Run a fresh matched control with publication verification after the Levanter smoke. |
| Levanter learner with vLLM | The integration is wired through the artifact main. CPU proofs cover learner updates, restart and Gloo weight replay. The first H100 smoke failed before model load because the worker inherited `JAX_PLATFORMS=cpu`. The fix selects CUDA explicitly and keeps runtime scratch in the compute region. | Finish the frozen 40-H100 smoke. Then compare both learners for 25 updates. |
| Learner on current `main` | The local port covers SPMD scoring, Snowball precision and positions, the SkyRL adapter, and synchronous BATCH presets. Eight CPU worker and publisher cases passed, including continuation after fresh-process restore. | Qualify the deployed worker and maintained preset on H100. |
| H100 smoke `2026.11.20.5` | Source `6936528c55`. Iris retried the preempted coordinator and created child `a1029950e470`. All five tasks are scheduling-gated. At 9:17 p.m., Kueue reported a shortfall of one GPU and one InfiniBand allocation. There are no GPU execution receipts for this child. | Admit the existing gang and collect native state, weight receiver and generation receipts. |
| Native Snowball capture and replay | A fresh learner reproduced all 555 sampled raw logprobs on the two-layer, four-TPUv4 fixture. Serving used paged inference and the fixture used an unfused loss. | Port the required code to current `main`, keep that numerical gate reproducible, then run two closed RL updates. |
| Native online loop on current `main` | The local prototype completed both updates on one and four CPU devices. Each run consumed eight binary frames and sampled 128 tokens. The second rollout used the exact first published weight fingerprint. The maximum sampled-to-learner logprob difference was `4.06e-6`. The first four-TPUv4 run failed the score comparison. A rerun with shared normalization is queued. | Complete both updates on four TPUv4 chips. |
| Native inventory backend | The branch contains a portable route payload and paged decode. Against one-token decode chunks, 32-token chunks cut finite-batch latency by 3.54× on the long fixture and 1.79× on the mixed fixture. | Verify weight installation, real tool turns, full checkpoint and production context, then run the matched serving comparison. |
| Typed binary protocol | The local native loop uses protobuf metadata and binary tensor buffers end to end between generation and learner batching. The existing learner pipe uses binary pickle, and expert-block publication uses binary NCCL. | Qualify cross-process native transport with bounded flow control, then migrate the learner pipe after learner acceptance. |

[Quick-loop branch status][learner-status] · [Regional Iris smoke][iris-smoke] ·
[First TPUv4 attempt][iris-native] · [TPUv4 normalization rerun][iris-native-rerun] ·
[Snowball qualification report][snowball-status] ·
[Historical quick-loop timing][timing]

## Desired system

SkyRL continues to run the synchronous loop. The existing artifact main
constructs a pinned experiment with an explicit role plan and emits the
`SkyRLLaunchConfig`. SkyRL owns task selection, rollout admission, tools,
grading, rewards, RLOO-N advantages, evaluation and run bookkeeping. Levanter
owns raw policy scoring, gradients, optimizer state and native checkpoints.
The serving backend owns sampling, KV state and an acknowledged policy
installation. One policy version is active for an admitted batch.

```text
SkyRL / RolloutEngine
  -> sample with the acknowledged policy version
  -> grade four answers per prompt and admit complete groups
  -> finalize advantages and whole-minibatch loss weights
  -> Levanter PolicyLearner scores old policy and applies one update
  -> publish the new policy and verify the serving install
  -> admit generation on the new version
```

RolloutEngine initially calls vLLM. A native Snowball backend implements the
inference and publication operations that RolloutEngine uses for this recipe.
The full recipe needs token-in/token-out generation, chat rendering with tool
continuation, stop handling, evaluation sampling and an admission barrier.
The native fixture can call the engine in process. It exercises the same
learner and policy version rules before adding remote serving and tools.
[Artifact launch contract][launch] · [Inference interface][engine-interface]

### Policy and batch contract

Use zero staleness, a single optimizer pass, token-mean clipped PPO, and
coordinator-computed neutral leave-one-out advantages (RLOO-N). The inventory
recipe uses 128 groups of four
answers per update. Admission sees four completed outcomes per group. An
infrastructure failure occupies a slot and is excluded from the baseline and
loss. A group needs at least two baseline contributors. Agent failures count
as contributors with zero reward. Groups whose eligible outcomes have equal
reward contribute zero advantage. Advantages remain fixed during the update.
Compute the loss denominator over the complete optimizer minibatch before sharding or
accumulating gradients. Prompt, padding and tool observations carry zero
objective weight and valid context tokens remain visible to attention. The
recipe uses `1e-6` AdamW, clip width 0.2, and zero KL and entropy coefficients.
[Recipe and objective][policy] · [RLOO-N implementation][rloo]

The batch carries exact rendered token IDs, positions, attention and response
alignment, loss weights, advantages and policy identity. Output token at
position `p + 1` is scored from predictor row `p`, including the first sampled
token after prefill and EOS when retained. With replay, bind the expert IDs,
route validity and the router's FP32 thresholds (`qb_alpha`) to those coordinates and the full
causal prefix. Carry them through padding, packing and microbatch slicing.
The learner fixes expert choices for scoring and update and recomputes combine
weights under its current parameters. Preserve the same frozen old scores
and objective weights through the update.

A sampling logprob describes the transformed distribution used to draw a
token. A raw policy logprob describes the model before temperature and
top-p. Inventory training and the native parity gate use temperature 1.0 and
top-p 1.0. The inventory evaluation uses temperature 0.6 and top-p 0.95. Track
each logprob kind explicitly and use frozen raw scores in the PPO ratio.
[Scoring implementation][scorer] · [Replay payload][replay]

### Policy installation and recovery

Give every published policy an identity scoped to the run, model and
initializer revision, with a monotonic version within the run attempt.
The verified serving tensor manifest has one immutable digest per identity.
It describes the installed compute view, including casts, dtype, shape and
a digest of each shard's bytes. A payload checksum establishes payload
integrity. The serving receipt identifies the installed weights. Keep sample
stamps consistent with the existing SkyRL convention
that weight version `v` produces token stamp `v + 1`.

At the synchronous boundary, close admission and drain in-flight requests.
Verify each serving rank installed its expected shards for the same version.
Clear or recreate KV and prefix state, then acknowledge the new cache epoch
before reopening admission. Fail the step if an install is partial or any
rank disagrees.
Treat a repeated publication of the same identity as idempotent. A retry
with different bytes at the same identity fails. Bind each trajectory to
the version and cache epoch reported by the serving backend when it sampled
the tokens. Reject mixed or stale identities at admission.

Save model, Adam state, RNG, step and model averaging in the native checkpoint.
Commit a step after the optimizer update and serving acknowledgement. Use
run, step and admitted batch identity to prevent duplicate updates within a
live worker. If the learner state is ambiguous after a failure, recover from
the last durable checkpoint and its batch cursor. Start a fresh publication
epoch when restoring an older optimizer step. Restore exact learner
fingerprints and resynchronize serving before accepting new rollouts. Archive
the artifact version, code and runtime pins, checkpoint metadata, and
publication and generation receipts. Native terminal HF export is an open deliverable for
consumers that require it. The RL gate uses the native checkpoint and an
acknowledged serving installation. [Checkpoint and publication checks][learner-status]

## Binary protocol

Use a checked-in protobuf schema for commands, receipts and route metadata.
Use a bounded length-delimited frame with a header and contiguous little-endian
tensor bytes. For each tensor the header specifies a field ID, dtype, shape,
offset and byte count. Keep the codec importable in both the SkyRL and Levanter
environments without importing Torch or JAX. Include a protocol version,
message kind, request or trajectory ID, and policy identity. Use stable dtype
codes and define exactly which ordered bytes enter the payload digest.

The rollout message contains the exact rendered token sequence, including
tool observations, and token spans for prompt, assistant and tool turns.
Include predictor positions or a defined contiguous position range, masks,
finish reason and sampling metadata. Index each logprob and route row by its
predictor position and distinguish both kinds of logprob in the header.
For route replay, require selected expert IDs, per-layer validity and QB
thresholds. The learner command carries the admitted batch and the same
policy identity. The response carries optimizer step, publication status or
an explicit error. Human-readable config, logs and receipts can remain JSON.
Keep the existing vLLM OpenAI edge and NCCL expert-block transport during
learner qualification.

Validate frame size before allocating tensor buffers. Validate declared
length against dtype and shape, tensor bounds and overlap, the supported
schema version, token and route alignment, and expected policy identity
before admitting data. Bound in-flight bytes and stream per trajectory,
turn or bounded chunk. Route size grows with sequence length, layer count
and selected experts. A full 512-answer update can have large capture volume
at the production context limit. Measure serialized bytes, peak host memory
and encode/decode latency before expanding capture coverage.

Use the binary format on the first cross-process native replay integration.
Keep historical replay fixtures readable where their qualification requires
it. Migrate the existing trusted per-host pickle pipe in a separate change
after the matched learner arms. Test round trips across both runtime
environments, exact token and tensor bytes, invalid lengths and versions,
stale policy identities, truncated frames and restart after publication.
[Current learner pipe][worker-pipe] · [Current replay encoding][replay]

## Implementation and acceptance

### 1. Qualify the Levanter learner on H100

Use the frozen smoke bundle and the existing `2026.11.20.5` artifact. The
job reserves four eight-H100 learner hosts and one eight-H100 vLLM host in
`cw-us-east-02a`. The learner uses data4/expert8. Keep the current gang while
it is queued and follow the coordinator if Iris retries the child.

- Verify all learner ranks select CUDA and load
  `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` at `fe666478454e`.
- Require two completed updates with model-change evidence for each. The
  frozen worker fingerprints the first change during its checkpoint gate.
  For the second update, compare identical trainable tensor slices in
  `native-validation/optimizer-step-1` and `global_step_2/policy`. At least
  one slice must change. An unchanged sample is inconclusive. Use regional,
  manifest-guided reads of bounded slices. Receiver byte receipts attest
  transfer equality within a version.
- Require zero mismatched bytes for the installed vLLM weight versions and
  nonempty generation stamped by the serving backend with the new policy.
  Verify all eight expected receivers for each publication. Require stamp 2
  on generation after version 1 and stamp 3 on final evaluation after
  version 2. Require a terminal native checkpoint at step 2, exact model,
  Adam, RNG, step and averaging fingerprints after fresh learner-process
  restore, and the further optimizer update after restore.
- After acceptance, run the planned 25-update Megatron/vLLM and Levanter/vLLM
  arms with distinct artifact versions and publication verification enabled in
  both. Hold the initializer, Snowball 100k selection `2026.11.04`, optimizer, 24,576-token request window, 8,192-token
  response cap, grader and fixed 100-task evaluation panel constant. Record
  training reward separately from fixed-panel reward. Record failed groups,
  generated and trained tokens, per-step timing, GPU-hours and output receipts.

Both 25-update recipes render with publication verification. Their only
rendered differences are the learner strategy, batch builder and Levanter
policy configuration. The original preflight verified the input artifact.
A fresh preflight from the current shell reached the selection metadata check
and requires the regional object-store credentials.

Current `main` uses the `experiments.post_training.rl` runner and named
Snowball presets. The local port adds `levanter-smoke`, `learner-control`
and `learner-levanter` presets for synchronous BATCH runs. Both Levanter
presets emit native checkpoints. The existing `learning-medium`
preset uses asynchronous rollouts, truncated importance sampling and HF
export. Those features each need explicit learner support before carrying
the new strategy into that preset. [Preset migration][preset-migration]

The unchanged frozen source reproduces a startup framing failure on two CPU
Gloo processes with JAX and JAXLIB 0.11.0. Python stdout redirection leaves
native writes on the frame stream, and the reader interprets the Gloo
message as a frame length. The current-main worker port separates those
file descriptors and passes the two-process test. In the H100 logs, inspect
the first frame read error before attributing later rendezvous failures.

A separate exact-pin CPU probe confirmed that a native child can receive
SIGTERM when its spawning thread exits while the host remains alive.
Check for this separately if a restored process exits immediately after
a matching checkpoint receipt. The CPU probes establish two concrete
diagnostics. A local serial Ray probe kept the creating thread and child
alive between method calls. The current-main port creates the child on its
persistent IPC thread and passed single- and two-process host restart
tests. CUDA/NCCL startup and restart remain unverified on the frozen H100
job.

[Exact smoke and comparison recipes][learner-status] ·
[Frozen checkpoint validation code][frozen-worker]

### 2. Close the smallest native RL loop

Port the Snowball paged model/cache path, engine capture, learner replay and
fixture arithmetic in reviewable slices onto current `main`. Reconcile
Snowball attention, model and loss changes with the learner branch. The
snapshot is 766 commits behind the `main` used for this design, and includes
unrelated Hero and Iris changes. [Source comparison][snowball-compare]

The bounded run on source `158bd86404` generated the first group and
scored its replay on four TPUv4 chips. At the first raw logprob comparison,
51 of 64 scores exceeded `rtol=1e-5` and `atol=1e-5`. The largest
absolute difference was `7.13e-5`. The assertion fired before the first
optimizer update, so the run produced no step or publication receipt.
[Native TPUv4 fixture][iris-native]

The failing source uses the current paged and fused scoring paths. The
earlier bitwise fixture used an opt-in fixed-schedule model and a shared
batch-invariant normalizer for the sampler and unfused learner loss.
The rerun uses source `d7ddcf3c67` and the shared fixed-row FP32 normalizer
for sampling and full-vocabulary learner scoring. It saves fused scorer
values alongside the serving and learner scores before applying the
original tolerance. The candidate completed two CPU updates with that
profile and passed 35 focused CPU tests. The four-chip TPUv4 run was
pending at the last check. [TPUv4 normalization rerun][iris-native-rerun] ·
[Earlier Snowball qualification][snowball-status]

A separate local scheduler slice passed heterogeneous queue and slot-reuse
replay on one and four CPU devices. The local policy owner also passed
draining in-flight generation, swapping to a fresh KV cache, preserving
the old policy on installation failure, and rejecting stale policy
requests. The four-device CPU loop through that owner published policy
stamps 1 and 2 and sampled its second group from stamp 1.

Start with the supported four-row, 256-token TPUv4 layout, a single-turn task
and a deterministic grader with mixed rewards within a group. Submit four
single-choice requests with distinct seeds under the same prompt ID.
Use the existing RLOO-N calculation and Levanter `PolicyLearner` through
`PolicyBatch`. Assemble a batch whose route tensors can be sliced and sharded
with its tokens. The quick-loop bridge rejects `routed_experts` today, and
the Snowball snapshot requires default contiguous positions at its public
replay boundary. Use the qualified layout for the first gate. Then verify
explicit positions, left-padded prompts and right-padded responses before
connecting the inventory batch builder.

Check matching routes and raw logprobs before the first update under identical
weights. Check finite loss and changed parameters. Publish the update, reset
serving state and confirm the next nonempty rollout used the new weights.
Complete a second graded optimizer update and publish again. Save the batch,
policy identities and receipts so the bounded acceptance can be replayed
without reconstructing sampled text. [Batch bridge][batch-bridge] ·
[Engine capture API][engine-replay]

### 3. Qualify native serving for the inventory recipe

- Connect the native backend to the RolloutEngine operations used by the
  recipe. Prove one actual model/tool/model rollout end to end, with sampled
  IDs, complete causal route coverage, predictor positions and zero loss on
  tool observations. Then exercise tool and infrastructure failure masks
  and a restart at the publication boundary.
- Load the pinned 67B checkpoint on the intended serving hardware. Qualify
  real context and batch sizes, expert sharding, precision, peak memory,
  compile time and sustained serving. Measure serve/score drift under the
  same weights and routes with the production fused scorer. Report raw
  logprob error and ratio/clipping distributions. Use the bitwise fixture to
  catch regressions and establish full-shape tolerances before the matched
  run.
- Run a matched Levanter/vLLM versus Levanter/native comparison. Hold learner,
  task mix, sampling and fixed evaluation panel constant. Record hardware
  and cost, time per admitted group and optimizer update, and fixed-panel
  reward. Break group readiness down into generation, tool execution and
  grading. Record publication separately in the per-update timing. In the
  prior FP8 run, mean decode time fell 13%, the wait for rollout groups rose
  6.1 minutes and time to the same evaluation target rose 5.7 minutes.
  [Prior timing comparison][timing]

## Open decisions

- Set learner promotion criteria for fixed-panel reward and update time or
  cost before changing the default learner.
- Choose the full-model native serving topology after measuring the bounded
  path and profiling the pinned checkpoint. The current native parity
  evidence covers one TPUv4 host, and the learner comparison uses H100s.
- Choose numerical margins and a throughput target from the new vLLM
  baseline before the native inventory comparison. The historical speed
  figures describe distinct experiments.
- Decide which checkpoint consumers need HF export before switching the
  default learner. Identify the export artifact and verify it with those
  consumers.

[learner-status]: https://github.com/marin-community/marin/blob/b8aa9e2af13d991b45c64e036cd0b4885222a118/experiments/post_training/inventory/levanter-quick-loop-status.md
[iris-smoke]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.5-coordinator-2ea612c2%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.5-a1029950e470
[snowball-status]: https://github.com/marin-community/marin/blob/5992abd0529d81bb90356e97bab3a8ac498daf68/docs/reports/snowball-inference-status-2026-11-20.md
[timing]: https://github.com/marin-community/marin/issues/11190
[launch]: ../references/rl-launching.md
[engine-interface]: https://github.com/marin-community/marin/blob/b8aa9e2af13d991b45c64e036cd0b4885222a118/vendor/skyrl/skyrl-train/skyrl_train/inference_engines/base.py#L126
[policy]: https://github.com/marin-community/marin/blob/b8aa9e2af13d991b45c64e036cd0b4885222a118/docs/references/policy-learning.md#skyrl-quick-loop
[scorer]: https://github.com/marin-community/marin/blob/b8aa9e2af13d991b45c64e036cd0b4885222a118/lib/levanter/src/levanter/policy_model.py#L135
[replay]: https://github.com/marin-community/marin/blob/fcc3167fef231ed2dacdd9e21cd48a564668e7ee/lib/levanter/src/levanter/inference/snowball_replay_payload.py
[worker-pipe]: https://github.com/marin-community/marin/blob/6936528c55df0212ec8ccd1fa619361863b4c420/lib/marin/src/marin/rl/levanter_worker_protocol.py#L18-L41
[frozen-worker]: https://github.com/marin-community/marin/blob/6936528c55df0212ec8ccd1fa619361863b4c420/vendor/skyrl/skyrl-train/skyrl_train/workers/levanter/levanter_worker.py#L156-L225
[snowball-compare]: https://github.com/marin-community/marin/compare/2bf3f721c9946150a02c3417c4c93e14b273a5e8...5992abd0529d81bb90356e97bab3a8ac498daf68
[batch-bridge]: https://github.com/marin-community/marin/blob/b8aa9e2af13d991b45c64e036cd0b4885222a118/vendor/skyrl/skyrl-train/skyrl_train/workers/levanter_batch.py#L46-L74
[engine-replay]: https://github.com/marin-community/marin/blob/fcc3167fef231ed2dacdd9e21cd48a564668e7ee/lib/levanter/src/levanter/inference/snowball_engine_replay.py#L32-L42
[rloo]: https://github.com/marin-community/marin/blob/6936528c55df0212ec8ccd1fa619361863b4c420/vendor/skyrl/skyrl-train/skyrl_train/utils/advantage_estimators.py#L176-L289
[preset-migration]: https://github.com/marin-community/marin/pull/10740
[iris-native]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-binary-proof-20261120-01
[iris-native-rerun]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-binary-proof-20261120-02
