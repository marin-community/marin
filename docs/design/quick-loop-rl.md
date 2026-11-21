# Levanter quick-loop RL

Draft for review. Accelerator status last checked November 21, 2026 at 7:04 a.m. PST.

Move the learner in the Snowball inventory quick loop from Megatron to
Levanter with SkyRL and vLLM in place. The inventory quick loop trains the
67B-A2B Snowball checkpoint on verifiable tasks from the Snowball 100k task
selection. Once that combination passes the full-model smoke and the matched
comparison, qualify Levanter's Snowball engine as the full-model rollout
backend. The learner jobs have first claim on H100 capacity. Run bounded
native-loop checks on TPU while the learner jobs wait. Use a typed binary
protocol for internal rollout and learner traffic.

## Status

| Milestone | Evidence | Next gate |
| --- | --- | --- |
| Megatron learner with vLLM | The inventory quick loop has historical 25-update runs. | Run a fresh matched control with publication verification after the Levanter smoke. |
| Levanter learner with vLLM | The integration is wired through the artifact main. CPU proofs cover learner updates, restart and Gloo weight replay. The first H100 smoke failed before model load because the worker inherited `JAX_PLATFORMS=cpu`. The fix selects CUDA explicitly and keeps runtime scratch in the compute region. | Finish the frozen 40-H100 smoke. Then compare both learners for 25 updates. |
| Learner port based on `main` | The local port covers SPMD scoring, Snowball precision and positions, the SkyRL adapter, and synchronous BATCH presets. Eight CPU worker and publisher cases passed, including continuation after fresh-process restore. | Qualify the deployed worker and maintained preset on H100. |
| 40-H100 learner smoke | Run `2026.11.20.5` joined all five Ray nodes and captured vLLM CUDA graphs. The learner failed during mesh initialization before any update. Corrected source `dba95c7f84` passed 15 local tests on JAX 0.11.0 and rendered the two-update recipe as version `2026.11.20.8`. | Submit the staged rerun once the inventory task service credentials are authorized for the CoreWeave job. The matched runs depend on smoke acceptance. |
| Native Snowball capture and replay | A fresh learner reproduced all 555 sampled raw logprobs on the two-layer, four-TPUv4 fixture. The local port passed the bounded optimizer gate and replayed four 24,576-token trajectories through the fused scorer on the small TPUv4 fixture. | Qualify production shapes and fused replay against the pinned checkpoint. |
| Native online loop based on `main` | On two four-chip TPUv4 hosts, the small BF16 fixture passed two tool episodes and optimizer updates, byte-exact binary replay, fresh learner restore and final generation under policy 2. One serving process handled all three policies. | Verify the deployed handoff from four learner hosts to one serving host at 67B. |
| Native inventory backend | Both inventory presets pass regional read-only plans. The standalone probe loaded the pinned 67B model on eight H100s and ran out of memory at full-size prefill. A query-tiled reference passed CPU parity tests at the same token capacity. | Verify full generation and fused replay on H100, then run the held-learner serving comparison. |
| Typed binary protocol | Native serving and the maintained isolated learner use protobuf headers with detached binary payloads. Both passed exact TPUv4 recovery. The typed learner also passed CPU Gloo expert-block publication. | Verify the production four-learner-host handoff and measure the typed learner at 67B. The frozen H100 learner keeps its original pipe for the smoke and matched comparison. |

[Quick-loop branch status][learner-status] · [Regional Iris smoke][iris-smoke] ·
[TPUv4 queue and cache proof][iris-native-install] · [TPUv4 separate-process proof][iris-native-runtime-reference] ·
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
Let a live serving process re-acknowledge an exact install receipt using
the current or immediately preceding policy identity. A conflicting
receipt for the same step fails. Bind each trajectory to the version and
cache epoch reported by the backend when it sampled the tokens. Reject
mixed or stale identities at admission.

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
tensor bytes. The header identifies each tensor and specifies its dtype,
shape and byte count. Segment order determines offsets in the learner pipe.
Keep the codec importable in both the SkyRL and Levanter environments without
importing Torch or JAX. Include a protocol version, message kind, request or
trajectory ID, and policy identity. Use stable dtype codes and define exactly
which ordered bytes enter the payload digest.

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

The maintained learner at `63fb947665` uses the typed pipe for all 13
subprocess commands, including the four expert-block publication commands.
A 64-bit frame length carries up to 16 GiB, with a 2 MiB protobuf header
limit and at most 128 MiB per binary segment. The writer sends each tensor,
replay frame or weight block separately. A framing test counted the writes
for a logical 6.7 GiB inventory batch without allocating that batch.
It checks length encoding and segmentation. Peak memory and latency at 67B
still need live measurement.

On CPU, the typed learner reproduced exact native restart state and live
replay after an interrupted checkpoint write. Two real Torch Gloo receivers
also received 7,876,160 bytes each per policy version across two versions,
with zero byte differences from the prior build after a real optimizer
update. H100 vLLM and NCCL remain accelerator gates. The frozen smoke and
matched learner runs keep the existing learner pipe.
[Typed learner source and proof receipts][typed-learner-source] ·
[Frozen learner pipe][worker-pipe] · [Replay encoding][replay]

A local process-boundary proof at `9fc0fd8afd` sent four generated frames
through a Unix socket with one unacknowledged frame at a time. A separate
learner validated frame integrity and policy identity, rebuilt the graded
batch, and reproduced its scores on a second pass. The follow-up at
`9809cafecf` published two optimizer updates. The separate serving process
restored each TensorStore checkpoint, verified the actual weight digest and
acknowledged only after replacing its KV state. It generated 64 targets at
each of versions 0, 1 and 2. The largest raw score difference was
`4.29e-6`. A wrong digest and stale identities were rejected. After the
failed install, the active policy reproduced its earlier frames
byte-for-byte. Both proofs ran on one and four CPU devices. See the local
controller and multi-process publication checks for the next integration
boundary.

The learner worker publishes immutable, weights-only TensorStore snapshots.
Each learner process hashes its owned logical slices and exchanges only
the digests. The `snowball-weights-v2` protocol at `4594784248` publishes
the learner's compute weights. Inventory uses BF16 with FP32 router
weights and router bias. The serving process restores the same bytes and
verifies their dtypes and digests before it acknowledges the new policy
and opens generation with fresh KV state. The receipt includes the
publisher's shard layout, so the serving host can verify the same slices
on a different local layout. Native learner checkpoints retain the FP32
master weights and optimizer state.

Local CPU tests installed versions 0 through 2 across the worker pipe and
rejected stale identities. They exercised publication from one and two
learner processes and a bitwise-identical next update after restoring the
trainer. A local Ray run installed two mixed-precision snapshots while
preserving exact binary replay across tool turns. The tool-turn check
across accelerator hosts remains open.

The local `snowball_rl` artifact main at `2a8cbaf1c7` renders two-update
presets named `levanter-inventory-native` and `levanter-inventory-vllm`.
Both use four eight-H100 learner hosts and one eight-H100 serving host.
They share the pinned initializer and tokenizer, the `2026.11.19.1` task
selection, and a fixed 100-task panel. Use this pair to qualify the
maintained inventory launcher. The frozen 25-update learner comparison
uses its original selection and remains a separate experiment.

The launcher takes the native serving device count and EP/TP dimensions
from the role plan. It places immutable snapshots in the launch attempt's
regional temporary storage and optimizer recovery in the checkpoint store.
Iris is configured to provision the same locked Levanter environment on
all five nodes. The native bootstrap checks that SkyRL can import the host
client and that the isolated JAX interpreter can import the binary endpoint.
These checks passed on CPU with JAX and JAXLIB 0.11.0. GPU startup remains a
qualification step.

Regional TensorStore is the first deployment transport. At 67.078B
parameters, including about 17 million FP32 routing values, one write and
one read of the compute weights move about 268 GB of logical data per
publication. The earlier FP32 transport would move about 537 GB. A native
run with 25 of these publications would move about 6.7 TB before
compression, initialization and retries. Measure upload, restore, byte
verification and cache allocation separately on H100.

The native serving endpoint at `40ce66daf4` uses the bounded Marin worker pipe. It sends
versioned protobuf commands with exact prompt IDs, seeds and generation
limits. Replies carry a protobuf header and concatenate the existing
replay frames as a separate binary body. Each result declares its exact
length. The host validates the total size before slicing the frames. The
child verifies checkpoint weights before installing fresh KV state. One
serving process installed three successive policies across two real CPU
optimizer updates. A separate CPU test exercised the SkyRL Ray worker
through two tool continuations with the same transport.
[CPU serving receipts in Filer][native-serving-wire-archive]

The local retry change at `f36f5761a5` verified a repeated install through
the actor and protobuf process after moving the snapshot directory.
Serving re-acknowledged the same receipt and kept producing the same
replay bytes. It rejected a different digest for the installed step and
a changed receipt for the same digest. The publisher test withheld the
first acknowledgement and kept the next lease closed until an exact
retry succeeded. This covers a live serving process.
[Retry source and CPU tests][native-install-retry-source]

At the pinned model's 26 layers and four selected experts, a synthetic
24,576-token frame occupied 13,650,628 bytes. A four-answer group of those
frames occupies about 52 MiB. In a five-repeat local codec probe, median
encode and decode took 17.8 ms and 8.1 ms per frame. Choose a byte budget at
the service boundary that admits this geometry. Measure frame lifetimes and
RAM across concurrent production requests before increasing concurrency.

The serving actor groups up to four training requests on the installed
policy. The child bounds the batch by KV capacity and the existing 65 MiB
IPC limit. It returns a separate replay frame for each request. Four
synthetic full-context frames occupied 52.3 MiB in the IPC envelope. On
four CPU devices, batched and serial calls matched exact tokens, per-layer
routes, QB thresholds, sampled scores, stop reasons and PRNG keys. The
actor drains admitted requests before installing the next policy. This
qualifies the scheduling and wire behavior on CPU. Sustained group
throughput and memory remain GPU measurements.

In a synthetic CPU probe of the response envelope alone, four replay
frames totaling 54.86 MB encoded in 17.3 ms and decoded in 17.8 ms with
the separate body. Embedding the same frames in protobuf took 155 ms to
encode and raised peak process RSS by about 109 MiB relative to the
separate body. Both forms preserved the frame digests. The probe measures
only the envelope codec.

## Implementation and acceptance

### 1. Qualify the Levanter learner on H100

Use the staged `2026.11.20.8` smoke pinned to `dba95c7f84`. The prior
`2026.11.20.5` attempt is terminal. The rerun reserves four eight-H100 learner
hosts and one eight-H100 vLLM host in `cw-us-east-02a`. The learner places
the four data shards across learner hosts and keeps the eight expert-axis
shards within each host.
Authorize the inventory task service credentials for the job before submission.
Follow the coordinator if Iris retries the child.

- Verify all learner ranks select CUDA and load
  `open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21` at `fe666478454e`.
  Require startup placement evidence from all four learner hosts. Retrieve
  the original attempt-scoped Ray worker logs if the aggregated startup
  stream omits a rank.
- Require two completed updates with model-change evidence for each. The
  frozen worker fingerprints the first change during its checkpoint gate.
  For the second update, compare identical trainable tensor slices in
  `native-validation/optimizer-step-1` and `global_step_2/policy`. At least
  one slice must change. An unchanged sample is inconclusive. Use regional,
  manifest-guided reads of bounded slices. Receiver byte receipts attest
  transfer equality within a version.
- Require zero mismatched bytes for the installed vLLM weight versions and
  nonempty generation stamped by the serving backend with the new policy.
  Verify expert-block receiver participants 32–39 for each publication. Require stamp 2
  on generation after version 1 and stamp 3 on final evaluation after
  version 2. Require a terminal native checkpoint at step 2, exact model,
  Adam, RNG, step and averaging fingerprints after fresh learner-process
  restore, and the further optimizer update after restore.
  [Expert-block participant numbering][frozen-receiver-numbering]
- After acceptance, run the planned 25-update Megatron/vLLM and Levanter/vLLM
  arms with distinct artifact versions and publication verification enabled in
  both. Hold the initializer, Snowball 100k selection `2026.11.04`, optimizer, 24,576-token request window, 8,192-token
  response cap, grader and fixed 100-task evaluation panel constant. Record
  training reward separately from fixed-panel reward. Record failed groups,
  generated and trained tokens, per-step timing, GPU-hours and output receipts.

Both 25-update recipes passed read-only preflight on the frozen source
against the regional task selection. Megatron uses version `2026.11.20.6`
and Levanter uses `2026.11.20.7`. Both enable publication verification.
The trainer differences are the learner strategy, batch builder and
Levanter policy configuration. Submit the control first after smoke acceptance.
Before each submission, confirm that Iris and the durable and scratch prefixes
have no prior attempt for that version. The launcher creates a fresh coordinator
ID and the trainer can resume from an existing checkpoint.

For each arm, collect Finelog rows for the successful child and its task attempts.
Match the `WANDB_MIRROR` metrics by optimizer step with rollout and update
completion lines. Require a step-0 start and one applied optimizer update in
`policy/policy_update_steps` at each step through 25, with finite optimizer
metrics. A step-completion line can accompany a skipped update. Check the
step-25 checkpoint marker and committed policy even when the successful
terminal receipt has `model: null`, as expected with HF export disabled.
Require zero byte mismatches on all eight receivers at version 25 and nonempty
generation with stamp 26 between the publication acknowledgement and the final
evaluation result. Verify the panel rows separately.
Report admitted groups, admitted response tokens, generated tokens when available
and consumed loss tokens separately. Filtering can change the workload each
learner sees.
Measure training and publication wall time from the initial evaluation
result to the acknowledgement of version 25. Report allocation wall time
separately. [Matched-run auditor and tests][h100-matched-audit]

The frozen evaluator writes per-dataset rows beneath `exports/dumped_evals`
with ordinal UIDs assigned after prompt-length filtering. Verify that no
rows were dropped from the pinned validation Parquet and that each of the
100 expected UIDs appears exactly once per panel. Compare sources and
prompt tokens across both arms at steps 0 and 25. Recompute
`eval/all/avg_verifier_score` from verified row verdicts and reconcile it
with the aggregate dump and the trainer log. Compare paired rewards only
with 100 verified scores per panel. Report incomplete panels with their
attempted and verified counts.
[Frozen evaluator UID assignment][frozen-validation-uid] · [Pinned validation panel in Filer][frozen-validation-panel]

The baseline at `2bf3f721c9` uses the `experiments.post_training.rl` runner and named
Snowball presets. The local port adds `levanter-smoke`, `learner-control`
and `learner-levanter` presets for synchronous BATCH runs. Both Levanter
presets emit native checkpoints. The existing `learning-medium`
preset uses asynchronous rollouts, truncated importance sampling and HF
export. Those features each need explicit learner support before carrying
the new strategy into that preset. [Preset migration][preset-migration]

The unchanged frozen source reproduces a startup framing failure on two CPU
Gloo processes with JAX and JAXLIB 0.11.0. Python stdout redirection leaves
native writes on the frame stream, and the reader interprets the Gloo
message as a frame length. The worker at local revision `4a6d1e7a3e`
separates those file descriptors and passed the two-process Gloo restart
test. In the H100 logs, inspect the first frame read error before
attributing later rendezvous failures.

The frozen H100 run stopped in mesh construction before model load. The
corrected CUDA layout keeps logical `data=4` across the four learner hosts
and `expert=8` within each host. This preserves the data-axis sharding of
Snowball parameters and Adam moments. The pinned model's storage layout puts
about 27.9 GiB of persistent state on each H100 before activations and
temporary buffers. The corrected branch preserves the binary learner
protocol, separates native stdout from replies, and keeps the child process
on its persistent IPC thread. The exact-version CPU tests verified fresh
restore and continuation. The four-host CUDA run still needs to prove
runtime placement, peak memory, publication, and a second update.

A separate exact-pin CPU probe confirmed that a native child can receive
SIGTERM when its spawning thread exits while the host remains alive.
Check for this separately if a restored process exits immediately after
a matching checkpoint receipt. The CPU probes establish two concrete
diagnostics. A local serial Ray probe kept the creating thread and child
alive between method calls. Revision `4a6d1e7a3e` creates the child on its
persistent IPC thread. Its CPU restart tests covered fresh learner and
host processes. CUDA/NCCL startup and restart remain unverified on the
frozen H100 job.

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

The failing source uses the paged and fused scoring paths. The earlier
bitwise fixture used an opt-in fixed-schedule model and a shared
batch-invariant normalizer for the sampler and unfused learner loss.
The rerun on `d7ddcf3c670b5c6aca19a111cba15e1082e8d560` used
`levanter.testing.snowball_quick_loop --backend tpu --projection full_vocabulary`
on four TPUv4 chips. Paged sampling and full-vocabulary learner scoring
used the same fixed-row FP32 normalizer. Both optimizer updates completed.
The second rollout's weight fingerprint matched the first publication, and
each update produced a distinct weight fingerprint. An independent decode of
the eight binary frames confirmed request IDs, weight identities, token IDs
and raw score coverage for all 128 sampled tokens. The largest raw score
error was `4.29e-6`, and both steps passed the original tolerance. The
diagnostic fused forward scorer also passed against both rollout groups with
the same maximum error. The optimizer used the full-vocabulary path.
[Successful TPUv4 proof][iris-native-rerun] · [Proof archive in Filer][native-archive] ·
[Earlier Snowball qualification][snowball-status]

The archive is
`gs://marin-us-central2/users/dlwh/quick-loop-rl/native-proof/2026.11.20.02/outputs.tar.zst`.
Its SHA-256 is `fe3d11318301799d201e4a744cc604ac42c4009f40a37d6ba08539c6720178f0`.

A separate local scheduler slice passed heterogeneous queue and slot-reuse
replay on one and four CPU devices. The local policy owner also passed
draining in-flight generation, swapping to a fresh KV cache, preserving
the old policy on installation failure, and rejecting stale policy
requests. The integrated revision `ff1fc5e833` completed both updates on
four CPU devices with queued generation and that policy owner in both
scoring modes. Each run decoded eight binary frames and verified the
`0 → 1 → 2` installed policy chain. The same revision passed the
full-vocabulary profile and fused forward diagnostic on four TPUv4 chips.
The three installed weight digests were distinct, the second group
used step 1, and the 128 sampled scores matched within `4.29e-6`. This
qualifies fresh local KV state across updates. The separate CPU publication
proof covers the process boundary. The later two-host fixture below
exercises publication across accelerator hosts.
[TPUv4 queue and cache proof][iris-native-install]

The integrated proof archive is
`gs://marin-us-central2/users/dlwh/quick-loop-rl/native-proof/2026.11.20.03/outputs.tar.zst`.
Its SHA-256 is `dfe5aadfd44f1d3c3111474cb5a8a528d0b55078df9b166b30dd8727c8e5d1c1`.
[Integrated proof archive in Filer][native-install-archive]

The follow-up at `a5cfd7cbcc` ran the same four-TPUv4 fixture on the
combined local source and used fused scoring for both actual optimizer
updates. Each group drew four independently seeded answers. The second
group sampled from the first update's verified weights, and both updates
changed the weight digest. An independent decode verified eight binary
frames and all 128 sampled tokens. The maximum absolute raw-score
difference was `4.29e-6`.
[Fused TPUv4 run][iris-native-fused] · [Fused proof archive in Filer][native-fused-archive]

The fused proof archive is
`gs://marin-us-central2/users/dlwh/quick-loop-rl/native-proof/2026.11.20.05/outputs.tar.zst`.
Its SHA-256 is `ef8261d13ab4e4a41a4860b24cffc83dd2a08ea519b41e4b4bc4dcfb2eb11136`.

The maintained learner and serving runtimes ran in six serial subprocesses
on four CPU devices at `ca21258b3f`. Fresh learners restored exact state at
steps 0 and 1 and then updated. Three verified mixed-dtype weight snapshots
bound the generated binary frames to their serving policy. On one TPUv4
host, the learner published step 0 and the separate serving process loaded
it. First prefill failed before an optimizer update because Mosaic rejected
FP32 contraction precision for the BF16 grouped expert matmul operands.
[TPUv4 runtime attempt][iris-native-runtime-first]

A TPU-only grouped-dot precision change at `7c72cffd0d` let the separate
serving worker generate four binary trajectories under step 0. The archive
contains 64 raw scores across four independently seeded requests. A fresh
learner restored step 0 exactly, then hit the BF16 contraction restriction
in Splash Attention during replay scoring. The optimizer had applied no
updates when the run exited. [TPUv4 grouped-dot retry][iris-native-runtime-precision]

The separate-process proof at the same revision used compiled reference
attention for the BF16 learner at 256 tokens. Six serial processes shared
four TPUv4 chips. Two fresh learners restored exact state at steps 0 and 1
and then applied real optimizer updates. Serving installed the verified
weights for policies 0, 1, and 2, rejected the previous identity after
each swap, and generated from policy 2. The eight binary replay frames
covered 128 sampled tokens. Serving and learner replay raw scores differed
by at most `5.88e-4`, with no PPO clip crossings.
[TPUv4 separate-process proof][iris-native-runtime-reference] · [Proof archive in Filer][native-runtime-reference-archive]

The proof archive is
`gs://marin-us-central2/users/dlwh/quick-loop-rl/native-proof/2026.11.21.08/outputs.tar.zst`.
Its SHA-256 is `d1f390022889410a25bba5a14972a290ec1c8e0b4e8b0f117bacf2c8fd774a42`.

A follow-up at `7c72cffd0d` used BF16 Splash Attention with default matmul
precision in the learner processes and `highest` in serving. It passed both restores and
updates, rejected stale identities after each install, and generated from
policy 2. The eight replay frames covered 128 sampled tokens. The maximum
absolute raw-score difference was `6.60e-4`, with no PPO clip crossings.
This run used the original pickle command pipe.
[BF16 Splash Attention run][iris-native-runtime-splash] · [Proof archive in Filer][native-runtime-splash-archive]

The Splash proof archive is
`gs://marin-us-central2/users/dlwh/quick-loop-rl/native-proof/2026.11.21.09/outputs.tar.zst`.
Its SHA-256 is `cbe4ae635fc9c620f7788d07e4338fbfd0e7609b4f73ae3f0805d99cae250cf9`.

The 256-token run at `40ce66daf4` repeated both restores and updates on
one four-chip TPUv4 host using the maintained protobuf serving client.
All eight training frames and the final generation frame matched the
earlier Splash run byte for byte. All three policy IDs also matched.
The final worker answered an evaluation request under policy 2 with seed
`20261020`, temperature `0.6` and top-p `0.95`. The captured reply held
sampled tokens, transformed logprobs and an empty replay body. The source bundle applies atop
`2bf3f721c9`.
[TPUv4 protobuf run][iris-native-runtime-protobuf] · [Verified output archive][native-runtime-protobuf-archive] · [Source bundle, harness and verifiers][native-runtime-protobuf-source]

Iris schedules single-host v4-8 replicas independently. In the two-host
TPU attempt, the learner reached the regional GCS rendezvous and timed
out while the serving replica was still pending. A follow-up used two
distinct CPU workers in `us-central2`. The learner published
three immutable BF16 TensorStore snapshots to regional GCS across two
optimizer updates. One serving process installed all three versions,
rejected stale requests and returned four evaluation tokens under policy 2.
An independent audit verified 128 scored targets, zero clip-boundary
crossings and eight byte-identical binary replay frames across the two
worker archives.
[Two-host TPU attempt][iris-native-cross-host] · [Two-host CPU run][iris-native-cross-host-cpu] ·
[Learner archive][native-cross-host-learner] · [Serving archive][native-cross-host-serving] · [Harness and verifier][native-cross-host-source]

The follow-up on `63fb947665` used separate four-chip TPUv4 hosts for
the learner and serving roles. Each role confirmed the peer's first attempt
before initializing its accelerator. Over regional GCS, the typed learner
published BF16 policies 0, 1 and 2 across two optimizer updates. Serving
acknowledged the exact installed identities, rejected stale requests and
sampled four evaluation tokens under policy 2. The two hosts agreed on all
eight binary replay frames. The fused scorer checked 128 targets with zero
clip crossings. This qualifies the handoff for the small 256-token fixture.
The production run still needs the four learner hosts, pinned 67B model and
a cross-host tool turn.
[Verified two-host TPU handoff][iris-native-cross-host-typed-tpu]

The maintained learner and native ports share the SPMD policy scorer and
learner commit `d52fd47db5`. A local merge at `fe0bb85486` preserves the
learner's explicit positions and attention precision alongside the native
paged and replay paths. On four CPU devices its two scoring profiles
produced the same tokens, scores and installed weight digests as the native
parent. The capture format still requires contiguous valid request positions.

The bounded gate uses the supported four-row, 256-token TPUv4 layout, a
single-turn task and a deterministic grader with mixed rewards within a
group. It submits four
single-choice requests with distinct seeds under the same prompt ID.
Use the existing RLOO-N calculation and Levanter `PolicyLearner` through
`PolicyBatch`. Assemble a batch whose route tensors can be sliced and sharded
with its tokens. The SkyRL bridge at `b8aa9e2af1` rejects
`routed_experts`. The Snowball replay API at `fcc3167fef` requires default
contiguous positions. Use the qualified layout for the first gate. Then
verify explicit positions, left-padded prompts and right-padded responses
at inventory shapes before connecting the inventory batch builder.

Check matching routes and raw logprobs before the first update under identical
weights. Check finite loss and changed parameters. Publish the update, reset
serving state and confirm the next nonempty rollout used the new weights.
Complete a second graded optimizer update and publish again. Save the batch,
policy identities and receipts so the bounded acceptance can be replayed
without reconstructing sampled text. [Batch bridge][batch-bridge] ·
[Engine capture API][engine-replay]

### 3. Qualify native serving for the inventory recipe

The regional HF export index lists 39 shards and 134.16 GB of tensor
bytes. Its config has 26 layers, 20 query heads, five KV heads, 256
experts and four selected experts per token. On eight devices the head
counts select tensor parallelism of one. A JAX trace of the full model
shape confirmed that the default layout shards each expert weight on its
hidden dimension. The explicit expert-parallel layout assigns 32 whole
experts per device. It estimates 18.3 GiB of BF16 parameter storage per
device and 1.52 GiB of replicated KV per full-context slot at 24,576
tokens. Eight full slots therefore need about 12.2 GiB for one cache and
24.4 GiB while two caches overlap during an install. These figures
exclude live activations, compile scratch and overlapping weights. The
trace used checkpoint metadata and abstract arrays.

The loader at `59398f200f` accepts
`LevanterEngineConfig(expert_parallel_size=8)`. A local HF export/load
round trip exercised the serving loader and verified identical BF16
expert bytes on the eight-device mesh. A small model with 20 query heads,
five KV heads and four selected experts out of 16 completed two paged
rollout and fused learner updates across eight CPU devices. The largest
raw score difference was `1.55e-5` within the original relative and
absolute tolerances. The current explicit layout needs the replay batch
and serving slot dimensions aligned to the combined token-sharding width.
Four configured slots failed that shape check. Eight slots and two groups
of four passed.

Use an eight-H100 expert-sharded host as the first candidate for a
same-hardware comparison. Start with the minimum aligned serving slots
and measure actual HBM while loading, generating and installing weights.
The current prefill scheduler admits whole prompts. Size its token pack
for the longest accepted prompt in the first full-context smoke and
measure prefill memory and latency on that shape.
The current BF16 paged attention selects the reference implementation on
CUDA and the TPU path also falls back for BF16. Benchmark that path at the
current context and compare with a qualified faster kernel before
concluding that native serving improves time per rollout group.

The local qualification command at `4fbb38e286` uses the regional 67B
export, its completion manifest, the pinned tokenizer revision and the
maintained inventory precision settings. This revision connects the
per-turn replay binder to the probe and places logical `data=4` across
the typed learner's four CUDA hosts while keeping `expert=8` host-local.
Thirty mesh, typed worker and checkpoint regressions passed on JAX `0.11.1`.
Its inventory case uses four
independently seeded samples with a 16,384-token prompt and an 8,192-token
response each. The engine keeps the inventory's 24,576-token prefill
capacity. The long-prompt case uses 24,544 prompt tokens and 32 generated
tokens for a shorter diagnostic.

The probe checks the regional completion manifest, config and shard index
against their recorded hashes. Safetensors headers from the pinned export
show BF16 storage for all 26 router weights and all 26 QB biases. Record
those storage dtypes and apply the inventory learner's exact promotion to
FP32 for router and QB compute. Check the loaded parameter count and the
actual compute dtypes.
It records the weight manifest identity separately from the tokenizer
revision. The paged generator uses its serving precision, and the FA4
learner forward uses the learner's launch precision and fused projection.
The staged inventory probe at `97d63da41a` checks that each frame matches
its request ID, sampling seed and exact prompt and finishes at the full
context length.
It checks the serving and learner score shapes before subtracting logprobs.
The gate requires all 32,768 real targets and zero PPO clip crossings.

The receipt records resolved inputs and loaded geometry before starting
generation. It captures per-device allocator reports at each phase and
separate first-invocation timings for KV allocation, generation, frame
encoding, replay binding and fused scoring. The timing includes any
compilation or cache reuse during that invocation. A two-device EP2 CPU
control preserved FP32 router and QB values from a mixed-precision export,
rejected rounded BF16 routing weights and mismatched metadata, and scored
all targets for two prompt/response splits. The pinned 67B accelerator
result remains open.

The first eight-H100 attempt verified the pinned weight metadata and
stopped during model loading when XLA's per-fusion autotuning cache used
an S3 path. A retry used a worker-local cache and progressed through
deserialization. It stopped at the probe's original FP32 storage assertion.
[Initial 67B attempt][iris-native-67b-first] ·
[Worker-local cache retry][iris-native-67b-local-cache]

The precision-corrected attempt loaded all 67,078,882,816 parameters in
129.65 seconds and verified FP32 router and QB compute weights. Its first
prefill requested a 44.01 GiB allocation and failed with a 59.39 GiB
allocator limit per H100. A retry with the inventory serving allocator
fraction of 90% measured a 71.26 GiB limit and 24.66 GiB resident per GPU
before prefill. The same allocation still failed. The reference attention
now has a staged version that processes queries in tiles while keeping the
full prompt capacity and each query's page order. CPU compilation reported
30.94 GiB of temporary memory for the original attention function and
1.97 GiB with query tiling at the same prefill shape. That measures the
individual CPU executable. Generation and fused scoring at the full shape
remain open on H100.
[67B model-load and prefill attempt][iris-native-67b-router] ·
[90% allocator attempt][iris-native-67b-cap90]

The typed serving and learner pipes passed at 8,192 and 24,576 tokens on
the 524K-parameter fixture on one four-chip TPUv4 host. Each run used
four distinct prompts that filled the requested context with 32 generated
tokens per prompt. An independent decoder checked all token IDs, positions,
per-layer routes, QB thresholds and the installed checkpoint identity.
The fused learner scored all 128 targets in each run with zero clip
crossings and rejected a deliberately misaligned position.

| Context tokens | Replay bytes across four requests | Maximum absolute logprob difference | Generation RPC including cold compilation | Typed fused forward |
| --- | --- | --- | --- | --- |
| 8,192 | 854,256 | `6.09e-4` | 44.25 s | 6.19 s |
| 24,576 | 2,558,232 | `6.11e-4` | 73.73 s | 18.58 s |

The 24,576-token test used 24,544 prompt tokens, a 32-token response and
local model publication between serial processes. Inventory qualification still
requires the pinned 67B weights, production response budget, accelerator
handoff and FA4 scorer.
[8K TPU context probe][iris-native-8k-context] ·
[24K TPU context probe][iris-native-24k-context]

On the eight-H100 regional worker, run the probe with the locked `pipeline`
environment used for the isolated CUDA learner, including JAX and JAXLIB
`0.11.0` and FA4. The staged job assumes authorized CoreWeave storage access
to the pinned checkpoint and the Iris output bucket. Check free eight-GPU
hosts as well as regional quota before reserving a host for this probe.

```bash
JAX_PLATFORMS=cuda JAX_DEFAULT_MATMUL_PRECISION=default \
  uv run --no-sync python -m experiments.post_training.snowball_native_qualification \
  --case inventory "$IRIS_OUTPUT_DIR/snowball-native-67b-inventory.json"
```

- Connect the native backend to the RolloutEngine operations used by the
  recipe. Use SkyRL's exact-token chat continuation helper to retain the
  first assistant turn's served tokens and append only the rendered boundary
  and tool suffix. A local probe at `1c34b22a42` sampled an arithmetic tool
  call from a tiny Snowball model, dispatched the real tool through
  RolloutEngine and fed its observation into a second model turn. On one
  and four CPU devices, learner replay matched both turns' raw scores and
  gave every added tool or boundary token zero objective gradient. Both
  frames used the same acknowledged installed policy.

  The two-host TPUv4 probe at `63fb947665` executed the tool and captured
  both binary replies under policy 0. All 24 overlapping tokens and
  positions matched, and all 48 overlapping expert assignments matched.
  The first turn's terminal token was captured through decode. The second
  turn recomputed it during prefill. Two QB threshold values differed at
  that shared token, with a maximum absolute difference of `1.91e-4`.
  The exact overlap assertion stopped the probe before learner scoring.
  The typed learner on CPU then rejected the same archived frames at
  the production overlap check, using the same weights and model binding.
  With the TPU Splash kernel in CPU interpret mode, the typed learner
  scored each archived turn independently. The two fused scores were within
  `1.58e-4` of the corresponding captured raw scores.
  [Cross-host tool-turn probe][iris-native-tool-turn-tpu]

  The typed learner replays each turn with its own causal prefix and exact
  captured expert routes and retains each frame's QB thresholds. It selects
  the predictor for each action on the original response axis, then runs one
  vocabulary projection and applies the original batch's RLOO-N advantages
  and normalized objective weights. The production loss gives identical
  values and gradients for the assembled scores and the sum of globally
  weighted turn contributions, with zero direct objective gradient at all
  nine tool tokens. The integrated forward on CPU, with Splash in interpret
  mode, matched the archived turns' independent fused scores bit for bit.
  With reference attention, the learner scored a mixed-turn batch across two
  microbatches and published weights from a real optimizer step. A separate
  CPU control on `e58727f2b0` completed two real SkyRL tool episodes and
  updates. A fresh learner restored step 1 exactly. One serving process
  installed v0 through v2, rejected stale requests after both updates and
  generated and evaluated under v2. The two-host TPUv4 replay reached the
  learner host, then timed out scheduling the serving host before the
  cross-host proof began. An atomic retry on the same source timed out
  with both replicas still unassigned. That gang requires two compatible
  ready workers in the same zone. The deployed autoscaler routes a gang
  demand to a single matching slice and cannot provision two independent
  one-VM slices for it. Production-context memory remains to be qualified.
  [TPUv4 tool-turn restart run][iris-native-tool-restart-tpu] ·
  [TPUv4 gang retry][iris-native-tool-gang-tpu]

  On one four-chip TPUv4 host at `4fbb38e286`, fresh learner and serving
  processes took turns using the slice. The first optimizer update published
  policy 1, and the second tool episode sampled that installed policy. The
  learner reported a second update and then the fixture rejected the loss
  `-0.9999975` for differing from `-1.0` by more than `1e-6`. The attempt
  ended before policy 2 publication and final generation.
  [TPUv4 serial tool-turn attempt][iris-native-tool-serial-tpu]

  The retry recomputed the clipped loss from frozen learner scores and
  normalized action weights with an FP32 roundoff bound. It passed two
  optimizer updates and four binary replay frames across six fresh serial
  processes on the same four TPUv4 chips. Splash learner scores differed
  from captured raw scores by at most `2.31e-4` across four action targets,
  with zero PPO clip-boundary crossings. A fresh learner restored model,
  optimizer, RNG, step and averaging fingerprints exactly before step 2.
  Serving rejected stale requests after each install, then generated and
  evaluated a fixture sample under policy 2. The fixture uses a 524,296
  parameter model and exercises sequential use of the TPU slice.
  [Successful TPUv4 serial tool-turn retry][iris-native-tool-serial-retry]

  The same source then passed on distinct TPUv4 learner and serving hosts
  in `us-central2-b`. Iris scheduled each replica independently. The proof
  began after both attempts exchanged readiness receipts. One live
  serving process installed policies 0 through 2 and rejected requests
  for the previous policy after each update. Each optimizer update used
  two tool-turn frames that matched the serving output byte for byte.
  Each turn retained its own router thresholds, and the shared prefix had
  identical expert choices. The maximum raw score difference was
  `2.31e-4` with zero PPO clip crossings. A fresh learner restored the
  exact step-1 state before training on the second episode. Changing a
  single tool observation token caused the learner to reject replay.
  The audit tied these checks and the final policy-2 generation to both
  original Iris attempts. [Cross-host TPUv4 tool-turn proof][iris-native-tool-split-tpu]
- The local `NativeReplayChat` adapter at `59dd3a0277` sends the exact
  prompt IDs from SkyRL's chat client and binds the returned binary frame to
  the controller's expected policy identity. The turn records carry those
  frames to trainer rank batches. The follow-up at `732ff5cdca` exercised
  the standard SkyRL task worker with a local tool provider and preserved
  the frame bytes through assembly. The handoff at `470c0a0034` validates
  admitted sequences, positions, masks and per-layer routes in the isolated
  JAX worker. It scores and updates with the same captured choices, and
  keeps raw old-policy scores separate from sampler probabilities. The
  real worker replay test passed on one and four CPU devices with
  JAX/JAXLIB 0.11.0.
- The controller at `658ee7e085` and `d4dcd378f5` opens lease `n+1`
  after the learners agree on weight version `n` and serving acknowledges
  its digest. A local Ray run installed two distinct snapshots and
  exercised an offline tool call in each lease. The standard task worker
  carried exact binary frames into ranked learner inputs with zero
  staleness. The serving process rejected the first lease after installing
  the second snapshot. That controller test used a deterministic weight
  perturbation. The learner and IPC tests separately exercised real
  optimizer changes.
- Use the two-update inventory launcher to verify the cross-host
  acknowledgement from the four learner hosts to the serving host. Check
  exact tool tokens and infrastructure failure masks on the same deployed
  path. Exercise continuation across a tool call with a separate two-turn
  probe because the inventory preset uses a one-turn budget. Restart the
  learner and serving processes from a durable checkpoint and verify that
  the next rollout uses the restored policy.
  [Exact chat continuation][chat-continuation]

  The two-update preset currently commits the SkyRL recovery checkpoint at
  train end. Its earlier worker roundtrip covers learner state alone.
  In a CPU reproduction, the real trainer committed step 2. A staged
  step-3 folder with one TensorStore file blocked resume before any
  learner loaded weights. Moving the staged folder outside the checkpoint
  root let a fresh learner restore the exact state and prompt cursor.
  [Callback recovery reproduction][native-callback-recovery-source]

  The maintained Levanter runtime at `c016fc3ae5` writes each save to a
  unique generation. It conditionally advances the index after all learner
  ranks agree on the path and version and the trainer and rollout files
  are durable.
  Resume and artifact lookup use the indexed path. Retention updates the
  index before deleting removed generations and checks the attempt owner
  before terminal cleanup.

  A four-device CPU test failed the step-2 rollout-state write after
  committing step 1. Fresh learner and serving processes restored the
  saved prompt cursor and installed policy 1. The four live replay frames
  matched the uninterrupted run after normalizing request IDs. Scores,
  optimizer, PRNG, model state and policy 2 also matched. The recovered
  trainer committed step 2 to a new generation and left the abandoned
  generation intact. The test used
  local TensorStore with in-process actor dispatch. Separate metadata
  probes passed stale-writer rejection and exact retry acknowledgement
  on CoreWeave S3 and GCS.
  [Durable CPU recovery source and tests][native-durable-recovery-source]

  The same recovery gate on `63fb947665` passed on one four-chip TPUv4
  host with the typed learner and serving pipes. Ten serial worker
  processes used the 256-token fixture with BF16 and Splash Attention
  in the learner. An injected step-2 write failure left step 1 committed.
  Fresh processes restored the exact prompt cursor and learner state,
  reran 64 scored targets and reproduced the uninterrupted step-2 model,
  Adam, PRNG, step and averaging fingerprints. The four live replay frames
  matched after normalizing request IDs. The retry committed a new
  generation, and the final server returned 16 tokens under policy 2.
  The test used local TensorStore and in-process actor dispatch. The
  four-learner-host handoff and restart on the pinned checkpoint remain
  deployment gates.
  [TPUv4 restart run][iris-native-tpu-recovery] ·
  [Verified frames and receipt][native-tpu-recovery-results] ·
  [Submitted harness and independent verifier][native-tpu-recovery-source]
- During native training, fixed-panel evaluation uses a separate binary
  sampling call bound to the installed weight digest. A local isolated
  CPU test matched an independently configured local sampler's token IDs and
  transformed logprobs at temperature 0.6, top-p 0.95 and seed 20261020.
  This call omits training replay frames. First check evaluation on the same
  pinned checkpoint on H100. Then run the fixed panel at the same optimizer
  steps using each arm's own acknowledged policy.
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
  reward. Both two-update plans at `4fbb38e286` resolved their regional
  inputs for artifact version `2026.11.21.2`. The primary fixed panel is
  `eval-lite@2026.11.17.1`. The published 100-task Parquet has SHA-256
  `557733f6d4f7f42c5c5072a29325da03d489a5fd77b456054bc34003b2f16e95`.
  Require all four attempt-scoped panels at steps 0 and 2, with each task
  ID, prompt token sequence and score reconciled against the successful
  attempts. Collect logs using the successful rank-zero attempt from each
  exact Iris coordinator. Match the receipts and panel paths to those
  attempts, and recheck the attempt identities after collection. Compare
  step 2 separately because step 1 includes the learner
  subprocess checkpoint round trip. Both maintained arms log
  `timing/rollout_admission_window_seconds`
  and `rollout/admission_window_groups` for fresh, full batches at zero
  staleness. The buffer's clock measures from policy lease opening through
  final group admission. Divide the sum of elapsed seconds by the sum of
  admitted groups across the same measured updates. Report the trainer's
  `timing/wait_for_generation_buffer` separately as buffer wait, along
  with `timing/step` and `timing/sync_weights`. Use trajectory timings
  to distinguish generation, tool execution and grading.

The prior FP8 comparison reported a 13% reduction in mean decode time
and 5.7 minutes longer to reach a mean verifier score of 0.2901, the
BF16 arm's final score. The archived trainer events reproduce 37.0
minutes of buffer wait for BF16 and 43.1 for FP8 through update 15. The
initial and update-15 dumps contain the same 100 task IDs across both
arms. [Prior timing comparison][timing]

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
[iris-smoke]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.5-coordinator-2ea612c2%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.5-a8ae164919e7
[frozen-validation-uid]: https://github.com/marin-community/marin/blob/6936528c55df0212ec8ccd1fa619361863b4c420/vendor/skyrl/skyrl-train/skyrl_train/dataset/dataset.py#L62-L105
[h100-matched-audit]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-matched-audit%2F2026.11.21.1%2Fh100-matched-audit.tar.zst
[frozen-validation-panel]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Frl-data%2Ftask-selection%2Fsnowball-100k%2F2026.11.04%2Fvalidation.parquet
[frozen-receiver-numbering]: https://github.com/marin-community/marin/blob/6936528c55df0212ec8ccd1fa619361863b4c420/vendor/skyrl/skyrl-train/skyrl_train/weight_sync/expert_block/schedule.py#L117-L118
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
[chat-continuation]: https://github.com/marin-community/marin/blob/2bf3f721c9946150a02c3417c4c93e14b273a5e8/vendor/skyrl/skyrl-train/skyrl_train/inference_engines/chat_continuation.py#L28
[rloo]: https://github.com/marin-community/marin/blob/6936528c55df0212ec8ccd1fa619361863b4c420/vendor/skyrl/skyrl-train/skyrl_train/utils/advantage_estimators.py#L176-L289
[preset-migration]: https://github.com/marin-community/marin/pull/10740
[iris-native]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-binary-proof-20261120-01
[iris-native-rerun]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-binary-proof-20261120-02
[iris-native-install]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-binary-proof-20261120-03
[iris-native-fused]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-fused-tpu-20261120-05
[iris-native-runtime-first]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-sequential-tpu-20261120-06
[iris-native-runtime-precision]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-sequential-tpu-20261121-07
[iris-native-runtime-reference]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-reference-tpu-20261121-08
[iris-native-runtime-splash]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-splash-tpu-20261121-09
[iris-native-runtime-protobuf]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-protobuf-tpu-20261121-11
[iris-native-cross-host]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-crosshost-tpu-20261121-12
[iris-native-tool-turn-tpu]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-tool-turn-tpu-20261121-26
[iris-native-tool-restart-tpu]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-tool-restart-tpu-20261121-27
[iris-native-tool-gang-tpu]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-tool-gang-tpu-20261121-28
[iris-native-tool-serial-tpu]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-tool-serial-tpu-20261121-29
[iris-native-tool-serial-retry]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-tool-serial-tpu-20261121-30
[iris-native-tool-split-tpu]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-tool-split-tpu-20261121-33
[iris-native-67b-first]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-67b-inventory-h100-20261121
[iris-native-67b-local-cache]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-67b-inventory-h100-20261121-local-cache
[iris-native-67b-router]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-67b-inventory-h100-20261121-router
[iris-native-67b-cap90]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-67b-inventory-h100-20261121-cap90
[iris-native-cross-host-typed-tpu]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-crosshost-tpu-20261121-25
[iris-native-cross-host-cpu]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-crosshost-cpu-20261121-13
[native-cross-host-learner]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.13%2Flearner.tar.zst
[native-cross-host-serving]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.13%2Fserving.tar.zst
[native-cross-host-source]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.13%2Fsource-inputs.tar.zst
[native-install-retry-source]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.14%2Finstall-retry-source-and-tests.tar.zst
[native-callback-recovery-source]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.15%2Fcallback-recovery-reproduction.tar.zst
[native-durable-recovery-source]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.16%2Fdurable-checkpoint-recovery.tar.zst
[typed-learner-source]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.17%2Ftyped-learner-wire.tar.zst
[iris-native-tpu-recovery]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-recovery-tpu-20261121-21
[native-tpu-recovery-results]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.21%2Foutputs.tar.zst
[native-tpu-recovery-source]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.21%2Ftpu-recovery-source.tar.zst
[iris-native-8k-context]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-8k-tpu-20261121-22
[iris-native-24k-context]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-24k-tpu-20261121-23
[native-archive]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.20.02%2Foutputs.tar.zst
[native-install-archive]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.20.03%2Foutputs.tar.zst
[native-fused-archive]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.20.05%2Foutputs.tar.zst
[native-runtime-reference-archive]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.08%2Foutputs.tar.zst
[native-runtime-splash-archive]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.09%2Foutputs.tar.zst
[native-runtime-protobuf-archive]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.11%2Foutputs.tar.zst
[native-runtime-protobuf-source]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.11%2Fsource-inputs.tar.zst
[native-serving-wire-archive]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-proof%2F2026.11.21.10%2Freceipts.tar.zst
