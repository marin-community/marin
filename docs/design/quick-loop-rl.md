# Levanter quick-loop RL

Draft for review.

Move the learner in the Snowball inventory quick loop from Megatron to
Levanter with SkyRL and vLLM in place. The inventory quick loop trains the
67B-A2B Snowball checkpoint on verifiable tasks from the Snowball 100k task
selection. Once that combination passes the full-model smoke and the matched
comparison, qualify Levanter's Snowball engine as the full-model rollout
backend. The learner jobs have first claim on H100 capacity. Run bounded
native-loop checks on available capacity while the learner jobs wait. Use a
typed binary protocol for rollout replay and isolated worker commands.

## Status

| Milestone | Evidence | Next gate |
| --- | --- | --- |
| Megatron learner with vLLM | The frozen control completed 25 updates on 40 H100s. It passed 208 receiver checks, committed step 25 and scored 0.1902 to 0.2700 on the fixed panel. | Use the frozen control to evaluate changes to the maintained learner. |
| Levanter learner with vLLM | The frozen 25-update arm scored 0.2310 to 0.3201 on 40 H100s. Its median warm update took 3.0 times the Megatron control’s wall time. The maintained two-update H100 arm also passed exact restore and all 24 receiver checks. | Profile maintained publication and run a longer comparison before changing the learner default. |
| Learner port based on `main` | The port covers SPMD scoring, Snowball precision and positions, the SkyRL adapter, and synchronous BATCH presets. Four deployed learner hosts applied two updates, including exact continuation after restart, with vLLM serving. | Profile warm updates and publication on the maintained Levanter/vLLM path. |
| 40-H100 learner smoke | Passed on the `2026.11.20.20` replacement. Both updates, exact restore on four fresh learners, all 24 receiver receipts, stamped generation under the trained policies, terminal checkpoint, and independent changed-weight sample passed on the same child. | — |
| Native Snowball capture and replay | A fresh learner reproduced all 555 sampled raw logprobs on the two-layer, four-TPUv4 fixture. On eight H100s, one pinned 67B diagnostic passed weight swap, a real tool turn, and all 32,768 full-context fused scorer targets. [Passing diagnostic][native-passing-diagnostic-evidence] | Repeat the 67B diagnostic before promoting native serving. Earlier diagnostics crossed the scorer's ratio bounds. |
| Native online loop based on `main` | On two four-chip TPUv4 hosts, the small BF16 fixture passed two tool episodes and optimizer updates, byte-exact binary replay, fresh learner restore and final generation under policy 2. One serving process handled all three policies. | Qualify the current port with the full 67B inventory workload. |
| Bounded 40-H100 backend pilot | Both arms completed two 32-group updates and two prompt-identical panels with a 512-token reply cap. Native took 35.73 lease seconds per admitted group, compared with 0.331 for vLLM. Policy-2 mean verifier scores were 0.08 and 0.20, respectively. | Keep vLLM serving for the quick loop. Test the optimized AgentX serving path on fixed RL bursts and tool-use prompts. |
| Typed binary protocol | Native serving and the maintained isolated learner use protobuf headers with detached binary payloads. The maintained learner passed exact recovery on TPUv4 and across four 67B learner hosts on H100. | Evaluate carrying the typed batch envelope across Ray. |

[Quick-loop branch status][learner-status] · [H100 readiness retry coordinator][iris-smoke-readiness] ·
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

A cold continuation can recompute different expert routes on shared tokens.
Retain each turn's complete captured prefix, including its routes and router
thresholds. Assign every sampled target to the turn that produced it. Tool
observations remain visible to later turns and carry zero objective weight.
Reuse a cached prefix only with the KV, routes and thresholds from that
execution and the same acknowledged policy.

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

Commit the trainer and rollout context with the learner at each completed
step in a recovery validation run. The context includes the prompt cursor,
retries, in-flight prompts and buffered grades. A learner validation
checkpoint supplies the state for subprocess recovery. Scheduler recovery
also needs the ordinary step index and committed coordinator context.

Assign each logical run attempt a stable GPU child ID before scheduling.
Persist that ID with the source bundle and resolved launch digest. A retried
CPU coordinator looks up the exact child and resumes observation while it
runs. A successful child supplies the same immutable terminal record for
validation and artifact completion. A failed child can start a new attempt
from the last committed checkpoint with a new recorded identity. The
coordinator uses Iris child preservation for scheduler preemption and
leaves the GPU child running when its foreground observer exits. Explicit
job cancellation still terminates the run.

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
an explicit error. Use protobuf for the native serving and isolated learner
subprocess contracts. Ray carries actor calls and rank batches, preserving
the replay frame bytes. Human-readable config, logs and archival receipts can
remain JSON. Keep vLLM HTTP, NCCL expert-block publication and TensorStore
snapshots on their existing transports.

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
preserving exact binary replay across tool turns. The small cross-host
TPUv4 loop also exercised both tool turns. The pinned 67B deployment still
needs to pass the same gate.

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

The frozen H100 serving actor groups up to four training requests on the
installed policy. The child bounds the batch by KV capacity and the existing 65 MiB
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

All five hosts joined in `2026.11.20.8` on `dba95c7f84`. vLLM captured CUDA
graphs and all four JAX processes advanced past mesh setup to checkpoint
loading. The learner on GPU task 4 was sampled in host BF16-to-FP32 widening
before the 1,800-second initialization deadline fired. Fresh synthetic casts
in that pod completed promptly. The run ended with zero optimizer updates.
[Mesh setup failure][iris-smoke] · [Initialization timeout][iris-smoke-rerun] ·
[Verified failure evidence][h100-init-failure]

Iris accepted `2026.11.20.9` at 12:44 p.m. PST on November 21 on exact
source `988a376b05`. The loader widens only the distinct local slices when
they cover part of a BF16 tensor. Two CPU processes verified exact FP32 bytes
for all 65,536 BF16 encodings across the tested sharding layouts. The retry
uses a 3,300-second initialization deadline within the driver's one-hour
setup deadline. The model, workload, topology and acceptance checks match
the earlier smoke. It requests four eight-H100 learner hosts and one
eight-H100 vLLM host in `cw-us-east-02a`, with four data shards across
learner hosts and eight expert-axis shards per learner host.
[Retry coordinator][iris-smoke-retry]

The last learner completed all 39 checkpoint shard reads. The following
`jit_load_from_state_dict` failed requesting 45.31 GiB on multiple H100s. XLA
reported that it replicated transposed expert tensors while moving from the
32-way shape-based staging layout to eight expert shards and four hidden
shards. The run ended before learner initialization or an optimizer update.
[Failed GPU child][iris-smoke-retry-child] · [Verified failure evidence][h100-layout-failure]

Iris accepted `2026.11.20.10` on source `0ecb355467`. The loader derives
source tensor shardings from the model's export transforms and uses them
when placing safetensors on the learner mesh. In a CPU compile of all 502
tensors on 32 logical devices using JAX 0.11.0, the new layout removed
92,904,574,212 bytes of compiler temporaries, 370 all-gathers and 52
all-to-alls. A small sharded checkpoint verified BF16-to-FP32 import and
weight parity. The frozen artifact preview matches `2026.11.20.9` after
normalizing versioned paths and identities. The stored launch configuration
matches after normalizing the actual version and attempt identities.
[Layout retry coordinator][iris-smoke-layout] · [Frozen source and compiler evidence][h100-layout-handoff]

All four learners loaded the production checkpoint and initialized in
`2026.11.20.10`. All eight vLLM receivers reported zero mismatches on
157,498,867,712 compared initializer bytes. Serving generated with policy
stamp 1 and finished the 100-task initial evaluation in 103 seconds. The
first batch admitted 128 rollout groups, 512 trajectories, and 1,646,883
response tokens. Levanter finished forward policy scoring before the first
`jit__train_step` requested 41.05 GiB and failed. No optimizer update
completed. Measure the training executable and its live buffers before
retrying the same batch geometry and objective.
[GPU child][iris-smoke-layout-child] · [Verified training failure][h100-train-failure]

Iris accepted `2026.11.20.11` on source `ebee07b144`. The learner keeps
the FP32 Adam moments in pinned host memory while the gradient executable
runs. After gradients finish, it restores those buffers to the GPU and
applies the same AdamW update in a separate executable. The initializer,
workload, serving engine and topology match `2026.11.20.10`. The optimizer
placement is the only recipe change. The clean source passed 17 tests,
including exact learner restore and continuation with one or two CPU
processes. An abstract CPU compile of the isolated 67B Adam executable
accounts for 46.51 GiB per device after aliasing. The run records CUDA
memory before each phase it reaches.
[Optimizer retry coordinator][iris-smoke-optimizer] · [Frozen source and audit tools][h100-optimizer-handoff]

Run `2026.11.20.11` verified startup, initial vLLM transfer, baseline
evaluation and scoring of the first 512 trajectories. Three learner ranks
reported 9.32–9.36 GiB in use per H100 with Adam in pinned host memory.
The fixed XLA projection enters a 200-second host agreement during gradient
tracing. Three ranks reached that agreement. Learner rank 2 never reached
it before the deadline. Its original log ends after forward scoring.
[Failed GPU child][iris-smoke-optimizer-child] · [Verified rendezvous failure][h100-readiness-failure]

Run `2026.11.20.12` used `f1ea01e18d`. It synchronizes the learner
ranks after batch placement within the existing command deadline, then
traces the same gradient executable. The frozen workload preview matches
`2026.11.20.11` after normalizing its run identity. On two CPU learner
processes, a seven-second delivery delay exceeded a test-only two-second
projection deadline on the previous source. The new source passed two
updates, exact fresh restore and byte-identical resumed continuation with
the same delay and deadline.
[Readiness retry coordinator][iris-smoke-readiness] · [Frozen source and timed reproducer][h100-readiness-handoff]

The four H100 learners passed readiness and the first gradient and optimizer
phases. The step-1 save began at 01:28:14 UTC on November 22. Rank 2 reached
the commit barrier at 01:31:13, followed by ranks 0 and 1. At 01:36:14 the
barrier hit JAX's 300-second default. Rank 3's archived log records entry
into the save and ends before local staging completion. The fresh learner
restart and version-1 vLLM publication were pending when the job exited.
[H100 attempt][iris-smoke-readiness-child] · [Original checkpoint failure evidence][h100-checkpoint-failure]

Iris accepted `2026.11.20.13` on `0a02a0e58`. The synchronous save now
uses the same 30-minute commit budget as Levanter's training checkpointer.
The save waits for all rank writes and the completion marker before
computing the saved-state fingerprint. The identical two-rank CPU control
with six seconds of skew timed out at the scaled two-second deadline and
restored exact model, Adam, and RNG state with the scaled twelve-second
deadline. The exact JAX 0.11.0 focused suite passed 15 tests. The 40-H100
configuration matches the previous smoke after normalizing version and
attempt identities.
[Checkpoint retry coordinator][iris-smoke-checkpoint] · [Frozen source and tests][h100-checkpoint-handoff]

The retry committed the step-1 checkpoint at 02:25:47 UTC on November 22,
16 minutes and 10 seconds after the save began. The last learner reached
the commit barrier 13 minutes and 16 seconds after the first. The step-1
metadata records policy version 1 and the pinned initializer. At 02:29:54
the parent reported a shutdown acknowledgement from the old learner, then
timed out waiting five seconds for exit. The failure occurred after the
save, state fingerprint and model-change check, before any fresh-process
restore or version-1 serving publication.
[Checkpoint retry attempt][iris-smoke-checkpoint-child] · [Original exit-timeout evidence][h100-shutdown-failure]

Iris accepted `2026.11.20.14` on `6bfb23794c`. The parent now uses one
monotonic deadline for both the shutdown acknowledgement and process exit.
It requires a zero exit code before starting a fresh learner. The delayed
exit reproduced the old timeout on CPU. All four focused tests on the
frozen JAX 0.11.0 source passed, including two-rank exact state recovery and
byte-identical continuation through the second update. The launch preview
matches the prior 40-H100 smoke after normalizing version and attempt IDs.
[Shutdown retry coordinator][iris-smoke-shutdown] · [Frozen source and tests][h100-shutdown-handoff]

Run `2026.11.20.14` accepted update 1 at 03:53:13 UTC on November 22,
after all four old learners exited with status zero and four fresh PIDs
restored model, Adam, RNG, step and averaging state byte-for-byte. At
03:56:01 vLLM published version 1. All eight receivers compared
157,498,867,712 bytes in total with zero mismatches for each of versions
0 and 1. The second rollout admitted 128 groups and 512 trajectories
with zero staleness. All 512 used policy version 1 and carried stamp 2.

All four restored ranks entered the second gradient phase. NCCL reported
CUDA out of memory on a 10 MiB allocation while splitting a communicator
at 03:59:21. XLA then waited for that subgroup at its rendezvous. The
1,800-second learner command deadline ended the job before update 2.
XLA reported a first-update peak of 51.14 GiB per H100 before Adam
execution. The main XLA pool cap was 71.26 GiB, leaving roughly 8 GiB
outside that cap. Physical free HBM at the failing split was unmeasured.
[Shutdown retry attempt][iris-smoke-shutdown-child] · [Verified restart and NCCL failure][h100-nccl-failure]

Run `2026.11.20.15` on `9e8664df3b` reduced the main Levanter XLA allocator
cap from 90% to 80%, or 63.35 GiB per H100. The typed recipe and launch
preview matched `.14` after normalizing the cap and run identities. All
four learners initialized from the pinned source. The eight vLLM receivers
passed the initial byte check, evaluation covered all 100 tasks, and the
driver admitted 512 trajectories. At 07:08:42 UTC on November 22, all eight
device allocators on learner rank 2 failed the same 32.21 GiB allocation
in the first gradient executable at sequence width 13,450. Iris marked the
attempt failed before any update. The original Ray and Kubernetes logs
lack failure-time bytes in use, bytes reserved and largest free block,
so they cannot distinguish pool exhaustion from fragmentation.
[NCCL headroom retry attempt][iris-smoke-nccl-headroom-child] · [Original failure evidence][h100-growing-pool-failure] · [Frozen source and checks][h100-nccl-handoff]

Iris accepted retry `2026.11.20.16` on `3f203da1db`. It keeps the 80% cap
and preallocates the CUDA learner pool.
The pinned XLA build uses one fixed address range for ordinary buffers
and XLA collective buffers in this mode. They allocate from opposite
ends and share the space between them. This tests whether providing one
range from startup accommodates the first gradient while retaining device
headroom for direct NCCL allocations after restore. The frozen launch
preview and live resolved configuration matched `.15` after normalizing
preallocation and run identities.
Fifteen CPU learner and recipe tests passed, including fresh-process
checkpoint continuation on JAX 0.11.0. The GPU acceptance criteria require
both phases in the same attempt.
[Preallocated learner attempt][iris-smoke-preallocated-child] · [Frozen source and checks][h100-preallocated-handoff] · [Pinned XLA allocator][xla-learner-allocator] · [Pinned XLA defaults][xla-learner-allocator-defaults]

Run `2026.11.20.16` loaded the pinned model on all four learner hosts.
All eight vLLM receivers compared 157,498,867,712 bytes for version 0
with zero mismatches. Initial evaluation completed 100 tasks. The trainer
admitted 128 groups and 512 trajectories with zero staleness. Learner rank 3
then exceeded the 1,800-second `model_fingerprint` deadline while staging
model shards in pageable host memory. The other three fingerprints had
finished. The readiness barrier kept all four learners out of the gradient
until the trainer shut down. This attempt leaves gradient memory at the
preallocated cap unmeasured.
[Preallocated learner attempt][iris-smoke-preallocated-child] · [Original failure evidence][h100-fingerprint-timeout]

Iris accepted `2026.11.20.17` on `5b8d1be3ab`. The fingerprint now hashes
the live disposable CPU staging view with the existing descriptors, order
and bytes. TensorStore still takes its own copy where a snapshot must
outlive staging. The exact fresh-process restore gate still hashes complete
model and optimizer state. On pinned JAX 0.11.0 the CPU fixture matched the
old digest across all 65,536 BF16 bit patterns, and the one- and two-rank
fresh-process restore tests passed. The frozen preview and live resolved
configuration matched `.16` after normalizing run identities. The GPU child
uses the same five H100 hosts as `.16`. The previously slow learner host
completed its first fingerprint in 85.256 seconds. All four ranks completed
the first gradient at sequence width 13,450, also the width of the failed
`.15` gradient. XLA reported a 50.82 GiB peak before Adam on the observed
ranks. The learner started writing the step-1 checkpoint at 08:54:10 UTC
and committed it across all four ranks at 09:01:39 UTC.

At 09:15:00 UTC all four fresh learners matched the saved model, Adam,
RNG, step and averaging fingerprints byte for byte. All four old processes
exited with status zero, and every rank's model bytes changed during update
1 without a layout change. The driver accepted the update and vLLM
published version 1 at 09:17:49 UTC. All eight receivers compared
157,498,867,712 bytes in total with zero mismatches. The second rollout
started under version 1.

The second rollout admitted 128 groups and 512 trajectories. The original
driver logs contained 19 independently checked nonempty generation receipts
with stamp 2. The fresh learners applied update 2 at 09:22:12 UTC, and
the reported peak before Adam was 51.35 GiB. On version-2 replay at
09:24:57 UTC, receiver 32 reported 1,035 mismatches out of 19,687,358,464
compared bytes. The driver had collected all eight reports and raised
before emitting any version-2 receipts. It kept generation paused, so
the attempt produced no version-2 evaluation or terminal checkpoint.
The other receiver outcomes and the affected tensors are unknown.
[Verified publication failure][h100-version-two-failure]

Iris accepted `2026.11.20.18` on `96f3d6ea0f`. Its preview matches
`.17` after normalizing the attempt identities. The sender checks each
payload's digest between installation and replay. A failure reports all
receiver totals, up to eight affected transfers per receiver, and any
source payloads whose digest changed. Normal publication still requires
zero mismatched bytes. Ten CPU transport and receipt tests passed,
including deliberate source and receiver byte changes.
[Replay diagnostic attempt][iris-smoke-replay-diagnostic] · [Frozen source and checks][h100-replay-diagnostic-handoff]

The live `.18` configuration also matches `.17` after normalizing run
identities. All eight receivers passed the version-0 check. The four
learners committed step 1 at 10:10:08 UTC and verified exact state in fresh
processes at 10:24:45 UTC. Each learner observed changed model bytes.
All eight receivers verified version 1 with zero mismatches. vLLM published
that version at 10:28:30 UTC and produced nonempty generation with stamp 2.
The original logs contain 21 verified generation receipts with that stamp.
The second rollout admitted 128 groups and 512 trajectories. The restored
learners applied update 2 at 10:32:53 UTC.

Version 2 failed verification at 10:36:33 UTC. Each of the eight vLLM
receivers reported the same 929 mismatched bytes out of 19,687,358,464.
Every mismatch belonged to one of seven BF16 attention K/V projections.
The 32 sender participants reported all 13,798 unique payloads. The same
seven immutable CPU payloads differed between exports. These are the input
bytes to the GPU copy and NCCL broadcast. The cause of the differing
export bytes remains open. The driver rejected the publication and kept
generation paused. The attempt failed before stamp-3 generation or a
terminal checkpoint.
[Verified diagnostic failure][h100-replay-diagnostic-failure]

Iris accepted `2026.11.20.19` on source `3e4c59c33c`. The frozen preview
matches the previous workload and retains the 80% preallocated learner pool.
With verification enabled, the sender writes the first export's raw
payloads to a temporary file on each learner host. It replays those bytes
over NCCL so every vLLM receiver can independently compare them with the
live weights before admitting generation. Each sender closes its file
when verification ends. The current schedule contains 134,191,857,664
unique source payload bytes across the four learner hosts.

Eleven CPU transport tests passed, including Gloo replay over two
versions, a staged-file fault, and a fault in the receiver's installed
weights. The CPU proof observed one learner export per version and
detected both faults. The check establishes equality between the exported
bytes and installed weights. The numerical investigation of the K/V
exports remains open.

On H100, the four original learner actors staged 13,798 source payloads
for version 0. Their 134,191,857,664 bytes match the prior schedule.
All eight original vLLM receivers compared 157,498,867,712 bytes total
with zero mismatches before the driver released version 0 at 11:09:30
UTC. Initial publication took 202.048 seconds, including 165.440 seconds
to install. The previous attempt took 254.647 seconds for its initial
publication on the same five physical hosts. The initial evaluation
completed all 100 tasks with a mean verifier score of 0.24.
All four learners completed the first gradient and committed the native
step-1 checkpoint at 11:21:39 UTC. The fresh processes first reloaded the
39-file HF initializer from the regional mirror, then loaded the native
state. At 11:35:07 UTC all four ranks verified exact model, Adam, RNG,
step and averaging fingerprints in their new processes. All four old
processes exited cleanly, and the first update changed model bytes on
every rank. The driver accepted update 1 and released the trained policy
at 11:38:06 UTC after all eight vLLM receivers compared the complete
installed weights with zero mismatches. Publication took 179.283 seconds.
The second rollout began under version 1 and produced nonempty generation
with stamp 2. It admitted 128 groups and 512 trajectories with zero
staleness. The restored learners applied update 2 at 11:42:33 UTC.

Version 2 was released at 11:45:19 UTC after all eight receivers
compared the full installed policy with zero mismatches. Publication
took 165.194 seconds. All three versions passed with 157,498,867,712
bytes compared per version. The step-2 native checkpoint committed at
11:48:23 UTC, and final evaluation generated with stamp 3 and completed
all 100 tasks. The mean verifier score was 0.26, compared with 0.24
at initialization. An independent regional read matched the committed
step-2 marker and both native manifests. The same 256-byte FP32 router
slice changed between checkpoints, within a 5 MiB chunk-read budget.

The job failed after evaluation while draining the completed checkpoint
upload. Trainer-state writes and the latest-checkpoint marker precede
the cleanup call. SkyRL asks the learner actor group for the Ray node
IDs it needs to locate each checkpoint cache. The Levanter host actor
lacked the `get_ray_node_id` RPC. The coordinator wrote no successful
terminal manifest.
[Immutable replay smoke][iris-smoke-canonical-replay] · [Frozen source and checks][h100-canonical-replay-handoff]

Run `2026.11.20.20` adds that RPC on source `82fd5cbeac`. A local Ray
probe reproduced the failure on the previous source. With the fix, the
same checkpoint commit and cleanup path finished using two real
Levanter actor handles, retained step 2, and pruned the older fixture
checkpoint. The native terminal-manifest test also passed. The live
retry's launch matches the previous workload after normalizing run
identities. Iris placed it on the same five H100 hosts with fresh
checkpoint roots. It staged the full publication on four learner hosts
and released version 0 at 12:13:18 UTC after all eight receivers matched
the installed bytes. Initial publication took 200.626 seconds. The
first rollout admitted 128 groups and 512 trajectories. The CPU
coordinator was preempted at 12:23:21 UTC during the step-1 checkpoint
write. Iris stopped all five child tasks, then retried the coordinator.
The learner processes received SIGTERM before the driver reported Ray
socket closure.

The coordinator submitted a new five-host GPU child at 12:27:31 UTC.
Its live resolved configuration matches the same frozen workload.
Regional checks before its startup found neither the latest-step marker
nor the step-1 native completion metadata. Its driver confirmed a start
at step 0. All eight receivers compared the full version-0 policy with
zero mismatches, and the driver released it at 12:41:41 UTC. Initial
publication took 197.883 seconds. The initial panel scored 0.30 across
100 tasks, and the first rollout admitted 128 groups and 512 trajectories.
The four learners applied the first update and committed the step-1
checkpoint at 12:53:11 UTC. At 13:05:56 UTC all four fresh learners
verified exact model, Adam, RNG, step and averaging state. Each had
changed model bytes from update 1, and the previous processes exited
cleanly. All eight receivers verified version 1 with zero mismatches.
The driver released it at 13:08:40 UTC, and the second rollout generated
nonempty output with stamp 2. All four restored learners applied update 2
at 13:13:02 UTC. The eight receivers verified version 2 with zero
mismatches, and the driver released it at 13:15:43 UTC. Each policy
version compared 157,498,867,712 bytes across its eight receivers.

The terminal step-2 checkpoint committed at 13:18:55 UTC. The final
100-task evaluation generated under stamp 3 and reported a mean verifier
score of 0.2711 and pass@1 of 0.29. The initial values were both 0.30.
The GPU child and CPU coordinator both succeeded. The original receipts
and terminal manifest matched the frozen source, model and attempt. An
independent checkpoint read found changed values in the same 256-byte
FP32 router slice across steps 1 and 2 within a 5 MiB chunk-read budget.
[Successful terminal manifest][h100-cleanup-terminal-manifest]
[Preempted child][iris-smoke-terminal-cleanup] · [Replacement child][iris-smoke-terminal-cleanup-replacement]

#### Read only the HF weight ranges a learner host needs

The current loader streams Hugging Face model IDs from the Hub through
`hf://` without a local disk cache. It determines the file read ranges
before using JAX sharding to place the addressable tensor slices. The frozen
H100 smoke points the same safetensors reader at a regional CoreWeave copy
of the HF export. Each learner host streams roughly 125 GiB of BF16
payload across all 39 files. Levanter previously prototyped direct GCS reads
per FSDP slice. The requests were small and numerous, so the merged loader
switched to large contiguous reads on each host.
[Direct Hub loading][hf-direct-loading] · [Current range planning][hf-current-range-planning] ·
[Original slice reader][original-hf-slice-reader] · [Loader history][hf-loader-history]

The 40-H100 native arm confirmed full reads of all 39 files on
each of its four original learner hosts. Three hosts finished in 2m05s to
2m34s. The fourth took 29m48s from its first shard log. The trainer began
publishing policy 0 after all four learners initialized. The shard times
measure this initializer and stay separate from rollout and update timings.
[Per-host startup receipt][held-native-matched-current-shards]

The held vLLM pilot uses distributed Run:ai streaming for the same
regional source. The eight processes on its inference host divide the
storage reads and exchange tensors. The original rank-zero log recorded
15.6 GiB of storage payload in 23.1 seconds and 25.6 seconds for model
loading. A plain Hub model ID follows vLLM's cache download path before
the streamer runs. The Levanter `hf://` path reads model weights directly
from the Hub.
[Held vLLM coordinator][iris-held-vllm-pilot] ·
[Distributed streaming][runai-distributed-streaming] · [vLLM model loader][vllm-runai-loader]

SkyRL's Megatron loader already coalesces direct reads for rank-owned HF
tensors and first-axis expert slices. Its Snowball comparison reduced
policy-worker startup from 209.9 s to 75.3 s versus per-tensor requests.
The same approach applies to Levanter weights with contiguous addressable
slices. Keep coalesced whole-tensor reads where the mesh would generate many
small requests. On the smoke mesh, the four-way data split gives each host a
contiguous span per routed down-projection expert. Reading those spans
alone would reduce logical weight bytes from about 125 to 94.5 GiB per
host. Each learner would read 256 contiguous ranges per layer, one per
expert. Across 26 layers and four learners, the down-projection reads
would fall from 162.5 to 40.625 GiB across 26,624 logical range requests.
The gate and up projections split the innermost HF axis, which
would produce about 17 million 1,280-byte spans per host if requested
individually. These estimates use the pinned tensor shapes and mesh and
exclude request coalescing and read amplification. Compare actual object
bytes and GET counts per host, cold time to learner readiness, and bitwise
equality of the loaded model against the smoke loader. Use the same pinned
BF16 export for both readers.
[SkyRL range prefetch][skyrl-hf-prefetch] · [Pinned export record][pinned-hf-export]

A synthetic safetensors fixture read 131,072 BF16 down-projection bytes in
64 local ranges on the 32-device CPU mesh. The shard-local FP32 arrays
matched the frozen whole-tensor loader bit for bit across all 65,536 BF16
bit patterns. Split whole-tensor coalescing at selectively read tensors
so it cannot fetch those bytes again. The proof uses local files.
[Reproducible range-read proof][down-rank-read-proof]

The prototype now runs through the frozen learner's model-layout HF
initializer. It plans selective reads from the explicit addressable
shardings, deduplicates local replicas and retains whole-tensor reads for
layouts with a split trailing axis. Runs with two and four JAX CPU
processes used two devices per process and the same four-file Snowball
export. Each process read 1 MiB with two processes and 0.5 MiB with four,
down from 2 MiB in the original loader. The processes read disjoint
ranges that exactly covered the routed down projections. The other 44
tensors kept the same reads, and all 46 initialized model leaves per
process matched the original loader byte for byte. The program recorded
actual local filesystem requests. Measure cold-start time and range
request overhead against the regional object store on H100.
[Initializer prototype and distributed CPU receipts][down-rank-initializer-proof]

A regional native checkpoint is also an option for repeated cold starts.
Convert the pinned export, including its finalized router biases, to a
model-only Levanter checkpoint. Verify the source hashes and the FP32
loaded weights against the direct HF initializer. The native TensorStore
reader loads addressable slices and can elect one reader per replicated
shard. Snowball supports model-only restore with fresh Adam state.
[Converter][hf-to-native-converter] · [Native shard reader][native-shard-reader] · [Model-only restore][model-only-restore]

The current fresh-process restore initializes from HF before loading the
native optimizer checkpoint. In the accepted maintained vLLM run, all four
learners logged two passes over the same 39 files. One pass covers about
125 GiB of logical object data per host, or about 500 GiB across the learners.
The logs record file sizes and read starts. Measure wire traffic separately.

A scratch test on the maintained source restored the complete checkpoint
with the HF initializer directory unavailable. The save receipt provided
each leaf's concrete sharding and memory kind. Inferring those from the model
shape alone gave the replicated router bias a different layout. The real
typed worker verified exact model, Adam, RNG, step and averaging state in
fresh one- and two-process CPU runs, retained Adam's pinned-host placement
and matched the next update. The worker rejected an incorrect initializer,
model binding or Adam fingerprint, and missing completion metadata. Existing
checkpoints from this pilot lack the new save receipt. Measure the restart
savings and verify exact state on H100 before adopting the path.
[Checkpoint-first CPU proof][checkpoint-first-cpu-proof]

The prior handoffs preserve the `2026.11.20.9` and `2026.11.20.8` source,
along with the held serving comparison. The restored serving plans reproduce
the original artifact roots with a repository-local Git tree abbreviation.
[Shard-local retry handoff][h100-retry-handoff] · [Earlier handoff][launch-handoff]

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

Both 25-update recipes passed the artifact main's no-run preflight with
the regional task selection pinned. The Megatron control retains source
`9e8664df3b` and fingerprint `ef11bf85`. The prospective Levanter arm
uses cleanup retry source `82fd5cbeac` and fingerprint `10242ebf`. It enables
preallocation and keeps the 80% learner cap and pinned-host optimizer
state. A clean restore reproduced the authored and composed configuration
checks. The Levanter arm uses the accepted smoke source.
Report publication, including any temporary staging and SHA-256 checks,
separately from optimizer time.
The Megatron control `2026.11.20.6` was submitted after the smoke passed
its receipt and weight-change audits.
Iris accepted Levanter `2026.11.20.7` as the second arm after the Megatron
control passed the terminal audit. Both use the same regional route and
publication verification. The trainer differences are the learner
strategy, batch builder and Levanter policy configuration.
[Accepted Megatron control][iris-matched-megatron-third] ·
[First Levanter child][iris-matched-levanter] ·
[Second Levanter child][iris-matched-levanter-retry] ·
[Third Levanter child][iris-matched-levanter-third]

The first control child started at step 0 on the same five physical H100 hosts. Its
initial panel scored 0.2602. The row-level audit verified all 100 tasks,
including exact prompt tokens for subsequent panel comparison. The
original Ray driver logs verified all eight receivers with zero mismatches
for versions 0 and 1, each covering 157,226,237,952 bytes. Its first two
training rollouts each admitted 128 groups and 512 trajectories with zero
staleness. Version-1
publication took 2.998 seconds. The accepted Levanter smoke took 164.066
seconds to publish version 1, including 128.147 seconds to install.
The Megatron install took 0.439 seconds. The Levanter expert-phase timer
accounted for 119.911 seconds of the install and includes source waits
and per-send CUDA synchronization. Each learner host staged approximately
3,450 individual payloads and 33.5 GB of source bytes. Megatron used
persistent GPU views and synchronized at phase boundaries. At the
receivers, the FP32 Levanter router raised total bytes by 0.17%. Time the
individual export, host copies, spool I/O, CUDA transfer, and replay
stages before selecting a publication optimization, and use both complete
matched runs to compare overall training and rollout time.

The first child completed 17 optimizer updates and all 144 receiver checks
through version 17. CoreWeave evicted the CPU coordinator to admit another
workload. Iris marked all five GPU tasks `Parent task retried`. The first
replacement lost its coordinator during initialization, before reporting a
step or publishing a checkpoint. Both evicted coordinators ran on the
Genoa CPU pool. The checkpoint-marker read after both interruptions found
no committed step. Keep each child's panel, timing and receiver evidence
together. [First control child][iris-matched-megatron] ·
[First replacement][iris-matched-megatron-retry]

The third child started at step 0. Its initial panel scored 0.1902 versus
0.2602 in the first child. All 100 task IDs and
prompt token sequences matched. Twenty response token sequences matched
exactly and received the same scores in both runs. Compare each completed
arm against its own initial panel and report per-task changes on the
common panel. [Third control child][iris-matched-megatron-third]

The third child and its coordinator succeeded. The audit tied 25 applied
updates, 208 zero-mismatch receiver receipts and 25 nonempty generation
witnesses stamped 26 to the same original GPU child. The step-25 checkpoint
marker committed, and the final panel scored 0.2700 over the same 100 tasks
and exact prompt tokens. The successful child consumed 38.12 million loss
tokens over 3,785 seconds of active training and used 56.26 H100-hours.
The final checkpoint phase took 505 seconds and the final evaluation took
94 seconds. The two interrupted children have separate evidence.

The first Levanter child used the same five physical H100 hosts in the same
Iris task order as the accepted control. Its initial score was 0.2900 across the same 100 exact
prompt-token sequences. The first training batch admitted 128 groups and
512 trajectories with zero staleness. Verified publication took 208.913
seconds at initialization and 163.672 seconds after update 1. The latter
included 127.550 seconds to install, compared with about three seconds
for the control's full verified publication. All eight receivers passed
at both candidate versions.

Update 2 gives the first warm timing comparison on the exact matched arms.
It consumed 1.65 million loss tokens with Megatron and 1.62 million with
Levanter. The four measured phases below account for the early step-time
gap. These values describe one update in each run.

| Update 2 phase | Megatron seconds | Levanter seconds |
| --- | ---: | ---: |
| Rollout group admission | 124.129 | 124.164 |
| Learner preparation | 5.457 | 28.195 |
| Policy training | 17.978 | 91.017 |
| Verified weight publication | 3.053 | 163.769 |
| Full step | 151.893 | 408.420 |

Across updates 2–16, median step time was 151.0 seconds for Megatron and
416.7 seconds for Levanter. Their mean group admission times were 0.972 and
0.959 seconds per group, with 1,920 admitted groups each. Averaged over
those fifteen updates, Levanter spent 159.2 additional seconds on
publication, 80.3 on policy training and 27.1 on learner preparation.
The expert phase accounted for about 119 seconds per Levanter publication
and combines source export, staging and transfer. Time those components
separately before changing the NCCL schedule or the byte-verification
path. These timings compare the same update numbers and exclude the
candidate interruption and any terminal checkpoint work.

The first Levanter child completed 16 applied updates and 136
zero-mismatch receiver checks through version 16. It consumed 24.61
million loss tokens over 6,723 seconds of active training. Its warm
median step took 416.7 seconds, with 162.1 seconds for verified
publication. Kubernetes preempted the CPU coordinator and Iris stopped
all five GPU tasks. The recipe schedules its regular checkpoint at step
25, and the regional checkpoint prefix held no committed step after
this interruption. The second child started on the same frozen source
and lost its CPU coordinator to Kueue preemption during initializer
streaming, before any initial policy publication or evaluation.
The third child started from step 0. Its initial publication took
212.953 seconds, with all eight receivers verifying 157,498,867,712 bytes
and zero mismatches. Its initial score was 0.231 over the same 100 tasks
and prompt token hashes as the completed Megatron control. Use the
first child's data for phase diagnosis. Compare each complete child's
terminal metrics with its own initial panel.
[First Levanter child][iris-matched-levanter] ·
[Second Levanter child][iris-matched-levanter-retry] ·
[Third Levanter child][iris-matched-levanter-third]

The third child's first five updates verified all 48 receiver checks
through version 5. Across the four warm updates, median step time was
465.3 seconds for Levanter and 153.1 seconds for Megatron. Group admission
was about a second per group in each. The mean extra Levanter time was
203.9 seconds in publication, 80.4 in policy training and 25.2 in learner
preparation. The independently captured 64 MiB NCCL all-to-all preflight
had similar median bus bandwidth on all 40 original ranks, 44.17 GB/s
for the third Levanter child and 43.47 GB/s for the accepted Megatron
child. The collective preflight measures a different path from export,
CPU staging and the per-expert publication sends. The frozen learner
copies each canonical expert slice to the host, serializes it through a
local pipe, stages it on the sender GPU and spools the same bytes for
receiver verification. The maintained binary pipe reached a median of
1.60 GiB/s versus 0.71 GiB/s for the frozen pickle pipe in a local ARM CPU
test with the production 6.25 and 12.5 MiB expert matrix sizes. This
measures framing and decoding.

The instrumented held vLLM arm measured its first H100 publication on all
four learner hosts. Per-host median install and replay wall time was 124.5
seconds. Source SHA-256 took 39.0 seconds and recorded broadcast synchronization
8.3 seconds. The time in per-payload requests outside the learner's `next`
method was 25.6 seconds. The learner's method accounted for 29.0 seconds,
including export staging, device-to-host copies and payload serialization.
The held arm's second publication took 109.7 seconds. The original warm
records on all four hosts put the two SHA-256 passes at a median 39.0 seconds,
the RPC outside the learner method at 27.3 seconds, the learner method at
19.5 seconds, spool I/O at 10.9 seconds and broadcast synchronization at
2.8 seconds. The frozen and maintained arms used different hosts and rank
placements, so their publication difference combines transport and runtime
effects.

A bounded hash pipeline overlaps both SHA-256 passes with the existing
sender loop and waits for every digest before acknowledging the version.
The prototype kept the broadcast order and passed three two-version CPU
Gloo cases, including source-spool and installed-receiver corruption. Measure
the hash wait on H100 before adopting it and keep exact byte verification
on the receiver. [Original timing audit and prototype][held-publication-overlap]

A separate candidate adds one-frame read-ahead within an expert layer
and projection. It fetches the next detached binary payload while the
sender stages and sends the current one. Reads across projection boundaries
and all dense reads stay on the sender thread. Each extra frame is bounded
at 64 MiB. The existing collective order, CUDA fences, both SHA-256 passes,
and receiver replay still apply. Two CPU sweeps with 900 MiB per trial
used the production binary pipe and identical host copy, hashing, and
spool work. Median wall time moved from 0.96 to 0.90 seconds in one sweep
and from 1.75 to 0.70 seconds in the other. All digests were exact. The
spread is large and H100 speedup remains unmeasured. The two-version CPU
Gloo cases passed with both source and receiver corruption detected.

Across the 23 warm updates from step 2 through 24, median step time was
450.4 seconds for Levanter and 150.1 seconds for Megatron. They processed
3,163 and 10,141 loss tokens per step second. Admission took 0.94 and
0.96 seconds per group. Both admitted 2,944 groups. Publication, policy
training and learner preparation accounted for 195.1, 80.6 and 26.1
additional seconds per Levanter update on average. Publication made up
65% of the total gap. The step-25 callback includes regular checkpoint
and evaluation work, so the warm comparison stops at step 24.

#### Completed frozen comparison

Both successful attempts used five eight-H100 hosts and started from the
pinned HF initializer. Each applied 25 updates, admitted 3,200 groups and
12,800 trajectories, and passed all 208 receiver checks from policy 0
through 25. The original serving streams carried stamp 26 on nonempty
final-evaluation generation. The marker and policy metadata confirmed the
committed step-25 checkpoints. The separate smoke covers exact fresh-process
restore. All four panels contain the same 100 tasks and exact prompt tokens.

| Measure | Megatron with vLLM | Levanter with vLLM |
| --- | ---: | ---: |
| Initial mean verifier score | 0.1902 | 0.2310 |
| Final mean verifier score | 0.2700 | 0.3201 |
| Score change | +0.0798 | +0.0891 |
| Active learner and rollout wall time | 63.1 min | 187.9 min |
| Consumed loss tokens | 38.12M | 35.57M |
| Loss tokens per active second | 10,071 | 3,156 |
| Median warm update | 150.1 s | 450.4 s |
| Median verified publication in a warm update | 3.1 s | 200.8 s |
| Group admission in warm updates | 0.96 s/group | 0.94 s/group |
| Allocated H100-hours for the successful attempt | 56.26 | 153.01 |

Active time runs from the initial evaluation result through the policy-25
acknowledgment. The final checkpoint phase took 505 seconds for Megatron
and 1,311 for Levanter. Final evaluations took 95 and 87 seconds.
Levanter's gain over its own baseline exceeded Megatron's by 0.0093.
Resampling the paired tasks within each dataset gave a 2.5th to 97.5th
percentile range of −0.1004 to +0.1190 for that difference. The resampling
captures variation across fixed-panel tasks. Each arm has one optimizer run.
[Accepted Megatron control][iris-matched-megatron-third] ·
[Accepted Levanter candidate][iris-matched-levanter-third] ·
[Matched audit summary in Filer][matched-final-evidence]

The built-in `hub` route submits through the Marin hub and delegates the
CPU coordinator to `cw-us-east-02a`. The controller rejects federation of a
nested GPU job from a hub-resident parent. A regional coordinator can use
Iris child preservation after the artifact launcher gains stable child IDs,
terminal-aware attachment and passive observation. The frozen launcher
generates a new child ID per coordinator attempt and cancels its active
GPU child when the foreground supervisor receives SIGTERM.
The direct regional GPU-root path reproduced the frozen five-host BATCH
request and the ArtifactStep fingerprint in a local fixture. The same
fixture interrupted an observer after the native terminal receipt and
reattached under the ArtifactStep lease to write the typed record. Live
admission and remote lease recovery still need verification. Use a fresh
output version if the existing coordinator and all GPU children reach a
terminal state and a direct-root retry is needed.

For a fresh artifact version, confirm that Iris and the durable and scratch
prefixes have no prior attempt. To reuse a version, confirm that every prior
owner has ended and check the executor lease, receipts and checkpoint index.
The ordinary launcher creates a fresh coordinator ID and the trainer can
resume from an existing checkpoint.
[Earlier matched launch and audit package][h100-matched-handoff]

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
separately. [Matched-run auditor and tests][h100-readiness-handoff]

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
host processes. CUDA/NCCL startup and restart remain unverified for the
maintained typed learner.

[Exact smoke and comparison recipes][learner-status] ·
[Frozen checkpoint validation code][frozen-worker]

### 2. Close the smallest native RL loop

Port the Snowball paged model/cache path, engine capture, learner replay and
fixture arithmetic onto `main` in reviewable slices. Reconcile Snowball
attention, model and loss changes with the learner branch. The reviewed
Snowball snapshot `5992abd052` was 766 commits behind baseline
`2bf3f721c9` and included unrelated Hero and Iris changes.
[Source comparison][snowball-compare]

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
frames and all 128 sampled tokens. This deterministic single-turn fixture
uses the all-eligible RLOO-N calculation on four rows padded to 256
positions. Its learner recomputes raw old-policy logprobs for the loss
and compares them with a separate array of serving scores. The maximum
absolute score difference was `4.29e-6`.
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
all targets for two prompt/response splits.

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
individual CPU executable. The tiled H100 attempt reduced the failed
allocation to 35.50 GiB. The allocator reported 24.66 GiB resident and a
71.26 GiB limit immediately before prefill. A dedicated preallocation
trial uses the same context and records a focused XLA memory report to
test whether the allocator's growth behavior contributes to the failure.
The tiled prefill compiler report records 59.95 GiB per GPU including
arguments and workspace. Dedicated preallocation let the probe generate
four 8,192-token replies from 16,384-token prompts in 43.81 minutes,
including cold compilation and prefill. The JAX allocator recorded 59.97
GiB peak in use per GPU. The learner ran the FA4 attention and fused XLA
projection on all 32,768 targets in 12.25 seconds. The p99 absolute raw
logprob difference was `2.322e-4`. One target reached a log-ratio of
`+0.183555`, corresponding to a ratio of `1.201481` against the `1.20`
upper PPO bound. The job failed the zero-crossing gate after scoring.
This measures one synthetic group with a fixed checkpoint. The receipt
retains four frame digests and omits the raw frames and outlier position.
[67B model-load and prefill attempt][iris-native-67b-router] ·
[90% allocator attempt][iris-native-67b-cap90] ·
[Tiled H100 attempt][iris-native-67b-tiled] ·
[Dedicated preallocation trial][iris-native-67b-prealloc] ·
[Byte-verified original archive][native-67b-prealloc-archive]

The next attention candidate skips padding-only query tiles and bounds the
scan to visible KV pages while preserving the accumulation order. It
passed 26 focused checks on JAX 0.11.0 CPU. For the qualification shape,
the page-loop count falls from 119,808 to 14,456 per prefill and from
40.9 million to 11.8 million across four responses. The one-host H100
run at `22a4e78e6a` used the same weights, seeds and context. It generated
all four replies in 24.86 minutes, 43.3% less wall time than the earlier
probe. The JAX allocator recorded 59.97 GiB peak in use per GPU. The
inventory scorer ran in 9.75 seconds and passed on 32,768 targets, with
signed log-ratios from `-0.004765` to `+0.069251` and p99 absolute error
`2.139e-4`. The four binary frame digests differ across the two runs. This
qualifies the bounded-page configuration on one fixed-checkpoint group.
It leaves the cause of the earlier crossing open.

The diagnostic compared four scorer variants on the second run's same
bound replay, varying matmul precision, projection dtype and expert
accumulation. All four had zero crossings. Use the pinned inventory scorer
for acceptance and retain signed outlier positions in follow-up probes.
The later ordered-gather run passed 67B tool-turn replay and exposed a
separate full-context score discrepancy after the live swap. The
five-host online deployment and held-learner measurements remain open.
[Scorer attribution run][iris-native-67b-profiles] ·
[Byte-verified successful archive][native-67b-profiles-archive]

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
  inputs for artifact version `2026.11.21.2`. The held pair at `07bddc836e`
  uses artifact version `2026.11.21.5` and the exact bounded-page kernel
  from the successful H100 probe. It keeps the same learner, checkpoint,
  training tasks and evaluation settings. The dedicated native serving
  subprocess in that pair preallocates 90% GPU memory. Submit after the
  priority learner smoke and matched 25-update comparison.
  Use the archived pair as the serving configuration reference. Port the
  qualified learner's loading, optimizer placement, readiness and recovery
  changes into the common typed learner before submitting either arm.
  A CPU port of the final-layout loader, readiness barrier, and checkpoint
  and shutdown deadlines matched 15 typed receipts from the deadline-only
  control on two JAX 0.11.0 learner processes. The receipts cover forward
  scores, three update calls, model, optimizer, RNG and layout fingerprints,
  and exact fresh checkpoint restore. The loader preserves legacy placement
  for unpartitioned source tensors. In a separate two-rank CPU test, split
  gradient and optimizer execution with pinned-host Adam matched the
  split-device control across all 137 checkpoint leaves. Both modes
  restored exactly and reproduced the next update. The original fused path
  differed from the split path by at most `1.86e-9` in model values and
  Adam first moments. The learner allocator choice, CUDA transfer cost,
  and full-model GPU qualification remain open.
  The combined CPU port includes the fingerprint staging change and the
  optional prefix implementation. Twenty replay and wire tests passed.
  All three optimizer modes reproduced their exact fresh restore and next
  update. The follow-up at `5fefb7ae6b` brings the accepted canonical
  byte replay and Ray finalization fix into the typed learner. Both
  version `2026.11.22.1` plans use the same 80% GPU preallocation,
  pinned-host Adam, and learner checkpoint roundtrip. The dry-run
  comparison matched initializer, training selection, evaluation,
  and learner fields. Eleven CPU transport cases and 34 typed learner,
  config, and serving cases passed on the port. The prospective native
  preset keeps prefix reuse disabled until the accelerator gate.
  Retain the binary frames and native snapshot install, then qualify the
  ported learner with vLLM. The archived preset uses one turn per episode.
  Use a separate 67B tool-turn fixture to test continuation and a live
  weight install. The current native path resets paged KV on each generation.
  On CPU, a two-turn fixture with a simulated tool observation rejected a
  future policy before installation and rejected the old policy and its
  replay frame after installation. The new-policy replay frame matched an
  independently started process byte for byte.
  Porting cross-request prefix reuse also requires preserving expert IDs
  and router thresholds alongside cached pages for training replay.
  A one-slot CPU prototype retained four prefix tokens under policy A,
  reused zero on the first policy B request after installing B, and reused
  six on a repeat under B. The cached and cold runs produced identical
  tokens, positions, validity masks, expert routes, routing thresholds,
  and logprobs. The prototype used a simulated tool continuation. A second
  CPU fixture ran eight small slots with a 40-page limit. The first slot
  exclusively reused pages retained by a previous single request. Each
  sibling allocated disjoint pages and matched its independent cold frame.
  The fixture retained prefixes from single-request calls. Production
  page geometry, a real 67B tool turn, and serving throughput remain open.

  A separate CPU fixture used the production SkyRL tool parser, rollout
  worker and typed native serving subprocess. The model sampled a
  `fetch_url` action and an offline provider returned the observation that
  SkyRL rendered into the second turn. The binary learner frames retained
  all 24 overlapping token positions and 96 expert assignments. All nine
  observation and boundary tokens carried zero objective weight. Recomputed
  raw policy logprobs matched the sampled tokens exactly.
  The first request after a live policy install reused zero prefix tokens.
  Its binary replay frame matched a cold subprocess's frame byte for byte.
  A repeat under the same policy reused 32 tokens. The 67B fixture pins
  the three tokenizer assets and raw model configuration used by the
  accepted H100 smoke. All four files were restored from the regional
  initializer mirror and their bytes match the recorded hashes.

  A CPU bridge restored the model subtree of two checkpoints written by
  the frozen learner. It reproduced the original FP32 master fingerprints,
  applied the learner's compute cast while retaining FP32 router and QB
  fields, and published typed manifests without changing the source files.
  Serving verified both snapshots on a different mesh. The regional
  preflight rechecked both accepted H100 checkpoint manifests, the
  step-2 completion marker, and the bounded changed-weight sample. The
  resolved configuration builds an abstract model of 67,078,882,816
  parameters. Convert the two checkpoints on the original Data4/EP8
  learner layout, then independently verify the typed snapshots on the
  one-host serving mesh. The staged Iris wiring starts the serving job
  only after the conversion succeeds and its immutable receipt validates.
  The first coordinator requested an elevated accelerator profile that
  Iris denied. The second used the ordinary profile but passed an
  explicit regional peer directive to the regional controller. Iris
  admitted four eight-H100 conversion ranks with implicit local placement
  on the third attempt. All four failed a launcher check for a preassigned
  JAX port before reading weights. The fourth attempt uses Iris's normal
  rank-zero port registration and discovery, which passed three focused
  tests. Four ranks restored step 1 with exact FP32 weight fingerprints
  and layout hashes from the accepted smoke. Each rank restored 67,240,597,504
  addressable bytes at each step. The step-2 fingerprints differed on every
  rank and the layouts matched. The region-local typed receipt names two
  distinct immutable policy IDs under the same model binding. All four
  conversion tasks succeeded. [Conversion job][iris-native-conversion] ·
  [Typed conversion receipt][native-typed-conversion-receipt]

  The one-host serving job saw all eight H100s and stopped when Iris's
  workspace extraction removed the telemetry wrapper's execute bit. The
  tested fix restores execute permission on the hash-checked scratch copy.
  The first serving retry verified policy 1 and completed the tool exchange.
  Its two binary frames matched on all 471 shared tokens, positions and
  validity masks. The defined cross-turn route-equality check found 3,647
  differences across 48,984 ordered expert assignments. Some positions
  changed expert membership, including positions in the original prompt on
  deeper layers. Prefix caching was disabled for this fixture.
  [Serving attempt and captured frames][iris-native-fused-gate]

  The production binder independently verified both frames against the
  same model and policy, preserved each turn's routes and QB thresholds
  bit for bit, and assigned 247 and 104 sampled targets to their original
  turns. All 46 tool observation and boundary tokens remained available
  as context with no target owner. The next qualification replays both
  original turns with the fused scorer and independently compares every
  action against its sampled raw policy score. The new per-turn scorer
  matched the raw scores exactly on an independent four-device CPU fixture
  using the actual SkyRL learner inputs. The post-swap CPU fixture replayed
  all 32 targets with zero clip crossings.

  A subsequent H100 sample exhausted the short fixture's 256-token reply
  budget before issuing the tool call. The retry uses up to 1,024 tokens
  and four fixed seeds for each short turn. The selected call and answer
  produced 224 and 69 sampled tokens. Their actual SkyRL learner inputs
  preserve all raw scores and exclude 46 tool observation and boundary tokens
  from the objective. A fresh eight-H100 worker independently restored
  policy 1 and scored all 293 sampled targets from those exact turns using
  FA4 and the fused vocabulary projection. Their absolute log-ratio p99
  values were 0.094 and 0.120, with zero PPO clip crossings.
  [Independent tool-turn scorer][iris-native-tool-scorer]

  The same serving process verified policy 2, allocated its replacement
  KV pool and released it. The full-context gate retains the original
  24,576-token shape and 8,192-token reply cap.
  Before the swap, device allocation was 24.48 GiB live. It reached
  49.00 GiB while both policies and KV pools were present, then fell to
  24.53 GiB after release. All eight BFC allocators failed the first
  post-swap request for a contiguous 35.50 GiB buffer. The allocator maps
  show free memory split across smaller regions. The isolated serving
  retry used JAX's CUDA async allocator on the same model and context
  and returned the first verified policy-2 reply in 8.32 seconds. The
  repeated request produced a different binary frame. The retry stopped
  before persisting both frames, so the difference could be sampled
  tokens or another captured field. The next run measures page references,
  live sequence and request slots, and retained prefix state at the
  install boundary and on cold restore. It preserves all three reply
  frames and verifies each one's request, positions and policy identity.
  The two independent fused scorers retain their original allocator.
  [Original live swap][iris-native-live-score] ·
  [CUDA async allocator retry][iris-native-allocator-retry] ·
  [Cache ownership and full-context run][iris-native-cache-ownership] ·
  [JAX allocator options][jax-allocator-options]

  The cache-ownership run installed policy 2 on the original serving
  process and reported zero page references, live sequence slots, active
  request slots and retained prefixes at release. The first reply took
  2.11 seconds. Two requests under that same policy and seed had 514
  identical prompt tokens and four identical sampled tokens. Their first
  sampled raw logprobs were `-2.1513` and `-1.5693`. Decode scores for the
  other three sampled tokens differed by at most `5.35e-4`. Expert-set
  differences appeared during prompt prefill, starting at layer 1,
  position 244. Across the prompt, 863 layer-position rows changed expert
  membership. The installed policy has FP32 router logits, but paged
  serving uses the ring implementation's BF16 scatter-add at EP8. The
  learner uses the gather-combine implementation, which sums local
  contributions in fixed order in FP32 before casting. Compare the same
  captured request and weights across both implementations. A separate
  cold policy-2 load reproduced the difference with the original ring
  path. All four repeats diverged in their binary frames. Enabling
  deterministic XLA with the original ring path produced four
  byte-identical frames. The ordered gather path also produced four
  byte-identical frames under ordinary XLA. These results cover one
  short H100 request. Fused learner parity still uses the actual tokens
  and routes from each individual reply.
  [Original paged serving path][native-paged-moe] ·
  [Controlled short repeatability probe][iris-native-repeat-prefill] ·
  [Cache ownership and full-context run][iris-native-cache-ownership]

  The typed install keeps the old model and KV pool live until it verifies
  the replacement weights and allocates the new pool. A CPU probe found
  that the old model, KV and optional prefix references dropped at the
  swap. On H100, sample live device bytes during restore and allocation,
  after the swap, and during the first request. The measured cold-run
  peak covers one resident policy.

  The post-swap fused probe captures four production-length responses
  under the replacement policy in the same live serving process. A
  fresh one-host scorer verifies the installed weights independently
  and scores the actual binary replay with FA4 and fused projection.
  The CPU fixture passed all 32 action targets with maximum absolute
  log-ratio `2.38e-7` and zero clip crossings. Require zero crossings
  on all 32,768 post-swap targets on H100.

  The original H100 process completed four 8,192-token responses under
  policy 2 at 24,576 tokens of context. Generation took 1,492.6 seconds
  and peaked at 59.96 GiB per H100. Independent readback verified all
  positions, captured routes, scores and the policy identity. A fresh
  process restored the same policy with all cache ownership counters at
  zero. Its short request diverged from the live request on the first
  sampled token. The original fused worker scored the 611 sampled
  policy-1 tool targets with zero PPO crossings, then stopped before
  scoring policy 2 because a reader still required an identical cold
  reply. The replay-only scorer uses the exact saved policy-2 bytes and
  the same FA4 and fused projection configuration. The first replay-only
  worker read the archived bytes and stopped on a missing `zstd` command.
  The replacement used the task's locked Python Zstandard reader and
  scored all 32,768 original targets. The independent audit recomputed
  one upper crossing at response offset 6,167 in request 60,001. Its
  ratio was 1.2528 against the 1.20 limit. The p99 absolute log-ratio
  was `1.58e-4`. Ordered gather used the same checkpoint bytes and clip
  gate. It released policy 2 with zero cache ownership, and the first,
  repeat and independent cold replies were byte-identical. Its four
  long responses finished in 1,476.0 seconds and captured identical
  positions, ordered routes, and thresholds across their common
  16,384-token prompt. The independent fused scorer passed 167 sampled
  tool-turn targets. It scored all 32,768 full-context targets with
  three crossings, two below 0.80 and one above 1.20. Each was a
  standalone period under a fully matching token history in the
  scatter and gather captures. Their predictor expert sets still
  differed between serving implementations. A saved-frame H100
  diagnostic reproduced the same scores bit for bit with and without
  the learner's named-axis mapping. FP32 projection retained all three
  crossings, including in a comparison that held the learner
  activations fixed. Highest matmul precision left two crossings.
  Native H100 serving streams paged K/V through the portable FP32
  reference attention path. The learner uses dense-sequence BF16 FA4.
  A second replay verified no expert assignments dropped at any
  learner layer. Switching the learner to blockwise BF16 XLA
  attention kept three crossings and changed which positions crossed.
  Changing blockwise attention contractions and state to FP32 improved
  the p99 error and moved the tail. At highest matmul precision, BF16
  and FP32 blockwise attention both had two crossings. The next test
  needs the actual serving activations at the persistent outlier and
  the corresponding learner activations.
  The serving-score parity gate compares scores from two runtimes.
  The held PPO objective uses its own learner-rescored old
  probabilities.
  [Full-context run][iris-native-cache-ownership] ·
  [Archived exact-task source][native-full-capture-archive] ·
  [Replay-only scorer][iris-native-b-scorer] ·
  [Full ordered-gather qualification][iris-native-ordered-gather] ·
  [Saved-frame precision diagnostic][iris-native-score-profiles] ·
  [Saved-frame BF16 attention diagnostic][iris-native-attention-profiles] ·
  [Saved-frame FP32 attention diagnostic][iris-native-attention-fp32]

  The matched 40-H100 pilot measures end-to-end rollout admission and optimizer
  timing with four learner hosts. Step 2 of the third frozen Levanter
  run generated 1,628,026 response tokens for 512 trajectories. At the
  22-token-per-second native rate measured on the full-context group,
  that volume corresponds to about 20 hours of generation. This
  extrapolation uses one long-context group. The bounded pilot uses
  the same 40-H100 layout, four answers per
  group, 32 groups per update, a 16,384-token prompt cap and a 24,576-token
  serving window. It uses common 512-token reply caps for rollout and
  evaluation in both arms. Keep the full-context production qualification
  alongside the bounded end-to-end timing and report the cap when
  interpreting panel rewards.

  The first bounded vLLM child installed policy 0, then lost its CPU
  coordinator before receiver verification completed. Kueue later
  preempted the replacement GPU group during initialization. Neither
  attempt reached the baseline evaluation. The timing-instrumented
  vLLM child was also preempted during setup. The regional controller
  accepted a direct five-host retry on the same frozen source and
  artifact at BATCH priority. Recovery confirmed terminal prior owners
  in both controllers, closed regional task attempts, and an empty
  checkpoint prefix with no active executor lease. A durable submission
  claim binds the worker configuration and Iris root. After regional
  acceptance, Kueue found zero eligible nodes because the prototype
  added an `iris.region` selector absent from the managed nodes. The next direct dispatch must rely on the
  already-selected regional controller for placement. The revised request
  matched the admitted ordinary job's managed-node selector in a local
  fixture. Iris confirmed that all five tasks in the first direct job
  were canceled without starting. The regional controller then accepted
  the corrected single replacement at the same source, allocation and
  priority. Kueue admitted all five original GPU tasks with the managed-node
  selector. The 40-rank NCCL gate passed. vLLM published policy 0 in 124.8
  seconds and all eight receivers verified 157,498,867,712 bytes with zero
  mismatches. The baseline scored all 100 tasks in 16.5 seconds with a mean
  verifier score of 0.1800. The first training window admitted 32 groups
  of four rollouts in 10.0 seconds at zero staleness. The supplemental
  fresh-learner recovery check then began writing a roughly 750 GiB optimizer
  checkpoint. Two of the four learner ranks reported local write completion
  before the 30-minute learner command deadline. The checkpoint prefix
  contained a partial TensorStore manifest and lacked committed metadata.
  The job ended before accepting update 1 or publishing policy 1.
  The retry plan allows up to 60 minutes for the learner command, retains
  the independent TensorStore commit deadline and requires exact restore.
  The official artifact main verified both new `2026.11.22.2` plans
  against their previous launch settings. The new vLLM child started on
  all five eight-H100 hosts and all 40 ranks passed the interconnect gate.
  Serving rank zero streamed its 15.6 GiB share of the regional HF export
  in 92.0 seconds. All eight serving ranks verified policy 0 in 150.7
  seconds with zero mismatches across 157,498,867,712 bytes. The run
  scored the 100-task baseline in 19.2 seconds and admitted the first 32
  groups at zero staleness. The retained baseline matched all 100 task IDs
  and prompt token sequences from the previous attempt. The mean verifier
  score changed from 0.18 to 0.23 on the fresh generation.
  Three learner ranks completed their local checkpoint writes. Thirty
  minutes after the first finished, the JAX TensorStore barrier reported
  that rank 1 had yet to arrive. The OCDBT manifest had changed three
  seconds before the error. The completion marker was absent, and the
  attempt ended with zero accepted updates. Scope the next deadline
  change to the supplemental validation save and retain the exact
  fresh-process state comparison.
  Both `2026.11.22.3` plans preserve the retry's launch settings and set
  a 45-minute inner commit deadline for the validation save within the
  existing 60-minute learner command window. Ordinary checkpoint saves
  retain the 30-minute inner deadline. The two-process CPU check passed
  exact fresh restore and a separate test propagated a real commit
  failure before emitting any fingerprints. Kueue preempted the first
  vLLM coordinator task during serving warmup. The coordinator's
  replacement started a fresh five-host child. All 40 ranks passed NCCL,
  and the child verified policy 0 in 113.1 seconds with eight exact
  receipts. It scored all 100 baseline tasks at a mean of 0.21 and
  admitted 32 groups of four rollouts at zero staleness. All four
  learner hosts committed the 749.66 GiB validation checkpoint in
  403 seconds. Fresh processes restored the model and optimizer state
  exactly before publishing policy 1. The same processes applied update
  2, published policy 2, and scored all 100 final tasks at a mean of
  0.20. All 24 receiver receipts had zero byte mismatches. The child
  and coordinator both succeeded. The independent audit verified the
  exact original attempts, both fixed panels and the committed step-2
  checkpoint. The post-restore update took 183.8 seconds including
  109.7 seconds of publication. The lease-to-admission window averaged
  0.305 seconds per group across all 32 groups.
  [Fresh held vLLM child][iris-held-vllm-retry]
  [Validation-deadline vLLM coordinator][iris-held-vllm-validation-timeout]
  · [Automatic GPU replacement][iris-held-vllm-validation-replacement] ·
  [Accepted held vLLM audit in Filer][held-vllm-accepted-audit] ·
  [Prior rendezvous evidence in Filer][held-vllm-rendezvous-proof]

  The native coordinator lost its first slot before the GPU group was
  admitted and its second during regional initializer streaming. Its
  third child on five H100 hosts committed a 124.98 GiB policy-0 snapshot
  across four learner hosts. Serving rejected the model binding before
  reading tensor weights. The learner encoded the capacity factor as
  integer `8`, and the typed serving decoder reconstructed float `8.0`.
  The values produced different JSON digests. An independent CPU
  reconstruction from the committed metadata matched all 502
  serving-template shapes and dtypes and isolated this field. Require
  the same typed config representation before hashing and keep the
  existing tensor and policy identity checks. The failed child stopped
  before admitting rollouts or starting evaluation.

  The repaired learner source `ab79d757b0` normalizes that field during
  model construction. A CPU regression reproduced the frozen learner's
  weights and optimizer results after a real update. Integer and float
  inputs produced identical binary replay after normalization. A changed
  capacity still failed the install check. The repaired pilot has its own
  artifact under tree `ac90c03` and uses the same learner algorithm, data,
  seed and 40-H100 layout. All four original learner processes initialized
  with the expected canonical config hash. The repaired policy-0 snapshot
  matched all 502 tensor inventories and shard hashes from the earlier
  snapshot. Serving verified and acknowledged that same policy identity.
  Writing the snapshot took 333.0 seconds and serving install took 158.8
  seconds. Serial baseline evaluation scored 32 of the 100 tasks before
  the 30-minute deadline. When training opened, the native bridge rejected
  an assistant-prefix request before admitting the first group. The job
  ended with policy 0 and zero accepted updates.
  [Repaired held native run][iris-held-native-repaired] ·
  [Source delta and regression proof][held-capacity-source-proof]

  A separate one-host H100 diagnostic restored the same step-2 snapshot
  and started the four full-context responses. A duplicate debug callback
  stopped generation before scoring. The next probe collects byte-identical
  repeats and defers rejection of conflicting observations until after
  capture. It compares the serving and learner activations only when all
  four replay frames and the original fused scores match exactly. The CPU
  regression preserved ordinary frames and scores with repeated callbacks.
  The revised eight-H100 probe generated all four full-context responses.
  It reproduced every sampled token ID, position and validity mask from
  the earlier ordered-gather run. The raw logprobs differed from the first
  sampled target and some expert routes differed. The frame hash check
  rejected the head comparison. The ordinary scorer finished with zero
  crossings across 242 tool-turn targets and one above the upper bound
  across 32,768 full-context targets. The earlier ordered-gather run found
  three full-context crossings. One target crossed in both runs with
  opposite signs. Compare head activations from the same generation call
  as the scored frames in the next diagnostic.
  [First head diagnostic][iris-native-head-comparison] ·
  [Verified failed-task archive][native-head-first-archive] ·
  [Revised diagnostic][iris-native-head-retry]

  The native baseline held the serving-process lock for each evaluation
  decode. The repair coalesces evaluation requests in a typed binary batch
  and drains evaluation and training requests before installing weights.
  On the small BF16 CPU fixture, the actual worker and client preserved
  the serial evaluation's tokens, score bits and finish reasons with
  independent seeds and stops. A separate CPU rollout carried the shared
  renderer's assistant-prefix tokens in context and attached replay scores
  to newly sampled tokens. Measure the 100 pinned prompts and check
  assistant-prefix replay on one eight-H100 host before retrying the
  five-host native pilot.
  The first calibration stopped before opening the snapshot. The regional
  Kubernetes pod omitted the worker-region field that the standalone
  Iris worker supplies. The original pod and node records confirm eight
  H100s on a managed host in the intended region. The revised guard checks
  the original task identity, the inherited regional storage prefix and
  the visible accelerator inventory. The calibration times the serving
  worker on the 100 exact prompt token sequences and leaves grading and
  Ray overhead to the full held run.
  The revised task restored the exact policy-0 snapshot in 153.5 seconds.
  Its typed batch capacity was four. From the cold evaluation start, it
  completed all 100 requests in 1,563.5 of the allowed 1,800 seconds,
  with 53,055 prompt tokens and 49,078 generated tokens. The same worker
  returned an assistant-prefix training frame with all 79 prompt tokens,
  contiguous positions and exactly ten raw logprobs for the ten sampled
  tokens. The warm comparison completed the same 16 prompts in 264.9
  seconds batched and 858.0 seconds unary. Four responses had identical
  token IDs, three also had bit-identical sampling logprobs and 15 had
  identical finish reasons. The maximum absolute sampling-logprob
  difference among the identical responses was 0.475. The instrument
  timed requests with the same policy and per-request sampling seed.
  The observed batch-shape sensitivity leaves the separate raw PPO score
  gate open.
  The matched `2026.11.22.3` native arm started after vLLM published
  policy 1. It uses the same learner commit, scoped checkpoint deadline,
  initializer and task selections. The new native child passed NCCL on
  all 40 H100s. Three learner hosts finished their initializer reads
  within four minutes. The fourth finished after about 26 minutes.
  The acknowledged policy-0 manifest matched all 502 tensors and shard
  digests from the previous initializer. Snapshot publication took
  149.4 seconds and serving install took 244.5 seconds. Kueue preempted
  the CPU coordinator as baseline evaluation began. This child recorded
  no scored baseline tasks or optimizer updates. Iris admitted the
  replacement on the same 40-H100 layout. All 40 ranks passed NCCL.
  The last learner finished streaming the HF export in about 30 minutes
  and all four workers reported ready. Serving acknowledged the pinned
  policy-0 identity and all 502 tensor inventories matched the previous
  initializer snapshot. Publishing the snapshot took 109.9 seconds and
  installing it took 244.9 seconds.

  The replacement scored all 100 baseline tasks in 1,647.1 seconds and
  opened its first 32-group training lease. The vLLM arm scored the same
  task IDs in 19.7 seconds. The original panel audit found 89 exact prompt
  token matches and 11 differences, all in the Nemotron tool-use tasks.
  The chat template rendered the same tool schema fields in a different
  order. Nine prompts also differed in the formatting of historical tool
  call arguments. The full-panel means were 0.12 for native and 0.21 for
  vLLM. Across the 89 matching prompts, the means were 0.135 and 0.157.
  Treat these as baseline observations and require exact prompts on all
  100 tasks before accepting a matched reward comparison.
  [Baseline prompt audit][held-native-baseline-prompt-audit]

  The second CPU coordinator preemption stopped all five H100 tasks
  during the first training admission window. That child emitted
  trajectories and recorded zero completed updates. Another child on
  the original source lost its GPU allocation to a higher priority
  workload, and the coordinator then failed.

  A one-file native renderer repair reproduced all 100 original vLLM prompt
  token arrays with the pinned tokenizer on CPU. It changed the 11 tool
  prompts and kept the 89 other arrays. The exact continuation test
  preserves already sampled token IDs. The offline artifact plan holds
  the learner, data, seed and 40-H100 role plan fixed. Require exact
  live-panel prompt parity and an audit of the source delta to reuse the
  completed vLLM arm. A preemption interrupted the first repaired child
  during initializer streaming. The replacement passed all 40 NCCL
  ranks, initialized the four learners and installed the same policy-0
  digest with all 502 tensors and their shard hashes verified.
  Snapshot publication took 817.7 seconds and serving install took
  547.4 seconds. The replacement scored all 100 baseline tasks in
  1,685.4 seconds and opened its first training lease. An independent
  audit matched all 100 original task IDs and prompt token sequences
  against the pinned vLLM control. The baseline means were 0.10 for
  native and 0.21 for vLLM. The comparison uses a single run per backend.
  [Reviewed renderer proof][held-native-prompt-parity-proof] ·
  [First prompt-aligned child][iris-held-native-prompt-aligned] ·
  [Prompt-aligned replacement][iris-held-native-prompt-aligned-replacement] ·
  [Verified live prompt parity][held-native-live-prompt-parity]

  The replacement filled its first training batch with 32 groups, 128
  trajectories and 55,301 response tokens at zero staleness. The trainer
  spent 2,336.8 seconds waiting for the batch, or 73.0 seconds per group.
  The held vLLM trainer spent 11.4 seconds, or 0.36 seconds per group,
  filling the equivalent batch. Some native model calls hit the
  1,800-second per-turn limit, which includes queue wait. The coordinator
  kept the lease open and filled the batch. All four learners started
  writing the step-one validation checkpoint. The checkpoint committed
  in 21 minutes. The fresh workers streamed the pinned HF export again,
  then each read about 187.4 GiB from TensorStore in 170 seconds. The
  original receipts verify changed model bytes and exact model, Adam,
  RNG, step and averaging state on four fresh processes. Policy-1
  publication started after the restore. The restore took about 36
  minutes from checkpoint commit to exact verification.
  [Verified first-update restore][held-native-step1-restore]

  Serving acknowledged policy 1 after 390.6 seconds of publication and
  install. Update one took 6,385.8 seconds end to end, including validation
  recovery. The second rollout produced trajectories under policy 1 before
  Kueue preempted the GPU driver and stopped the four remaining tasks.
  The coordinator ended in failure. The validation checkpoint and policy
  receipts are preserved. The ordinary step index and rollout cursor
  checkpoint are absent, so the exact two-update comparison needs a fresh
  run. Final matched reward and steady-state update time remain open.
  [Verified interrupted outcome][held-native-interrupted-result]

  The fresh recovery plan uses the same corrected native transport,
  initializer, selection, fixed panel, optimizer, seed and five-host role
  plan. The official artifact main rendered a clean candidate whose only
  authored configuration differences are its run and job IDs and a checkpoint
  every update. The regional Iris and storage namespaces were empty at
  preflight. The pinned `442f21b` candidate started on 40 H100s on
  November 23 at 10:24 UTC.
  [Verified per-update checkpoint plan][held-native-recovery-plan]

  The staged regional root launcher uses the same artifact fingerprint
  and five-host BATCH request as the nested launch. Its request uses the
  managed H100 pool directly, which avoids the inherited region selector
  that blocked an earlier submission. The local test suite covered a
  single exact submission and passive finalization under the artifact
  lease after an interrupted result write. The finalizer captures the
  original step-2 checkpoint marker before retention prunes step 1.
  Removing the CPU coordinator from the GPU group's lifecycle reduces
  one source of eviction. Kueue treats its H100 leafgroup as a placement
  preference. The scheduler placed all five original tasks on the same
  InfiniBand leafgroup. All 40 ranks passed the InfiniBand GDRDMA and
  PyNCCL gate. The first-attempt manifest pins all five task identities
  for checkpoint and panel verification.
  [Regional request fixture][held-native-recovery-root] ·
  [Exact first-attempt launcher][held-native-direct-operator] ·
  [Original native matched arm][iris-held-native-matched-preempted] ·
  [Verified 40-rank startup][held-native-matched-current-startup]

  The conventional vLLM launch receipt records the configuration before
  driver validation. The native direct-root receipt already includes
  derived driver fields. Replaying validation confirmed the two actual
  workers used the same model, train and evaluation data, and resolved
  learner settings. Both original logs confirm four RLOO-N samples, at
  least two eligible samples, and a 16,896-token request limit.
  [Effective workload parity proof][held-native-matched-current-config]

  All four learners initialized, and the serving host acknowledged policy 0
  at 11:04 UTC. Snapshot publication took 100.0 seconds and installation
  took 159.3 seconds. The 502 tensor records and weight digest matched the
  earlier verified policy 0. The fixed baseline evaluation then started
  under that installed identity.
  [Original policy-0 installation proof][held-native-matched-current-policy0]

  The fixed baseline scored all 100 tasks in 1,615.3 seconds. An
  independent read of the original panel found exact prompt token matches
  on all 100 tasks against the held vLLM control. The policy-0 mean
  verifier score was 0.12 for native and 0.21 for vLLM. The native trainer
  opened its first 32-group admission window at 11:31 UTC.
  [Original matched baseline receipt][held-native-matched-current-panel0]

  The 11 tool-use tasks accounted for seven of vLLM's nine additional
  passes. Native reached the 512-token cap on all 11. vLLM emitted 10
  structured final calls and passed seven tasks. Replaying the saved
  native tokens with tool parsing enabled recovered five complete calls.
  The original final-action grader scored all five zero. The replay also
  reproduced the function names and arguments of all 10 saved vLLM calls
  through the native parser and reproduced their original scores. The
  declared sampling settings matched across engine-specific samplers.

  All 11 tasks enforce a single-call contract. The first complete call
  in each saved native reply occurred at a median of 88 tokens. Grading
  those exact prefixes passed five tasks, which would raise the original
  100-task panel score from 0.12 to 0.17 in this offline replay. The first
  closing delimiter matched the valid prefix in all 11 replies.

  A local CPU prototype requests a structured stop over typed evaluation
  IPC when the request supplies tools and explicitly disables parallel
  calls. The serving child uses the staged policy tokenizer and the
  Snowball tool-call grammar. It returns the sampled token IDs and
  logprobs through the first complete call outside reasoning, then retires
  that request before the next decode chunk. Delimiters inside quoted
  arguments are treated as argument text. At the default 32 rounds, the
  last chunk can already have sampled up to 31 extra tokens for that
  request. The response excludes that suffix. In a mixed CPU batch,
  subsequent chunks advanced only the other request, whose tokens and
  scores matched the baseline. The 11 archived prefixes ranged from 60
  to 424 tokens. Grading after a typed-codec round trip reproduced all
  five passes.

  The prototype has its own source diff. The sealed 40-H100 comparison
  still uses the frozen evaluator. Qualify the stop with a new artifact
  version and measure both device work and end-to-end panel time on H100.

  Kueue evicted the trainer at 12:03 UTC to admit another workload in
  the same queue. The other four tasks ended with it. The original trainer
  recorded 128 generated responses with 50,154 tokens and rejected five
  fully masked groups. Preemption ended the admission window after 31m57s,
  before the batch closed or an optimizer update ran. The policy-0 snapshot
  remains available. The ordinary checkpoint marker was absent after exit,
  so the two-update comparison needs a fresh attempt.
  [Verified interrupted attempt][held-native-matched-current-preemption]

  The inventory preset sets the artifact revision. The outer `--version`
  flag alone still resolved to the spent artifact namespace. An offline
  render with an explicit preset revision produced a fresh namespace and
  the same bound workload and Iris resources. It retained both evaluation
  panels and per-update checkpoints. The live operator still needs a fresh
  claim and a pin to that exact plan.

  A second offline render pins the combined serving source and enables
  blocked attention in the native recipe. It preserves the frozen inputs,
  two 32-group updates with four choices per prompt, both 100-task panels,
  and the exact 40-H100 scheduling request. The extracted Iris workspace
  validated the worker config and round-tripped the v2 binary initializer.
  The page and replay budget fits five choices at the held run's maximum
  request length. Eight fit with prompts up to 11,776 tokens and 512-token
  replies. Evaluation keeps its four-choice limit. These bounds describe
  admission capacity. H100 throughput and reward require the live run.

  Iris identifies the workspace by the SHA-256 of the uploaded archive.
  In an offline launch rehearsal, the stock SDK emitted the exact sealed
  archive and an otherwise identical 40-H100 request. A fresh checkout
  first produced a different digest. After the rehearsal verified all
  4,688 bundled files and restored their archived modification times in
  a disposable checkout, the stock SDK emitted the same request byte for
  byte. In both historical jobs, the regional controller returned the
  original bundle ID and clean source provenance while masking the
  separate Git environment stamp and embedded worker config. The
  acceptance check binds the candidate archive digest to the registered
  job and verifies the worker-written resolved config against the plan.

  A fresh attempt allocated by the normal recipe was prepared and
  reopened locally in a separate process. The authored plan, bound
  worker bytes, workspace digest, and exact parentless regional job
  ID reproduced. Read-only checks found the candidate's resolved config,
  terminal receipt, and executor marker absent. The live launch needs a
  current namespace check and an exclusive claim for that attempt.
  An offline Iris controller rehearsal dropped the reply after committing
  the job. The SDK retried the same request, and the controller kept one
  root with five tasks. Readback recovered the registered bundle and
  clean source identity. After an ambiguous reply, keep the claim and
  observe the saved regional job ID, including when the first read
  returns `NOT_FOUND`.
  [Interrupted native attempt][iris-held-native-matched-preempted] ·
  [Accepted vLLM attempt][iris-held-vllm-validation-replacement]

  The sealed request reserves 1,800 GiB of RAM and 2 TiB of ephemeral
  storage per host. Iris parses the authored size strings as binary
  units. Forwarded Finelog rows matched all 15 original task pods across
  the held vLLM control and two native attempts. The largest learner
  working-set sample was 323 GiB for the native attempt interrupted
  before its first update. The earlier one-update native arm reached
  559 GiB in the optimizer window, 593 GiB during the validation save,
  and 600 GiB during fresh restore. The completed two-update vLLM arm
  reached 608 GiB on a learner. These are minute-resolution samples of
  task-container working set. Measure the combined native source through
  ordinary per-update checkpointing and final evaluation before sizing
  a smaller memory limit.
  [Verified first-update restore][held-native-step1-restore] ·
  [Accepted vLLM attempt][iris-held-vllm-validation-replacement]

  The sealed request treats a common leafgroup as a preference. Kueue
  checks wider topology domains when a leafgroup cannot fit all five
  pods. At 18:52 UTC, eight idle hosts in one superpod met the full
  request across leafgroups of four, two, and two. At 18:53 UTC, the
  existing GPU reservations were already bound to pods and accounted
  for in the capacity scan. This was a physical fit at the time of
  inspection. By 19:04 UTC no idle host met the full request, while
  Kueue still had 173 H100s of unreserved quota. Check current node
  capacity and quota before submission. The BATCH request remains
  subject to preemption. Record the realized topology alongside the
  rollout and learner timing.
  [Kueue v0.18.3 topology scheduling][kueue-preferred-topology]

  The accepted vLLM control's archived pod placement joins to four
  learner hosts on one leafgroup and its serving host on another leafgroup
  in the same superpod. All five current Node objects and the provider's
  managed-field entries for those labels predate the run. The join uses
  original pod placement and current label values. The earlier
  native attempt's live capture places all five of its hosts on one
  leafgroup in the same superpod. Record the learner and serving domains
  separately on the sealed run when interpreting update and publication
  timings.
  [Accepted vLLM attempt][iris-held-vllm-validation-replacement] ·
  [Original native attempt][iris-held-native-matched-preempted]

  Regional telemetry recorded 32 group calls and 128 model waits in the
  first lease. Five group calls finished just after the 1,800-second model
  deadline and reported four projected tokens each. The buffer separately
  recorded five fully masked groups totaling 20 projected tokens. The two
  series have no shared group identifier. The model deadline starts before
  the Ray request is queued, and the logs show it expired before eviction.
  [Original native attempt][iris-held-native-matched-preempted]

  A local cancellation fix forwards the expired training request's
  cancellation to Ray. In the frozen CPU reproduction, the expired request
  stayed queued and was sent to the serving subprocess with a live request.
  The fix removed it before dispatch and preserved the live binary replies
  byte for byte. Already-dispatched work still drains the framed pipe. This
  removes work after a caller expires. The effect on H100 admission time
  remains unmeasured.

  With rollout spans enabled, a local timing candidate joins native training
  requests to the trainer's admission and disposition records by lease ID.
  It measures time waiting for a serving slot, time queued inside the actor,
  and wall time for the shared binary worker exchange. The caller's RPC
  duration minus the actor's duration gives the combined time outside the
  actor, including Ray scheduling and reply delivery. Each process records
  local elapsed time. The comparison needs neither synchronized host clocks
  nor replay frame changes. A loopback run through the real Ray processes and
  a committed CPU snapshot joined all four choices to one admitted group and
  traced one queued cancellation. H100 timing still needs a matched run.

  A two-CPU rehearsal combined the serving changes with the production
  rollout coordinator, task worker, and admission buffer. The reference
  and blocked-attention arms each graded two quartets per update, applied
  two Levanter optimizer updates, installed policy versions 0 through 2,
  and generated under policy 2. All 16 paired binary replays matched after
  normalizing the actor-assigned request ID. Raw serving scores, fused
  learner scores, model weights, and optimizer state matched in both arms.
  The blocked arm happened to dispatch eight choices in each training
  exchange. This fixture establishes correctness for the combined changes.

  A separate reference-mode CPU run exercised `RayPPOTrainer.train()` and
  the real Levanter learner subprocess through a local placement adapter.
  The trainer admitted two four-choice groups at each step, applied both
  optimizer updates, installed versions 0 through 2, and committed the
  ordinary native checkpoint index through step 2. This covers the
  trainer and checkpoint callbacks on the combined source.

  The same stock trainer also recovered a committed step-1 checkpoint in
  a fresh CPU learner and serving process with blocked attention enabled.
  The restored model, optimizer, training key, step, and model averaging
  matched the uninterrupted control. The second lease continued at the next
  two prompts and installed the same step-2 policy after the update. All
  16 paired binary replays matched after normalizing request IDs. The new
  attempt appended its checkpoint to the original step-1 lineage. This
  rehearsal stopped after a completed checkpoint commit. Its fixture
  supplied the sampling seeds.

  A separate native candidate stores a draw index with each selected
  prompt. Checkpointed prompts keep their index when regenerated, and
  fresh selections take the next index. The worker combines the trainer
  seed, draw index, installed policy digest and model-call ordinal, then
  gives each repetition a distinct seed. The binary replay records the
  key actually used by serving. Prompt selection pauses during the
  buffer snapshot so the saved cursor and counter cover the same draws.

  With one prompt group in flight at a time, the stock CPU trainer
  reproduced all 16 paired replays and the terminal learner and serving
  state after a fresh-process restore. Each group still sampled four
  choices. A concurrent two-group rehearsal also reproduced its 16
  replay pairs and the restored step-1 state. Its step-2 learner hashes
  differed. That rehearsal recorded reply completion order.

  A second two-group CPU run captured the learner's binary scoring and
  training requests. The continuous learner received the third prompt
  group before the fourth. The resumed learner received those groups in
  reverse order. After aligning rows by sampled choice, all packed
  arrays, including old logprobs, advantages and policy weights,
  matched byte for byte. Four fresh learners restored the same step-1
  checkpoint. For each captured training payload, one learner replayed
  its original order and one replayed it with the two groups swapped.
  Changing only group order reproduced the other run's model and Adam
  hashes in both directions. The measured gradient norm was the same.

  A native driver candidate implements that ordering at batch assembly.
  It sorts the completed selected groups by persisted draw ID, carrying
  each group's row facts and existing repetition order together. The
  sort runs after the admission window closes and counts toward batch
  assembly time. Native Levanter uses the driver batch builder.

  A stock CPU restart fixture kept two groups in flight with four
  choices each. The fixture let both groups generate and grade before
  committing them to the buffer in opposite orders at step 2. Both
  learners received the same ordered scoring and training arrays and
  the same replay frames after normalizing the actor-assigned numeric
  request ID. The resumed learner matched the continuous run's step-2
  model, optimizer, training key, step, averaging state and installed
  policy. Under the same fixture, the previous source fed the learner
  opposite group orders and produced different model and optimizer
  hashes. This tests exact restart with the same selected groups.
  Measure the changes to training draws and batch order against the completed
  H100 comparison in a separate arm.

  The integrated candidate combines persisted draw IDs and deterministic
  batch ordering with evaluation stopping at a complete tool call when
  parallel tool calls are disabled. The two-group CPU restart fixture
  passed on the combined source. A fresh serving process loaded the
  exact step-2 published policy. Its typed evaluation response kept the
  complete tool call's token and score prefix and preserved the ordinary
  sibling response. The same training request returned byte-identical
  replay before and after the mixed evaluation batch.

  The stock launcher rendered `2026.11.23.91` offline with the same inputs,
  two 100-task panels, checkpoint cadence, and five eight-H100 hosts as
  the sealed `2026.11.23.90` run. Two fresh renders bound the same digest
  from the stock Iris workspace archive. With sockets blocked, the
  extracted rollout and serving tokenizer loaders read the same locally
  pinned production assets and matched vocabulary and decoding. The
  candidate changes training draws, batch order, and evaluation
  stopping, so its live comparison needs a separate arm.

  A separate offline rehearsal replayed the archived CPU checkpoints
  through stock retention. It reproduced the archived step-2 index bytes and joined the
  retained index to the same generation and four object metadata records.
  Capture the original step-1 index, the two-entry index after step 2,
  and the terminal retention index in order during the H100 run. The
  regional collector checks an object's ETag before and after a
  bounded conditional GET. It reads only the index and policy metadata
  bodies and takes HEAD records for the four checkpoint objects. Its
  offline tests passed all three captures through the custody verifier
  and exercised the routed S3 client's conditional requests. Capture
  the two-entry index before invoking the direct-root terminal
  attachment, which calls stock retention. This ordering preserves the
  original commit even when the last save completes after evaluation.

  A fresh-start diagnostic can skip the initial evaluation callback and
  keep the final panel. The callback ran for 26m55s in the interrupted
  attempt. The production worker composer reproduced the archived
  configuration and rendered the diagnostic configuration offline. The
  verifier rechecked all 100 archived scores and the 502-tensor policy
  manifest. It permits changes only to the initial callback flag, the
  exact policy-0 pin, and the artifact and attempt paths. At runtime the
  pin requires the learner hosts to agree with the archived weight digest
  and model binding before serving can install policy 0 and open lease 1.
  The normal native launcher exercised that guard locally using committed
  CPU snapshots. It rejected a wrong binding before serving started and
  generated under both acknowledged policies as the lease advanced.

  Dispatch for this diagnostic would also need to prove the new checkpoint
  namespace is empty. The archived baseline stays with the original attempt.
  Skipping evaluation changes serving warmup, and training draws vary with
  request IDs. The diagnostic would measure a cold training start. The held
  comparison still requires its initial and final evaluations in the same
  attempt and remains open.

  A CPU probe ran the normal step-end callback with one and two native
  learner processes, local storage and a local Ray dispatch adapter. It
  restored exact learner state and the next prompt selection from the
  committed index. The next update matched the uninterrupted reference,
  including its model fingerprint and serving digest. An interrupted
  sidecar write left the previous index intact. The candidate also passed
  17 focused CPU tests. A separate candidate raises the ordinary
  TensorStore barrier from 30 to 45 minutes for the pilot's 60-minute
  worker command budget. Its authored workload and per-update checkpoint
  cadence match the initial recovery candidate. Six exact-source CPU
  cases passed, including injected commit failures that left completion
  metadata absent. The H100 save still needs regional validation. A
  restart can repeat the pilot's deliberate validation round trip on the
  next update because that flag lives in the process.
  [CPU recovery source and test results][held-native-recovery-cpu] ·
  [Scoped checkpoint deadline proof][held-native-recovery-deadline]

  Wait for the ordinary native index to commit a checkpoint generation
  before counting an update as recoverable. The policy can already be
  serving while the trainer and rollout sidecars and index are uploading.
  Preserve the first index and receipts before final retention can prune the
  generation. If Iris evicts the group, restart the same artifact with a
  fresh attempt ID and automatic latest-checkpoint recovery. This preserves
  the rollout cursor. The official cross-artifact initializer path starts
  the dataset from the beginning. Require the successor to load the exact
  indexed checkpoint, restore the trainer and rollout state, and reinstall
  the same policy digest before training resumes. The local artifact
  rehearsal retained the failed root's attempt record, assigned a fresh
  root ID under the same artifact, linked it to the first committed
  checkpoint and finalized the second generation. The staged recovery
  command pins the five predecessor tasks and the committed index's ETag,
  submits one successor under the same artifact, and verifies that step 2
  descends from the pinned checkpoint. It passed 14 offline tests.
  [Same-artifact continuation rehearsal][held-native-continuation] ·
  [Step-1 recovery command and tests][held-native-step-one-operator]

  Prefer an uninterrupted allocation to measure native update two at
  steady state. If the comparison spans attempts, report requeue, restore,
  initial publication and the resumed evaluation separately. Archive each
  optimizer update and its preceding admission window from the attempt that
  ran it. Compare the core update wall alongside the full step wall and
  report checkpoint time separately. Use policy 0 for the baseline panel
  and policy 2 for the terminal panel, including when a restart from a
  committed step 2 runs the final evaluation without another optimizer update.
  The paired audit replayed the accepted vLLM receipt byte for byte and
  independently verified all four learner logs from the interrupted native
  update. The audit binds the two training updates and their elapsed
  times to the original attempts, including when the final evaluation
  runs on a separate attempt.
  [Matched-pair acceptance checker and proof][held-native-pair-acceptance]

  The prepared `2026.11.23.90` run now has a local verifier for a
  preemption after the step-1 checkpoint commits. It binds both captures to
  the sealed source and Iris workspace. The verifier checks that the
  successor restores step 1 and reinstalls a policy with the same digest
  before admitting step 2. Keep the predecessor's resolved launch before
  its successor overwrites the shared URI. Capture the step-2 index with entries `[1, 2]` before
  terminal retention prunes it to `[2]`. The verifier reads panel 0
  from the predecessor and panel 2 from the successor. It reports
  requeue, restore, and the resumed policy-1 evaluation separately
  from update timing. The local test reverified the frozen vLLM arm
  and exercised checkpoint lineage with synthetic native captures.
  The matched result requires the original H100 evidence from both
  native updates.

  A CPU rehearsal interrupted final evaluation after step 2 had committed.
  A fresh learner restored identical state, and fresh serving installed
  the same policy-2 digest and reproduced the interrupted sample. The
  evaluator completed the two-prompt fixture without opening a training
  lease or running an optimizer update. Its terminal receipt retained the
  original checkpoint index, ETag and owner. The audit checks that
  lineage before accepting a recovered 100-task panel. The native H100
  comparison completed in a single allocation. Recovering the final
  evaluation after a regional H100 interruption remains unverified.
  [Final evaluation recovery proof][held-native-final-panel-recovery]

  The earlier prompt-aligned replacement and the held vLLM control both
  completed their first admission windows without tool calls. The native
  worker served batches of up to four requests serially. Its logs combine
  queueing with model execution. On one eight-H100 host, profile an identical
  captured prompt
  group cold and warm, then repeat with a bounded queue. Time enqueue to
  dispatch, record realized batch size, and measure prefill, decode and
  replay serialization. The serve-only fixture uses four matched short
  evaluation prompts at the 512-token training response cap. An exact
  two-CPU test reconciled all 16 typed frames and showed that the first
  repeat compiled another prefill executable. The initial H100 attempt
  stopped during policy installation because the snapshot fixture lacked its
  compute dtype. The retry used the verified BF16 metadata and completed
  on eight H100s. Policy installation took 174.8 seconds. The worker checked
  16 binary replay frames and returned 512 sampled tokens per request. Cold
  and repeat tokens and sampling logprobs matched byte for byte.

  Cold batch service took 79.3 seconds and the first repeat took 67.9
  seconds. Prefill compiled in both batches. Each subsequent batch took
  58.2 seconds, with 57.0 seconds in synchronized decode and 1.1 seconds
  in prefill. Encoding and writing the binary reply took under 10 ms per
  batch, including encoding the replay frames. With eight requests queued,
  the second four waited a median of 58.2 seconds behind the first four.
  Decode and the four-request batch limit account for the delay on this
  fixture. The probe times synchronized calls to a single serving process.
  The matched run measures admission and update times end to end, including
  Ray, grading, tools, the learner and policy publication.
  [Scoped serving diagnostic and CPU proof][held-native-admission-probe] ·
  [Fixture repair proof][held-native-admission-dtype] ·
  [Successful H100 attempt][iris-held-native-admission-retry] ·
  [Verified H100 profile archive][held-native-admission-h100]

  The worker derives its four-request limit from the wire request count,
  worst-case replay frame size and full-context KV reservation. Its eight
  logical slots share 768 pages of 128 tokens. Eight requests using the
  short probe prompts and a 512-token response cap need 40 pages. Eight
  full 24,576-token contexts need 1,536 pages. Use the prompt lengths and
  response caps to bound the combined KV pages and binary reply. Split
  prefill into packs within its token budget. On two CPU devices, the small
  model filled all eight slots over two prefills and returned the same
  sampled tokens, selected experts and sampling logprobs as two batches of
  four. The frozen wire caps batches at four requests. The CPU test raised
  that count locally and confirmed that eight full-context frames exceed
  the 65 MiB envelope. Decode already packs eight positions.
  [Eight-slot CPU replay and wire checks][held-native-eight-slot-cpu]

  An isolated prototype passes the installed limits to the actor over
  typed IPC. The actor dispatches the longest FIFO prefix within the slot,
  KV and reply budgets. The worker checks the batch again before generation.
  On two CPU devices, eight short requests and a mix of two full-context
  and six short requests each ran in one batch. Eight full-context
  requests split into two batches of four. All 24 replay frames matched
  the frozen four-request actor byte for byte. A direct oversized request
  to the worker was rejected, and the worker returned another valid frame
  under the same policy.
  [Prototype source and CPU receipts][held-native-length-aware-cpu]

  On H100, compare a warmed batch of eight active requests against the
  measured 116.4 seconds for two warmed batches of four. Pin the short
  inputs and seeds and check replay, occupancy and peak memory. The
  earlier reference-attention probe used four requests. The completed
  backend pilot allowed eight training requests and four evaluation requests.

  The warmed 57.0-second device decode measurement includes attention,
  expert computation and route capture. The engine already runs up to
  32 generation rounds per compiled call. Within each round, the CUDA
  BF16 reference attention reads one KV page at a time. The opt-in
  `reference_blocked` prototype gathers four pages and batches their FP32 query-key
  products for static packs of up to 32 queries. It applies the same
  ordered FP32 softmax and value recurrence. Large prefill packs follow
  the reference path. Lowering the eight-slot H100 geometry verified the
  bounded gather and four-page product. For the archived short inputs,
  the page geometry gives 615 outer block iterations per layer in place
  of 1,692 single-page iterations across 512 decode positions. The last
  block can include masked padding.

  Two real CPU serving processes used identical weights and their own
  model configuration digests. Ten typed replay frames matched in
  every other field, including exact tokens, positions, routes, sampled
  scores and an early stop. Measure synchronized decode time and peak
  device memory on H100 before enabling the prototype for the inventory
  comparison.
  [Page batching source, binary frames and lowering][held-native-page-block-cpu]

  A two-device CPU run applied both serving prototypes together and
  completed two generate, grade, update and publish cycles. Each update
  consumed two independently sampled quartets and 128 response targets.
  The optimized actor dispatched one batch of eight, and the frozen actor
  dispatched two batches of four. Both used production RLOO-N and the
  fused learner scorer, and installed the same three weight identities.
  The serving processes rejected stale requests and returned a fresh
  sample under policy 2. Sixteen training replay frames and the terminal
  frame matched after accounting for each configuration digest. The
  independent learner scores stayed within 0.00048 of serving scores
  with zero PPO clip crossings.

  An isolated port exposes `inference_attention_implementation` in
  `generator.native_snowball.engine_config`. The normal artifact launcher
  carries the choice over the existing typed initialization message.
  The initial and replacement engines apply the serving-only option
  after verifying the snapshot and preserve the original model binding.
  On local Ray, omission and `reference_blocked` both loaded the same
  committed policies 0, 1 and 2. All 12 corresponding binary replies
  matched byte for byte, including their configuration and weight
  identities. Both actors rejected the old policy after each swap.
  Worker traces confirmed the selected decode mode.

  A staged single-host H100 profile isolates the kernel and batch-size
  effects on the prior BF16 snapshot and exact short prompts. The two
  attention modes use sequential workers on the same host. Each mode
  warms both geometries and measures 4+4 and 8 in alternating order.
  The profiler records synchronized prefill and decode, IPC wall time
  and per-device lifetime allocator peaks. Eight requests reserve 40 of
  768 KV pages under the production admission budget. The snapshot's
  502-entry tensor manifest matched its original receipt. A CPU rehearsal
  verified all 64 timed binary replies, detected compilation during
  warm-up and found none in the measured waves. Both H100 modes completed
  their batches. The byte-exact audit rejected a replay frame for the first
  request. The audit compared all batch geometries and both attention
  modes together, so the first differing field and its cause remain open.
  The harness retained aggregate timing only after a passing audit.
  Preserve per-case timing and frame deltas for the next replay check.

  A separate eight-H100 diagnostic keeps the existing PPO
  thresholds. It binds the serving head observations and ordinary fused
  scores to the same typed frames from one original attempt. The first
  diagnostic installed both policies and started the full-context
  group. Its CPU coordinator was preempted at the same time as the
  held native coordinator. The replacement stopped at the 1,800-second
  policy-2 install deadline. The original task archive contains both
  tool-turn frames and the install timeout. The full-context group and
  scorer never ran. A new eight-H100 retry uses 3,600 seconds for the
  live policy-2 install. It resets the original deadline before the next
  RPC. The tool replay, generation bounds and PPO thresholds carry over.
  The retry verified policy 2 in 154.7 seconds and returned all four
  full-context trajectories after 1,492.1 seconds. The optional head
  diagnostic rejected one predictor with conflicting score observations.
  The short tool-turn scorer passed all 167 sampled targets and masked
  46 tool observation tokens. The ordinary full-context scorer found
  two ratios below 0.8 and one above 1.2, with extrema 0.5553 and 1.2350.
  Independent recomputation from the original arrays reproduced all three.
  The four replay frames and raw score arrays also matched an earlier
  failed attempt byte for byte. A scratch CPU diagnostic now links the
  head vector and score to the exact decoder output row used for the
  emitted token. It preserved all four ordinary frames and their scores.
  The first H100 attempt of that diagnostic lost its host to a higher
  priority workload during startup. The byte-identical diagnostic retry
  completed the live swap and exact cold replay. It decoded four
  24,576-token trajectories in 1,498.0 seconds. The ordinary scorer passed all
  32,768 targets with ratios between 0.9425 and 1.1554 inclusive. The
  real tool turn passed all 364 sampled targets and masked 46 observation
  tokens. The same-attempt observer matched both target tokens and raw
  scores to the emitted frames. A shared head projection reproduced
  0.0413 of the 0.0592 absolute score difference at the previously
  unstable predictor. This identifies a pre-head contribution on that
  attempt. The earlier failures still motivate repeated qualification
  before promotion.
  [Verified score evidence][native-full-context-ratio-evidence] ·
  [Diagnostic preemption evidence][native-head-preemption-evidence] ·
  [Emitted-row diagnostic retry][iris-native-emission-retry] ·
  [Verified passing diagnostic][native-passing-diagnostic-evidence]
  [Original calibration][iris-native-serving-eval-first] ·
  [Revised calibration][iris-native-serving-eval-retry] ·
  [Verified calibration archive][native-serving-eval-archive] ·
  [First matched native child][iris-held-native-validation-retry] ·
  [Matched native replacement][iris-held-native-validation-replacement] ·
  [Same-attempt score diagnostic][iris-native-same-attempt-score] ·
  [Original install-timeout archive][native-same-attempt-timeout-archive] ·
  [Scoped install retry][iris-native-install-deadline-retry]

  Both plans use a 24,576-token serving limit and a 16,896-token request
  limit. Native uses the cache geometry from the 67B live-swap run. Use the same learner, selection and fixed panel to
  compare the completed attempts.
  [First held vLLM coordinator][iris-held-vllm-pilot] ·
  [Preempted replacement child][iris-held-vllm-replacement] ·
  [Timing-instrumented vLLM coordinator][iris-held-vllm-profile] ·
  [Unstarted first direct job][iris-held-vllm-direct] ·
  [Corrected held vLLM job][iris-held-vllm-direct-corrected] ·
  [Third held native GPU child][iris-held-native-third]

  [Earlier handoff][launch-handoff]
  The primary fixed panel is `eval-lite@2026.11.17.1`. The published 100-task
  Parquet has SHA-256
  `557733f6d4f7f42c5c5072a29325da03d489a5fd77b456054bc34003b2f16e95`.
  Require all four attempt-scoped panels at steps 0 and 2, with each task
  ID, prompt token sequence and score reconciled against the successful
  attempts. For each native panel, join the driver's evaluation policy
  ID to the install and publication for that step. The typed serving
  client checks that each reply echoes that ID. For a resumed run,
  check the successor's startup evaluation binding too.

  Match the model digest, layout digest and addressable bytes before
  the first update on each of the four logical JAX learner ranks.
  Collect logs from the successful rank-zero attempt in each Iris GPU job.
  Match the receipts and panel paths to those attempts, and recheck the
  attempt identities after collection. Compare step 2 separately because
  step 1 includes the learner
  subprocess checkpoint round trip. Both maintained arms log
  `timing/rollout_admission_window_seconds`
  and `rollout/admission_window_groups` for fresh, full batches at zero
  staleness. The buffer's clock measures from policy lease opening through
  final group admission. Divide the sum of elapsed seconds by the sum of
  admitted groups across the same measured updates. Report the trainer's
  `timing/wait_for_generation_buffer` separately as buffer wait, along
  with `timing/step` and `timing/sync_weights`. The final comparison
  reports both per-update windows, total lease seconds, total admitted
  groups, and the pooled rate. The frozen vLLM run admitted 64 groups
  in 21.185 lease seconds across updates 1 and 2, or 0.331 seconds per
  group. The driver log timestamps also include work after admission
  and are recorded separately. For a resumed native attempt, label
  update 2 as the successor's first training window and report requeue,
  restore, and resumed evaluation separately. Use trajectory timings
  to distinguish generation, tool execution and grading.

  The completed native H100 arm used reference-blocked attention. All five
  original tasks succeeded. The pre-update model digest, layout and
  addressable bytes matched the vLLM control on all four learners. The
  same native attempt applied both updates, verified exact learner restore,
  installed policy versions 0 through 2, and committed both checkpoint
  generations. All 100 prompt token sequences matched across both backends
  and both evaluation steps. [Accepted vLLM control][iris-held-vllm-validation-replacement]

  | Measurement | Levanter/vLLM | Levanter/Snowball |
  | --- | ---: | ---: |
  | Update 1 admission, seconds per group | 0.357 | 36.492 |
  | Update 2 admission, seconds per group | 0.305 | 34.975 |
  | Pooled admission, seconds per group | 0.331 | 35.734 |
  | Update 1 wall, seconds | 1,544.8 | 3,383.0 |
  | Update 2 wall, seconds | 183.8 | 1,667.9 |
  | Update 2 wall excluding checkpoint time, seconds | 183.8 | 1,481.9 |
  | Policy-0 mean verifier score | 0.21 | 0.10 |
  | Policy-2 mean verifier score | 0.20 | 0.08 |

  Each arm admitted 64 groups at zero policy staleness across two updates.
  The lease windows totaled 21.185 seconds for vLLM and 2,286.953 seconds
  for native. The native arm checkpointed after each update. Its learner
  and serving hosts shared a leaf group. The vLLM serving host occupied a
  different leaf group from its learners. The table covers one attempt per
  backend.

  Both arms trained on 256 single-turn, tool-free trajectories. Native
  admitted 113,218 response tokens and vLLM admitted 110,787. The native
  trace joined all 256 model calls to 34 serialized serving exchanges.
  Thirty exchanges ran at the eight-request cap. The worker calls took
  2,283.266 seconds, or 99.84% of the 2,286.953-second admission time.
  Native dispatch and return each took under 0.2 ms at the median. The
  earlier four-request H100 probe assigned 57.0 of 58.2 seconds to
  synchronized decode. The trace of the matched pilot times the whole
  worker call.

  Across all eight native serving H100s in the second admission window,
  sampled GPU utilization averaged 93.8%, while the tensor pipelines
  were active 0.13% of the time. The reference-blocked path batches FP32
  query-key products over four pages and processes the softmax and value
  recurrence one page at a time. The vLLM logs reported FlashAttention 3
  and Triton MoE. The device counters cover the native serving process;
  they do not distinguish individual kernels.

  Native serves one batch at a time and returns after the batch drains.
  vLLM continuously schedules requests across eight data-parallel ranks
  and reported 127 running requests at peak in the second update. Both
  warm optimizer calls took about 48 seconds. Of the second update's
  1,484-second wall gap, 1,109 seconds came from rollout admission,
  186 from publication, and 186 from checkpointing.

  The AgentX H100 series on the inference branch used a different
  checkpoint and a 64K serving workload at concurrency 128. Its first
  fixed-batch arm achieved 47.82 output tokens per second. Continuous
  admission with a larger KV cache reached 164.26. Later changes added
  per-layer KV buffers, smaller expert kernels, shard-local sampling,
  and two four-H100 replicas. The singleton decode repeat achieved 621.18
  output tokens per second, against 590.24 for compiled plain vLLM on the
  same AgentX workload. The reference-blocked RL worker admitted 49.51
  response tokens per second. The frozen-policy test below measures
  transfer to the RL request burst.
  [AgentX H100 series][agentx-h100-serving]

  The native training sampler sorts the full vocabulary twice even at
  `top_p=1`. In a CPU lowering of the eight-position, 128,256-token
  vocabulary sampler, omitting nucleus filtering removed both sorts and
  preserved sampled IDs and scores on the same inputs and keys. The
  inference branch already chooses the full-distribution fast path before
  mapping the sampler over positions. Measure its effect on H100 with
  the paged attention and expert kernels held fixed. [Batched sampler][agentx-batch-sampler]

  Tool use accounts for seven points of the eleven-point baseline score
  gap and five points of the twelve-point policy-2 gap. All 11 native
  tool-use responses reached the 512-token cap in each panel. A replay
  that stopped at the first complete tool call recovered three correct
  calls per panel. The same evaluator left eight and nine points of gap
  at steps 0 and 2. At policy 2, seven of eight non-tool tasks that passed
  only for vLLM hit the native cap and ended normally for vLLM. Compare
  greedy outputs and teacher-forced raw scores on the same policy-0
  weights, then vary sampling, tool stopping, and the reply cap separately.

  A frozen-policy test on one eight-H100 host served the original 128
  training requests from 32 prompt groups through two EP4 replicas. Each
  replica verified all 5,952 original weight shard hashes. The first
  verification path reached the 500 GiB host limit while copying full
  JAX arrays to NumPy. Startup memory peaked at 92.7 GiB after changing
  verification to stage one shard at a time in pageable memory.

  Each replay mode completed 64 groups across two timed bursts. Route
  capture produced 1,783 output tokens per second and 0.950 groups per
  second, compared with 1,751 and 0.925 with capture off. The 1.8% token
  throughput difference was smaller than the spread between repeats.
  Exact completions matched for 3 and 5 of 128 requests across modes
  and for 6 and 7 of 128 within each mode. Exact raw-score matches
  numbered zero in both cross-mode comparisons. Use fixed batches and
  teacher-forced continuations to isolate score and router differences. The AgentX
  singleton path requires replay to be disabled.

  The separate 11-prompt tool-stop test kept route capture off. Stopping
  at the first complete tool call raised correct calls from 0 to 5 and
  reduced output from 5,632 to 1,912 tokens. Ten responses ended at a
  complete call. Three preserved the baseline token sequence through
  the stop. Reconcile position and route fields against the selected
  learner's replay input before evaluating reward.
  Use plain compiled vLLM for the backend comparison and vLLM with four
  DSpark proposals for the cutover comparison. Gate the Megatron canary on
  the live receiver, verified publication, and invalidation of old caches.
  [Route-capable AgentX source][agentx-route-capable] ·
  [Native checkpoint publication][native-checkpoint-publication] ·
  [Megatron-to-Levanter handoff][megatron-levanter-handoff] ·
  [Singleton replay guard][agentx-singleton-replay] ·
  [DSpark comparison][dspark-h100-control]

The prior FP8 comparison reported a 13% reduction in mean decode time
and 5.7 minutes longer to reach a mean verifier score of 0.2901, the
BF16 arm's final score. The archived trainer events reproduce 37.0
minutes of buffer wait for BF16 and 43.1 for FP8 through update 15. The
initial and update-15 dumps contain the same 100 task IDs across both
arms. [Prior timing comparison][timing]

## Open decisions

- Agree on fixed-panel reward and update time or cost margins for the
  maintained Levanter/vLLM build. Use the frozen pair as the baseline.
  Evaluate bounded source hashing on the maintained H100 path and track
  hash wait, receiver replay and total publication time. Change the
  learner default once the maintained build meets the margins and passes
  H100 recovery and weight publication.
- Keep native serving opt-in until the fixed-checkpoint shape gate and the
  five-host deployment pass. Use the fresh vLLM baseline to agree on
  sustained group throughput, update time, reward and recovery criteria
  before promoting it. Decide on serving topology after measuring the
  pinned checkpoint and expected concurrency.
- Decide whether to carry the typed learner batch envelope across Ray as
  well as the subprocess boundary. The current binary replay frame already
  crosses both without re-encoding.
- Decide which checkpoint consumers need HF export before switching the
  default learner. Identify the export artifact and verify it with those
  consumers.

[learner-status]: https://github.com/marin-community/marin/blob/b8aa9e2af13d991b45c64e036cd0b4885222a118/experiments/post_training/inventory/levanter-quick-loop-status.md
[kueue-preferred-topology]: https://github.com/kubernetes-sigs/kueue/blob/v0.18.3/site/content/en/docs/concepts/topology_aware_scheduling.md#user-facing-apis
[launch-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fhandoff%2F2026.11.21.1%2Fquick-loop-launch-handoff.tar.zst
[iris-smoke]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.5-coordinator-2ea612c2%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.5-a8ae164919e7
[iris-smoke-rerun]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.8-coordinator-5050ee06
[iris-smoke-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.9-coordinator-151d77eb
[iris-smoke-retry-child]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.9-coordinator-151d77eb%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.9-aa8d6bc47005
[iris-smoke-layout]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.10-coordinator-c66407b6
[iris-smoke-layout-child]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.10-coordinator-c66407b6%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.10-a954f7d0dae3
[iris-smoke-optimizer]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.11-coordinator-3ba28fae
[iris-smoke-optimizer-child]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.11-coordinator-3ba28fae%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.11-aa5a21c03703
[iris-smoke-readiness]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.12-coordinator-8b936a37
[iris-smoke-readiness-child]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.12-coordinator-8b936a37%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.12-a2e820308486
[iris-smoke-checkpoint]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.13-coordinator-22237d08
[iris-smoke-checkpoint-child]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.13-coordinator-22237d08%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.13-a6da35eb027d
[iris-smoke-shutdown]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.14-coordinator-93324d26
[iris-smoke-shutdown-child]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.14-coordinator-93324d26%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.14-ae0b645fbb4e
[iris-smoke-nccl-headroom-child]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.15-coordinator-69a6cbbb%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.15-a453f4b331ee
[iris-smoke-preallocated-child]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.16-coordinator-317129fa%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.16-ac3d40727bb4
[iris-smoke-fingerprint]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.17-coordinator-88b26859%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.17-a9c7f68045d0
[iris-smoke-replay-diagnostic]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.18-coordinator-631ad9f7%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.18-a4e74365b4d3
[iris-smoke-canonical-replay]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.19-coordinator-b37693ff%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.19-abe09138e0b3
[iris-smoke-terminal-cleanup]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.20-coordinator-8633ad51%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.20-a747a82c7558
[iris-smoke-terminal-cleanup-replacement]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.20-coordinator-8633ad51%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.20-acbbca3b8eb3
[iris-matched-megatron]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.6-coordinator-46214fd0%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.6-a30288e64eea
[iris-matched-megatron-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.6-coordinator-46214fd0%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.6-a361bc746b85
[iris-matched-megatron-third]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.6-coordinator-46214fd0%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.6-ab4ae1aae5e9
[iris-matched-levanter]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.7-coordinator-6d5448c8%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.7-af8929ab8ff2
[iris-matched-levanter-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.7-coordinator-6d5448c8%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.7-ae4bda9bd4c7
[matched-final-evidence]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.22.1%2Fmatched-h100-25-update-comparison.json
[iris-matched-levanter-third]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.7-coordinator-6d5448c8%2Fusers-dlwh-checkpoints-snowball-inventory-2026.11.20.7-af6dc8d549b1
[iris-native-fused-gate]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122e
[iris-native-live-score]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122g
[iris-native-allocator-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122h
[iris-native-cache-ownership]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122i
[iris-native-tool-scorer]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-original-tool-scorer-20261122a
[iris-native-repeat-prefill]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-repeat-prefill-20261122a
[iris-native-b-scorer]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-original-b-scorer-20261122b
[iris-native-ordered-gather]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122j
[iris-native-score-profiles]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-saved-24k-score-profiles-20261122a
[iris-native-attention-profiles]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-saved-24k-attention-20261122a
[iris-native-attention-fp32]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-saved-24k-attention-fp32-20261122a
[iris-held-vllm-pilot]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-e8ff805-in-424011b4-2026.11.22.1-coordinator-c033eabc
[iris-held-vllm-replacement]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-e8ff805-in-424011b4-2026.11.22.1-coordinator-c033eabc%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-e8ff805-in-424011b4-2026.11.22.1-ae1c053f89fb
[iris-held-vllm-profile]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-38f335b-in-424011b4-2026.11.22.1-coordinator-5490d623
[held-capacity-source-proof]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.22.1%2Fheld-capacity-source-delta-comparator.tar.zst
[native-head-first-archive]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Ftmp%2Fttl%3D7d%2Firis%2Ftask-outputs%2Fdlwh%2Fnative-regional-native-fused-20261122k%2Fnative-native-fused-20261122k-tool-fused%2F0%2F709a5c5cf7fb819e%2Foutputs.tar.zst
[iris-held-vllm-direct-corrected]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-38f335b-in-424011b4-2026.11.22.1-acd9904b74c0
[iris-held-vllm-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-52324ce-in-424011b4-2026.11.22.2-coordinator-3c9d26e6%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-52324ce-in-424011b4-2026.11.22.2-a08d6abd1b30
[iris-held-vllm-validation-timeout]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-aabb141-in-424011b4-2026.11.22.3-coordinator-c303053f
[iris-held-vllm-validation-replacement]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-aabb141-in-424011b4-2026.11.22.3-coordinator-c303053f%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-aabb141-in-424011b4-2026.11.22.3-ab5ce0f492e1
[iris-native-serving-eval-first]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-serving-67b-eval-20261123a
[iris-native-serving-eval-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-serving-67b-eval-20261123b
[native-serving-eval-archive]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fnative-serving-67b-eval-20261123b%2Foriginal-task-outputs.tar.zst
[held-vllm-rendezvous-proof]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-vllm-a08d6abd1b30%2Ftensorstore-rendezvous-terminal-proof.json
[held-vllm-accepted-audit]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-vllm-ab5ce0f492e1%2Faccepted-vllm-arm.json
[iris-held-native-validation-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-aabb141-in-424011b4-2026.11.22.3-coordinator-1f166fa9%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-aabb141-in-424011b4-2026.11.22.3-a9875414cfa5
[iris-held-native-validation-replacement]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-aabb141-in-424011b4-2026.11.22.3-coordinator-1f166fa9%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-aabb141-in-424011b4-2026.11.22.3-aa86eb747b53
[iris-native-same-attempt-score]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122m%2Fnative-native-fused-20261122m-tool-fused
[native-same-attempt-timeout-archive]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fnative-fused-20261122m%2Fb14120ce5f263e2c-original-task-outputs.tar.zst
[iris-native-install-deadline-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122n
[checkpoint-first-cpu-proof]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-checkpoint-first-cpu-proof.tar.zst
[held-native-baseline-prompt-audit]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-aa86eb747b53%2Fverified-baseline-prompt-mismatch.json
[native-full-context-ratio-evidence]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fnative-fused-20261122n%2Fverified-ordinary-fused-evidence.json
[native-head-preemption-evidence]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fnative-fused-20261122o%2Fverified-preemption.json
[held-native-prompt-parity-proof]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-079e4e4%2Fprompt-parity-proof.tar.zst
[iris-held-native-prompt-aligned]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-9a7e3ee-in-424011b4-2026.11.22.3-coordinator-f0713ab7%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-9a7e3ee-in-424011b4-2026.11.22.3-a6705fa4611d
[iris-held-native-prompt-aligned-replacement]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-9a7e3ee-in-424011b4-2026.11.22.3-coordinator-f0713ab7%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-9a7e3ee-in-424011b4-2026.11.22.3-afb17423cacb
[held-native-live-prompt-parity]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-afb17423cacb%2Fverified-baseline-prompt-parity.json
[held-native-step1-restore]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-afb17423cacb%2Fverified-step1-four-fresh-learners.json
[held-native-interrupted-result]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-afb17423cacb%2Fverified-interrupted-held-native.json
[held-native-recovery-plan]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fverified-recovery-plan.json
[held-native-recovery-cpu]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fordinary-checkpoint-cpu-proof.tar.zst
[held-native-recovery-root]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fparentless-regional-launch-proof.tar.zst
[held-native-direct-operator]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-first-attempt-7de5313.tar.gz
[iris-held-native-matched-preempted]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-7de5313-in-424011b4-2026.11.23.2-a7416ff4d7c8
[held-native-matched-current-startup]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-matched-a7416ff4d7c8-startup-proof.tar.zst
[held-native-matched-current-config]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-matched-a7416ff4d7c8-effective-workload.tar.zst
[held-native-matched-current-shards]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-matched-a7416ff4d7c8-shard-load-proof.tar.zst
[held-native-matched-current-policy0]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-matched-a7416ff4d7c8-policy0-proof.tar.zst
[held-native-matched-current-panel0]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-matched-a7416ff4d7c8-policy0-panel-proof.tar.zst
[held-native-matched-current-preemption]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-matched-a7416ff4d7c8-preempted-proof.tar.zst
[held-native-continuation]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-direct-same-artifact-rehearsal.tar.zst
[held-native-step-one-operator]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-same-run-latest-7de5313.tar.gz
[held-native-pair-acceptance]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-held-pair-acceptance-proof.tar.zst
[held-native-final-panel-recovery]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-final-panel-recovery-proof.tar.zst
[held-native-recovery-deadline]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fordinary-checkpoint-deadline-proof.tar.zst
[held-native-admission-probe]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Foffline-native-admission-probe.tar.zst
[held-native-admission-dtype]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-admission-dtype-repair-proof.tar.zst
[iris-held-native-admission-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-admission-67b-h100-20261123b
[held-native-admission-h100]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-admission-h100-profile-proof.tar.zst
[held-native-eight-slot-cpu]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-eight-slot-cpu-proof.tar.zst
[held-native-length-aware-cpu]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-length-aware-batch-cpu-proof.tar.zst
[held-native-page-block-cpu]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fnative-blocked-paged-decode-cpu-proof.tar.zst
[held-publication-overlap]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fverified-publication-overlap-proof.tar.zst
[iris-native-emission-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122p
[native-passing-diagnostic-evidence]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fnative-fused-20261122p%2Findependently-verified-passing-diagnostic.json
[iris-native-head-retry]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122l%2Fnative-native-fused-20261122l-tool-fused
[iris-native-head-comparison]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122k%2Fnative-native-fused-20261122k-tool-fused
[iris-held-native-repaired]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-ac90c03-in-424011b4-2026.11.22.1-coordinator-f7d86071
[iris-held-native-third]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-38f335b-in-424011b4-2026.11.22.1-coordinator-eaec3017%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-native-pilot-38f335b-in-424011b4-2026.11.22.1-a830de53136f
[iris-held-vllm-direct]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fusers-dlwh-checkpoints-snowball-rl-levanter-inventory-vllm-pilot-38f335b-in-424011b4-2026.11.22.1-a028a6627d0d
[native-full-capture-archive]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-08a%2Ftmp%2Fttl%3D7d%2Firis%2Ftask-outputs%2Fdlwh%2Fnative-regional-native-fused-20261122i%2Fnative-native-fused-20261122i-tool-fused%2F0%2F2d4106a1e0593f89%2Foutputs.tar.zst
[native-paged-moe]: https://github.com/marin-community/marin/blob/fcc3167fef231ed2dacdd9e21cd48a564668e7ee/lib/levanter/src/levanter/models/snowball.py#L1932-L1943
[jax-allocator-options]: https://docs.jax.dev/en/latest/gpu_memory_allocation.html
[iris-native-conversion]: https://iris-cw-us-east-02a.oa.dev/#/job/%2Fdlwh%2Fnative-regional-native-fused-20261122d%2Fnative-native-fused-20261122d-convert
[native-typed-conversion-receipt]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-tool-67b%2F2026.11.22.1%2Fnative-fused-20261122d%2Fevidence%2Ftyped-snapshot-receipts.json
[xla-learner-allocator]: https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/pjrt/gpu/se_gpu_pjrt_client.cc#L1343-L1380
[xla-learner-allocator-defaults]: https://github.com/openxla/xla/blob/131bf41acb4650e4391a640c3f1859c1c86ad74b/xla/debug_options_flags.cc#L308-L315
[h100-init-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.8%2Fobserved-failure.json
[frozen-validation-uid]: https://github.com/marin-community/marin/blob/6936528c55df0212ec8ccd1fa619361863b4c420/vendor/skyrl/skyrl-train/skyrl_train/dataset/dataset.py#L62-L105
[h100-retry-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.9%2Fh100-learner-retry.tar.zst
[h100-layout-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.9%2Fobserved-failure.json
[h100-layout-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.10%2Fh100-parameter-layout-retry.tar.zst
[h100-train-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.10%2Fobserved-failure.json
[h100-optimizer-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.11%2Fh100-pinned-optimizer-retry.tar.zst
[h100-readiness-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.11%2Fobserved-failure.json
[h100-readiness-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.12%2Fh100-readiness-retry.tar.zst
[h100-checkpoint-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.12%2Fobserved-failure.json
[h100-checkpoint-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.13%2Fh100-checkpoint-commit-retry.tar.zst
[h100-shutdown-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.13%2Fobserved-failure.json
[h100-shutdown-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.14%2Fh100-shutdown-retry.tar.zst
[h100-nccl-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.14%2Fobserved-failure.json
[h100-nccl-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.15%2Fh100-nccl-headroom-retry.tar.zst
[h100-growing-pool-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.15%2Fobserved-failure.json
[h100-preallocated-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.16%2Fh100-learner-preallocation-retry.tar.zst
[h100-fingerprint-timeout]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.16%2Fobserved-failure.json
[h100-fingerprint-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.17%2Fh100-fingerprint-retry.tar.zst
[h100-version-two-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.17%2Fobserved-failure.json
[h100-replay-diagnostic-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.18%2Fh100-publication-diagnostics-retry.tar.zst
[h100-replay-diagnostic-failure]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.18%2Fobserved-failure.json
[h100-canonical-replay-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.19%2Fh100-immutable-publication-retry.tar.zst
[h100-cleanup-terminal-manifest]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fcheckpoints%2Fsnowball-inventory%2F2026.11.20.20%2Fterminal.json
[h100-matched-handoff]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fh100-learner-smoke%2F2026.11.20.15%2Fquick-loop-matched-2026.11.20.15.tar.gz
[agentx-h100-serving]: https://github.com/marin-community/marin/blob/15bc5e751c75f41743b15ec8c99597f077fc9d26/docs/reports/snowball-inference-status-2026-11-20.md#agentx-at-concurrency-128
[agentx-batch-sampler]: https://github.com/marin-community/marin/blob/15bc5e751c75f41743b15ec8c99597f077fc9d26/lib/levanter/src/levanter/layers/sampler.py#L95-L110
[agentx-singleton-replay]: https://github.com/marin-community/marin/blob/15bc5e751c75f41743b15ec8c99597f077fc9d26/lib/levanter/src/levanter/inference/engine.py#L1804-L1810
[agentx-route-capable]: https://github.com/marin-community/marin/commit/3c9d26717a3108307f7f84eac9bf7605204e0d08
[native-checkpoint-publication]: https://github.com/marin-community/marin/pull/12090
[megatron-levanter-handoff]: https://github.com/marin-community/marin/blob/4d481270a1c1e0ec70a9e28c4a1a29b8f5693b31/docs/reports/snowball-megatron-levanter-handoff.md
[dspark-h100-control]: https://github.com/marin-community/marin/pull/11975
[original-hf-slice-reader]: https://github.com/marin-community/levanter/commit/fee2b33fced800a165672dcdffaaee14f16ff241
[hf-loader-history]: https://github.com/marin-community/levanter/blob/10a4038ce305eb0100f1812958c2a8078b5a8a22/.agents/projects/gcs_serialization.md
[hf-direct-loading]: https://github.com/marin-community/marin/pull/2348
[hf-current-range-planning]: https://github.com/marin-community/marin/blob/main/lib/levanter/src/levanter/compat/fsspec_safetensor.py#L195-L258
[skyrl-hf-prefetch]: https://github.com/marin-community/marin/pull/10662
[runai-distributed-streaming]: https://github.com/dsx-ai-factory/model-streamer/blob/master/docs/src/usage.md#distributed-streaming
[vllm-runai-loader]: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/model_loader/runai_streamer_loader.py
[down-rank-read-proof]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fhandoff%2F2026.11.21.1%2Fproofs%2Fsnowball-down-rank-reads-586d56160770.tar.gz
[down-rank-initializer-proof]: https://filer.marina.oa.dev/#s3%3A%2F%2Fmarin-us-east-02a%2Fmarin%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fevidence%2F2026.11.23.1%2Fheld-native-resumable%2Fhf-rank-local-initializer-cpu-proof.tar.zst
[native-shard-reader]: https://github.com/marin-community/marin/blob/6936528c55df0212ec8ccd1fa619361863b4c420/lib/levanter/src/levanter/tensorstore_serialization.py#L988-L1024
[hf-to-native-converter]: https://github.com/marin-community/marin/blob/6936528c55df0212ec8ccd1fa619361863b4c420/lib/marin/src/marin/experiment/checkpoints.py#L125
[model-only-restore]: https://github.com/marin-community/marin/pull/9475
[pinned-hf-export]: https://huggingface.co/open-athena/Grug-67B-A2B-Datakit-SFT-262K-2026.09.21/blob/fe666478454e79fe6027a0bab0388d9abd68f51a/export_complete.json
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
[iris-native-67b-tiled]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-67b-inventory-h100-20261121-tiled
[iris-native-67b-prealloc]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-67b-inventory-h100-20261121-prealloc
[iris-native-67b-profiles]: https://iris.oa.dev/#/job/%2Fdlwh%2Fsnowball-native-67b-inventory-h100-20261121-profiles
[native-67b-prealloc-archive]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-67b-inventory%2F2026.11.21.1%2Foutputs.tar.zst
[native-67b-profiles-archive]: https://filer.marina.oa.dev/#gs%3A%2F%2Fmarin-us-central2%2Fusers%2Fdlwh%2Fquick-loop-rl%2Fnative-67b-inventory%2F2026.11.21.2%2Foutputs.tar.zst
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
