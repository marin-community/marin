# Alternating Multi-Host RL on a Single TPU Pod

**Status**: in progress
**Audience**: engineers who know basic JAX and TPU concepts but are new to Marin

## Summary

This document proposes a new RL execution mode for Marin that uses **one TPU pod allocation** and alternates between:

1. a **sampling phase** that runs one independent vLLM replica per TPU host
2. a **training phase** that runs Levanter across the **full pod mesh**

The key design choice is:

- **reuse the TPU allocation**
- **do not reuse the TPU runtime**

We start and stop separate phase-specific processes on the same TPU VMs. We do **not** try to keep training and sampling alive in one long-lived multi-host JAX runtime.

That gives us all three properties we want:

- a single TPU allocation rather than separate trainer and sampler TPUs
- phase-alternating RL rather than concurrent trainer/sampler roles
- stable multi-host training and stable multi-host sampling

This mode is primarily for **single-allocation** or **availability-constrained**
deployments. It is **not** intended to replace the existing concurrent two-TPU
on-demand RL path when two TPU allocations are available and wall-clock speed or
policy freshness is the top priority.

## Branch Direction

This document now reflects the implementation direction we are actively taking.

- the alternating RL implementation in this worktree is the base branch
- competing sketches that reconstruct config independently per phase or mix
  sampling/training contracts loosely are not the direction forward
- ideas from those sketches may still be ported selectively when they improve
  this branch without weakening state management, validation, or testability

The most important example is the materializer:

- this branch keeps the controller/runtime/state design
- spill-to-disk materialization is being folded into this branch as an internal
  improvement, not as a reason to switch architectures

## Problem

Marin currently has two useful but separate pieces:

- a stable Levanter-based RL trainer path in `lib/marin/src/marin/rl/train_worker.py`
- a stable vLLM-based sampler path in `lib/marin/src/marin/rl/rollout_worker.py`
- a two-TPU manual mode in `experiments/exp2039_rl_math500.py` and `ON_DEMAND_RL.md`

Those pieces are currently wired for **concurrent** trainer and sampler roles:

- `RLJob.run()` always launches one trainer plus one or more rollout workers at the same time in `lib/marin/src/marin/rl/rl_job.py:192-261`
- `TrainWorker.train()` immediately serves initial weights and waits for live rollouts in `lib/marin/src/marin/rl/train_worker.py:299-339`
- `RolloutWorker.run()` repeatedly does `SYNC_WEIGHTS -> GENERATE -> WRITE_ROLLOUT` in `lib/marin/src/marin/rl/rollout_worker.py:706-851`

That design works when training and sampling are separate jobs. It is the wrong shape for a single TPU pod that alternates between phases.

At the same time, past attempts to keep training and rollout inside one multi-host JAX process group while assigning them different device subsets run into a hard runtime constraint:

- a normal multi-host JAX program expects all hosts to participate in the same compiled program
- disjoint role meshes are not an independent scheduler
- weight transfer via ordinary cross-mesh resharding bottoms out in `jax.device_put`, which is not a general-purpose transport between independently running disjoint host groups

That means the target should **not** be:

- “one long-lived process that changes roles”
- “one multi-host JAX runtime with trainer chips and sampler chips”

The target should instead be:

- one TPU pod allocation
- explicit phase boundaries
- independent phase processes
- file/object-store handoff at phase boundaries

## Why Current Marin Pieces Are Not Enough As-Is

This section is important because it explains why the proposal introduces new phase runners instead of adding one more flag to the existing workers.

### 1. `RLJob` encodes concurrent roles

`RLJob.run()` creates TPU resources for a trainer and rollout workers and launches them together in `lib/marin/src/marin/rl/rl_job.py:201-261`.

That is a good fit for:

- Ray mode
- manual two-TPU trainer/sampler mode

It is not a good fit for:

- one pod that alternates between “all hosts are samplers” and “all hosts are trainers”

The `RunConfig.num_train_slices` field exists in `lib/marin/src/marin/rl/rl_job.py:55`, but `RLJob.run()` never uses it. The existing RL job abstraction does not currently express “same pod, different phase topology”.

### 2. `TrainWorker` expects live rollouts and live weight serving

`TrainWorker` does three things that are specifically designed for concurrent RL:

- it creates a replay loader over live rollout storage in `lib/marin/src/marin/rl/train_worker.py:202-221`
- it starts a weight transfer server in `lib/marin/src/marin/rl/train_worker.py:222-226`
- it serves step `-1` weights and waits for initial rollouts in `lib/marin/src/marin/rl/train_worker.py:320-326`

In an alternating design:

- there is no live sampler during training
- there is no reason to run Arrow Flight during training
- there is no reason to wait for rollouts to arrive while training is already running

### 3. `RolloutWorker` expects live weight polling

`RolloutWorker` is built around periodic weight sync from a live trainer:

- it creates a weight transfer client in `lib/marin/src/marin/rl/rollout_worker.py:325-329`
- in sync mode it polls for new weights before each rollout cycle in `lib/marin/src/marin/rl/rollout_worker.py:709-725`

In an alternating design:

- the policy is frozen for the whole sampling phase
- there is no live trainer process to poll
- the correct bootstrap is “load phase policy once, then sample until quota is met”

### 4. The current rollout ingestion path is not explicit enough for multi-host training

The current file reader says training workers “read all files, and use their `process_index()` to slice out the appropriate subset” in `lib/marin/src/marin/rl/rollout_storage.py:24-30`.

But the actual implementation:

- reads all unseen files in `FileRolloutReader.read_all_available()` in `lib/marin/src/marin/rl/rollout_storage.py:215-243`
- adds all of them into one local replay buffer in `ReplayDataLoader._collect_rollouts()` in `lib/marin/src/marin/rl/replay_buffer.py:388-401`
- does not use `ReplayBuffer.total_processes` for sampling in `lib/marin/src/marin/rl/replay_buffer.py:61-123` and `lib/marin/src/marin/rl/replay_buffer.py:242-299`
- builds a full batch from whatever the local process sampled, then shards it in `lib/marin/src/marin/rl/train_worker.py:139-163`

That path might be workable for a single-process or loosely synchronized setup. It is not the input path we should build a new multi-host phase-based system on top of.

## Goals

- Run RL using **one TPU pod allocation** instead of separate trainer and sampler TPUs.
- Support both **single-host** and **multi-host** TPUs.
- Keep multi-host **training** on the proven Levanter path.
- Keep multi-host **sampling** on the proven “one independent vLLM replica per host” path.
- Make phase boundaries explicit and durable through object-store state.
- Preserve fast vLLM startup using Marin’s existing fast bootstrap logic.
- Make resume/restart behavior legible from GCS state without needing a live coordinator actor.
- Make the system understandable to newcomers without requiring them to understand Ray internals or JAX multi-controller edge cases.

## Non-Goals

- We are not trying to overlap training and sampling on different chips within the same pod.
- We are not trying to keep one long-lived TPU Python process alive across both phases.
- We are not trying to do live in-flight weight updates in this design.
- We are not trying to make the current concurrent `TrainWorker` / `RolloutWorker` path disappear.
- We are not chasing every possible phase-boundary optimization in v1. Correctness and clarity come first, but
  persistent compilation caching and bounded-memory materialization are mandatory day-1 requirements.

## Choosing Between This Mode And Concurrent Two-TPU RL

This design solves a different problem than the current concurrent on-demand RL
path. The choice should be explicit.

| Axis | Alternating Single-Pod RL | Concurrent Two-TPU RL |
|------|----------------------------|------------------------|
| TPU allocations required | 1 pod | 2 pods |
| Wall-clock efficiency | Lower due to phase boundaries | Higher due to overlap |
| Policy freshness | Lag = `steps_per_phase` | Near-stepwise |
| Runtime semantics | Simpler, phase-based, durable manifests | More live coordination |
| Live weight transfer | None | Required |
| Multi-host sampling topology | Independent per-host vLLM replicas | Separate sampler TPU(s) |
| Debuggability / resume | Strong | Moderate |
| Allocation-hours per wall-clock hour | ~1 pod-hour/hour, minus boundary idle | ~2 pod-hours/hour, but better overlap |
| Best fit | TPU scarcity, one-allocation limit, simpler operations | Highest throughput, freshest on-policy data |

Use alternating RL when one of the following is true:

- you can only get one TPU pod allocation
- quota or availability makes two pods unreliable
- you value a simpler, file-backed control plane more than maximum wall-clock throughput

Prefer concurrent two-TPU RL when one of the following is true:

- two TPU allocations are readily available
- you want the closest behavior to the current on-policy live-sync regime
- policy freshness matters more than allocation footprint

## Hard Requirements And Day-1 Invariants

These are not optional improvements. If we cannot satisfy them, this mode
should not graduate.

### 1. Persistent compilation cache for both phases

Both training and sampling must use persistent XLA compilation caches.

Training support already exists:

- Marin sets `JAX_COMPILATION_CACHE_DIR` in `lib/marin/src/marin/training/training.py:254-258`
- Levanter honors `jax_compilation_cache_dir` in `lib/levanter/src/levanter/trainer.py:1016-1017`

Sampling support already exists:

- Marin vLLM wiring sets `JAX_COMPILATION_CACHE_DIR` and `VLLM_XLA_CACHE_PATH` in
  `lib/marin/src/marin/inference/vllm_server.py:748-795`

The container runtime already mounts a persistent Docker volume at `/home/levanter`
in `lib/levanter/src/levanter/infra/docker.py:218-240`. For v1, the cache should
live on that mounted volume so it survives container restarts on the same TPU pod:

- training cache: `/home/levanter/compilation-cache/train`
- sampling cache: `/home/levanter/compilation-cache/vllm`

The cache location must be stable across phases. Do not generate phase-specific
cache paths.

### 2. Stable compile signatures across phases

Warm cache reuse only helps if the compiled programs are actually identical.
Therefore the following must stay fixed across phases for a given run family:

- TPU type and host count
- Levanter mesh and axis mappings
- training batch size and sequence shape
- vLLM local tensor-parallel size
- vLLM `max_model_len`
- model config and tokenizer
- container image digest
- JAX / vLLM / tpu_inference versions

Also, dynamic phase metadata must not leak into JIT static arguments or traced
shape logic. In particular:

- checkpoint paths
- manifest paths
- phase ids
- policy version strings

must remain runtime data, not compile-shape data.

### 3. Warm-cache validation is a blocker, not a hope

We must explicitly measure the second identical transition on the same pod.

If the second training start or the second sampling start still behaves like a
cold compile, the economics of this design change substantially and we should
not treat it as production-ready.

### 4. Bounded-memory materializer

The materializer must never assume it can hold all raw rollouts for a phase in
RAM at once. It must stream and, if needed, spill intermediate data to local
disk on worker 0.

### 5. Phase-boundary TPU lock hygiene

Every phase transition must:

- remove the previous phase container by fixed name
- verify the container is gone
- run a small health probe before starting the next phase

This is required to avoid orphaned processes holding TPU device state across
phase boundaries.

### 6. Image digest pinning

The controller must resolve the exact container image digest once at bootstrap,
record it in `run_state.json`, and use that same digest for all later phase
launches.

This is required for two reasons:

- compile-cache stability
- semantic stability across a long-running RL job

If someone rebuilds the image mid-run, the controller should fail fast instead
of silently mixing versions across phases.

## Core Design Decision

The system will reuse the same TPU pod allocation but switch between **two incompatible runtime topologies**:

### Sampling topology

- each TPU host is isolated as its own local JAX cluster
- each host runs one vLLM instance
- tensor parallelism stays inside the host
- there is no cross-host JAX coordination

### Training topology

- all TPU hosts participate in one Levanter mesh
- JAX distributed initialization spans the whole pod
- the model and optimizer state resume from Levanter checkpoints

These topologies have different process-level assumptions. Therefore the phase boundary must be a **process boundary**.

The allowed sequence is:

1. stop all sampling processes
2. run any CPU-only aggregation/materialization
3. start training processes
4. stop all training processes
5. export the next policy
6. start sampling processes again

The forbidden sequence is:

1. initialize a multi-host TPU JAX runtime
2. then try to fork or reshape it into independent vLLM per-host runtimes

The existing warning in `create_inference_context()` is the clearest local signal for this rule. `AsyncvLLMInferenceContext` uses a fork-based engine path, and the comment in `lib/marin/src/marin/rl/rollout_worker.py:242-248` explicitly warns that forking after JAX TPU initialization can deadlock on TPU device locks.

## Existing Code We Intend To Reuse

This proposal is not a rewrite. We should reuse the stable parts and replace only the coordination model.

### Reuse from Marin RL

- `load_model_from_checkpoint()` in `lib/marin/src/marin/rl/model_utils.py`
- `create_inference_context()` and `vLLMInferenceContext` in `lib/marin/src/marin/rl/rollout_worker.py:235-270` and `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
- `create_training_batch_from_rollouts()` in `lib/marin/src/marin/rl/train_batch.py`
- `RLLossModule.compute_advantages()` via the loss module already used by `ReplayBuffer`
- curriculum state/logic in `lib/marin/src/marin/rl/curriculum.py`
- rollout serialization format in `lib/marin/src/marin/rl/types.py` and `lib/marin/src/marin/rl/rollout_storage.py`

### Reuse from Levanter

- multi-host trainer initialization in `lib/levanter/src/levanter/trainer.py:910-945`
- multi-host checkpoint restore in `lib/levanter/src/levanter/checkpoint.py:370-430`
- HF/safetensors export callback in `lib/levanter/src/levanter/compat/hf_checkpoints.py:1147-1182`

### Reuse from current on-demand RL

- single shared GCS root layout from `experiments/exp2039_rl_math500.py:252-292`
- same-region shared-root discipline from `experiments/exp2039_rl_math500.py:210-225`
- fast vLLM bootstrap from `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:181-205` and `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:550-579`

## Terms

To avoid ambiguity, this document uses these terms precisely:

- **pod**: the entire TPU allocation, potentially with multiple hosts
- **host**: one TPU VM in the pod
- **phase**: one contiguous runtime mode, either sampling or training
- **policy version**: an integer identifying the frozen policy used for one sampling phase
- **global trainer step**: Levanter’s monotonically increasing optimizer step
- **run root**: the shared `gs://...` path for one RL run

## High-Level Architecture

The new design introduces a controller and three explicit phase runners:

1. **controller**
2. **sampling host runner**
3. **materializer**
4. **training phase runner**
5. **policy exporter**

In practice, “policy exporter” should be implemented as a helper invoked by the
training phase runner before the full training mesh tears down. It is listed
separately here because it has a distinct artifact contract.

### Controller responsibilities

- create or reuse the TPU pod
- write run state and per-phase manifests
- launch sampling on each host
- wait for all hosts to finish
- run the materializer
- launch training on the whole pod
- export the next policy
- update run state
- handle resume/retry

### Sampling host runner responsibilities

- load one frozen policy version
- run one local vLLM engine on one TPU host
- sample rollouts until its per-host quota is met
- optionally run eval quota at the start of the phase
- write raw rollout batches and host status files

### Materializer responsibilities

- read all raw rollout files for one policy version
- compute advantages
- deterministically sample and pack training batches
- write a finite set of **materialized training batch files**

### Training phase runner responsibilities

- initialize Levanter across the full pod
- restore the latest Levanter checkpoint
- train for exactly `N` steps using the materialized batches
- save a new Levanter checkpoint

### Policy exporter responsibilities

- export the trained policy to HF/safetensors
- write a policy manifest pointing to the export and the source Levanter checkpoint

## Why This Avoids The Multi-Host Failure Mode

The design works because it does **not** ask JAX to do something it is bad at.

During the sampling phase:

- there is no cross-host TPU program
- each host is independent
- no cross-host resharding happens

During the training phase:

- all hosts participate in the same Levanter mesh
- this is exactly the mode Levanter already supports

At the phase boundary:

- policy transfer is done through **files/object storage**
- not through a live disjoint-mesh TPU-to-TPU transfer

This turns the hard runtime problem into an ordinary artifact handoff problem.

## Phase Transition Budget And Economics

This mode is expected to be slower in wall-clock time than concurrent two-TPU
RL. The reason to build it is not “fastest possible RL”; the reason is “best
single-allocation multi-host RL mode that we can make correct and operable”.

Even with perfect warm-cache behavior, each phase has some irreducible
non-training / non-sampling work:

| Step | Warm-cache target |
|------|-------------------|
| Stop sampling containers and collect status | 5-15s |
| Materialize raw rollouts into training batches | 20-90s |
| Start training: distributed init + checkpoint restore + residual compile | 30-90s |
| Export final policy for the phase | 20-60s |
| Start sampling: fast bootstrap + residual compile | 15-45s |
| Total per phase boundary | 90-180s target |

These numbers are targets, not promises. They must be measured on the second
and third phase transitions with caches already warm.

Cold-cache behavior is not acceptable as the steady-state assumption. If we see
repeated full cold compiles after the first phase, alternating RL is still a
correct design, but probably not an economical default execution mode.

The correct framing is:

- concurrent two-TPU RL: higher allocation footprint, better wall-clock
- alternating single-pod RL: lower allocation footprint, simpler runtime model, lower wall-clock efficiency

This tradeoff is acceptable when allocation count is the binding constraint.

## Run Root Layout

Every alternating RL run has one shared root:

```text
gs://<bucket>/alternating-rl/<run_id>/
  state/
    run_state.json
    failure.json
  curriculum/
    curriculum_state.json
    curriculum_metrics_phase_0000.json
    curriculum_metrics_phase_0001.json
  policies/
    policy_0000/
      manifest.json
      config.json
      tokenizer.json
      model-00001-of-000NN.safetensors
      model.safetensors.index.json
    policy_0001/
      ...
  levanter_checkpoints/
    phase_0000/
      ...
    phase_0001/
      ...
  sampling/
    phase_0000/
      manifest.json
      host_000/
        status.json
        eval/
          000000.pkl
          000001.pkl
        train/
          000000.pkl
          000001.pkl
      host_001/
        ...
    phase_0001/
      ...
  materialized/
    phase_0000/
      manifest.json
      batch_000000.pkl
      batch_000001.pkl
      ...
    phase_0001/
      ...
  exports/
    policy_0001.json
    policy_0002.json
  logs/
    controller/
    sampling/
    training/
```

The important point is that there are **two distinct checkpoint families**:

- `levanter_checkpoints/`: full training-state checkpoints, used to continue optimization
- `policies/`: HF/safetensors exports, used to bootstrap vLLM

We must not collapse those into one thing. vLLM needs HF/safetensors. Training continuity needs Levanter optimizer state.

## Run State

`state/run_state.json` is the source of truth for the controller and for resume logic.

Example:

```json
{
  "run_id": "math500-alt-20260320-120000",
  "status": "sampling",
  "phase_id": 3,
  "policy_version": 3,
  "source_global_step": 120,
  "num_hosts": 4,
  "tpu_name": "math500-alt-20260320-120000",
  "tpu_type": "v5p-16",
  "zone": "us-east5-a",
  "image_digest": "us-east5-docker.pkg.dev/project/levanter/image@sha256:...",
  "current_policy_path": "gs://bucket/alternating-rl/math500-alt-20260320-120000/policies/policy_0003",
  "current_levanter_checkpoint_path": "gs://bucket/alternating-rl/math500-alt-20260320-120000/levanter_checkpoints/phase_0002",
  "current_sampling_manifest": "gs://bucket/alternating-rl/math500-alt-20260320-120000/sampling/phase_0003/manifest.json",
  "current_materialized_manifest": null,
  "last_completed_phase": 2
}
```

Allowed `status` values:

- `bootstrapping`
- `sampling`
- `materializing`
- `training`
- `exporting`
- `completed`
- `failed`

We should use one enum for this in code. Do not use multiple booleans like `is_sampling`, `is_training`, `did_export`.

The controller writes `image_digest` at bootstrap and must refuse to launch any
later phase with a different image digest. Mid-run image drift is not allowed.

## Policy Manifest

Each policy export gets its own manifest:

```json
{
  "policy_version": 3,
  "phase_id": 2,
  "source_global_step": 120,
  "hf_export_path": "gs://bucket/.../policies/policy_0003",
  "levanter_checkpoint_path": "gs://bucket/.../levanter_checkpoints/phase_0002",
  "model_name": "meta-llama/Llama-3.1-8B-Instruct",
  "created_at": "2026-03-20T12:00:00Z"
}
```

Sampling never reads a bare directory path directly. It reads a policy manifest so the controller can sanity-check versioning and provenance.

## Sampling Manifest

The controller writes one sampling manifest per phase:

```json
{
  "phase_id": 3,
  "policy_version": 3,
  "policy_manifest_path": "gs://bucket/.../policies/policy_0003/manifest.json",
  "curriculum_state_path": "gs://bucket/.../curriculum/curriculum_state.json",
  "num_hosts": 4,
  "local_tensor_parallel_size": 4,
  "host_assignments": [
    {"host_ordinal": 0, "seed": 1000, "target_train_groups": 2048, "target_eval_groups": 128},
    {"host_ordinal": 1, "seed": 1001, "target_train_groups": 2048, "target_eval_groups": 128},
    {"host_ordinal": 2, "seed": 1002, "target_train_groups": 2048, "target_eval_groups": 128},
    {"host_ordinal": 3, "seed": 1003, "target_train_groups": 2048, "target_eval_groups": 128}
  ],
  "frozen_lesson_weights": {
    "algebra": 0.4,
    "geometry": 0.35,
    "number_theory": 0.25
  },
  "output_root": "gs://bucket/.../sampling/phase_0003"
}
```

The lesson weights are frozen for the entire phase. That is a deliberate simplification. The old concurrent design updated curriculum online. The alternating design updates curriculum only at phase boundaries.

## Materialization Manifest

The materializer writes one manifest per phase:

```json
{
  "phase_id": 3,
  "policy_version": 3,
  "num_input_rollout_files": 512,
  "num_rollout_groups": 8192,
  "num_individual_rollouts": 32768,
  "num_training_batches": 120,
  "global_batch_size": 256,
  "max_seq_len": 4096,
  "batch_paths": [
    "gs://bucket/.../materialized/phase_0003/batch_000000.pkl",
    "gs://bucket/.../materialized/phase_0003/batch_000001.pkl"
  ]
}
```

The controller never guesses what the materializer produced. It waits for this manifest and uses it as the next input.

## Proposed New Metadata On Rollouts

Today `RolloutMetadata` contains:

- `worker_id`
- `timestamp`
- `weight_step`

See `lib/marin/src/marin/rl/types.py:34-45`.

For alternating RL, `weight_step` is too ambiguous. We need explicit phase provenance:

```python
@dataclass(frozen=True)
class RolloutMetadata:
    worker_id: str = ""
    timestamp: float = 0.0
    weight_step: int = -1
    policy_version: int = -1
    phase_id: int = -1
    source_global_step: int = -1
```

Why keep `weight_step` at all?

- the existing concurrent path already uses it
- it is useful when the policy export came from a specific trainer step

But the alternating path should key its filtering logic on `policy_version`, not on `weight_step`.

## Detailed Control Flow

This is the exact phase loop the controller runs.

### Phase 0 bootstrap

1. Build and push the Docker image using the same infrastructure flow as current `lib/levanter/infra/launch.py`.
2. Create one TPU pod allocation.
3. Resolve the initial model:
   - if the user provided an HF checkpoint, export it immediately to `policies/policy_0000/`
   - if the user provided a Levanter checkpoint, export policy `0000` from that checkpoint
4. Initialize `curriculum/curriculum_state.json`.
5. Write `run_state.json` with `policy_version = 0`, `phase_id = 0`, `status = sampling`.

### For each phase `p`

1. Read `run_state.json`.
2. Launch sampling on every host using `policy_version = p`.
3. Wait for every sampling host to write `status.json`.
4. Aggregate eval stats and update curriculum state.
5. Run materialization for phase `p`.
6. Update `run_state.json` to `status = training`.
7. Launch full-pod training using the materialized batches for phase `p`.
8. Wait for training completion.
9. Export policy `p + 1` to HF/safetensors.
10. Update `run_state.json` with new `policy_version = p + 1`, new Levanter checkpoint pointer, and `status = sampling`.
11. Repeat until `global_step >= trainer.num_train_steps` or `phase_id == max_phases`.

## Sampling Phase In Detail

The sampling phase is where the design differs most from today’s `RolloutWorker`.

### Runtime model

There is one sampling process per host. Each process:

- runs only on local TPU chips
- runs one vLLM engine
- samples lessons using frozen phase weights
- computes rewards locally
- writes rollouts locally to shared storage

There is no live Arrow Flight server and no live trainer peer.

### Why one host process, not one central router

For RL, we do not need a centralized prompt router. We only need enough rollout throughput. Independent host-local rollout workers are simpler and map cleanly to the current environment API:

- each host already knows how to call `env.sample(...)`
- each host can write `RolloutBatch` files independently
- the materializer can merge them afterward

This is effectively “data-parallel sampling”, but implemented as independent rollout workers rather than one logically centralized inference service.

### Local TPU topology

Each sampling host must be launched with environment variables that make it behave like a one-host TPU cluster. The exact values depend on TPU type and local chip layout, but the shape is:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0
TPU_BACKEND_TYPE=jax
PJRT_DEVICE=TPU
CLOUD_TPU_TASK_ID=0
TPU_PROCESS_BOUNDS=<one-host-bounds>
TPU_CHIPS_PER_PROCESS_BOUNDS=<local-chip-bounds>
TPU_VISIBLE_CHIPS=<local-chip-list>
```

The important invariant is:

- each host sees only its own local chips
- no host tries to join a multi-host JAX process group during sampling

We should implement this through one helper:

```python
def local_vllm_topology(tpu_type: str) -> LocalVllmTopology:
    ...
```

That helper should return:

- `tensor_parallel_size`
- `TPU_PROCESS_BOUNDS`
- `TPU_CHIPS_PER_PROCESS_BOUNDS`
- `TPU_VISIBLE_CHIPS`

Do not scatter these strings through scripts.

### Sampling host main loop

The host runner should look like this:

```python
def run_sampling_host(config: SamplingHostConfig) -> None:
    manifest = read_sampling_manifest(config.manifest_path)
    policy = read_policy_manifest(manifest.policy_manifest_path)
    curriculum = load_curriculum_state(manifest.curriculum_state_path)
    policy_ctx = create_vllm_policy_context(policy, config.local_topology)

    run_eval_quota(policy_ctx, curriculum, manifest.host_assignment)
    run_train_quota(policy_ctx, curriculum, manifest.host_assignment)
    write_host_status(...)
```

This runner should reuse the useful parts of `RolloutWorker`:

- inference context creation
- `_sample_batch(...)` logic
- rollout writing

But it should **not** reuse:

- weight transfer client
- background weight sync thread
- live curriculum actor calls

### Curriculum behavior during sampling

The curriculum is frozen for the phase:

1. controller loads saved curriculum state
2. controller computes `frozen_lesson_weights`
3. controller writes those weights into the sampling manifest
4. every host samples lessons from that frozen distribution

This means:

- no cross-host curriculum actor is needed
- no online lesson unlocking happens mid-phase
- all curriculum updates happen between phases, not during them

That is a good tradeoff for clarity and determinism.

### Eval at the start of each phase

We still want curriculum eval signals, because curriculum graduation logic in `lib/marin/src/marin/rl/curriculum.py:489-527` uses eval stats, not just training stats.

The cleanest place to get those eval stats is:

- immediately after loading the frozen policy for the phase
- before generating training rollouts

That gives us:

- eval stats for the current policy
- a chance to update/unlock/graduate lessons before choosing training mixture weights

So each sampling phase has two sub-steps:

1. **eval prelude**
2. **train rollout generation**

### Host quotas

The controller assigns each host an explicit quota, but in v1 those quotas
should be **derived** from `steps_per_phase`, not hand-tuned independently.

The key sizing rule is:

- v1 should run with `max_samples = 1` for the alternating path

That keeps the semantics closest to batched on-policy training: each individual
rollout is consumed at most once during the next training phase.

Given:

- `S = steps_per_phase`
- `B = global_batch_size`
- `G = n_generations_per_prompt`
- `H = num_hosts`

the minimum number of individual rollouts needed for one phase is:

```text
required_individual_rollouts = S * B
```

The minimum number of rollout groups is:

```text
required_groups = ceil(required_individual_rollouts / G)
```

And the per-host train quota is:

```text
target_train_groups_per_host = ceil(required_groups / H)
```

On top of that, the controller adds a separate eval quota.

So the manifest should carry:

- `target_eval_groups`
- `target_train_groups`

Why explicit quotas?

- they make termination unambiguous
- they make throughput accounting easy
- they avoid one fast host sampling forever while another lags

Each host should keep generating until it has written at least its quota. If one final batch overshoots the quota slightly, that is fine. The materializer reads what actually exists.

### Why `max_samples = 1` is more than a sizing detail

This is not just an implementation convenience. It is an algorithmic choice.

With `max_samples = 1`:

- each sampled rollout influences training at most once
- the alternating path behaves like batched policy iteration with no replay reuse
- low-quality or uninformative rollouts still consume part of the next training phase budget

This has two implications:

- it keeps the semantics close to “generate fresh data, then train once on it”
- it removes one of the stabilizing knobs the concurrent replay buffer has, namely
  the ability to reuse or oversample especially informative rollouts

So learning dynamics in alternating RL are affected by **both**:

- policy lag from `steps_per_phase`
- lack of rollout reuse from `max_samples = 1`

That combination needs empirical validation, not assumption.

### Sampling outputs

Each host writes:

- `eval/*.pkl`
- `train/*.pkl`
- `status.json`

`status.json` should include:

```json
{
  "host_ordinal": 2,
  "phase_id": 3,
  "policy_version": 3,
  "eval_groups_written": 128,
  "train_groups_written": 2055,
  "success": true,
  "started_at": "...",
  "finished_at": "..."
}
```

The controller does not infer host completion from missing files. It waits for `status.json`.

## Materialization Phase In Detail

This phase exists because the current live replay ingestion path is not the right foundation for multi-host alternating training.

### Where the materializer runs

The materializer must run **in-region**, not on the user’s laptop.

The safe first implementation is:

- stop all sampling containers
- launch a CPU-only materializer process on **worker 0 of the existing TPU pod**
- let it read raw rollout files from the same-region GCS bucket
- let it write materialized batches back to the same run root

Why this matters:

- rollout files can be large
- the repo guidance explicitly warns against moving large data across regions
- the controller should orchestrate, not download RL artifacts to the local machine

So the controller may run locally, but the data-heavy phase-boundary work should run on the pod.

### Bounded-memory requirement

Worker 0 is the right place to host the materializer in v1, but worker 0 is not
allowed to become a giant in-memory aggregation point.

The materializer must keep memory bounded by:

- one input rollout file
- small per-environment buffers
- metadata for its local spill files

If it needs to hold more than that, it should spill to local disk under a
phase-specific temporary directory on worker 0.

### Inputs

The materializer reads:

- all `sampling/phase_<p>/host_*/train/*.pkl`
- the phase config
- the RL loss module
- the training batch size and max sequence length

### Output

It writes a deterministic finite batch set:

- `materialized/phase_<p>/batch_000000.pkl`
- `materialized/phase_<p>/batch_000001.pkl`
- ...
- `materialized/phase_<p>/manifest.json`

Each `batch_*.pkl` contains one full `TrainingBatch`, not a raw `RolloutBatch`.

### Why materialize `TrainingBatch` instead of raw rollouts

Because `TrainingBatch` is already the exact object the trainer needs after:

- padding
- truncation handling
- loss mask creation
- policy logprob packing

Those semantics already live in `create_training_batch_from_rollouts()` and should be applied once, deterministically, at the phase boundary.

### Old-policy logprobs are a required input, not optional metadata

The training loss uses the generating policy’s logprobs for the importance
sampling ratio. That means the sampling phase must persist old-policy per-token
logprobs and the materializer must preserve them into the final packed training
batch.

Current Marin already has this path:

- `Rollout.response_logprobs` in `lib/marin/src/marin/rl/types.py:63-64`
- `create_training_batch_from_rollouts()` packs them into
  `TrainingBatch.policy_logprobs` in `lib/marin/src/marin/rl/train_batch.py:69-86`
  and `lib/marin/src/marin/rl/train_batch.py:156-166`

So this is not a new field we need to invent. But it is a hard contract for the
alternating path:

- raw rollout files must contain `response_logprobs`
- spill files must retain them
- `batch_<step>.pkl` must retain them

If any phase drops those logprobs, the alternating training path is invalid.

### Materialization algorithm

The materializer should be implemented as a **streaming two-pass pipeline**.

### Current implementation status

At the time of writing, this branch has moved the materializer partway toward
the intended two-pass design:

- pass 1 now spills advantage-annotated rollout records to local temporary shard
  files on the worker running materialization
- pass 2 now plans deterministic batch assignments from lightweight per-env
  index metadata, then streams spill shards into per-batch append files before
  packing final `TrainingBatch` payloads

So the current state matches the intended two-pass architecture much more
closely: rollout objects are no longer reloaded wholesale into RAM during pass
2, and the main in-memory state is now the lightweight assignment/index plan.

The remaining question is empirical rather than architectural: whether the
local append-file pattern is fast enough on real TPU VMs, or whether we should
later collapse the per-batch append files into a more specialized local format.

#### Pass 1: ingest and spool

1. Iterate raw rollout files in deterministic sorted order.
2. For each file:
   - deserialize one `RolloutBatch`
   - verify `policy_version == p`
   - compute advantages for each rollout group immediately
   - flatten to `RolloutWithAdvantage` records
   - append those records to a local per-environment spill file
3. Record per-environment counts and spill-file metadata in a small local index.

The spill directory can look like:

```text
/tmp/marin-alt-materializer/phase_0003/
  index.json
  algebra/
    shard_000.pkl
    shard_001.pkl
  geometry/
    shard_000.pkl
  number_theory/
    shard_000.pkl
```

This pass does all expensive reward/advantage normalization exactly once while
keeping RSS bounded.

#### Pass 2: sample and pack

1. Open the local spill index.
2. Reuse the current replay sampling policy:
   - environment-balanced sampling
   - recency bias via `alpha`
   - max usage count via `max_samples`
3. Plan exactly `steps_per_phase` output batches using rollout indices and
   per-env usage counts only.
4. Stream spill shards once, routing selected rollout records into local
   per-batch append files.
5. For each output batch:
   - read the local per-batch append file
   - restore the intended sample order using the saved slot indices
   - build one `TrainingBatch`
   - write `batch_<step>.pkl`

This preserves the useful parts of `ReplayBuffer` while moving them into an
explicit offline boundary step.

Implementation note:

- the current branch now implements both the spill step and the planned
  streaming pass 2 described above
- the remaining work is performance validation on real TPU VMs, not a missing
  architecture step

### Sampling policy during materialization

The first implementation should preserve the current replay semantics as closely as possible:

- balance across environments
- recency bias using `alpha`
- retire examples after `max_samples`
- ignore stale-rollout logic based on trainer step, because phase boundaries already enforce policy freshness

The cleanest implementation is likely:

- factor the sampling logic from `ReplayBuffer.sample_rollouts()` into a reusable helper
- feed that helper with local spill-file metadata instead of an in-memory list

### Why not overlap materialization with straggling samplers in v1

The architecture could eventually support overlapping materialization with late
sampling hosts, but v1 should not do that.

Waiting for all host `status.json` files before starting materialization keeps
the phase boundary easy to reason about:

- all sampling inputs are final
- there is one manifest for the whole phase
- resume logic does not need to reason about partially materialized input sets

### Determinism

Materialization should be deterministic given:

- phase id
- policy version
- materializer seed
- set of input rollout files

Why?

- it makes resume and debugging much easier
- it lets us regenerate a phase’s training batches exactly

The materializer must therefore:

- sort input files deterministically
- seed its RNG from the phase id
- write batch files with fixed names

## Training Phase In Detail

This phase uses the entire pod as one Levanter mesh.

### Runtime model

All hosts run the same training entrypoint. Levanter initializes distributed TPU JAX in the normal way via `TrainerConfig.initialize()` and `TrainerConfig.device_mesh` in `lib/levanter/src/levanter/trainer.py:910-945`.

This is the multi-host mode Marin already trusts.

### Input contract

The training phase reads:

- the materialization manifest
- the latest Levanter checkpoint path
- the trainer config

It does **not** read raw rollout files.

### Loader model

Because Levanter expects an iterator of batches, we introduce a simple finite loader:

```python
class MaterializedTrainingLoader:
    def __iter__(self):
        for path in self.batch_paths:
            yield load_training_batch(path)
```

All hosts read the **same** `TrainingBatch` files in the first implementation.

This is intentionally simple:

- every host sees the same global batch
- every host calls `hax.shard(...)` on that same batch
- there is no ambiguity about per-process slicing

This duplicates GCS reads across hosts, but it keeps correctness obvious. If that becomes a bottleneck later, we can optimize to per-process batch shards. That optimization is future work, not day-1 scope.

### Trainer state continuity

The training phase should always resume from the latest full Levanter checkpoint if one exists.

That means:

- optimizer state continues across phases
- LR schedule continues across phases
- `global_step` stays monotonic
- logging and checkpointing remain consistent

Only phase 0 is special:

- if the run starts from an HF checkpoint, we initialize model weights from HF and optimizer state fresh
- after the first training phase, all subsequent phases resume from Levanter checkpoints

### Training step budget per phase

The controller must make this explicit. Each training phase runs for a fixed number of optimizer steps:

- `steps_per_phase`

The training phase should not “train until data runs out”. The controller already knows how many materialized batches exist and how many steps should be consumed.

Recommended rule:

- materializer writes exactly `steps_per_phase` batch files
- training consumes exactly `steps_per_phase` batch files

That makes the phase boundary legible and removes all ambiguity.

### `steps_per_phase` is the main economic and algorithmic knob

This parameter controls two competing forces:

- **phase overhead amortization**
- **policy lag**
- **single-use data**

For the alternating design:

```text
policy_lag_in_steps = steps_per_phase
```

because one frozen policy generates the entire next training phase’s data.

At the same time, larger phases amortize the fixed boundary cost better.

Using a simple training-side approximation:

- assume warm-cache phase boundary cost `O = 120s`
- assume training step time `T = 10s`

then the boundary fraction relative to training time alone is:

```text
boundary_fraction ~= O / (O + steps_per_phase * T)
```

Example values:

| `steps_per_phase` | Training time only | Boundary fraction |
|-------------------|--------------------|-------------------|
| 40 | 400s | 23% |
| 80 | 800s | 13% |
| 160 | 1600s | 7% |

These are intentionally conservative because they ignore sampling compute. They
still show the important shape of the tradeoff: very small phases are expensive.

The other side of the tradeoff is learning dynamics:

- 20-40 steps: freshest policy, but likely poor utilization
- 80-160 steps: likely the first useful search range
- >200 steps: high-lag regime, only acceptable if experiments show stable learning

This is not only a policy-lag question. With `max_samples = 1`, larger phases
also mean more of the run’s learning signal comes from one large frozen batch of
single-use rollouts. That is a meaningful regime change from the existing
concurrent replay-based path.

So the document should not pretend there is a universally correct setting. The
correct initial plan is to empirically sweep this knob.

### Initial recommendation for v1 experiments

Do not bake in a single unquestioned default. Instead:

1. start with `steps_per_phase` in `{40, 80, 160}`
2. measure warm-cache phase overhead
3. measure learning quality versus the concurrent baseline
4. only promote a default after both utilization and learning dynamics look acceptable

If the algorithm degrades sharply once policy lag reaches 80-160 steps, this
execution mode may still be valuable as an availability fallback, but it should
not be presented as a preferred research default.

### Checkpointing during training

We should keep normal Levanter checkpointing during the phase, not just at the end.

Why?

- preemption
- debugging
- resume within a long phase

At phase completion, the controller stores:

- the base Levanter checkpoint path for that phase
- the exact global step reached

### Exact batch loading contract

The first implementation should make the training input contract painfully explicit:

1. every host reads the same `batch_<step>.pkl`
2. every host deserializes the same `TrainingBatch`
3. every host enters the same Levanter step with identical host input
4. `hax.shard(...)` produces the global sharded batch over the already-initialized full pod mesh

If this turns out to be awkward in practice, the fallback is:

- write per-process batch shards
- reconstruct a global array with JAX multihost utilities

But that is a fallback. The day-1 implementation should optimize for an obvious correctness story.

### This `hax.shard()` path needs explicit validation

This loading path is simple, but it is not the same path as Levanter’s usual
per-process sliced dataset flow.

So we need an early correctness test that:

1. constructs one known `TrainingBatch`
2. has every host deserialize the same batch
3. applies `hax.shard(...)` on every host
4. verifies the resulting global sharded arrays are numerically identical to an
   expected reference layout

This is a validation requirement, not an optional nice-to-have. If this path is
wrong, training could silently produce bad gradients.

## Policy Export In Detail

The next sampling phase needs HF/safetensors, not a Levanter optimizer checkpoint.

### Existing capability

Levanter already has an HF export callback in `lib/levanter/src/levanter/compat/hf_checkpoints.py:1147-1182`.

We should reuse that machinery to export one policy directory per completed phase:

- `policies/policy_0001/`
- `policies/policy_0002/`
- ...

### Where export runs

The export should run while the **full training mesh is still alive**.

This is an important design constraint.

A post-hoc export process that runs on only one host is not a general solution,
because the trained model may require the full pod mesh to load correctly. For
small models a one-host export might work accidentally; for larger models it is
the wrong design.

So the recommended sequence is:

1. training phase restores the model on the full pod
2. training phase runs `steps_per_phase` optimizer steps
3. training phase saves the new Levanter checkpoint
4. training phase immediately exports HF/safetensors before process teardown
5. training phase writes the policy manifest
6. only then does the training phase exit

The controller should treat “training complete” as meaning:

- Levanter checkpoint written
- policy export written
- policy manifest written

### Export timing

Export should happen **after** training phase completion, not at every training step.

Why?

- only the final policy of the phase is needed for the next sampling phase
- exporting every step would add unnecessary I/O

### Export output contract

Each policy export must contain:

- `config.json`
- tokenizer files
- safetensors shard(s)
- `model.safetensors.index.json` when sharded
- `manifest.json`

The sampling phase then uses the existing fast bootstrap logic:

- stage metadata locally
- start vLLM with `load_format="dummy"`
- inject weights from safetensors

That logic is already implemented in `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:181-205` and `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py:550-579`.

## Failure And Resume Model

The system must be resumable from shared state alone.

### Controller crash

If the controller dies:

- no TPU process should keep mutating shared state forever without the controller knowing
- all phase outputs already written remain durable
- a new controller process can read `run_state.json` and continue

### Sampling host crash

If one sampling host crashes:

- its `status.json` is missing or says `success = false`
- the controller marks the run failed or relaunches just that host for the same phase
- because the policy is frozen, rerunning one host is safe

### Training crash

If training crashes:

- `run_state.json` remains `status = training`
- the controller reads the latest successful Levanter checkpoint under the current phase path
- the controller can rerun the training phase from that checkpoint

### Export crash

If export crashes:

- the source Levanter checkpoint still exists
- the controller reruns the training phase in “export-only” mode or from the final checkpoint while the full mesh is available
- sampling does not begin until `policy_<n+1>/manifest.json` exists

### Idempotence rule

Every phase output should be safe to recompute for the same `(run_id, phase_id)`.

That means:

- manifests are written last
- partial directories without manifests are ignored on resume

## Infrastructure And Launching

The controller needs a way to:

- build/push the image once
- create one TPU pod once
- run commands on all workers for training
- run commands on individual workers for sampling
- run CPU-only controller/materializer commands on worker 0
- stop containers cleanly between phases

### Why the current `launch.py` path is not enough by itself

`lib/levanter/infra/launch.py` and `lib/levanter/src/levanter/infra/tpus.py` currently implement:

- build image
- create TPU
- run one command on all workers

That is perfect for “launch one job on one TPU”. It is not enough for alternating phases on an existing pod because we also need:

- per-worker commands
- repeated command launches on the same pod without recreating it
- one-worker commands for materialization and other control tasks
- explicit stop/restart of phase containers

### Proposed infra helpers

Add helpers in `lib/levanter/src/levanter/infra/tpus.py`:

- `ssh_tpu_worker(tpu_name, zone, worker_ordinal, *args)`
- `run_container_on_worker(...)`
- `run_container_all_workers(...)`
- `stop_container_on_worker(...)`
- `stop_container_all_workers(...)`
- `describe_tpu_workers(...)`
- `run_python_on_worker(...)`

The controller can then do:

1. `ensure_tpu_exists(...)`
2. for each worker ordinal, launch one sampling container with host-specific env
3. wait for sampling completion
4. stop sampling containers
5. launch materializer on worker 0
6. wait for materializer completion
7. launch training container on all workers
8. wait for training completion
9. stop training container

The important operational rule is:

- **do not delete the TPU between phases**

We are reusing the allocation. We are only changing the phase process running inside it.

## Proposed New Code Structure

This is the concrete file layout I recommend.

### New files

- `experiments/alternating_rl_math500.py`
  - user-facing entrypoint for this execution mode
- `lib/marin/src/marin/rl/alternating/config.py`
  - top-level config dataclasses
- `lib/marin/src/marin/rl/alternating/state.py`
  - run state enums and manifest dataclasses
- `lib/marin/src/marin/rl/alternating/controller.py`
  - phase orchestration
- `lib/marin/src/marin/rl/alternating/sampling_host.py`
  - one-host sampling process
- `lib/marin/src/marin/rl/alternating/materializer.py`
  - raw rollout -> materialized training batch conversion
- `lib/marin/src/marin/rl/alternating/training_phase.py`
  - full-pod Levanter training over materialized batches
- `lib/marin/src/marin/rl/alternating/policy_export.py`
  - HF/safetensors export + manifest writing
- `lib/marin/src/marin/rl/alternating/local_topology.py`
  - TPU-type -> local vLLM topology mapping

### Entry point modes

`experiments/alternating_rl_math500.py` should expose explicit modes:

- `controller`
  - orchestration only; may run locally or on a CPU VM
- `sampling-host`
  - launched once per TPU host during the sampling phase
- `materialize`
  - launched on worker 0 after sampling completes
- `train-phase`
  - launched on all hosts for full-mesh training
- `export-only`
  - optional recovery mode to export a policy from the latest phase checkpoint without rerunning sampling

These modes should be explicit CLI choices, not inferred from a pile of environment variables.

### Example CLI shape

User-facing controller invocation:

```bash
uv run python experiments/alternating_rl_math500.py \
  --mode controller \
  --run-id math500-alt-20260320 \
  --shared-root gs://my-bucket/alternating-rl/math500-alt-20260320 \
  --tpu-name math500-alt-20260320 \
  --tpu-type v5p-16 \
  --zone us-east5-a
```

What the controller launches on each sampling host:

```bash
uv run python experiments/alternating_rl_math500.py \
  --mode sampling-host \
  --manifest-path gs://.../sampling/phase_0003/manifest.json \
  --host-ordinal 2
```

What the controller launches for materialization on worker 0:

```bash
uv run python experiments/alternating_rl_math500.py \
  --mode materialize \
  --manifest-path gs://.../sampling/phase_0003/manifest.json
```

What the controller launches on all hosts for training:

```bash
uv run python experiments/alternating_rl_math500.py \
  --mode train-phase \
  --manifest-path gs://.../materialized/phase_0003/manifest.json \
  --run-state-path gs://.../state/run_state.json
```

What the controller launches for export-only recovery:

```bash
uv run python experiments/alternating_rl_math500.py \
  --mode export-only \
  --run-state-path gs://.../state/run_state.json
```

These exact flags can evolve, but the overall split should stay this explicit.

### Existing files to modify

- `lib/marin/src/marin/rl/types.py`
  - extend `RolloutMetadata`
- `lib/marin/src/marin/rl/train_batch.py`
  - make sure `TrainingBatch` is safely serializable for materialized files
- `lib/marin/src/marin/rl/replay_buffer.py`
  - factor environment-balanced sampling logic into a reusable helper
- `lib/levanter/src/levanter/infra/tpus.py`
  - add existing-pod and per-worker helpers

## Concrete Dataclasses

These are the core new config/state objects.

```python
@dataclass
class AlternatingRLConfig:
    run_id: str
    shared_root: str
    tpu_name: str
    tpu_type: str
    zone: str
    seed: int
    model: LmConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    loss: RLLossModule
    curriculum: CurriculumConfig
    tokenizer: str
    initial_checkpoint: str
    steps_per_phase: int
    training_compilation_cache_dir: str
    sampling_compilation_cache_dir: str
    train_groups_per_host: int | None
    eval_groups_per_host: int

@dataclass
class AlternatingRunState:
    status: RunStatus
    phase_id: int
    policy_version: int
    source_global_step: int
    current_policy_path: str
    current_levanter_checkpoint_path: str | None
```

```python
@dataclass
class SamplingHostAssignment:
    host_ordinal: int
    seed: int
    target_eval_groups: int
    target_train_groups: int

@dataclass
class SamplingManifest:
    phase_id: int
    policy_version: int
    policy_manifest_path: str
    curriculum_state_path: str
    frozen_lesson_weights: dict[str, float]
    host_assignments: list[SamplingHostAssignment]
    output_root: str
```

The controller writes these as JSON. Do not invent parallel ad-hoc JSON structures in different modules.

`train_groups_per_host` should be `None` in the normal path so the controller
derives it from `steps_per_phase`. It exists only as an escape hatch for
advanced experiments.

## Recommended Container Names

Use fixed container names so phase cleanup is easy and scripts stay legible:

- sampling host container: `marin-alt-sampler`
- materializer container: `marin-alt-materializer`
- training container: `marin-alt-trainer`

Phase transitions should begin by forcibly removing the previous phase’s known container name on the relevant workers before launching the next one.

## Example Phase Launch Sequence

This is what one controller iteration should do operationally.

### Sampling phase

For each worker ordinal `i`:

1. `stop_container_on_worker(..., name="marin-alt-sampler")`
2. verify the container is gone
3. run a small worker health probe
4. `run_container_on_worker(..., worker_ordinal=i, name="marin-alt-sampler", command="... --mode sampling-host ... --host-ordinal i")`

Then:

3. poll `sampling/phase_<p>/host_*/status.json`
4. fail or relaunch if any host does not report success

### Materialization

On worker `0` only:

1. `stop_container_on_worker(..., name="marin-alt-materializer")`
2. verify the container is gone
3. run a small worker health probe
4. `run_container_on_worker(..., worker_ordinal=0, name="marin-alt-materializer", command="... --mode materialize ...")`
5. wait for `materialized/phase_<p>/manifest.json`

### Training

On all workers:

1. `stop_container_all_workers(..., name="marin-alt-trainer")`
2. verify the containers are gone
3. run a small all-worker health probe
4. `run_container_all_workers(..., name="marin-alt-trainer", command="... --mode train-phase ...")`
5. wait for training completion marker or process exit
6. verify new Levanter checkpoint path and new policy manifest exist

## Detailed Controller Algorithm

The controller loop should be explicit enough that a newcomer can trace it line-by-line.

```python
def run_controller(config: AlternatingRLConfig) -> None:
    ensure_image_and_tpu(config)
    state = bootstrap_or_resume(config)

    while not should_stop(state, config):
        sampling_manifest = write_sampling_manifest(config, state)
        launch_sampling_phase(config, sampling_manifest)
        wait_for_sampling_hosts(config, sampling_manifest)

        curriculum_state = aggregate_phase_eval_and_update_curriculum(config, sampling_manifest, state)
        materialized = materialize_training_batches(config, sampling_manifest, curriculum_state, state)

        training_result = run_training_phase(config, materialized, state)
        next_policy = export_policy(config, training_result, state)

        state = advance_run_state(state, training_result, next_policy)
        write_run_state(config, state)
```

Every helper in that loop should correspond to one small module or one small cluster of functions. Do not bury the whole control plane inside one 800-line script.

## Why We Freeze The Policy Per Phase

This is worth stating plainly.

Freezing the policy for the whole sampling phase means:

- no live weight sync
- no Arrow Flight dependency
- no “rollouts from mixed policy versions” ambiguity
- no trainer starvation because sampling is finite
- no need to reason about stale live rollout filtering based on trainer step delay

The cost is:

- some policy lag between rollouts and training

That tradeoff is acceptable here because the goal is a correct, stable multi-host alternating system, not a fully asynchronous on-policy engine.

## Why We Freeze Curriculum Per Phase

The current concurrent system updates curriculum online. We should not copy that behavior blindly.

Phase-frozen curriculum gives us:

- no distributed curriculum actor
- deterministic sampling weights for the phase
- easier resume and reproducibility

The update cadence becomes:

1. eval current policy
2. update curriculum
3. freeze lesson weights
4. sample training rollouts

That is easy to explain and easy to debug.

## What We Deliberately Do Not Reuse

These existing pieces are good in their current context but should not be the foundation of the new path.

### Not reused as core orchestration

- `RLJob.run()`
- live Arrow Flight weight transfer
- `RolloutWorker._sync_weights()`
- `TrainWorker._wait_for_initial_rollouts()`

### Reused only as logic source

- replay sampling policy from `ReplayBuffer`
- rollout batch construction from `RolloutWorker`
- training batch construction from `train_batch.py`

This split is important. Reusing logic is good. Reusing the wrong control plane is not.

## Single-Host Behavior

On a single-host TPU pod, the design degenerates cleanly:

- sampling phase launches exactly one host runner
- training phase still launches full-pod Levanter, which is just one host
- all manifests and state logic remain identical

That means this design is not “multi-host only”. It is one design that covers both cases.

## Go / No-Go Validation Before Adoption

This mode should not be promoted based on architectural cleanliness alone. We
need evidence in four areas.

### 1. Warm-cache validation

Run two identical sampling starts and two identical training starts on the same
TPU pod with persistent cache paths.

We want to see:

- the second training start avoids a full cold compile
- the second sampling start avoids a full cold compile
- warm-start timings are materially lower than cold-start timings

If the second start still looks cold, treat that as a blocker.

### 1b. Identical-input sharding validation

Run a dedicated multi-host test for the new loader contract:

- every host reads the same serialized `TrainingBatch`
- every host calls `hax.shard(...)`
- the resulting sharded arrays match an expected reference

If this path does not validate cleanly, do not proceed with the alternating
training implementation.

### 2. Phase-boundary budget validation

Measure the second full transition:

- sampling stop
- materialization
- training start
- training finish
- export
- sampling start

Record the warm-cache boundary time. If this remains much higher than the
target `90-180s` range, the design may still be correct but should not be
marketed as efficient.

### 3. `steps_per_phase` sweep

Run a small sweep over:

- `40`
- `80`
- `160`

For each point, measure:

- phase overhead fraction
- rollout throughput
- trainer throughput
- learning quality versus the concurrent baseline

This sweep determines whether the alternating design has a usable operating
range or only functions as a quota-constrained fallback.

The sweep should be interpreted as measuring both:

- policy lag sensitivity
- sensitivity to the `max_samples = 1` single-use data regime

### 4. Materializer throughput and memory

On a representative phase size, measure:

- worker 0 RSS
- wall-clock materialization time
- bytes read from GCS
- bytes written back

If worker 0 RSS grows linearly with total rollout volume, the materializer
implementation has violated the bounded-memory requirement.

## Tests

We should not treat this as complete without end-to-end tests. At minimum:

### Unit tests

- run state JSON roundtrip
- manifest JSON roundtrip
- local topology mapping for supported TPU types
- materializer determinism for a fixed seed
- rollout metadata carries `policy_version` and `phase_id`

### Integration tests

- single-host alternating run with tiny model and tiny curriculum
- materializer produces exact `steps_per_phase` batches
- training phase resumes from prior Levanter checkpoint
- export produces a valid HF/safetensors policy directory
- sampling phase can bootstrap vLLM from exported policy

### Recovery tests

- crash after sampling, resume into materialization
- crash after materialization, resume into training
- crash after training, resume into export
- crash after export, resume into next sampling phase

### Performance validation

- measure sampling throughput per host
- measure total phase transition overhead
- compare “full batch read on every host” vs “per-process shard” once correctness is stable

## Concrete Implementation Order

This is the order I would actually build it in.

1. Add persistent compile-cache wiring and fixed cache paths for both training and sampling.
2. Add the existing-pod TPU infra helpers in Levanter.
3. Validate warm-cache reuse on repeated sampling and training starts on the same pod.
4. Validate the identical-input `hax.shard(...)` training loader path on a small multi-host test.
5. Add run state + manifest dataclasses and JSON I/O.
6. Add policy export support for “one export per phase” inside the training phase process.
7. Add the one-host sampling runner with frozen policy and frozen curriculum.
8. Add the controller that launches sampling on every host and waits for `status.json`.
9. Add the streaming offline materializer that writes `TrainingBatch` files.
10. Add the training phase runner over materialized batches.
11. Add resume logic.
12. Add eval prelude and curriculum phase-boundary update.
13. Add end-to-end integration tests and the `steps_per_phase` sweep.

This order keeps the riskiest concurrency questions out of the first passes.

## Notes For The Future Implementer

- Keep training and sampling in separate top-level entrypoints or separate `--mode`s. Do not hide phase dispatch inside complicated conditionals.
- Keep all phase state in one run root. Debugging gets much harder if phase outputs are spread across unrelated prefixes.
- Keep the first implementation simple even if it duplicates some GCS reads. Correctness and explainability matter more than early micro-optimization.
- Do not reintroduce live TPU-to-TPU weight transfer into this design. That would put us back in the same runtime trap we are intentionally avoiding.
- Do not let sampling read “latest policy” by listing directories. Always go through `run_state.json` and the policy manifest.
- Do not let training read raw rollout files directly. Always go through the materialization manifest.

## Future Work

- Replace “all hosts read the same materialized batch files” with per-process batch shards if GCS fan-out becomes a bottleneck.
- Add a controller mode that runs on a small CPU VM instead of the user’s terminal.
- Add richer per-phase metrics dashboards.
- Add a policy export cache if export time becomes material.
- Add optional shorter subphases if policy lag becomes too large.
- Overlap materialization with late-arriving samplers once the basic phase-boundary model is proven stable.

## Final Recommendation

Implement alternating RL as a **new execution path**, not as a patch on top of the existing concurrent trainer/sampler workers.

The right abstraction is:

- **one TPU pod**
- **one controller**
- **two runtime topologies**
- **durable file-based phase boundaries**

That is the simplest design that matches the code Marin already has, the
training mode Levanter already supports, and the multi-host sampling pattern
that already appears to work well for vLLM.

But this should be presented honestly:

- it is the best design for **single-allocation** multi-host RL
- it is not automatically the best wall-clock RL mode when two TPU allocations are available

So the recommendation is:

1. build this as a separate execution mode
2. require warm-cache validation before claiming it is economical
3. compare it head-to-head against the concurrent two-TPU baseline
4. treat it as the preferred option when allocation count or availability is the binding constraint
