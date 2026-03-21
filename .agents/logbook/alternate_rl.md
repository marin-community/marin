# Alternating RL Logbook

Reference design doc: [`.agents/projects/alternating-multihost-rl.md`](/Users/ahmed/code/marin/.claude/worktrees/enchanted-crafting-wave/.agents/projects/alternating-multihost-rl.md)

This logbook is the running record for all ongoing experiments, smoke tests, failures, measurements, and follow-up fixes required to get alternating single-allocation multi-host RL working properly on real TPUs.

## Scope

Track:

- single-host smoke tests
- multi-host smoke tests
- compile-cache behavior across phase boundaries
- vLLM bootstrap and host-local sampling behavior
- materialization correctness and throughput
- Levanter training resume/export behavior
- controller/container lifecycle failures
- algorithmic behavior such as policy lag and `steps_per_phase` sensitivity

Do not use this file for speculative design notes that belong in the main design doc. Use it for execution history, observed behavior, concrete hypotheses, and next actions.

## Current Status

- Code status: controller/runtime/materialization/training path implemented in this worktree
- Recent implementation milestones:
  - added per-phase durable metrics manifests for prepare-sampling, sampling, curriculum update, materialization, training, and export timings
  - added sampler container liveness detection for the "host died without writing status" case
  - added export-only recovery mode via `export-policy`
  - optimized resumed zero-KL training phases to skip reloading the original HF checkpoint when only a correctly-shaped model tree is needed
  - made alternating phase metrics resilient to trainers without an attached tracker and added controller-side per-phase timing summaries in logs
- Local validation status:
  - alternating unit tests passing
  - repo pre-commit passing
- Next milestone:
  - first real TPU smoke test

## Success Criteria

We consider alternating RL viable only if all of the following are demonstrated on real TPU runs:

1. One full phase completes end to end: sampling -> materialization -> training -> export -> next policy manifest.
2. Warm-cache phase restarts reuse XLA caches well enough that boundary overhead is acceptable.
3. Multi-host training consumes materialized batches correctly with the current `hax.shard()` loader path.
4. Multi-phase runs complete without controller/state drift or container lifecycle failures.
5. Learning dynamics remain usable at some practical `steps_per_phase`.

## Experiment Template

Copy this block for each real run:

```md
## Run <ID>

- Date:
- Owner:
- Goal:
- TPU:
- Hosts:
- Image:
- Code revision:
- Command:
- Phase quotas:
- Expected outcome:

### Result

- Status:
- Completed phases:
- Wall-clock:
- Boundary overhead:
- Warm-cache behavior:
- Sampling throughput:
- Training throughput:
- Export result:

### Evidence

- Run state path:
- Policy manifest path:
- Materialized manifest path:
- Logs:
- Metrics:

### Findings

-

### Follow-up

- [ ]
```

## Hypotheses To Validate

### H1: Warm cache makes phase restarts acceptable

We expect persistent `JAX_COMPILATION_CACHE_DIR` and `VLLM_XLA_CACHE_PATH` to convert later phase starts from cold compile behavior into mostly warm-cache starts.

Evidence needed:

- first phase startup time
- second phase startup time
- whether the second phase still triggers large compilations

### H2: The materialized-batch loader is correct on multi-host TPU

The current design has each host read the same `TrainingBatch` pickle and then call `hax.shard(...)`. This is simple, but it must be validated on real distributed TPU runtime.

Evidence needed:

- no shape/sharding errors
- no silent gradient corruption symptoms
- stable multi-host training behavior on a tiny run

### H3: A practical `steps_per_phase` sweet spot exists

We expect there to be a workable range where:

- policy lag is not too large
- phase-boundary overhead is amortized enough

Evidence needed:

- at least a small sweep over `steps_per_phase`
- notes on wall-clock overhead and training behavior

## Run Ledger

## 2026-03-21 Local Milestone

- Status:
  - local code path updated and revalidated before first TPU smoke test
- Validation:
  - `uv run pytest tests/rl/test_alternating_state.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_training_phase.py tests/rl/test_alternating_runtime.py tests/rl/test_train_batch.py -o addopts=`
  - `./infra/pre-commit.py --all-files --fix`
  - `uv run python experiments/alternating_rl_math500.py --help`
- New operational fixes:
  - sampler liveness now fails when a host container disappears without writing status
  - alternating-specific phase timings are persisted durably, logged to the trainer tracker when present, and summarized by the controller after each completed phase
  - export-only policy recovery is now available from the experiment CLI
- Remaining TPU-only unknowns:
  - warm-cache reuse across real phase restarts
  - multi-host `hax.shard()` correctness on TPU runtime
  - actual vLLM TPU bootstrap behavior in the launch image

## Open Issues

- Warm-cache reuse is not yet validated on real TPU.
- Multi-host loader correctness is not yet validated on real TPU.
- First real vLLM TPU bootstrap in the target image is not yet validated under the alternating controller.
- Alternating-specific metrics now exist, but their usefulness still needs to be checked in the first real WandB-backed TPU run.

## Decisions

- 2026-03-21: Use the alternating controller/runtime/state architecture in this worktree as the base implementation.
- 2026-03-21: Treat this mode as a single-allocation execution mode, not a replacement for concurrent two-TPU RL.
- 2026-03-21: Require compile-cache persistence from day 1.
- 2026-03-21: Keep the multi-host `hax.shard()` loader path for the first TPU smoke test, but treat it as an empirical validation gate rather than a proven assumption.

## 2026-03-21 Planned TPU Experiment Sequence

The next TPU work should be run in this order. Do not skip ahead unless a prior
experiment already proves the required precondition.

### ALT-TPU-001: Single-host one-phase smoke test

- Goal:
  - prove that one complete alternating phase works on a real TPU VM with the
    current controller/runtime path
- Hypothesis:
  - with `num_hosts=1`, `steps_per_phase=1`, and `num_train_steps=1`, the run
    completes `sampling -> materialization -> training -> export` and writes
    the next policy manifest without manual intervention
- Suggested launch:

```bash
uv run python experiments/alternating_rl_math500.py controller \
  --run-id alt-tpu-001 \
  --shared-root gs://<bucket>/alternating-rl \
  --image <image> \
  --tpu-name alt-tpu-001 \
  --tpu-type v5p-8 \
  --zone us-east5-a \
  --num-hosts 1 \
  --local-tensor-parallel-size 4 \
  --steps-per-phase 1 \
  --num-train-steps 1
```

- Evidence to collect:
  - `state/run_state.json`
  - `state/phase_metrics/phase_0000.json`
  - `policies/policy_0001/manifest.json`
  - controller logs
  - sampling host status file
- If proven:
  - move immediately to `ALT-TPU-002`
- If disproven:
  - classify the failure first:
    - if sampling/vLLM bootstrap fails, debug the sampling container/image path
      before any multi-host test
    - if training/export fails, debug the Levanter resume/export path on
      single-host before any multi-host test
    - if controller/container lifecycle fails, fix runtime orchestration before
      spending more TPU time

### ALT-TPU-002: Single-host two-phase warm-cache validation

- Goal:
  - validate that phase 2 restart cost is materially lower than phase 1 because
    compile caches are being reused
- Hypothesis:
  - with `steps_per_phase=1` and `num_train_steps=2`, phase 1 pays the cold
    start cost and phase 2 reuses caches well enough that boundary time drops
    noticeably
- Suggested launch:

```bash
uv run python experiments/alternating_rl_math500.py controller \
  --run-id alt-tpu-002 \
  --shared-root gs://<bucket>/alternating-rl \
  --image <image> \
  --tpu-name alt-tpu-002 \
  --tpu-type v5p-8 \
  --zone us-east5-a \
  --num-hosts 1 \
  --local-tensor-parallel-size 4 \
  --steps-per-phase 1 \
  --num-train-steps 2
```

- Evidence to collect:
  - `state/phase_metrics/phase_0000.json`
  - `state/phase_metrics/phase_0001.json`
  - controller phase timing summaries in logs
  - any compile-cache hit/miss logging from training and sampling startup
- Success criterion:
  - phase 2 boundary cost is clearly lower than phase 1 and does not look like
    a full cold compile again
- If proven:
  - move to multi-host validation in `ALT-TPU-003`
- If disproven:
  - block all cost/economics claims for alternating RL
  - inspect cache path stability, image digest stability, shape stability, and
    startup env differences across phases
  - rerun this experiment only after a concrete cache-fix change lands

### ALT-TPU-003: Minimal multi-host one-phase sharding validation

- Goal:
  - validate the current identical-input `TrainingBatch` loader path on a real
    multi-host pod
- Hypothesis:
  - on a two-host pod, every host can deserialize the same materialized batch
    file, call `hax.shard(...)`, and train one phase without sharding or step
    consistency errors
- Suggested launch:

```bash
uv run python experiments/alternating_rl_math500.py controller \
  --run-id alt-tpu-003 \
  --shared-root gs://<bucket>/alternating-rl \
  --image <image> \
  --tpu-name alt-tpu-003 \
  --tpu-type v5p-16 \
  --zone us-east5-a \
  --num-hosts 2 \
  --local-tensor-parallel-size 4 \
  --steps-per-phase 1 \
  --num-train-steps 1
```

- Evidence to collect:
  - `state/run_state.json`
  - `state/phase_metrics/phase_0000.json`
  - `materialized/phase_0000/manifest.json`
  - training logs from all hosts
  - next policy manifest
- Success criterion:
  - one full multi-host phase completes and the exported policy manifest reports
    the expected next `source_global_step`
- If proven:
  - move to `ALT-TPU-004`
- If disproven:
  - treat the current `hax.shard()` path as invalid for multi-host
  - next engineering step is a new loader path that slices batches per process
    explicitly instead of relying on identical full-batch deserialization on
    every host

### ALT-TPU-004: Minimal multi-host two-phase reliability run

- Goal:
  - validate repeated phase alternation on a real pod, including cache reuse,
    container restarts, and next-policy advancement across more than one phase
- Hypothesis:
  - with `steps_per_phase=1` and `num_train_steps=2`, the pod survives two full
    alternating phases and writes `policy_0002`
- Suggested launch:

```bash
uv run python experiments/alternating_rl_math500.py controller \
  --run-id alt-tpu-004 \
  --shared-root gs://<bucket>/alternating-rl \
  --image <image> \
  --tpu-name alt-tpu-004 \
  --tpu-type v5p-16 \
  --zone us-east5-a \
  --num-hosts 2 \
  --local-tensor-parallel-size 4 \
  --steps-per-phase 1 \
  --num-train-steps 2
```

- Evidence to collect:
  - both phase metrics manifests
  - both policy manifests
  - controller logs showing per-phase timing summaries
  - any missing-status / dead-container failures
- If proven:
  - alternating RL is ready for an economic and algorithmic sweep
- If disproven:
  - classify the failure:
    - cache/restart problem -> stay on `steps_per_phase=1` until lifecycle is
      stable
    - export-only gap -> use the new `export-policy` path to recover, then fix
      the controller path
    - sampler liveness / stale container issue -> improve runtime health checks
      before longer runs

### ALT-TPU-005 / 006 / 007: `steps_per_phase` sweep on multi-host

- Goal:
  - find whether a practical operating point exists where phase-boundary cost
    is amortized without making policy lag unacceptable
- Hypothesis:
  - one of `steps_per_phase in {40, 80, 160}` gives a usable tradeoff for
    alternating RL on the target workload
- Suggested launches:
  - `ALT-TPU-005`: `steps_per_phase=40`
  - `ALT-TPU-006`: `steps_per_phase=80`
  - `ALT-TPU-007`: `steps_per_phase=160`
  - keep the same TPU type, image, and curriculum; vary only
    `steps_per_phase` and set `num_train_steps` large enough to complete at
    least two phases per run
- Evidence to collect:
  - boundary fraction from phase metrics
  - sampling/training throughput
  - reward/loss behavior from tracker logs
  - qualitative notes on instability, clipping pathologies, or obvious lag
- If proven:
  - pick the best point and document it as the default alternating RL operating
    regime
- If disproven:
  - keep alternating RL as an availability fallback only
  - prefer concurrent two-allocation RL whenever wall-clock or policy freshness
    is the priority

## 2026-03-21 ALT-TPU-001 Continuation: v5p re-investigation

- Owner: codex session
- Goal:
  - determine whether the current alternating setup can run on v5p with a
    correct image/runtime combination, instead of abandoning v5p prematurely
- Starting point:
  - Claude's TPU bring-up notes suggest three real issues:
    - wrong in-container experiment path
    - missing host-local TPU topology env on sampling containers
    - possible v5p image / libtpu / jax mismatch
  - The first two are controller/runtime bugs; the third is still a hypothesis
- Working hypothesis:
  - v5p should still be a viable target, but the current launch image and/or
    TPU-enabled Python stack may not match the TPU runtime actually provisioned
- Initial execution plan:
  1. verify which of Claude's fixes are already present in this worktree
  2. inspect the image/build chain that the alternating path is actually using
  3. inspect current local diffs tied to the TPU attempt
  4. try to reproduce or refute the "v5p image is missing TPU-enabled jax/libtpu" hypothesis with direct evidence
  5. only then decide whether to rerun a v5p smoke test or patch code/image first

### Updates

- 2026-03-21T00:00Z:
  - Read the main logbook and Claude's side logbook.
  - Verified the current worktree already includes the topology-env propagation
    path in `runtime.py` for sampling containers.
  - Verified the current worktree has uncommitted changes in:
    - `lib/marin/src/marin/rl/alternating/runtime.py`
    - `lib/marin/src/marin/rl/rl_experiment_utils.py`
  - Confirmed there are no active local Docker containers.
  - First `gcloud compute tpus queued-resources list` attempt failed because no
    default zone was configured; need to query explicit TPU zones instead of
    assuming global defaults.
- 2026-03-21T00:10Z:
  - Inspected the two uncommitted TPU-attempt diffs:
    - `lib/marin/src/marin/rl/alternating/runtime.py`
    - `lib/marin/src/marin/rl/rl_experiment_utils.py`
  - Verified Claude's first two TPU bring-up fixes are already present in the
    worktree:
    - phase commands now use the absolute experiment path inside Docker
    - sampling containers now receive the host-local topology env before Python starts
  - Inspected the alternating image build chain:
    - `lib/levanter/docker/tpu/Dockerfile.base`
    - `lib/levanter/docker/tpu/Dockerfile.incremental`
    - root `rl` extra -> `levanter[serve,tpu]` + `marin[vllm,rl,tpu]`
  - Conclusion:
    - alternating is not using a special one-off Docker path; it uses the same
      TPU image family as other repo paths
- 2026-03-21T00:18Z:
  - Compared the current alternating image (`9825f29cad7b`) against a known
    recent on-demand v5p image (`5bcffb3ddf72`).
  - Important finding:
    - under `uv run python`, both images expose the same TPU stack:
      - `jax==0.8.0`
      - `jaxlib==0.8.0`
      - `libtpu==0.0.24`
      - `vllm-tpu==0.13.2.post6`
  - This directly weakens the earlier hypothesis that "the image simply does
    not have libtpu for v5p".
  - More precise conclusion:
    - the failure is more likely about TPU runtime metadata / container env
      than about the image lacking TPU packages entirely
- 2026-03-21T00:24Z:
  - Read TPU container env docs in `lib/iris/README.md`.
  - Key repo-local finding:
    - Iris treats these as important for TPU containers:
      - `TPU_SKIP_MDS_QUERY=1`
      - `TPU_ACCELERATOR_TYPE`
      - `TPU_TYPE`
      - `TPU_WORKER_ID`
      - `TPU_WORKER_HOSTNAMES`
      - `TPU_CHIPS_PER_HOST_BOUNDS`
  - The alternating host-local topology contract did NOT set these env vars;
    it only set `PJRT_DEVICE`, `CLOUD_TPU_TASK_ID`, process bounds, and visible chips.
  - Direct container probe:
    - when I manually added `TPU_ACCELERATOR_TYPE=v5p-8` and
      `TPU_WORKER_HOSTNAMES=localhost`, libtpu/JAX initialization behavior
      changed materially versus the baseline
  - Working conclusion:
    - v5p is still plausible; the missing piece is likely a more explicit
      single-host TPU metadata contract for sampler containers
- 2026-03-21T00:32Z:
  - Patched `lib/marin/src/marin/rl/alternating/topology.py` so the
    host-local vLLM topology now exports:
    - `TPU_SKIP_MDS_QUERY=1`
    - `TPU_ACCELERATOR_TYPE`
    - `TPU_TYPE`
    - `TPU_NAME`
    - `TPU_WORKER_ID=0`
    - `TPU_WORKER_HOSTNAMES=127.0.0.1`
    - `TPU_CHIPS_PER_HOST_BOUNDS`
    - plus the existing single-host process-bounds / visible-chip envs
  - Added regression coverage in `tests/rl/test_alternating_topology.py`.
  - Validation:
    - `uv run pytest tests/rl/test_alternating_topology.py tests/rl/test_alternating_runtime.py tests/rl/test_alternating_state.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_training_phase.py tests/rl/test_train_batch.py -o addopts=`
    - result: `28 passed`
- 2026-03-21T00:36Z:
  - Current decision:
    - do NOT abandon v5p yet
    - next step should be a minimal real v5p probe before rerunning the full alternating controller
  - Planned next probe:
    - create or reuse a fresh v5p-8 in `us-east5-a`
    - run a minimal containerized `uv run python -c 'import jax; print(jax.devices())'`
      with the new explicit topology env
    - if that succeeds, immediately rebuild the alternating image and retry `ALT-TPU-001` on v5p
- 2026-03-21T08:04Z:
  - Reserved probe name: `alt-v5p-probe-001`
  - Checked `gcloud alpha compute tpus queued-resources describe alt-v5p-probe-001 --zone=us-east5-a`
  - Result:
    - `NOT_FOUND`
  - Decision:
    - safe to create a fresh dedicated v5p-8 probe TPU under that name
- 2026-03-21T08:05Z:
  - Created queued resource:
    - `gcloud alpha compute tpus queued-resources create alt-v5p-probe-001 --accelerator-type=v5p-8 --zone=us-east5-a --runtime-version=v2-alpha-tpuv5 --spot --node-id=alt-v5p-probe-001 --quiet`
  - Result:
    - queued resource creation succeeded
  - Next:
    - wait for `ACTIVE`
    - inspect raw host TPU env on the VM before running any container
- 2026-03-21T08:08Z:
  - Probe TPU state progressed to `PROVISIONING`
  - Additional evidence gathered while waiting:
    - official Google Cloud TPU container guidance independently recommends
      `TPU_SKIP_MDS_QUERY=1`, `TPU_WORKER_HOSTNAMES=localhost`,
      `TPU_WORKER_ID=0`, `TPU_ACCELERATOR_TYPE`, and
      `TPU_CHIPS_PER_HOST_BOUNDS` for single-host TPU containers
  - Interpretation:
    - the new topology env patch is consistent with both repo-local TPU
      container conventions and current Google Cloud guidance
- 2026-03-21T08:12Z:
  - `gcloud alpha compute tpus tpu-vm list --zone=us-east5-a` now shows:
    - `alt-v5p-probe-001   v5p-8   CREATING`
  - Interpretation:
    - queued resource provisioning has advanced into TPU VM creation
  - Next:
    - wait for `READY`
    - then inspect host TPU env / metadata immediately
- 2026-03-21T08:14Z:
  - Additional TPU-runtime inspection:
    - `vllm` TPU platform code explicitly forwards both
      `TPU_CHIPS_PER_HOST_BOUNDS` and `TPU_HOST_BOUNDS`
    - Ray TPU helper code also sets `TPU_HOST_BOUNDS=1,1,1` for single-host subsets
  - Follow-up code change:
    - extended `LocalVllmTopology.env()` to include `TPU_HOST_BOUNDS=1,1,1`
  - Validation:
    - `uv run pytest tests/rl/test_alternating_topology.py -o addopts=`
    - result: `3 passed`
- 2026-03-21T08:18Z:
  - Probe TPU reached `READY` / `HEALTHY`.
  - Host metadata inspection on `alt-v5p-probe-001` showed:
    - `accelerator-type = v5p-8`
    - `agent-worker-number = 0`
    - `worker-network-endpoints = unknown:unknown:10.202.1.156`
    - `tpu-env` includes:
      - `HOST_BOUNDS='1,1,1'`
      - `CHIPS_PER_HOST_BOUNDS='2,2,1'`
      - `TPU_PROCESS_BOUNDS='1,1,1'`
      - `TPU_CHIPS_PER_PROCESS_BOUNDS='2,2,1'`
  - Important host finding:
    - a plain host `env` dump did not expose TPU/JAX env vars directly, so the
      container cannot rely on inheriting enough TPU metadata automatically
- 2026-03-21T08:21Z:
  - First attempt to launch a minimal Dockerized probe via inline `gcloud ... --command=...`
    failed locally due shell quoting before reaching the TPU.
  - Recovery:
    - copied a small probe script to the TPU VM and ran it there instead
- 2026-03-21T08:24Z:
  - Real v5p container probe result:
    - image: `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774079266`
    - explicit env:
      - `TPU_SKIP_MDS_QUERY=1`
      - `TPU_ACCELERATOR_TYPE=v5p-8`
      - `TPU_TYPE=v5p-8`
      - `TPU_WORKER_ID=0`
      - `TPU_WORKER_HOSTNAMES=127.0.0.1`
      - `TPU_HOST_BOUNDS=1,1,1`
      - `TPU_CHIPS_PER_HOST_BOUNDS=2,2,1`
      - `CLOUD_TPU_TASK_ID=0`
      - `TPU_PROCESS_BOUNDS=1,1,1`
      - `TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1`
      - `TPU_VISIBLE_CHIPS=0,1,2,3`
    - command inside container:
      - `/opt/marin/.venv/bin/python -c 'import jax; print(jax.devices()); print(jax.default_backend())'`
    - output:
      - 4 TPU devices visible
      - backend = `tpu`
  - Conclusion:
    - v5p absolutely works with this software stack when the container gets a
      complete explicit single-host TPU metadata contract
    - the previous "image doesn't support v5p" hypothesis is disproven
  - Immediate next step:
    - rebuild the alternating image with the patched topology helper
    - rerun `ALT-TPU-001` on v5p instead of abandoning the platform
- 2026-03-21T08:33Z:
  - Resumed the thread with the same goal:
    - keep working directly from this branch
    - treat `.agents/logbook/alternate_rl.md` as append-only
    - reuse `alt-v5p-probe-001` rather than provisioning another TPU
  - Current local code state checked:
    - `runtime.py` already contains the earlier fixes for:
      - absolute in-container experiment path `/opt/marin/experiments/alternating_rl_math500.py`
      - topology env injection for both `prepare-sampling` and `sampling-host`
    - `topology.py` contains the explicit single-host TPU metadata contract
    - `rl_experiment_utils.py` still contains the local-controller `SamplingParams` stub fallback
  - Current infrastructure state:
    - `gcloud alpha compute tpus queued-resources describe alt-v5p-probe-001 --zone=us-east5-a`
      reports `ACTIVE`
  - Next:
    - run repo validation on the current branch
    - build and push a fresh TPU image containing the topology fix
    - launch the real alternating controller on the ready `v5p-8` probe
- 2026-03-21T08:37Z:
  - Validation checkpoint before image rebuild:
    - `uv run pytest tests/rl/test_alternating_topology.py tests/rl/test_alternating_runtime.py tests/rl/test_alternating_state.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_training_phase.py tests/rl/test_train_batch.py -o addopts=`
    - result: `28 passed`
    - `./infra/pre-commit.py --all-files --fix`
    - result: `OK`
  - Interpretation:
    - current local branch is in a clean enough state to cut a new TPU image
    - next step is a fresh incremental image build that actually contains the
      topology patch proven by the manual v5p probe
- 2026-03-21T08:39Z:
  - Starting fresh TPU image build for the controller retry.
  - Planned command:
    - `uv run python lib/levanter/infra/push_docker.py --docker_target gcp --project hai-gcp-models --region us-east5 --repository levanter --image levanter-ahmed --tag alt-rl-1774080929 --docker_file lib/levanter/docker/tpu/Dockerfile.incremental`
  - Expected outcome:
    - new Artifact Registry image in `us-east5` containing the patched
      `LocalVllmTopology` env contract
- 2026-03-21T08:44Z:
  - Image build/push result:
    - image:
      - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774080929`
    - digest:
      - `sha256:b939d3cb1567c5281d10f5944ce348b6cb45b3ecfb045430143b2463440bcd76`
  - Verification:
    - `gcloud artifacts docker images list us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed --include-tags --filter='tags:alt-rl-1774080929'`
    - result showed the expected tag and digest
  - Next:
    - inspect controller CLI one more time for the exact minimal single-host
      smoke-test flags
    - launch the real controller on `alt-v5p-probe-001`
- 2026-03-21T08:47Z:
  - Pre-launch checks:
    - `gcloud storage ls --recursive gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry`
    - result: no objects
    - `gcloud alpha compute tpus tpu-vm ssh alt-v5p-probe-001 --zone=us-east5-a --worker=all --command='bash -lc \"hostname; sudo docker ps -a --format \\\"table {{.Names}}\\\\t{{.Status}}\\\"\"'`
    - result:
      - TPU VM reachable
      - only system containers running
      - no stale alternating sampler/trainer containers
  - Planned controller launch:
    - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-001-v5p-retry --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774080929 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 1`
  - Expected outcome:
    - first real single-host alternating phase on `v5p-8` reaches at least
      `sampling -> materialization -> training`, or fails with a new
      controller-visible error that is narrower than the earlier topology
      failure
- 2026-03-21T08:53Z:
  - `ALT-TPU-001` controller retry result:
    - controller launched successfully against `alt-v5p-probe-001`
    - bootstrap artifacts were written:
      - `state/run_state.json`
      - `policies/policy_0000/manifest.json`
      - `sampling/phase_0000/manifest.json`
    - `run_state.json` advanced to `status: "sampling"`
    - sampler container started successfully on the TPU VM
  - Important positive finding:
    - the previous `Failed to get global TPU topology` error is gone
    - `prepare-sampling` now initializes vLLM on `v5p-8`, sees
      `alternating-local-vllm`, and reaches model construction
  - New failure:
    - sampler container exited during `prepare-sampling`
    - controller marked the run `status: "failed"`
    - stack trace ends inside
      `tpu_inference.models.jax.llama3.LlamaModel`
    - exact error:
      - `ValueError: Found unexpected Arrays on value of type <class 'list'> in static attribute 'layers'`
  - Root-cause narrowing:
    - inspected the image contents directly
    - confirmed the image contains:
      - `flax-0.11.1`
      - `jax-0.8.0`
      - `tpu_inference-0.13.2.post6`
    - inspected `tpu_inference/models/jax/llama3.py`
    - confirmed `LlamaModel.__init__` assigns:
      - `self.start_layer, self.end_layer, self.layers = make_layers(...)`
    - inspected `tpu_inference/layers/jax/pp_utils.py`
    - confirmed `make_layers(...)` returns a plain Python `list[nnx.Module]`
  - Interpretation:
    - this is no longer a TPU-topology or container-orchestration problem
    - the active blocker is an NNX compatibility bug in the TPU vLLM stack for
      Llama on the current image/runtime combination
  - Next:
    - patch the TPU-inference compatibility shim in Marin so `make_layers(...)`
      produces an `nnx.List` for the Llama path before vLLM model
      construction
    - rebuild the image and rerun the same `ALT-TPU-001` controller command on
      the same `v5p-8` probe
- 2026-03-21T09:01Z:
  - Implemented a Marin-side TPU-inference compatibility shim:
    - added `_patch_tpu_inference_llama_nnx_compat(...)` in
      `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
    - the shim wraps `tpu_inference.models.jax.llama3.LlamaModel.layers` in
      `nnx.List` before Flax NNX finishes model construction
    - wired the shim into `_patch_tpu_inference_registry()`
  - Added regression coverage:
    - `tests/rl/test_inference_ctx.py` now checks that the patch converts a
      dummy Llama layer list into `nnx.List` and remains idempotent
  - Validation:
    - `uv run pytest tests/rl/test_inference_ctx.py tests/rl/test_alternating_topology.py tests/rl/test_alternating_runtime.py tests/rl/test_alternating_state.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_training_phase.py tests/rl/test_train_batch.py -o addopts=`
    - result: `37 passed`
    - `./infra/pre-commit.py --all-files --fix`
    - result: `OK`
  - Next:
    - build a new image containing the NNX compatibility shim
    - rerun the same single-host `ALT-TPU-001` controller command on
      `alt-v5p-probe-001`
- 2026-03-21T09:03Z:
  - Starting second image rebuild with the Llama NNX compatibility shim.
  - Planned command:
    - `uv run python lib/levanter/infra/push_docker.py --docker_target gcp --project hai-gcp-models --region us-east5 --repository levanter --image levanter-ahmed --tag alt-rl-1774081435 --docker_file lib/levanter/docker/tpu/Dockerfile.incremental`
- 2026-03-21T09:06Z:
  - Second image build/push result:
    - image:
      - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774081435`
    - digest:
      - `sha256:bc9c51115d065d4e52acc91bfbc5960200881c2604181a5b8c2e2b1967db9710`
  - Infrastructure interruption:
    - `alt-v5p-probe-001` transitioned to `state: SUSPENDED`
    - `stateInitiator: SERVICE`
    - `queued-resources reset` is not allowed from `SUSPENDED`
    - `tpu-vm start alt-v5p-probe-001` fails because the underlying node no
      longer exists
  - Interpretation:
    - this specific spot TPU allocation needs to be recreated, not resumed
  - Next:
    - delete and recreate `alt-v5p-probe-001`
    - once the probe is back to `ACTIVE` / `READY`, rerun the controller
      against the new image digest
- 2026-03-21T09:08Z:
  - Recovery actions:
    - deleted suspended queued resource `alt-v5p-probe-001`
    - recreated `alt-v5p-probe-001` with:
      - `v5p-8`
      - `us-east5-a`
      - `v2-alpha-tpuv5`
      - `spot`
  - Current state after recreation:
    - queued resource initially returned to `WAITING_FOR_RESOURCES`
  - Next:
    - wait for the recreated probe to reach `ACTIVE` / TPU VM `READY`
    - rerun the controller as a fresh run id, preserving the failed
      `alt-tpu-001-v5p-retry` artifacts for comparison
- 2026-03-21T09:10Z:
  - Side investigation: the sampler warning
    - `Tensorflow library not found ... GCS paths cannot be accessed`
      does not currently look like the primary blocker for alternating RL
  - Evidence:
    - phase-0 `prepare-sampling` for `policy_0000` uses the Hugging Face model
      name directly, not a `gs://` bootstrap checkpoint
    - later fast-bootstrap code in
      `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py` stages
      metadata and safetensors through `url_to_fs(...)` and fsspec, not
      `tensorflow.io.gfile`
  - Current conclusion:
    - keep the warning in mind, but do not treat it as the next fix until a TPU
      run actually fails on an fsspec/GCS read path
- 2026-03-21T09:12Z:
  - Additional hardening fix landed while waiting for TPU capacity:
    - patched `lib/levanter/src/levanter/infra/tpus.py` so printed/failure
      command strings redact sensitive env values for:
      - `HF_TOKEN`
      - `WANDB_API_KEY`
      - `OPENAI_API_KEY`
      - `ANTHROPIC_API_KEY`
      - `GOOGLE_API_KEY`
    - added regression coverage in `lib/levanter/tests/test_tpus.py`
  - Validation:
    - `uv run --directory lib/levanter --group test python -m pytest tests/test_tpus.py`
    - result: `12 passed`
    - `./infra/pre-commit.py --all-files --fix`
    - result: `OK`
- 2026-03-21T09:14Z:
  - Recovery progress on the recreated probe:
    - queued-resource polling moved from `WAITING_FOR_RESOURCES` to
      `PROVISIONING`
    - `gcloud alpha compute tpus tpu-vm list --zone=us-east5-a` now shows:
      - `alt-v5p-probe-001 ... STATUS=CREATING`
  - Next:
    - wait for `alt-v5p-probe-001` to become `ACTIVE` / `READY`
    - launch a fresh controller retry using:
      - image digest `sha256:bc9c51115d065d4e52acc91bfbc5960200881c2604181a5b8c2e2b1967db9710`
      - run id `alt-tpu-001-v5p-retry2`
- 2026-03-21T09:16Z:
  - Probe recovery complete:
    - queued resource `alt-v5p-probe-001` returned to `ACTIVE`
    - TPU VM `alt-v5p-probe-001` reached `state: READY`
    - health: `HEALTHY`
  - Pre-launch checks for retry 2:
    - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry2` is empty
    - fresh TPU VM only has system containers running
  - Planned controller launch:
    - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-001-v5p-retry2 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774081435 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 1`
- 2026-03-21T09:20Z:
  - `ALT-TPU-001` retry 2 result:
    - bootstrap completed again
    - sampler container started on the recreated `v5p-8` host
    - the previous secret leak in controller traces is fixed; failure output now
      redacts API tokens correctly
  - Failure:
    - `prepare-sampling` still failed during Llama model construction
    - new exact error:
      - `ValueError: Cannot assign data value of type '<class 'flax.nnx.helpers.List'>' to static attribute 'layers'`
  - Interpretation:
    - the previous patch proved the target object was correct (`nnx.List`)
    - but patching `LlamaModel.__init__` is too late because Flax has already
      classified `layers` as static by the time we reassign it
  - Follow-up fix:
    - moved the compatibility shim earlier so it patches
      `tpu_inference.models.jax.llama3.make_layers(...)` instead of reassigning
      `self.layers` after construction
  - Local validation of the corrected fix:
    - `uv run pytest tests/rl/test_inference_ctx.py tests/rl/test_alternating_topology.py tests/rl/test_alternating_runtime.py tests/rl/test_alternating_state.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_training_phase.py tests/rl/test_train_batch.py -o addopts=`
    - result: `37 passed`
    - `./infra/pre-commit.py --all-files --fix`
    - result: `OK`
  - Next:
    - build a third image with the corrected `make_layers(...)` patch
    - rerun again as `alt-tpu-001-v5p-retry3` on the same still-ready
      `alt-v5p-probe-001`
- 2026-03-21T09:22Z:
  - Starting third image rebuild with the corrected `make_layers(...)` patch.
  - Planned command:
    - `uv run python lib/levanter/infra/push_docker.py --docker_target gcp --project hai-gcp-models --region us-east5 --repository levanter --image levanter-ahmed --tag alt-rl-1774082157 --docker_file lib/levanter/docker/tpu/Dockerfile.incremental`
- 2026-03-21T09:23Z:
  - Third image build/push result:
    - image:
      - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774082157`
    - digest:
      - `sha256:523307ff54ae06ecf5f0a8844fb5f3b16ac6b557208eb5d1263ae84774d32c13`
  - Current TPU state:
    - `alt-v5p-probe-001` still `ACTIVE`
  - Next:
    - launch `alt-tpu-001-v5p-retry3` on the same probe using this new image
- 2026-03-21T09:41Z:
  - `ALT-TPU-001` retry 3 result:
    - TPU bootstrap and local vLLM initialization progressed further than retry
      2
    - the run reached:
      - TPU device discovery on `v5p-8`
      - vLLM engine initialization
      - mesh setup
      - compile-cache enablement
      - Llama model construction
      - MATH-500 dataset loading
    - this disproves the earlier claim that the image/runtime stack is
      fundamentally incompatible with `v5p`
  - New failure:
    - `prepare-sampling` crashed inside `batch_completions(...)` with:
      - `AttributeError: 'SamplingParams' object has no attribute 'stop'`
  - Interpretation:
    - the remaining blocker is now at the controller-config serialization
      boundary, not TPU bring-up
    - the controller serialized a sampling-params object that did not round-trip
      cleanly into the TPU worker environment
  - Fix in progress:
    - replace foreign / environment-dependent sampling-params objects in
      `vLLMInferenceContextConfig` with a Marin-owned serializable
      `VllmSamplingConfig`
    - coerce any legacy object into that config on TPU worker load
    - add a regression test that controller config pickling preserves the
      sampling config across the local-controller -> TPU-worker boundary
  - Next:
    - run focused alternating tests
    - rebuild/push a fourth image with the serialization fix
    - rerun `ALT-TPU-001` on the same `v5p-8`
- 2026-03-21T09:46Z:
  - Serialization-fix validation:
    - focused alternating test suite:
      - `uv run pytest tests/rl/test_inference_ctx.py tests/rl/test_alternating_runtime.py tests/rl/test_alternating_topology.py tests/rl/test_alternating_state.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_training_phase.py tests/rl/test_train_batch.py -o addopts=`
      - result: `39 passed`
    - repo lint / type / formatting gate:
      - `./infra/pre-commit.py --all-files --fix`
      - result: `OK`
  - Fourth image build/push result:
    - image:
      - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774082534`
    - digest:
      - `sha256:4c23656b3be8febb3dd9685a74b2e1cb8d7286094d72ee9ac610fe717efd962f`
  - Current TPU state before retry 4:
    - queued resource `alt-v5p-probe-001` is still `ACTIVE`
    - TPU VM is still `READY`
    - health: `HEALTHY`
  - Planned controller launch:
    - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-001-v5p-retry4 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774082534 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 1`
- 2026-03-21T09:54Z:
  - `ALT-TPU-001` retry 4 live progress:
    - controller bootstrap succeeded on the existing `v5p-8` probe
    - `prepare-sampling` completed the full 500-example eval prelude
    - the first real compile during prelude showed the expected cold-start
      behavior:
      - first prompt batch initially advanced at `~112s/it`
      - throughput normalized after compile and completed all 500 eval prompts
        in `~3m12s`
    - after the prelude, the controller launched a fresh host sampling
      container
    - that second container completed vLLM warmup and started the actual
      rollout-generation step for the 64 training prompts
  - Most important conclusion:
    - alternating RL now provably runs on `v5p-8` through:
      - TPU bring-up
      - local vLLM init
      - `prepare-sampling`
      - handoff into actual host-local rollout generation
    - this clears the prior `SamplingParams.stop` serialization failure and the
      earlier TPU/NNX bootstrap failures
  - Current state:
    - run id `alt-tpu-001-v5p-retry4` still in progress
    - controller state remains `sampling`
    - waiting for `sampling/phase_0000/host_0000/status.json` and then
      materialization / training
  - Next:
    - keep the run alive until it either writes host status and advances to
      materialization, or produces the next blocking error
- 2026-03-21T10:01Z:
  - `ALT-TPU-001` retry 4 interruption:
    - the run survived long enough to begin real rollout generation, but the
      TPU allocation was reclaimed mid-phase by the service
    - observed control-plane state:
      - queued resource transitioned to `SUSPENDING`
      - `stateInitiator: SERVICE`
    - observed runtime symptom:
      - local controller liveness probes started failing because
        `gcloud alpha compute tpus tpu-vm ssh ...` returned non-zero while the
        TPU was being deleted
      - the controller raised from `wait_for_sampling_phase(...)` while polling
        `container_exists_on_worker(...)`
  - Interpretation:
    - this was not a model/runtime correctness failure
    - it was an infrastructure-loss case that the controller classified
      poorly
  - Follow-up code fix:
    - patched the sampling wait loop to catch worker-SSH probe failures,
      inspect queued-resource state, and raise a clear infrastructure-loss
      error when the TPU enters:
      - `SUSPENDING`
      - `SUSPENDED`
      - `DELETING`
      - `FAILED`
    - added a regression test for this path
  - Validation:
    - `uv run pytest tests/rl/test_alternating_runtime.py tests/rl/test_inference_ctx.py tests/rl/test_alternating_topology.py tests/rl/test_alternating_state.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_training_phase.py tests/rl/test_train_batch.py -o addopts=`
    - result: `40 passed`
  - Next:
    - wait for `alt-v5p-probe-001` to become deletable / absent
    - recreate the same `v5p-8`
    - rerun `ALT-TPU-001` as `alt-tpu-001-v5p-retry5`
- 2026-03-21T10:08Z:
  - Post-interruption recovery progress:
    - deleted the service-reclaimed queued resource:
      - `alt-v5p-probe-001`
    - reran repo gates after the infrastructure-loss controller fix:
      - `./infra/pre-commit.py --all-files --fix`
      - result: `OK`
    - built and pushed the next image:
      - image:
        - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774083474`
      - digest:
        - `sha256:2a39a51cfd56011c984cfe50594aa92a77a42838a9cf15d58803aded745efcdc`
    - recreated the same spot queued resource:
      - `alt-v5p-probe-001`
      - `v5p-8`
      - `us-east5-a`
      - `runtime-version=v2-alpha-tpuv5`
      - `--spot`
  - Current state:
    - queued resource is back in `WAITING_FOR_RESOURCES`
  - Planned next launch:
    - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-001-v5p-retry5 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774083474 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 1`
- 2026-03-21T10:14Z:
  - Retry 5 launch status:
    - replacement queued resource became:
      - `ACTIVE`
    - replacement TPU VM became:
      - `READY`
      - `HEALTHY`
    - SSH + Docker readiness probe succeeded on worker 0
    - `alt-tpu-001-v5p-retry5` controller launch started successfully on the
      new node
    - controller has already:
      - written `state/run_state.json`
      - written `sampling/phase_0000/manifest.json`
      - restored curriculum state
      - begun the sampler launch
    - current sampler state:
      - pulling image digest
        `sha256:2a39a51cfd56011c984cfe50594aa92a77a42838a9cf15d58803aded745efcdc`
      - no new runtime failure yet
  - Current state:
    - `alt-tpu-001-v5p-retry5` is live
    - controller state is `sampling`
  - Next:
    - keep monitoring retry 5 through the same checkpoints:
      - TPU bootstrap
      - `prepare-sampling`
      - actual rollout generation
      - materialization / training if the TPU survives
- 2026-03-21T10:20Z:
  - Retry 5 live milestone:
    - retry 5 reached the same healthy phase-0 checkpoint as retry 4
    - observed sequence:
      - TPU bootstrap succeeded on the replacement `v5p-8`
      - local vLLM init succeeded
      - full 500-example `prepare-sampling` eval prelude completed
      - prelude container exited cleanly with code `0`
      - controller launched the real background host sampler immediately after
    - durable controller evidence:
      - `state/run_state.json` remains in `status: sampling`
      - `state/phase_metrics/phase_0000.json` now records:
        - `prepare_sampling_seconds: 386.5650899410248`
    - durable sampling evidence:
      - `sampling/phase_0000/manifest.json`
      - `sampling/phase_0000/curriculum_snapshot.json`
      - no host status file yet, which means the second sampler container is
        still running the real rollout-generation step
  - Interpretation:
    - retry 5 confirms the reclaim / recreate / relaunch loop is working
    - the alternating path is repeatedly reaching real rollout generation on
      `v5p-8`, not just one lucky attempt
  - Next:
    - keep tailing the second sampler container until it either:
      - writes `host_0000/status.json` and advances to materialization
      - or surfaces the next concrete failure
- 2026-03-21T10:31Z:
  - Retry 5 interruption:
    - retry 5 also survived through:
      - TPU bootstrap
      - local vLLM init
      - full `prepare-sampling`
      - launch of the real background host sampler
    - then the service reclaimed the `v5p-8` again during the real rollout
      generation step
    - observed control-plane state:
      - queued resource transitioned to `SUSPENDING`
      - `stateInitiator: SERVICE`
    - observed runtime symptom:
      - the controller now fails with the intended explicit classification:
        - `RuntimeError: TPU became unavailable while waiting for sampling host completion: tpu_name=alt-v5p-probe-001, host_ordinal=0, queued_state=SUSPENDING`
    - durable evidence preserved before interruption:
      - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry5/state/run_state.json`
      - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry5/state/phase_metrics/phase_0000.json`
      - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry5/sampling/phase_0000/manifest.json`
      - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry5/sampling/phase_0000/curriculum_snapshot.json`
      - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry5/policies/policy_0000/manifest.json`
  - Interpretation:
    - this is the second consecutive `v5p-8` run that reaches real rollout
      generation before infrastructure loss
    - the blocking issue is currently spot-resource stability, not TPU/vLLM
      correctness up to the start of sampling
    - the first smoke profile is still too slow for this spot environment:
      - `prepare-sampling` alone costs about `386s`
      - phase 0 is carrying a much larger eval + rollout load than needed for
        first proof-of-life
  - Next:
    - reduce the `ALT-TPU-001` smoke profile substantially before retry 6:
      - smaller eval prelude
      - fewer training prompts
      - fewer generations per prompt
      - smaller train batch size
    - rebuild, reprovision, and relaunch on `v5p-8`
- 2026-03-21T10:46Z:
  - Smoke-profile reduction landed for retry 6:
    - added explicit controller overrides to
      `experiments/alternating_rl_math500.py` for:
      - `--train-batch-size`
      - `--n-prompts`
      - `--n-generations-per-prompt`
      - `--eval-examples-per-lesson`
      - `--max-input-tokens`
      - `--max-output-tokens`
    - added coverage in:
      - `tests/rl/test_alternating_math500_experiment.py`
  - Validation:
    - `uv run pytest tests/rl/test_alternating_math500_experiment.py tests/rl/test_alternating_runtime.py tests/rl/test_alternating_state.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_training_phase.py tests/rl/test_train_batch.py -o addopts=`
    - result: `30 passed`
    - `uv run python experiments/alternating_rl_math500.py controller --help`
    - result:
      - new smoke-profile flags are exposed on the controller CLI
  - Current TPU state:
    - queued resource `alt-v5p-probe-001` is now `SUSPENDED`
    - this means it can be deleted and recreated cleanly for the next retry
  - Retry 6 plan:
    - keep the TPU type at `v5p-8`
    - keep the region/zone at `us-east5-a`
    - keep the run single-host / one-phase
    - shrink phase-0 load to the minimum credible proof-of-life profile:
      - `--steps-per-phase 1`
      - `--num-train-steps 1`
      - `--train-batch-size 16`
      - `--n-prompts 4`
      - `--n-generations-per-prompt 4`
      - `--eval-examples-per-lesson 8`
      - `--max-input-tokens 512`
      - `--max-output-tokens 256`
  - Hypothesis:
    - this profile will cut the long `prepare-sampling` wall-clock enough that
      the first complete alternating phase has a materially better chance to
      finish before another spot reclaim
  - Next:
    - run repo lint gates
    - build and push a new image
    - delete and recreate the `v5p-8`
    - launch `alt-tpu-001-v5p-retry6` with the reduced smoke profile
- 2026-03-21T10:56Z:
  - Retry 6 bring-up started:
    - repo gate rerun:
      - `./infra/pre-commit.py --all-files --fix`
      - result: `OK`
    - issued delete for the now-suspended queued resource:
      - `gcloud alpha compute tpus queued-resources delete alt-v5p-probe-001 --zone=us-east5-a --force --quiet`
    - started a fresh incremental image build/push:
      - `uv run python lib/levanter/infra/push_docker.py --docker_target gcp --project hai-gcp-models --region us-east5 --repository levanter --image levanter-ahmed --tag alt-rl-1774084854 --docker_file lib/levanter/docker/tpu/Dockerfile.incremental`
  - Current state:
    - waiting for the old queued resource delete to finish
    - waiting for the new image push to finish
  - Next:
    - resolve the pushed image digest
    - recreate `alt-v5p-probe-001` as `v5p-8`
    - launch `alt-tpu-001-v5p-retry6` on the reduced smoke profile
- 2026-03-21T10:59Z:
  - Retry 6 infrastructure status:
    - confirmed pushed image tag:
      - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774084854`
    - confirmed digest:
      - `sha256:61639731c9ab619a127a6a0c14dfa425cf37ffb7d25d1d89d427a866813dec0b`
    - recreated queued resource:
      - `alt-v5p-probe-001`
      - `v5p-8`
      - `us-east5-a`
      - `runtime-version=v2-alpha-tpuv5`
      - `--spot`
  - Current state:
    - queued resource is back in `WAITING_FOR_RESOURCES`
    - holding the replacement request open until it reaches `ACTIVE`, then
      waiting for TPU VM `READY`
  - Next:
    - immediately launch retry 6 once the TPU is live
- 2026-03-21T11:02Z:
  - Opportunistic retry-loop hardening while waiting for capacity:
    - found a real helper bug in
      `lib/levanter/src/levanter/infra/tpus.py`:
      - `start_tpu_vm_queued_resources(...)` returned immediately when a queued
        resource already existed in `WAITING_FOR_RESOURCES`, which would make
        bootstrap race into SSH/Docker setup if the controller were launched too
        early
    - fixed the helper so existing non-`ACTIVE` queued resources wait for
      `ACTIVE` instead of returning early
    - added regression coverage in:
      - `lib/levanter/tests/test_tpus.py`
  - Validation:
    - `uv run --directory lib/levanter --group test python -m pytest tests/test_tpus.py`
    - result: `14 passed`
  - Current TPU state:
    - replacement `alt-v5p-probe-001` moved from `WAITING_FOR_RESOURCES` to
      `PROVISIONING`
  - Next:
    - keep waiting for `ACTIVE` / TPU VM `READY`
    - then launch `alt-tpu-001-v5p-retry6` immediately
- 2026-03-21T11:06Z:
  - Retry 6 controller launch:
    - launched against the existing replacement queued resource after the local
      `start_tpu_vm_queued_resources(...)` wait fix
    - exact command:
      - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-001-v5p-retry6 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774084854 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 1 --train-batch-size 16 --n-prompts 4 --n-generations-per-prompt 4 --eval-examples-per-lesson 8 --max-input-tokens 512 --max-output-tokens 256`
  - Current state:
    - controller process is live
    - queued resource is still transitioning through provisioning
    - the reduced smoke profile is now armed and waiting for the pod to finish
      coming up
  - Next:
    - watch retry 6 through:
      - TPU `ACTIVE` / VM `READY`
      - sampler bootstrap
      - reduced `prepare-sampling`
      - host-local rollout generation
- 2026-03-21T11:10Z:
  - Retry 6 progress:
    - controller successfully observed the replacement queued resource as
      `ACTIVE` and continued through SSH/Docker setup on the `v5p-8`
    - first sampler container launched and completed the reduced
      `prepare-sampling` prelude
    - durable phase metrics now show:
      - `prepare_sampling_seconds: 190.98283004760742`
    - comparison to retries 4 and 5:
      - old prelude was about `386.6s`
      - retry 6 prelude is about `191.0s`
      - this is roughly a 2x reduction from the smoke-profile change
    - after the prelude, the controller launched the real background sampler
      container and it is currently `Up`
  - Evidence:
    - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry6/state/phase_metrics/phase_0000.json`
    - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry6/sampling/phase_0000/manifest.json`
    - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry6/policies/policy_0000/manifest.json`
  - Interpretation:
    - the reduced smoke profile materially improved time-to-real-sampling on
      `v5p-8`
    - retry 6 is now in the same critical stage that retries 4 and 5 reached,
      but it reached it much faster
  - Next:
    - keep tailing the background sampler
    - look for `sampling/phase_0000/host_0000/status.json`
    - if the TPU survives, the next boundary is materialization
- 2026-03-21T11:14Z:
  - Retry 6 crossed the prior spot-failure boundary:
    - `sampling/phase_0000/host_0000/status.json` now exists with:
      - `status: succeeded`
      - `num_train_groups: 4`
      - one rollout file under
        `sampling/phase_0000/host_0000/rollouts/`
    - `run_state.json` advanced to `status: materializing`
    - materialization completed and wrote:
      - `materialized/phase_0000/manifest.json`
      - one batch file:
        - `materialized/phase_0000/batches/batch_000000.pkl`
    - materialized manifest contents:
      - `num_rollout_groups: 4`
      - `num_individual_rollouts: 16`
      - `num_training_batches: 1`
      - `global_batch_size: 16`
      - `max_seq_len: 768`
    - the trainer container is now `Up`
  - Most important conclusion:
    - retry 6 is the first `v5p-8` run to get past:
      - rollout generation
      - host status write
      - materialization
    - alternating RL on `v5p-8` is now demonstrably beyond the stage where
      retries 4 and 5 were reclaimed
  - Next:
    - tail trainer startup
    - confirm first training step
    - then watch for export and the phase-0 completion boundary
- 2026-03-21T11:19Z:
  - Retry 6 phase-0 failure:
    - training container exited immediately with:
      - `ValueError: train_batch_size (16) must be divisible by per_device_parallelism * data_axis_size (16, 4)`
    - controller then failed the phase while the pod itself remained:
      - `ACTIVE`
      - `READY`
      - `HEALTHY`
  - Diagnosis:
    - this is not a TPU/runtime failure
    - it is a smoke-profile configuration error:
      - on this `v5p-8` setup, the current trainer settings imply a minimum
        valid global batch of `64`
      - the reduced smoke profile used `train_batch_size=16`, which can never
        satisfy Levanter's batch divisibility rule on this mesh
  - Fix landed:
    - added controller-side validation so alternating RL now rejects this
      configuration before launching sampling/training:
      - `lib/marin/src/marin/rl/alternating/config.py`
    - added regression coverage:
      - `tests/rl/test_alternating_controller.py`
    - validation:
      - `uv run pytest tests/rl/test_alternating_math500_experiment.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_runtime.py tests/rl/test_alternating_state.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_training_phase.py tests/rl/test_train_batch.py -o addopts=`
      - result: `31 passed`
  - Retry 7 plan:
    - reuse the same still-healthy `v5p-8`
    - keep the same image
    - correct the training/sampling shape while keeping only `4` rollout groups:
      - `--train-batch-size 64`
      - `--n-prompts 4`
      - `--n-generations-per-prompt 16`
      - `--eval-examples-per-lesson 8`
      - `--max-input-tokens 512`
      - `--max-output-tokens 256`
    - rationale:
      - `64` satisfies the trainer divisibility rule
      - `64 / 16 = 4` rollout groups, so the next smoke run still keeps the
        same low group count that worked in retry 6
  - Next:
    - launch `alt-tpu-001-v5p-retry7` on the same pod immediately
- 2026-03-21T11:22Z:
  - Retry 7 launched on the same healthy `v5p-8`:
    - exact command:
      - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-001-v5p-retry7 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774084854 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 1 --train-batch-size 64 --n-prompts 4 --n-generations-per-prompt 16 --eval-examples-per-lesson 8 --max-input-tokens 512 --max-output-tokens 256`
  - Current state:
    - pod reuse path
    - no reprovisioning wait
    - same image / same caches / corrected batch shape
  - Next:
    - watch whether retry 7 preserves the retry 6 progress through
      materialization and then clears training startup
- 2026-03-21T11:27Z:
  - Retry 7 progress and failure:
    - warm-restart sampling improved substantially on the reused pod:
      - `prepare_sampling_seconds: 67.01760411262512`
      - `sampling_seconds: 106.89343905448914`
      - `curriculum_update_seconds: 1.4585700035095215`
      - `materialization_seconds: 49.39280605316162`
    - corrected rollout shape succeeded:
      - host status wrote successfully
      - `num_train_groups: 4`
      - `num_individual_rollouts: 64`
      - `global_batch_size: 64`
    - training startup also succeeded:
      - W&B run initialized
      - HF checkpoint loading completed
      - first train step completed in about `17s`
    - new phase-0 failure:
      - after the train step, alternating failed because no Levanter checkpoint
        existed under:
        - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry7/levanter_checkpoints/alt-tpu-001-v5p-retry7-alternating-train`
  - Diagnosis:
    - this is a general Levanter checkpoint hook issue, not an alternating-only
      bug
    - `Trainer.train()` calls `run_hooks(..., force=True)` at the end, but the
      default checkpoint hook was discarding the `force` argument
    - result:
      - short one-step runs complete training
      - no final forced checkpoint is written
      - alternating cannot export because there is nothing to export from
  - Fix landed:
    - patched `lib/levanter/src/levanter/trainer.py` so the default checkpoint
      hook forwards `force=True` and uses `info.next_step` when saving the final
      checkpoint
    - added regression coverage in:
      - `lib/levanter/tests/test_metrics.py`
    - validation:
      - `uv run --directory lib/levanter --group test python -m pytest tests/test_tpus.py tests/test_metrics.py`
      - result: `49 passed`
      - `./infra/pre-commit.py --all-files --fix`
      - result: `OK`
  - Next:
    - build and push a fresh incremental image with the Levanter checkpoint fix
    - rerun on the same still-healthy `v5p-8` as `alt-tpu-001-v5p-retry8`
- 2026-03-21T11:31Z:
  - Retry 8 bring-up prerequisites complete:
    - pod is still:
      - `ACTIVE`
      - `READY`
      - `HEALTHY`
    - pushed fresh image with the Levanter checkpoint fix:
      - tag:
        - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774086271`
      - digest:
        - `sha256:e04b6bd8e816040da646b89114e296b6ab543801a2e4433df0aca3ebc60a7eb7`
  - Next:
    - launch `alt-tpu-001-v5p-retry8` on the same pod with the same corrected
      smoke profile as retry 7, but using the new image digest
- 2026-03-21T11:32Z:
  - Retry 8 launched:
    - exact command:
      - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-001-v5p-retry8 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774086271 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 1 --train-batch-size 64 --n-prompts 4 --n-generations-per-prompt 16 --eval-examples-per-lesson 8 --max-input-tokens 512 --max-output-tokens 256`
  - Current state:
    - same active pod
    - new image with final-checkpoint hook fix
    - same corrected rollout / batch geometry that already cleared training startup
  - Next:
    - confirm retry 8 reproduces retry 7 through the first train step
    - then check for:
      - checkpoint creation
      - policy export
      - phase completion
- 2026-03-21T11:35Z:
  - Retry 8 early progress:
    - new image pull succeeded on the still-live pod
    - first sampler container on the new image reached the same warm-prelude
      path as retry 7
    - observed prelude behavior so far:
      - local TPU bootstrap succeeded
      - vLLM engine init succeeded
      - `8` eval requests were added for `prepare-sampling`
      - the reduced `prepare-sampling` prompt loop completed successfully
  - Interpretation:
    - the checkpoint-hook image rebuild did not regress the already-working
      single-host `v5p-8` sampling path
    - retry 8 is on track to test only the checkpoint/export fix, not to
      re-debug sampling again
  - Next:
    - keep driving retry 8 through:
      - host status write
      - materialization
      - training step
      - final checkpoint creation
      - export
- 2026-03-21T11:39Z:
  - Retry 8 checkpoint milestone:
    - retry 8 reproduced the full retry 7 path through:
      - sampling
      - materialization
      - trainer startup
      - first train step
    - the new behavior after the Levanter fix:
      - trainer now logs:
        - `Saving checkpoint at step 1.`
        - `Saving checkpoint at step 1 to gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry8/levanter_checkpoints/alt-tpu-001-v5p-retry8-alternating-train/step-1`
      - checkpoint path is now visible in storage:
        - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry8/levanter_checkpoints/alt-tpu-001-v5p-retry8-alternating-train/step-1/`
  - Most important conclusion:
    - the previous retry-7 blocker is resolved
    - the remaining question for `ALT-TPU-001` is now whether export and final
      phase-state advancement succeed after the checkpoint lands
  - Next:
    - keep tailing retry 8 through export
    - confirm `policy_0001/manifest.json`
    - confirm `run_state.json` reaches `completed`
- 2026-03-21T11:47Z:
  - Retry 8 final diagnosis:
    - retry 8 did not fail on checkpoint creation
    - it failed after training because alternating checked for the checkpoint too
      early, while the Levanter async checkpoint commit was still becoming
      visible in storage
    - direct evidence:
      - alternating raised:
        - `FileNotFoundError: training phase completed without a checkpoint under gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry8/levanter_checkpoints/alt-tpu-001-v5p-retry8-alternating-train`
      - shortly afterward, the expected checkpoint path existed:
        - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry8/levanter_checkpoints/alt-tpu-001-v5p-retry8-alternating-train/step-1/`
  - Interpretation:
    - this is not a `v5p` runtime failure
    - this is not a missing-checkpoint failure
    - it is a post-train visibility/commit race between:
      - Levanter async checkpoint completion
      - alternating export kickoff
  - Fix landed:
    - alternating training now waits for the target `step-N` checkpoint to
      become visible before export:
      - `lib/marin/src/marin/rl/alternating/training_phase.py`
    - Levanter `Trainer.train()` now waits for the default checkpointer to
      finish its forced final save before returning:
      - `lib/levanter/src/levanter/trainer.py`
    - new regressions:
      - `tests/rl/test_alternating_training_phase.py`
      - `lib/levanter/tests/test_metrics.py`
  - Validation:
    - alternating targeted tests:
      - `uv run pytest tests/rl/test_alternating_training_phase.py tests/rl/test_alternating_runtime.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_state.py tests/rl/test_train_batch.py tests/rl/test_alternating_math500_experiment.py -o addopts=`
      - result: `32 passed`
    - Levanter targeted tests:
      - `uv run --directory lib/levanter --group test python -m pytest tests/test_metrics.py tests/test_tpus.py`
      - result: `50 passed`
    - repo validation:
      - `./infra/pre-commit.py --all-files --fix`
      - result: `OK`
  - Current infra state:
    - queued resource `alt-v5p-probe-001` is still:
      - `ACTIVE`
    - plan is to reuse the same pod again
  - Retry 9 plan:
    - build and push one more incremental image containing the
      checkpoint-visibility fix
    - relaunch the same single-host smoke profile as:
      - `alt-tpu-001-v5p-retry9`
    - success condition:
      - `policy_0001/manifest.json` is written
      - `run_state.json` reaches `completed`
    - if retry 9 clears:
      - move immediately to the next planned experiment:
        - `ALT-TPU-002` single-host two-phase warm-cache validation
- 2026-03-21T11:55Z:
  - Retry 9 launch prerequisites complete:
    - queued resource `alt-v5p-probe-001` is still:
      - `ACTIVE`
    - pushed fresh incremental image containing:
      - alternating checkpoint-visibility wait
      - Levanter final-checkpointer wait
    - image tag:
      - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774087580`
    - image digest:
      - `sha256:81d2e8031c852456066c8a089804409134e3bf9d84ea4ac38ce1b15d279c0e56`
  - Next:
    - launch `alt-tpu-001-v5p-retry9` on the same `v5p-8`
    - hold all smoke-test knobs fixed from retry 8 so only the checkpoint/export
      boundary changes
- 2026-03-21T11:56Z:
  - Retry 9 launched:
    - exact command:
      - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-001-v5p-retry9 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774087580 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 1 --train-batch-size 64 --n-prompts 4 --n-generations-per-prompt 16 --eval-examples-per-lesson 8 --max-input-tokens 512 --max-output-tokens 256`
  - Current state:
    - same pod
    - same smoke geometry as retry 8
    - only new variable is the checkpoint/export boundary fix
  - Success condition:
    - `policy_0001/manifest.json` exists
    - run status reaches `completed`
- 2026-03-21T12:05Z:
  - Retry 9 intermediate milestone:
    - `retry9` has already re-cleared the entire pre-export path on the same
      `v5p-8`:
      - sampling
      - curriculum update
      - materialization
      - trainer launch
    - durable artifacts now present:
      - `materialized/phase_0000/manifest.json`
      - `state/run_state.json` with `status: training`
    - materialized manifest contents:
      - `num_rollout_groups: 4`
      - `num_individual_rollouts: 64`
      - `num_training_batches: 1`
      - `global_batch_size: 64`
      - `max_seq_len: 768`
    - phase-0 metrics so far:
      - `prepare_sampling_seconds: 83.73838686943054`
      - `sampling_seconds: 71.1819121837616`
      - `curriculum_update_seconds: 1.5072476863861084`
      - `materialization_seconds: 49.93586301803589`
  - Interpretation:
    - the checkpoint/export fix did not regress the already-working sampling or
      materialization path
    - the run is now testing only the training/checkpoint/export boundary
  - Next:
    - wait for:
      - first train step
      - checkpoint save
      - export
      - final run completion
- 2026-03-21T12:14Z:
  - Retry 9 export interruption:
    - phase 0 training completed and checkpoint commit finished successfully
    - export began and wrote into `policies/policy_0001`
    - before export finished, the queued resource was reclaimed again:
      - queued resource state:
        - `SUSPENDED`
      - initiator:
        - `SERVICE`
    - this was another spot-capacity interruption, not a code regression
  - Positive evidence captured before suspension:
    - training reached the previously failing boundary and passed it:
      - `Finished committing to storage layer by process: 0`
      - `Saved checkpoint to .../step-1 for step 1`
      - `Loading checkpoint from .../step-1...`
      - `Saving HF-compatible checkpoint to gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry9/policies/policy_0001`
    - committed checkpoint metadata exists:
      - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry9/levanter_checkpoints/alt-tpu-001-v5p-retry9-alternating-train/step-1/metadata.json`
  - Recovery plan chosen:
    - do not rerun sampling or training
    - delete the incomplete `policy_0001` directory
    - reprovision the same spot `v5p-8`
    - run `export-policy` only against phase `0`
    - once `policy_0001/manifest.json` exists, rerun the controller once so it
      observes the next policy manifest and advances `run_state.json` to
      `completed`
- 2026-03-21T12:16Z:
  - Retry 9 export-only recovery launched:
    - partial `policy_0001` cleaned up
    - exact recovery subcommand:
      - `export-policy --config-path gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry9/state/controller_config.pkl --phase-id 0`
    - launch mechanism:
      - reprovision same `alt-v5p-probe-001` spot `v5p-8` and run only the
        export container
  - Next:
    - wait for `policies/policy_0001/manifest.json`
    - rerun the controller once to finalize the run state
- 2026-03-21T12:33Z:
  - ALT-TPU-001 result:
    - passed on `v5p-8`, with export completed via export-only recovery after a
      spot interruption during the original export attempt
  - Final durable evidence:
    - policy manifest:
      - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry9/policies/policy_0001/manifest.json`
    - run state:
      - `status: completed`
      - `phase_id: 1`
      - `policy_version: 1`
      - `source_global_step: 1`
      - `last_completed_phase: 0`
    - retained checkpoint:
      - `gs://marin-us-east5/alternating-rl/alt-tpu-001-v5p-retry9/levanter_checkpoints/alt-tpu-001-v5p-retry9-alternating-train/step-1`
  - Final phase-0 timings:
    - `prepare_sampling_seconds: 83.73838686943054`
    - `sampling_seconds: 71.1819121837616`
    - `curriculum_update_seconds: 1.5072476863861084`
    - `materialization_seconds: 49.93586301803589`
    - `training_seconds: 156.22245407104492`
    - `export_seconds: 531.1827099323273`
  - Conclusions:
    - alternating single-host RL now works end to end on a real `v5p-8`
    - the original post-train checkpoint race is fixed
    - the remaining operational risk for spot runs is infra preemption during
      large exports, but export-only recovery from a committed Levanter
      checkpoint works
  - Next:
    - move immediately to `ALT-TPU-002`
    - reuse the still-active `alt-v5p-probe-001`
    - keep the same image and same smoke geometry
    - increase only `num_train_steps` from `1` to `2` so phase 2 can measure
      warm-cache boundary cost
- 2026-03-21T12:35Z:
  - ALT-TPU-002 launched:
    - run id:
      - `alt-tpu-002-v5p-retry1`
    - exact command:
      - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-002-v5p-retry1 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774087580 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 2 --train-batch-size 64 --n-prompts 4 --n-generations-per-prompt 16 --eval-examples-per-lesson 8 --max-input-tokens 512 --max-output-tokens 256`
  - Experiment goal:
    - validate warm-cache reuse across a second alternating phase on the same
      live `v5p-8`
  - Success criterion:
    - two full phases complete
    - `state/phase_metrics/phase_0001.json` exists
    - phase-1 boundary cost is materially lower than phase-0 cold-start cost
- 2026-03-21T12:37Z:
  - ALT-TPU-002 retry 1 result:
    - null result due spot interruption before phase-0 sampling actually
      launched
  - Evidence:
    - the existing `alt-v5p-probe-001` spot pod flipped to service-driven
      reclaim immediately after controller bootstrap:
      - queued resource state observed:
        - `SUSPENDING`
        - then TPU VM SSH reported `DELETING`
    - controller failure:
      - failed while trying to stop the old sampler container because the TPU
        was already being deleted
  - Interpretation:
    - this does not say anything about warm-cache reuse yet
    - it is a pure capacity interruption and should be retried with the same
      experiment config
  - Next:
    - relaunch unchanged as `alt-tpu-002-v5p-retry2`
    - let the helper delete/recreate the queued resource if needed
- 2026-03-21T12:39Z:
  - ALT-TPU-002 retry 2 launched:
    - run id:
      - `alt-tpu-002-v5p-retry2`
    - config unchanged from retry 1
  - Goal:
    - first successful two-phase warm-cache measurement on `v5p-8`
- 2026-03-21T12:42Z:
  - TPU helper fix while chasing ALT-TPU-002:
    - found a local orchestration bug:
      - `start_tpu_vm_queued_resources()` treated `SUSPENDING` like a normal
        "wait for ACTIVE" state
      - for reclaimed spot TPUs, that is wrong and can waste minutes on a
        resource that is already being deleted
    - fix landed:
      - treat `SUSPENDING` and `DELETING` like reclaim states
      - delete or wait-for-absence, then recreate
      - file:
        - `lib/levanter/src/levanter/infra/tpus.py`
    - regression:
      - `lib/levanter/tests/test_tpus.py`
      - result: `15 passed`
  - Next:
    - relaunch ALT-TPU-002 as `alt-tpu-002-v5p-retry3` from the fixed helper
- 2026-03-21T12:43Z:
  - ALT-TPU-002 retry 3 launched:
    - run id:
      - `alt-tpu-002-v5p-retry3`
    - same experiment config as retry 1/2
    - local controller now includes the reclaim-state helper fix
- 2026-03-21T13:02Z:
  - ALT-TPU-002 retry 3 durable-state refresh:
    - `state/run_state.json` confirms retry 3 completed phase 0 and then failed
      in phase 1:
      - `status: failed`
      - `phase_id: 1`
      - `policy_version: 1`
      - `source_global_step: 1`
      - `last_completed_phase: 0`
      - `current_policy_manifest_path: gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry3/policies/policy_0001/manifest.json`
      - `current_levanter_checkpoint_path: gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry3/levanter_checkpoints/alt-tpu-002-v5p-retry3-alternating-train/step-1`
      - `current_sampling_manifest: gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry3/sampling/phase_0001/manifest.json`
    - phase-0 metrics:
      - `prepare_sampling_seconds: 193.23569416999817`
      - `sampling_seconds: 106.31922316551208`
      - `curriculum_update_seconds: 1.5463597774505615`
      - `materialization_seconds: 49.71469068527222`
      - `training_seconds: 219.281813621521`
      - `export_seconds: 512.4452035427094`
    - `policy_0001/manifest.json` confirms phase 1 switches to the exported
      policy with fast bootstrap enabled:
      - `enable_fast_bootstrap: true`
      - `policy_path: gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry3/policies/policy_0001`
      - `model_name: meta-llama/Llama-3.1-8B-Instruct`
  - Interpretation update:
    - the new phase-1 blocker is more likely a fast-bootstrap memory sizing
      issue than stale trainer HBM
    - phase 0 uses the Hugging Face checkpoint path directly and succeeds
    - phase 1 is the first time the alternating path exercises:
      - exported GCS checkpoint
      - `enable_fast_bootstrap=true`
      - vLLM `load_format="dummy"` bootstrap path
    - the observed failure:
      - `ValueError: total_hbm_used_gb=359.65GiB exceeds total_hbm_limit_cap_gb=344.68GiB by 14.97GiB`
      is therefore plausibly specific to fast bootstrap at
      `gpu_memory_utilization=0.90`
  - Next:
    - expose an explicit alternating experiment knob for
      `inference_gpu_memory_utilization`
    - rerun ALT-TPU-002 on the same `v5p-8` smoke geometry with a higher
      vLLM memory utilization before assuming the architecture needs more
      invasive cleanup logic
- 2026-03-21T13:08Z:
  - Code change to test the phase-1 fast-bootstrap HBM hypothesis:
    - added controller CLI override:
      - `--inference-gpu-memory-utilization`
      - file: `experiments/alternating_rl_math500.py`
    - wired the override into `RLExperimentConfig` construction so the exact
      vLLM memory-utilization value is persisted in the pickled controller config
    - added alternating config validation that
      `inference.gpu_memory_utilization` stays in `(0, 1]`
      - file: `lib/marin/src/marin/rl/alternating/config.py`
  - Regression coverage:
    - `tests/rl/test_alternating_math500_experiment.py`
      - verifies the new override survives argument parsing and experiment-config
        construction
    - `tests/rl/test_alternating_controller.py`
      - verifies alternating config validation rejects invalid memory
        utilization
  - Validation:
    - `uv run pytest tests/rl/test_alternating_math500_experiment.py tests/rl/test_alternating_controller.py tests/rl/test_alternating_runtime.py tests/rl/test_alternating_training_phase.py tests/rl/test_alternating_materialization.py tests/rl/test_alternating_state.py tests/rl/test_train_batch.py -o addopts=`
      - result: `33 passed`
    - `uv run python experiments/alternating_rl_math500.py controller --help`
      - confirms the new CLI flag is exposed
  - Planned next run:
    - `alt-tpu-002-v5p-retry4`
    - exact command:
      - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-002-v5p-retry4 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774087580 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 2 --train-batch-size 64 --n-prompts 4 --n-generations-per-prompt 16 --eval-examples-per-lesson 8 --max-input-tokens 512 --max-output-tokens 256 --inference-gpu-memory-utilization 0.95`
  - Hypothesis for retry 4:
    - if the phase-1 failure is a fast-bootstrap memory-sizing issue,
      `0.95` should allow `prepare-sampling --phase-id 1` to clear on the same
      `v5p-8`
  - Decision rule:
    - if retry 4 clears phase 1 and reaches phase-1 sampling or training, keep
      pushing warm-cache validation on this path
    - if retry 4 fails with the same HBM signature, inspect whether alternating
      needs a different fast-bootstrap path or a stronger between-phase TPU
      runtime cleanup step
- 2026-03-21T13:12Z:
  - ALT-TPU-002 retry 4 launched from the local controller:
    - run id:
      - `alt-tpu-002-v5p-retry4`
    - exact command:
      - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-002-v5p-retry4 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774087580 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 2 --train-batch-size 64 --n-prompts 4 --n-generations-per-prompt 16 --eval-examples-per-lesson 8 --max-input-tokens 512 --max-output-tokens 256 --inference-gpu-memory-utilization 0.95`
  - Allocator behavior at launch:
    - detected the previous queued resource as `SUSPENDED`
    - deleted it automatically
    - recreated `alt-v5p-probe-001`
    - current queued-resource state after recreate:
      - `WAITING_FOR_RESOURCES`
  - What this run is testing:
    - same `v5p-8` warm-cache profile as retry 3
    - only material change is higher vLLM memory utilization for the sampling
      engine
  - Immediate next checkpoint:
    - wait for the recreated pod to become `ACTIVE`
    - then tail phase 0 through export and specifically watch whether
      `prepare-sampling --phase-id 1` clears the previous fast-bootstrap HBM
      failure
- 2026-03-21T14:41Z:
  - ALT-TPU-002 retry 4 in-progress checkpoint:
    - allocator path succeeded:
      - deleted the suspended spot resource
      - recreated `alt-v5p-probe-001`
      - waited through `WAITING_FOR_RESOURCES` -> `PROVISIONING` -> `ACTIVE`
    - durable run state now exists and shows the controller actively running
      phase 0:
      - `status: sampling`
      - `phase_id: 0`
      - `policy_version: 0`
    - current evidence from controller logs:
      - vLLM init on `v5p-8` is using `gpu_memory_utilization=0.95`
      - phase-0 HBM profiling passed:
        - `total_hbm_limit_cap_gb=363.82GiB`
        - `total_hbm_used_gb=14.97GiB`
        - `total_hbm_avail_gb=348.85GiB`
      - phase-0 `prepare-sampling` completed the reduced 8-example eval prelude
      - per-host sampling launched afterward and generated the phase-0 rollouts
      - controller has already updated curriculum stats from training rollouts
    - storage evidence:
      - `state/phase_metrics/phase_0000.json.tmp` now exists, so phase-0 timing
        output is being written
  - Current interpretation:
    - the increased vLLM memory-utilization setting did not break phase 0
    - retry 4 is still live and has progressed beyond the setup stages that
      were previously sensitive to reprovisioning noise
  - Next:
    - continue tailing retry 4 through:
      - phase-0 materialization
      - phase-0 training
      - phase-0 export
      - phase-1 `prepare-sampling`
    - phase-1 `prepare-sampling` remains the decisive checkpoint for the
      fast-bootstrap HBM hypothesis
- 2026-03-21T14:49Z:
  - ALT-TPU-002 retry 4 phase-0 training checkpoint:
    - `run_state.json` has advanced from `sampling` to `training`
    - finalized phase-0 metrics so far:
      - `prepare_sampling_seconds: 213.5215620994568`
      - `sampling_seconds: 106.91997528076172`
      - `curriculum_update_seconds: 1.4848949909210205`
      - `materialization_seconds: 50.23589015007019`
      - `training_seconds: null` (still in progress at the time of this note)
      - `export_seconds: null`
    - materialized phase-0 manifest:
      - `num_rollout_groups: 4`
      - `num_individual_rollouts: 64`
      - `num_training_batches: 1`
      - `global_batch_size: 64`
      - `max_seq_len: 768`
    - trainer evidence from controller logs:
      - W&B run initialized successfully:
        - `https://wandb.ai/marin-community/marin_post_training/runs/alt-tpu-002-v5p-retry4-alternating-train`
      - full HF checkpoint load from `meta-llama/Llama-3.1-8B-Instruct` completed
      - first train step finished in about `16.9s`
      - checkpoint save for phase 0 started:
        - `Saving checkpoint at step 1 to gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry4/levanter_checkpoints/alt-tpu-002-v5p-retry4-alternating-train/step-1`
      - storage commit has begun:
        - `Starting commit to storage layer by process: 0`
  - Interpretation:
    - retry 4 has now re-cleared the entire phase-0 path up through training on
      the recreated `v5p-8`
    - the remaining checkpoints are:
      - phase-0 checkpoint commit completion
      - phase-0 export completion
      - phase-1 `prepare-sampling` under fast bootstrap with `0.95`
- 2026-03-21T14:55Z:
  - ALT-TPU-002 retry 4 export in progress:
    - phase-0 checkpoint commit finished successfully:
      - `Saved checkpoint to gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry4/levanter_checkpoints/alt-tpu-002-v5p-retry4-alternating-train/step-1 for step 1`
    - export immediately started from the committed checkpoint:
      - `Loading checkpoint from .../step-1`
      - `Saving HF-compatible checkpoint to gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry4/policies/policy_0001`
    - export has already written at least the first shards of `policy_0001`:
      - shard `model-00001-of-00007.safetensors` saved and copied
      - shard `model-00002-of-00007.safetensors` saved and copied
      - shard `model-00003-of-00007.safetensors` started and is copying
  - Interpretation:
    - retry 4 has passed the original phase-0 training/checkpoint boundary and
      is now in the same long export window that was previously vulnerable to
      spot reclaim
    - if spot capacity survives export, the next decisive signal is phase-1
      `prepare-sampling` under fast bootstrap at `0.95`
    - if spot capacity does not survive export, phase-0 is still recoverable by
      the existing export-only recovery path
- 2026-03-21T15:03Z:
  - ALT-TPU-002 retry 4 result:
    - phase 0 completed fully on the recreated `v5p-8`:
      - training finished
      - checkpoint commit finished
      - export finished
      - `policy_0001/manifest.json` exists
      - `run_state.json` advanced to phase 1 before failure:
        - `last_completed_phase: 0`
        - `policy_version: 1`
        - `source_global_step: 1`
        - `current_policy_manifest_path: .../policy_0001/manifest.json`
        - `current_levanter_checkpoint_path: .../step-1`
    - final phase-0 timings:
      - `prepare_sampling_seconds: 213.5215620994568`
      - `sampling_seconds: 106.91997528076172`
      - `curriculum_update_seconds: 1.4848949909210205`
      - `materialization_seconds: 50.23589015007019`
      - `training_seconds: 170.10359048843384`
      - `export_seconds: 547.071382522583`
      - total:
        - `1089.3372955322266`
  - Failure signature:
    - phase-1 `prepare-sampling` no longer failed at vLLM's initial
      `total_hbm_limit_cap_gb` check
    - with `gpu_memory_utilization=0.95`, phase-1 engine init succeeded and
      fast-bootstrap checkpoint shards started streaming from GCS
    - the actual failure moved into fast-bootstrap weight injection:
      - `jax.errors.JaxRuntimeError: RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 1002.00M. That was not possible. There are 897.41M free.; (0x0x0_HBM0)`
      - stack:
        - `marin.rl.environments.inference_ctx.vllm._bootstrap_weights_into_engine`
        - `tpu_inference.worker.tpu_worker.sync_weights`
        - `transfer_state_with_mappings`
    - after that first failure, the current image's broad fallback path tried to
      initialize again and hit a second OOM with only ~23MB free; that fallback
      is not semantically valid for alternating RL because it can only fall back
      to the base model, not the newly trained policy
  - Interpretation update:
    - `0.95` is above the engine-start threshold but too high for the temporary
      HBM overhead of `sync_weights`
    - the plausible sweet spot is between:
      - `0.90` (too low for startup)
      - `0.95` (too high for weight injection headroom)
    - the next obvious probe is `0.94`
  - Code follow-up:
    - local patch landed in `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
      so fast-bootstrap failure now raises instead of silently falling back to
      the base model
    - new regression coverage added in `tests/rl/test_inference_ctx.py`
  - Next:
    - validate the new local fast-bootstrap test path
    - relaunch unchanged as `alt-tpu-002-v5p-retry5` with:
      - `--inference-gpu-memory-utilization 0.94`
    - keep the same image for the immediate memory-sweet-spot experiment if we
      want the fastest signal
    - if `0.94` clears phase 1, then decide whether to push a fresh image with
      the fast-bootstrap no-fallback fix before moving to the next experiment
- 2026-03-21T15:07Z:
  - Local validation before retry 5:
    - targeted tests passed after the fast-bootstrap no-fallback patch:
      - `uv run pytest tests/rl/test_inference_ctx.py tests/rl/test_alternating_math500_experiment.py tests/rl/test_alternating_controller.py -o addopts=`
      - result: `21 passed`
    - queued resource `alt-v5p-probe-001` is still `ACTIVE`, so retry 5 can
      reuse the same live `v5p-8`
  - Retry 5 plan:
    - run id:
      - `alt-tpu-002-v5p-retry5`
    - exact command:
      - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-002-v5p-retry5 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774087580 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 2 --train-batch-size 64 --n-prompts 4 --n-generations-per-prompt 16 --eval-examples-per-lesson 8 --max-input-tokens 512 --max-output-tokens 256 --inference-gpu-memory-utilization 0.94`
  - Hypothesis:
    - `0.94` is high enough to satisfy phase-1 engine startup but low enough to
      leave enough HBM headroom for `sync_weights`
- 2026-03-21T15:10Z:
  - ALT-TPU-002 retry 5 launched:
    - run id:
      - `alt-tpu-002-v5p-retry5`
    - reuse behavior:
      - same `alt-v5p-probe-001` queued resource stayed `ACTIVE`
      - no reprovision needed
    - durable state:
      - `state/run_state.json` already exists with:
        - `status: sampling`
        - `phase_id: 0`
        - `policy_version: 0`
        - `last_completed_phase: -1`
  - Immediate checkpoint:
    - wait for phase 0 to clear quickly on the reused pod
    - then watch phase-1 `prepare-sampling` at `0.94`
- 2026-03-21T16:33Z:
  - ALT-TPU-002 retry 5 early evidence:
    - phase-0 sampler startup is using:
      - `gpu_memory_utilization=0.94`
    - phase-0 memory profile on the live `v5p-8`:
      - `total_hbm_limit_cap_gb=359.99GiB`
      - `total_hbm_used_gb=14.97GiB`
      - `total_hbm_avail_gb=345.02GiB`
      - per-chip post-KV HBM:
        - `90.0 / 95.74 GiB`
    - comparison against retry 4 at `0.95`:
      - retry 4 cap:
        - `363.82GiB`
      - retry 4 available before KV:
        - `348.85GiB`
      - retry 5 therefore leaves about `3.83GiB` more total headroom, roughly
        `0.96GiB` per chip
    - phase-0 reduced `prepare-sampling` completed successfully and launched the
      per-host sampler container on the reused pod
  - Parallel build work:
    - started pushing a fresh incremental image tag so the local fast-bootstrap
      no-fallback fix can be used on TPU once the immediate `0.94` probe is
      resolved:
      - tag:
        - `alt-rl-1774110754`
      - command:
        - `uv run python lib/levanter/infra/push_docker.py --docker_target gcp --project hai-gcp-models --region us-east5 --repository levanter --image levanter-ahmed --tag alt-rl-1774110754 --docker_file lib/levanter/docker/tpu/Dockerfile.incremental`
- 2026-03-21T17:17Z:
  - ALT-TPU-002 retry 5 result:
    - phase 0 completed fully on the reused `v5p-8`:
      - training finished
      - checkpoint commit finished
      - export finished
      - `policy_0001/manifest.json` exists
      - `run_state.json` advanced to phase 1 before failure:
        - `last_completed_phase: 0`
        - `policy_version: 1`
        - `source_global_step: 1`
        - `current_policy_manifest_path: .../policy_0001/manifest.json`
        - `current_levanter_checkpoint_path: .../step-1`
    - final phase-0 timings:
      - `prepare_sampling_seconds: 74.33631992340088`
      - `sampling_seconds: 106.5502290725708`
      - `curriculum_update_seconds: 1.5024440288543701`
      - `materialization_seconds: 50.57193088531494`
      - `training_seconds: 196.11715722084045`
      - `export_seconds: 540.2181739807129`
      - total:
        - `969.2962551116943`
  - Failure signature:
    - phase-1 `prepare-sampling` again failed inside fast-bootstrap
      `sync_weights`, not at vLLM startup
    - first OOM:
      - `ValueError: RESOURCE_EXHAUSTED: Error allocating device buffer: Attempting to allocate 1.96G. That was not possible. There are 626.91M free.; (0x0x0_HBM0)`
      - stack still routes through:
        - `marin.rl.environments.inference_ctx.vllm._bootstrap_weights_into_engine`
        - `tpu_inference.worker.tpu_worker.sync_weights`
        - `tpu_inference.runner.tpu_runner.TPUModelRunner._sync_weights`
        - `tpu_inference.models.jax.utils.weight_utils.transfer_state_with_mappings`
    - the current TPU image was still the old digest
      `sha256:81d2e803...`, so after the first failure it still fell into the
      old broad fallback path and hit a second tiny-allocation OOM on a nearly
      exhausted device; that second error is noise, not the primary blocker
  - New direct code insight:
    - inspected `tpu_inference` inside the image
    - `_sync_weights()` is a thin wrapper around
      `transfer_state_with_mappings(...)`
    - `transfer_state_with_mappings(...)` does:
      - `new_v = jnp.copy(v.value)`
      - optional transpose / dtype cast
      - `shard_put(v_maybe_t, sharding)` into the live model state
    - interpretation:
      - phase-1 fast bootstrap is paying temporary HBM overhead while copying
        each new weight into a model that already owns:
        - dummy-initialized params
        - a KV cache sized from `gpu_memory_utilization`
      - this makes the remaining HBM headroom the critical control knob
  - Image status:
    - fresh incremental image tag is now published:
      - `alt-rl-1774110754`
      - digest:
        - `sha256:5b25f33e6211edbee04001545a0280b6e8e8543706dddd44ac7633aab6d22d5d`
    - this image includes the fast-bootstrap no-fallback patch, so the next TPU
      retry will emit only the primary failure instead of masking it with a
      semantically invalid base-model fallback
  - Cleanup checkpoint:
    - live pod `alt-v5p-probe-001` stayed `ACTIVE`
    - no reprovision needed
    - stale `marin-alt-sampler` container is exited, not running
  - Next:
    - relaunch as `alt-tpu-002-v5p-retry6`
    - switch to the new image tag `alt-rl-1774110754`
    - lower `--inference-gpu-memory-utilization` to `0.92`
  - Hypothesis:
    - `0.92` is low enough to buy the extra temporary HBM headroom required by
      `sync_weights`, while still staying above the known-bad startup floor at
      `0.90`
- 2026-03-21T17:19Z:
  - ALT-TPU-002 retry 6 launched:
    - run id:
      - `alt-tpu-002-v5p-retry6`
    - exact command:
      - `uv run python experiments/alternating_rl_math500.py controller --run-id alt-tpu-002-v5p-retry6 --shared-root gs://marin-us-east5/alternating-rl --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774110754 --tpu-name alt-v5p-probe-001 --tpu-type v5p-8 --zone us-east5-a --num-hosts 1 --local-tensor-parallel-size 4 --capacity-type spot --runtime-version v2-alpha-tpuv5 --steps-per-phase 1 --num-train-steps 2 --train-batch-size 64 --n-prompts 4 --n-generations-per-prompt 16 --eval-examples-per-lesson 8 --max-input-tokens 512 --max-output-tokens 256 --inference-gpu-memory-utilization 0.92`
    - image digest:
      - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:5b25f33e6211edbee04001545a0280b6e8e8543706dddd44ac7633aab6d22d5d`
    - reuse behavior:
      - same `alt-v5p-probe-001` queued resource stayed `ACTIVE`
      - controller observed only an exited stale `marin-alt-sampler`; no live
        alternating containers were left behind
    - early controller output:
      - curriculum checkpoint for the new run wrote successfully
      - curriculum checkpoint restored successfully on the worker
  - Immediate checkpoint:
    - wait for phase-0 `prepare-sampling` under `0.92`
    - if startup clears, continue watching phase-1 `prepare-sampling` because
      that remains the decisive bottleneck
- 2026-03-21T17:20Z:
  - ALT-TPU-002 retry 6 early evidence:
    - phase-0 sampler startup is using:
      - `gpu_memory_utilization=0.92`
    - phase-0 memory profile on the live `v5p-8`:
      - `total_hbm_limit_cap_gb=352.33GiB`
      - `total_hbm_used_gb=14.97GiB`
      - `total_hbm_avail_gb=337.36GiB`
      - per-chip post-KV HBM:
        - `88.08 / 95.74 GiB`
    - vLLM engine init completed successfully:
      - `init engine (profile, create kv cache, warmup model) took 2.85 seconds`
  - Interpretation update:
    - `0.92` is still above the phase-0 startup floor
    - the experiment remains viable
    - the decisive checkpoint is unchanged:
      - phase-1 fast bootstrap `sync_weights`
- 2026-03-21T17:34Z:
  - ALT-TPU-002 retry 6 result so far:
    - `0.92` cleared phase-0 startup and phase-0 completed substantially
      farther than any prior retry:
      - sampling completed
      - host status for `sampling/phase_0000/host_0000/status.json` exists
      - materialization completed
      - `materialized/phase_0000/manifest.json` exists
      - training completed its single step
      - checkpoint commit completed
      - durable checkpoint exists under:
        - `gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/levanter_checkpoints/alt-tpu-002-v5p-retry6-alternating-train/step-1`
    - export started and wrote at least the first HF shard:
      - `policies/policy_0001/model-00001-of-00007.safetensors` exists
  - Failure signature:
    - the TPU was reclaimed by the service during HF export, not during
      sampling, materialization, training, checkpoint commit, or fast bootstrap
    - queued resource state changed to:
      - `SUSPENDING`
      - initiator:
        - `SERVICE`
    - controller logs show repeated SSH resets while the pod is disappearing
  - Interpretation update:
    - this retry did not answer the phase-1 fast-bootstrap question because the
      run never reached phase 1
    - it did prove a stronger statement than before:
      - `v5p-8` with `gpu_memory_utilization=0.92` can clear phase-0 startup,
        generation, materialization, training, and checkpoint commit
    - the active blocker for this exact run is now export recovery after a spot
      reclaim
  - Recovery plan:
    - do not throw away the run
    - use export-only recovery against phase 0 once `alt-v5p-probe-001` is back
      or recreated
    - reason:
      - the checkpoint is complete and durable
      - export-only recovery discovers the latest checkpoint under the trainer
        checkpoint root and can finish `policy_0001` without rerunning phase 0
  - Next:
    - wait for the queued resource to finish suspending / be recyclable
    - bring `alt-v5p-probe-001` back
    - run `export-policy --phase-id 0` for `alt-tpu-002-v5p-retry6`
    - if export-only succeeds, resume the next experiment from the recovered
      `policy_0001` state rather than rerunning phase 0
- 2026-03-21T17:35Z:
  - Export-only recovery attempt queued for `retry6`:
    - goal:
      - salvage `policy_0001` from the already-committed phase-0 checkpoint
      - avoid rerunning sampling/materialization/training after a spot reclaim
        during export
    - launcher strategy:
      - use the existing saved controller config at
        `gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/state/controller_config.pkl`
      - recreate / reactivate `alt-v5p-probe-001` if needed
      - run the remote subcommand:
        - `export-policy --phase-id 0`
      - use the training compilation cache path, not the sampling cache path
  - Hypothesis:
    - export-only recovery should succeed because:
      - the phase-0 checkpoint is complete and discoverable
      - `export_policy_only(...)` discovers the latest checkpoint under the
        trainer checkpoint root even when `run_state.current_levanter_checkpoint_path`
        is stale or null
- 2026-03-21T17:37Z:
  - Export-only recovery is in progress:
    - local recovery launcher started successfully
    - it observed the queued resource had already moved to:
      - `SUSPENDED`
    - helper behavior:
      - deleted the suspended queued resource cleanly
      - recreated `alt-v5p-probe-001`
    - current infra state:
      - queued resource exists again
      - state:
        - `WAITING_FOR_RESOURCES`
  - Immediate next step:
    - once the new `v5p-8` becomes `ACTIVE`, run the single-host
      `export-policy --phase-id 0` recovery container and finish `policy_0001`
- 2026-03-21T17:58Z:
  - Export-only recovery succeeded for `retry6`:
    - recreated `alt-v5p-probe-001` reached `ACTIVE`
    - single-host `export-policy --phase-id 0` completed on the new pod
    - `policy_0001` is now complete and durable:
      - `manifest.json` exists
      - all `model-00001-of-00007` through `model-00007-of-00007` shards exist
      - tokenizer/config metadata exists
    - policy manifest fields:
      - `phase_id: 0`
      - `policy_version: 1`
      - `source_global_step: 1`
      - `levanter_checkpoint_path: .../step-1`
  - Interpretation update:
    - `retry6` no longer needs a full phase-0 rerun
    - the correct next action is to resume the original controller in place
      from the recovered manifest
    - controller behavior should be:
      - observe `run_state.status == training`
      - observe next policy manifest already exists
      - call `advance_run_state(...)`
      - continue at phase 1
  - Next:
    - rerun the `retry6` controller command unchanged
    - use the resumed run to test the real outstanding question:
      - phase-1 fast bootstrap at `gpu_memory_utilization=0.92`
- 2026-03-21T18:02Z:
  - Resume gap found and fixed:
    - rerunning the recovered controller initially exited immediately because
      `run_state.json` was still `status: failed`
    - root cause:
      - `export_policy_only(...)` wrote the recovered next policy manifest but
        did not heal durable controller state
      - controller returns early on `RunStatus.FAILED`, so it never got a chance
        to notice the recovered `policy_0001`
  - Code fix:
    - updated `lib/marin/src/marin/rl/alternating/training_phase.py`
    - `export_policy_only(...)` now rewrites `run_state.json` back to:
      - `status: training`
      - `current_levanter_checkpoint_path: <recovered checkpoint>`
    - this recovery is idempotent:
      - it also heals state when rerun after `policy_0001` already exists
    - regression coverage added in:
      - `tests/rl/test_alternating_training_phase.py`
  - Validation:
    - `uv run pytest tests/rl/test_alternating_training_phase.py -o addopts=`
      passed
    - `./infra/pre-commit.py --all-files --fix` passed
  - Live recovery:
    - reran local `export-policy --phase-id 0` for `retry6`
    - confirmed `run_state.json` now shows:
      - `status: training`
      - `current_levanter_checkpoint_path: .../step-1`
    - reran the unchanged `retry6` controller command
    - controller resumed correctly:
      - logged recovered phase-0 timings
      - moved back into the normal loop
      - began phase-1 startup
  - Next:
    - keep `retry6` running
    - watch phase-1 `prepare-sampling` at `gpu_memory_utilization=0.92`
- 2026-03-21T18:04Z:
  - `retry6` is now truly resumed into phase 1:
    - `run_state.json` shows:
      - `status: sampling`
      - `phase_id: 1`
      - `policy_version: 1`
      - `source_global_step: 1`
      - `current_policy_manifest_path: .../policy_0001/manifest.json`
      - `current_sampling_manifest: .../sampling/phase_0001/manifest.json`
      - `last_completed_phase: 0`
    - live controller logs show phase-1 sampler startup on `v5p-8`
  - Immediate next checkpoint:
    - whether phase-1 `prepare-sampling` clears fast bootstrap at
      `gpu_memory_utilization=0.92`
- 2026-03-21T18:20Z:
  - Phase-1 fast bootstrap result for `retry6`:
    - phase-1 `prepare-sampling` reached fast bootstrap on `v5p-8`
    - this is no longer an OOM / HBM-headroom failure
    - the new failure is a deterministic shape/transpose bug during weight
      injection:
      - `ValueError: axis 2 is out of bounds for array of dimension 2`
      - stack:
        - `tpu_inference.models.jax.utils.weight_utils.transfer_state_with_mappings`
        - `jnp.transpose(new_v, transpose_keys[src_keys[-1]])`
    - surfaced through Marin as:
      - `RuntimeError: Fast bootstrap failed while loading policy weights. Alternating RL cannot safely fall back to the base model for a later policy version.`
  - Interpretation update:
    - the `0.92` memory setting is good enough to reach the actual weight
      transfer path
    - the current blocker is now a mapping/reshape/transpose mismatch in the
      bootstrap conversion path, not TPU allocation, cache warmup, or OOM
    - likely direction:
      - one of the projected tensors arrives 2D while the static transpose map
        still assumes a 3D tensor
      - candidate keys include attention projections / projection biases in the
        exported HF checkpoint path
  - Immediate next step:
    - patch the fast-bootstrap conversion path to log the offending key/shape
      deterministically and then fix the transpose/padding logic
    - after the patch, rerun phase-1 `prepare-sampling` on the same `v5p-8`
      setup rather than changing TPU parameters again
- 2026-03-21T18:41Z:
  - ALT-TPU-002A local diagnosis result:
    - the failure is fully explained without another TPU probe
    - async RL and alternating fast bootstrap were feeding different tensor
      formats into the same `sync_weights(...)` contract:
      - async RL path:
        - transfers live Levanter state
        - attention projections are still articulated:
          - `q_proj`: `(KVHeads, QHeadsPerGroup, HeadSize, Embed)`
          - `k_proj` / `v_proj`: `(KVHeads, HeadSize, Embed)`
          - `o_proj`: `(Embed, Heads, HeadSize)`
      - alternating fast-bootstrap path:
        - loads exported HF safetensors from `policy_0001`
        - those linears were flattened for torch export:
          - `q_proj`: `(4096, 4096)`
          - `k_proj` / `v_proj`: `(1024, 4096)`
          - `o_proj`: `(4096, 4096)`
    - the existing transpose map in `MODEL_TRANSPOSE_KEYS` is correct for the
      live Levanter-shaped tensors used by async RL, but not for the flattened
      HF export tensors used by bootstrap:
      - `q_proj` / `k_proj` / `v_proj`: `(2, 0, 1)`
      - `o_proj`: `(1, 2, 0)`
    - consequence:
      - bootstrap was reusing the async/live converter on the wrong input
        format
      - `transfer_state_with_mappings(...)` then tried to apply a 3D transpose
        to a 2D exported tensor and failed exactly as observed
  - Local evidence:
    - confirmed against recovered `retry6` export:
      - `model.layers.0.self_attn.q_proj.weight`: `(4096, 4096)`
      - `model.layers.0.self_attn.k_proj.weight`: `(1024, 4096)`
      - `model.layers.0.self_attn.v_proj.weight`: `(1024, 4096)`
      - `model.layers.0.self_attn.o_proj.weight`: `(4096, 4096)`
    - scanned the full exported checkpoint against the current conversion +
      transpose contract:
      - `128` mismatches found
      - this is all `32` layers times the `4` attention projection weights
      - therefore the bug is systematic, not a one-off tensor
  - Code change in progress:
    - split the conversion paths explicitly:
      - keep the existing live Levanter -> NNX conversion for async RL
      - add a dedicated torch/HF-export -> Levanter-shaped conversion step for
        fast bootstrap before calling the existing NNX adapter
    - regression coverage added locally:
      - exported attention weights now round-trip back to the original
        Levanter-shaped tensors
      - bootstrap now proves it hands grouped/padded 3D attention tensors to
        `sync_weights(...)` instead of the flattened 2D export tensors
  - Local validation:
    - `uv run pytest tests/rl/test_inference_ctx.py -o addopts=`
      - result: `14 passed`
    - `./infra/pre-commit.py --all-files --fix`
      - result: `OK`
  - Next:
    - push the conversion split to the TPU image
    - rerun phase-1 `prepare-sampling` on the recovered single-host `retry6`
      state with the same `v5p-8` and `gpu_memory_utilization=0.92`

## Next Experiments After Phase-1 Fast-Bootstrap Transpose Failure

### ALT-TPU-002A: Identify the exact offending key/shape locally

- Goal:
  - turn the current generic transpose failure into a precise failing tensor key
    plus input/output shapes
- Hypothesis:
  - phase-1 fast bootstrap is applying a 3D transpose rule to a 2D tensor
    (most likely a projection bias or a projection tensor that is already
    flattened/padded)
- Method:
  - patch the fast-bootstrap conversion path to log:
    - source key
    - source tensor shape
    - requested transpose axes
    - destination key / destination shape if available
  - run a local reproduction against the recovered `policy_0001` export from
    `alt-tpu-002-v5p-retry6`
- Success criterion:
  - log names the exact offending key and shows why the transpose is invalid
- If proven:
  - implement the narrow mapping/transpose/padding fix in code
- If disproven:
  - inspect reshape/pad logic before transpose and compare against the target
    vLLM/tpu-inference parameter shapes

### ALT-TPU-002B: Add a deterministic unit test for the recovered bug

- Goal:
  - prevent regressions once the transpose fix lands
- Hypothesis:
  - the failing key can be reproduced in a small pure-Python/JAX unit test
    without needing a TPU
- Method:
  - add a regression test in `tests/rl/test_inference_ctx.py` or
    `tests/rl/test_vllm_fast_bootstrap.py`
  - encode the exact bad shape/key combination and assert the fixed conversion
    path succeeds
- Success criterion:
  - test fails before the fix and passes after the fix
- If proven:
  - treat the TPU rerun as validation of the full integrated path only
- If disproven:
  - keep the TPU rerun, but add richer runtime logging first

### ALT-TPU-002C: Rerun phase-1 prepare-sampling on single-host `v5p-8`

- Goal:
  - verify the transpose/mapping fix on the exact real hardware path that is
    currently blocked
- Hypothesis:
  - with the mapping fix in place, phase-1 fast bootstrap at
    `gpu_memory_utilization=0.92` will complete on `v5p-8`
- Method:
  - reuse the existing `alt-tpu-002-v5p-retry6` recovered state if possible
  - if the state is awkward to reuse cleanly, launch `alt-tpu-002-v5p-retry7`
    with the same parameters:
    - single host
    - `v5p-8`
    - `gpu_memory_utilization=0.92`
    - same image digest plus the transpose-fix patch
- Success criterion:
  - phase-1 `prepare-sampling` completes and the host writes
    `sampling/phase_0001/host_0000/status.json`
- If proven:
  - continue the same run through phase-1 materialization, training, and export
- If disproven:
  - use the new failure signature to decide whether the bug is still
    shape-specific or whether another resource/pathology is exposed

### ALT-TPU-002D: Complete a full two-phase single-host run on `v5p-8`

- Goal:
  - actually close out the original warm-cache / alternating-runtime objective
    for the single-host case
- Hypothesis:
  - once phase-1 fast bootstrap is fixed, the rest of phase 1 will behave like
    phase 0; the remaining risk is spot reclaim during export, not core runtime
    correctness
- Method:
  - run through:
    - phase-1 sampling
    - phase-1 materialization
    - phase-1 training
    - phase-1 export
  - if export is interrupted by spot reclaim again, use export-only recovery
    instead of rerunning the whole phase
- Success criterion:
  - durable `policy_0002/manifest.json` exists
  - controller finishes `ALT-TPU-002` as a completed two-phase single-host run
- If proven:
  - promote `0.92` as the current single-host `v5p-8` bootstrap setting for
    Llama-3.1-8B alternating RL
- If disproven:
  - classify the failure:
    - export-only / spot issue
    - training/materialization issue
    - fast-bootstrap issue

### ALT-TPU-003: Minimal multi-host one-phase validation after single-host is stable

- Goal:
  - only after the single-host two-phase path is stable, move to the first
    multi-host check
- Hypothesis:
  - the alternating controller/materialized-batch path will work on multi-host
    once the single-host fast-bootstrap path is fixed
- Method:
  - run the smallest multi-host one-phase configuration
  - validate:
    - per-host sampling startup
    - materialization
    - multi-host train step
    - export
- Success criterion:
  - one full multi-host phase completes and exports `policy_0001`
- If proven:
  - continue to the planned multi-host two-phase reliability run
- If disproven:
  - split the failure into:
    - multi-host input sharding / `hax.shard`
    - TPU topology/env
    - export/runtime issues

## 2026-03-21 19:02 UTC — ALT-TPU-002C execution start

- Goal:
  - rerun phase-1 `prepare-sampling` on the recovered
    `alt-tpu-002-v5p-retry6` state using the export-shape fix, without
    relaunching the controller
- Recovered state:
  - `config_path=gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/state/controller_config.pkl`
  - `sampling_manifest=gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/sampling/phase_0001/manifest.json`
  - `policy_path=gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/policies/policy_0001`
  - `tpu_name=alt-v5p-probe-001`
  - `zone=us-east5-a`
  - `phase_id=1`
  - `local_tensor_parallel_size=4`
- Image under test:
  - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:df9805f64343caef47e7ce4c3342a14e2f4a005e71a31de404ad0bf943d21242`
- Method:
  - clear stale `marin-alt-sampler` on worker 0
  - reuse the controller's own `prepare-sampling` command and environment
  - run foreground on worker 0 so the first post-fix failure or success is
    visible immediately

## 2026-03-21 18:51 UTC — ALT-TPU-002C prepare-sampling result

- Result:
  - `prepare-sampling --phase-id 1` completed successfully on
    `alt-v5p-probe-001` with image
    `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:df9805f64343caef47e7ce4c3342a14e2f4a005e71a31de404ad0bf943d21242`
- Evidence:
  - the previous `axis 2 is out of bounds for array of dimension 2` failure
    did not recur
  - live weight-sync logs showed phase-1 attention projections arriving in
    reconstructed Levanter geometry:
    - `k_proj`: `(8, 128, 4096)`
    - `q_proj`: `(32, 128, 4096)`
    - `v_proj`: `(8, 128, 4096)`
    - `o_proj`: `(4096, 32, 128)`
  - `sampling/phase_0001/manifest.json` now has
    `frozen_lesson_weights={'math_full': 1.0}`
  - `curriculum_snapshot_path` exists
- Next action:
  - run `sampling-host --phase-id 1 --host-ordinal 0` on the same recovered
    state to produce `sampling/phase_0001/host_0000/status.json`

## 2026-03-21 18:56 UTC — ALT-TPU-002C complete, ALT-TPU-002D execution start

- `ALT-TPU-002C` result:
  - `sampling/phase_0001/host_0000/status.json` exists and reports
    `status=succeeded`
  - host 0 produced `num_train_groups=4`
  - a rollout pickle was written under
    `sampling/phase_0001/host_0000/rollouts/`
- Interpretation:
  - the robust export/import fix is now validated on the exact real blocked
    path, not just in a unit test:
    - phase-1 fast bootstrap succeeds
    - phase-1 sampling succeeds
    - the recovered `retry6` state is usable again
- `ALT-TPU-002D` method:
  - update the saved controller config to the fixed image digest
  - heal `run_state.json` from `failed` back to the correct `sampling`
    checkpoint for phase 1
  - resume the normal controller so it performs:
    - curriculum update
    - materialization
    - one remaining training step
    - export of `policy_0002`

## 2026-03-21 19:12 UTC — ALT-TPU-002D result

- Result:
  - resumed `alt-tpu-002-v5p-retry6` completed successfully
  - final durable state:
    - `run_state.status=completed`
    - `phase_id=2`
    - `policy_version=2`
    - `source_global_step=2`
  - `policy_0002/manifest.json` exists at
    `gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/policies/policy_0002/manifest.json`
- What was required:
  - reconstruct exported HF attention projections back into native Levanter
    tensor geometry before the existing NNX/vLLM transpose mapping
  - validate the fix on the exact recovered phase-1 bootstrap path
  - heal the saved controller config and `run_state.json` to resume from the
    completed phase-1 sampling point under the fixed image digest
- Phase-1 timings from durable metrics:
  - `curriculum_update_seconds=1.81`
  - `materialization_seconds=51.24`
  - `training_seconds=210.77`
  - `export_seconds=434.27`
- Key artifacts:
  - phase-1 sampling host status:
    `gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/sampling/phase_0001/host_0000/status.json`
  - phase-1 materialized batch:
    `gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/materialized/phase_0001/batches/batch_000000.pkl`
  - final Levanter checkpoint:
    `gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/levanter_checkpoints/alt-tpu-002-v5p-retry6-alternating-train/step-2`
- Interpretation:
  - the blocker was not TPU memory at `0.92`; it was the export/import
    contract mismatch between flattened HF attention tensors and the live
    Levanter-to-vLLM sync path
  - once that contract was fixed, the full single-host two-phase `v5p-8` run
    completed end-to-end
- Next step:
  - promote this fix and use it as the new baseline before moving on to the
    planned minimal multi-host validation in `ALT-TPU-003`

## 2026-03-21 19:21 UTC — Command record for `ALT-TPU-002C` / `ALT-TPU-002D`

- Fixed image digest used for all TPU revalidation and resume steps:
  - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:df9805f64343caef47e7ce4c3342a14e2f4a005e71a31de404ad0bf943d21242`
- Exact phase entrypoints exercised against the recovered `retry6` state:
  - `prepare-sampling --config-path gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/state/controller_config.pkl --phase-id 1`
  - `sampling-host --config-path gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/state/controller_config.pkl --phase-id 1 --host-ordinal 0`
- Controller resume command:
  - local `uv run python` invoking
    `run_controller(read_pickle("gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/state/controller_config.pkl"), ExistingPodPhaseHooks())`
- State-heal details before controller resume:
  - updated `controller_config.pkl` in place to the fixed image digest
  - rewrote `run_state.json` from:
    - `status=failed`
    - `phase_id=1`
    - `policy_version=1`
  - to:
    - `status=sampling`
    - `phase_id=1`
    - `policy_version=1`
    - `current_sampling_manifest=gs://marin-us-east5/alternating-rl/alt-tpu-002-v5p-retry6/sampling/phase_0001/manifest.json`
    - `current_materialized_manifest=None`
- Resume behavior observed:
  - controller reused the completed phase-1 sampling artifacts
  - controller wrote
    `materialized/phase_0001/manifest.json`
  - controller resumed Levanter training from
    `levanter_checkpoints/alt-tpu-002-v5p-retry6-alternating-train/step-1`
  - controller saved `step-2` and exported `policy_0002`
- Operational note:
  - an attempted backup path built with `PurePosixPath` normalized
    `gs://...` incorrectly to a local `./gs:/...` scratch path
  - that local scratch directory was removed immediately
  - no durable backup artifact from that attempt should be relied on

## 2026-03-21 19:28 UTC — Proposed next experiment: `ALT-TPU-002E`

- Goal:
  - prove uninterrupted single-host phase alternation beyond the tiny
    two-phase recovery case by running at least five consecutive
    sample/materialize/train/export cycles
- Why this comes before multi-host:
  - `ALT-TPU-002D` proved end-to-end correctness on single-host, but not a
    longer uninterrupted soak
  - for a clean reliability signal, this run should remove spot reclaim as a
    confounder
- Proposed configuration:
  - single-host `v5p-8`
  - on-demand capacity
  - fixed image digest:
    `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:df9805f64343caef47e7ce4c3342a14e2f4a005e71a31de404ad0bf943d21242`
  - `steps_per_phase=1`
  - `num_train_steps=5`
  - same small smoke-scale workload as `ALT-TPU-002`:
    - `train_batch_size=64`
    - `n_prompts=4`
    - `n_generations_per_prompt=16`
    - `eval_examples_per_lesson=8`
    - `max_input_tokens=512`
    - `max_output_tokens=256`
    - `inference_gpu_memory_utilization=0.92`
- Exact controller launch:

```bash
uv run python experiments/alternating_rl_math500.py controller \
  --run-id alt-tpu-002e-v5p-soak5 \
  --shared-root gs://marin-us-east5/alternating-rl \
  --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:df9805f64343caef47e7ce4c3342a14e2f4a005e71a31de404ad0bf943d21242 \
  --tpu-name alt-v5p-soak-001 \
  --tpu-type v5p-8 \
  --zone us-east5-a \
  --num-hosts 1 \
  --local-tensor-parallel-size 4 \
  --capacity-type on-demand \
  --runtime-version v2-alpha-tpuv5 \
  --steps-per-phase 1 \
  --num-train-steps 5 \
  --train-batch-size 64 \
  --n-prompts 4 \
  --n-generations-per-prompt 16 \
  --eval-examples-per-lesson 8 \
  --max-input-tokens 512 \
  --max-output-tokens 256 \
  --inference-gpu-memory-utilization 0.92
```

- Success criterion:
  - no manual recovery
  - durable final state reports:
    - `status=completed`
    - `phase_id=5`
    - `policy_version=5`
    - `source_global_step=5`
  - `policy_0005/manifest.json` exists
- If proven:
  - single-host alternation is not just correct, but stable across repeated
    phase swaps
  - then move to `ALT-TPU-003` multi-host validation
- If disproven:
  - classify whether the failure is:
    - cumulative cache/runtime drift across phases
    - export-only overhead / artifact growth
    - training checkpoint / resume fragility
    - TPU infrastructure instability unrelated to alternating logic

## 2026-03-21 19:30 UTC — Tracking change: dedicate alternating RL W&B project

- Decision:
  - move alternating RL runs off the shared `marin_post_training` W&B project
    onto a dedicated `alternate_rl` project
- Reason:
  - alternating runtime debugging and soak tests now have enough volume and
    distinct operational concerns that they should not be mixed with the
    broader RL training runs
- Code change:
  - `experiments/alternating_rl_math500.py` now overrides the inherited RL
    experiment default and sets `project_name="alternate_rl"`
  - this changes both the trainer run and rollout tracker project for future
    alternating launches
- Implication for next runs:
  - `ALT-TPU-002E` and later alternating experiments should appear under
    `marin-community/alternate_rl`

## 2026-03-21 19:33 UTC — ALT-TPU-002E execution start

- Launch decision:
  - use the existing fixed TPU image digest
  - do not rebuild the TPU image just for the W&B project rename
  - rely on the local controller code to write `project_name="alternate_rl"`
    into `state/controller_config.pkl`
- Exact launch:

```bash
uv run python experiments/alternating_rl_math500.py controller \
  --run-id alt-tpu-002e-v5p-soak5 \
  --shared-root gs://marin-us-east5/alternating-rl \
  --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:df9805f64343caef47e7ce4c3342a14e2f4a005e71a31de404ad0bf943d21242 \
  --tpu-name alt-v5p-soak-001 \
  --tpu-type v5p-8 \
  --zone us-east5-a \
  --num-hosts 1 \
  --local-tensor-parallel-size 4 \
  --capacity-type on-demand \
  --runtime-version v2-alpha-tpuv5 \
  --steps-per-phase 1 \
  --num-train-steps 5 \
  --train-batch-size 64 \
  --n-prompts 4 \
  --n-generations-per-prompt 16 \
  --eval-examples-per-lesson 8 \
  --max-input-tokens 512 \
  --max-output-tokens 256 \
  --inference-gpu-memory-utilization 0.92
```

- Expected early signals:
  - queued resource `alt-v5p-soak-001` becomes `ACTIVE`
  - controller writes `state/controller_config.pkl` under the new run root
  - W&B run lands under `marin-community/alternate_rl`

## 2026-03-21 19:35 UTC — ALT-TPU-002E first launch failed before bootstrap

- Result:
  - on-demand `v5p-8` creation in `us-east5-a` failed immediately with
    `Insufficient capacity`
- Failure surface:
  - `gcloud alpha compute tpus queued-resources create alt-v5p-soak-001 --accelerator-type=v5p-8 --zone=us-east5-a ...`
- State impact:
  - `state/controller_config.pkl` was written
  - `state/run_state.json` was not written
  - no TPU queued resource remains for `alt-v5p-soak-001` in `us-east5-a`
- Next action:
  - retry the exact same soak under the same `run_id`, but switch only the zone
    to `us-east5-b`

## 2026-03-21 19:36 UTC — ALT-TPU-002E second launch also blocked before bootstrap

- Result:
  - retrying the on-demand soak in `us-east5-b` failed immediately on quota
- Failure surface:
  - `TPUV5PPerProjectPerZoneForTPUAPI` quota is `0` in `us-east5-b` for this
    project
- State impact:
  - `state/controller_config.pkl` can still be overwritten safely
  - `state/run_state.json` still does not exist
- Practical decision:
  - the clean on-demand soak is blocked by current infra limits
  - the simplest available launch is to reuse the already-`ACTIVE`
    `alt-v5p-probe-001` spot TPU in `us-east5-a`
  - this keeps the same five-phase controller configuration, but any result
    must be interpreted with spot reclaim as a confounder

## 2026-03-21 19:37 UTC — ALT-TPU-002E fallback launch on active spot probe

- Launch:
  - same `run_id=alt-tpu-002e-v5p-soak5`
  - same five-phase controller settings
  - fallback TPU target:
    - `tpu_name=alt-v5p-probe-001`
    - `zone=us-east5-a`
    - `capacity_type=spot`
- Early verification:
  - `state/controller_config.pkl` now points at:
    - `tpu_name=alt-v5p-probe-001`
    - `zone=us-east5-a`
    - `capacity_type=spot`
    - `image_digest=us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:df9805f64343caef47e7ce4c3342a14e2f4a005e71a31de404ad0bf943d21242`
    - `trainer.tracker.project=alternate_rl`
  - durable `run_state.json` exists and reports:
    - `status=sampling`
    - `phase_id=0`
    - `policy_version=0`
    - `current_sampling_manifest=gs://marin-us-east5/alternating-rl/alt-tpu-002e-v5p-soak5/sampling/phase_0000/manifest.json`
  - live TPU logs show phase-0 sampler startup on single-host `v5p-8`
- Interpretation:
  - the soak is now genuinely running
  - if it completes, it will still be useful operational evidence, but not a
    clean on-demand reliability proof because spot reclaim remains possible

## 2026-03-21 19:42 UTC — ALT-TPU-002E confirmed on `alt-v5p-probe-001`

- User direction:
  - stop trying to allocate any new on-demand TPU
  - run only on the already-`ACTIVE` single-host probe:
    - `alt-v5p-probe-001`
    - `zone=us-east5-a`
    - `tpu_type=v5p-8`
- Infra verification:
  - `gcloud alpha compute tpus queued-resources list --zone=us-east5-a --filter='name:alt-v5p-probe-001'`
    reports the queued resource still `ACTIVE`
- Durable controller state:
  - `gs://marin-us-east5/alternating-rl/alt-tpu-002e-v5p-soak5/state/run_state.json`
    now shows:
    - `status=training`
    - `phase_id=0`
    - `policy_version=0`
    - `tpu_name=alt-v5p-probe-001`
    - `zone=us-east5-a`
    - `current_materialized_manifest=gs://marin-us-east5/alternating-rl/alt-tpu-002e-v5p-soak5/materialized/phase_0000/manifest.json`
- Live controller signal:
  - phase-0 training ran on the active probe
  - the single train step completed in about `16.8s`
  - checkpoint save for `step-1` started at:
    - `gs://marin-us-east5/alternating-rl/alt-tpu-002e-v5p-soak5/levanter_checkpoints/alt-tpu-002e-v5p-soak5-alternating-train/step-1`
- Interpretation:
  - the soak is definitely running on the requested existing TPU
  - no new TPU allocation is needed for this run unless the active probe dies

## 2026-03-21 20:08 UTC — Alternating W&B lifecycle bug identified and fixed locally

- Symptom:
  - the durable controller state for `alt-tpu-002e-v5p-soak5` advanced to:
    - `status=sampling`
    - `phase_id=1`
    - `policy_version=1`
    - `source_global_step=1`
  - but the linked trainer run in W&B:
    - `https://wandb.ai/marin-community/alternate_rl/runs/alt-tpu-002e-v5p-soak5-alternating-train`
    was marked `crashed`
  - W&B API for that run reported:
    - `state=crashed`
    - `summary_runtime=None`
    - `summary_step=None`
- Root cause:
  - alternating training was creating a trainer tracker with run id
    `f"{config.run_id}-alternating-train"` but never explicitly calling
    `tracker.finish()` on the successful phase path
  - this differs from normal Levanter subprocess entrypoints like
    `train_lm.py`, which explicitly call `trainer.tracker.finish()` after
    training
  - `Trainer.__exit__` only unwinds context managers; it does not finish the
    tracker
- Local code fix:
  - `lib/marin/src/marin/rl/alternating/training_phase.py`
    - after logging alternating phase metrics, call `tracker.finish()` before
      returning the exported policy manifest
  - regression coverage added in:
    - `tests/rl/test_alternating_training_phase.py`
    - new test verifies the alternating phase tracker logs metrics and then
      finishes
  - debug log captured in:
    - `docs/debug-log-alternating-wandb-lifecycle.md`
- Validation:
  - `uv run pytest tests/rl/test_alternating_training_phase.py -o addopts=`
    passed (`6 passed`)
  - `./infra/pre-commit.py --all-files --fix` passed
- Important caveat:
  - the currently-running soak on `alt-v5p-probe-001` is still using image
    digest
    `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:df9805f64343caef47e7ce4c3342a14e2f4a005e71a31de404ad0bf943d21242`
  - so this W&B lifecycle fix is landed in repo, but not yet present in the
    active TPU runtime image
  - future TPU runs need a new image digest, or the controller would need to be
    resumed against a rebuilt image, for W&B to reflect this fix

## 2026-03-21 20:15 UTC — Refined next correctness experiments

- Current correctness status:
  - proven:
    - single-host alternating correctness across two full phases on `v5p-8`
    - recovery/resume correctness on the same single-host path
    - robust alternating fast bootstrap from exported HF weights back into TPU
      vLLM
  - not yet proven:
    - repeated single-host phase swaps without manual recovery
    - any multi-host alternating phase
    - repeated multi-host phase swaps
    - W&B lifecycle correctness in the deployed TPU image
- Ordered next experiments:

### ALT-TPU-002E: Keep running the active single-host five-phase soak

- Purpose:
  - prove repeated single-host phase swaps on real TPU, not just the two-phase
    end-to-end path
- Current live target:
  - `run_id=alt-tpu-002e-v5p-soak5`
  - `tpu_name=alt-v5p-probe-001`
  - `zone=us-east5-a`
  - `tpu_type=v5p-8`
  - `steps_per_phase=1`
  - `num_train_steps=5`
- Success criterion:
  - durable final state reports:
    - `status=completed`
    - `phase_id=5`
    - `policy_version=5`
    - `source_global_step=5`
  - `policy_0005/manifest.json` exists
- Interpretation:
  - if it completes cleanly, repeated single-host correctness is proven
  - if it is interrupted only by spot reclaim, treat the result as infra
    confounded but do not treat alternating correctness as disproven
  - if it fails due to alternating logic, pause multi-host until that bug is
    understood and fixed

### ALT-TPU-003: Minimal multi-host one-phase correctness gate

- Goal:
  - first proof that the alternating controller, materialization path, and
    training path all work on a real two-host pod
- Planned configuration:
  - `tpu_type=v5p-16`
  - `num_hosts=2`
  - `steps_per_phase=1`
  - `num_train_steps=1`
  - reuse the tiny Math500/smoke-sized quotas from `ALT-TPU-002E`
  - reuse the current single-host bootstrap setting
    `gpu_memory_utilization=0.92` unless disproven
- Preferred image:
  - use a rebuilt image digest that includes the W&B tracker-finish fix
  - however, W&B lifecycle correctness is not a blocker for this alternating
    correctness gate
- Success criterion:
  - one full multi-host phase completes
  - exported policy manifest exists at `policy_0001/manifest.json`
  - durable state reaches:
    - `policy_version=1`
    - `source_global_step=1`

### ALT-TPU-004: Minimal multi-host two-phase correctness gate

- Goal:
  - prove that multi-host correctness survives at least one real phase boundary
    and bootstrap into the next phase
- Planned configuration:
  - same pod shape as `ALT-TPU-003`
  - `steps_per_phase=1`
  - `num_train_steps=2`
- Success criterion:
  - the run reaches:
    - `status=completed`
    - `phase_id=2`
    - `policy_version=2`
    - `source_global_step=2`
  - `policy_0002/manifest.json` exists
- Failure split if disproven:
  - multi-host sampling startup / topology
  - multi-host materialization or batch sharding
  - multi-host training/checkpoint visibility
  - multi-host export/bootstrap boundary

### ALT-TPU-005: Only after `ALT-TPU-004` passes

- Decision point:
  - choose between:
    - longer multi-host soak for robustness
    - separate correctness thread for persistent async TPU workers with live
      weight sync
- Important framing:
  - the current alternating thread is now a correctness harness
  - it should not be confused with the final fast async architecture

## 2026-03-21 20:32 UTC

### W&B observability fix: alternating controller visibility

- Trigger:
  - active soak visibility in W&B was still incomplete
  - only the trainer subprocess run was visible
  - sampling/materialization/controller progress only existed in durable state
    files and TPU logs
- Root cause:
  - alternating training had a tracker
  - alternating controller/runtime never initialized a W&B run at all
  - therefore W&B went dark between training phases even when the soak was
    healthy
- Code changes:
  - added `lib/marin/src/marin/rl/alternating/wandb.py`
  - updated `lib/marin/src/marin/rl/alternating/controller.py` to create a
    resumed controller run and log:
    - controller start/resume
    - sampling manifest readiness
    - sampling completion and host counts
    - materialization start/completion
    - training completion
    - phase advance
    - terminal completed/failed summary
    - per-phase timing metrics from `state/phase_metrics/*.json`
  - updated `lib/marin/src/marin/rl/alternating/training_phase.py` so future
    trainer subprocess runs have explicit alternating metadata:
    - display name `<run_id>-alternating-train`
    - group `<run_id>`
    - tags include `alternating` plus role-specific tag
- Expected W&B shape on the next rebuilt image:
  - `<run_id>-alternating-controller`
  - `<run_id>-alternating-train`
- Validation:
  - `uv run pytest tests/rl/test_alternating_controller.py -o addopts=`
  - `uv run pytest tests/rl/test_alternating_training_phase.py -o addopts=`
  - `./infra/pre-commit.py --all-files --fix`
- Important caveat:
  - the currently running `ALT-TPU-002E` soak is on the older TPU image
  - it will not retroactively gain the controller W&B run
  - this fix is ready for the next image build / TPU launch

## 2026-03-21 20:36 UTC — Planned `ALT-TPU-002E` clean retry with full W&B coverage

- User direction:
  - stop the failed / stale `ALT-TPU-002E` attempt
  - rerun the five-phase single-host soak on the already-`ACTIVE`
    `alt-v5p-probe-001`
  - rebuild the TPU image if required so W&B shows the full alternating run
- Current pre-launch state:
  - durable run state for `alt-tpu-002e-v5p-soak5` is terminal:
    - `status=failed`
    - `phase_id=2`
    - `policy_version=2`
    - `source_global_step=2`
  - TPU queued resource remains:
    - `alt-v5p-probe-001`
    - `zone=us-east5-a`
    - `state=ACTIVE`
  - no sampler or trainer containers are live on the TPU at relaunch prep time
- Relaunch decision:
  - do not overwrite the failed run root
  - launch a fresh retry run id so controller state and W&B ids do not resume
    into the failed attempt
- Planned retry run:
  - experiment ID: `ALT-TPU-002E`
  - run id: `alt-tpu-002e-v5p-soak5-retry1`
  - TPU target:
    - `tpu_name=alt-v5p-probe-001`
    - `zone=us-east5-a`
    - `tpu_type=v5p-8`
    - `capacity_type=spot`
- Planned image build command:

```bash
uv run python lib/levanter/infra/push_docker.py \
  --docker_target gcp \
  --project hai-gcp-models \
  --region us-east5 \
  --repository levanter \
  --image levanter-ahmed \
  --tag <to-fill-after-build> \
  --docker_file lib/levanter/docker/tpu/Dockerfile.incremental
```

- Planned controller launch command:

```bash
uv run python experiments/alternating_rl_math500.py controller \
  --run-id alt-tpu-002e-v5p-soak5-retry1 \
  --shared-root gs://marin-us-east5/alternating-rl \
  --image <to-fill-after-build> \
  --tpu-name alt-v5p-probe-001 \
  --tpu-type v5p-8 \
  --zone us-east5-a \
  --num-hosts 1 \
  --local-tensor-parallel-size 4 \
  --capacity-type spot \
  --runtime-version v2-alpha-tpuv5 \
  --steps-per-phase 1 \
  --num-train-steps 5 \
  --train-batch-size 64 \
  --n-prompts 4 \
  --n-generations-per-prompt 16 \
  --eval-examples-per-lesson 8 \
  --max-input-tokens 512 \
  --max-output-tokens 256 \
  --inference-gpu-memory-utilization 0.92
```

- Success criterion for the retry launch:
  - controller W&B run appears:
    - `<run_id>-alternating-controller`
  - trainer W&B run appears when phase-0 training starts:
    - `<run_id>-alternating-train`
  - durable state enters phase-0 sampling on the existing TPU

## 2026-03-21 20:45 UTC — `ALT-TPU-002E` relaunch sequence and active retry

- Retry 1:
  - `run_id=alt-tpu-002e-v5p-soak5-retry1`
  - image digest:
    - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:54972a8f5a94e031b7869a883a680f70850c1dc8fa24f543fbd3ab015da1cffe`
  - result:
    - failed immediately in local controller W&B setup
  - root cause:
    - `lib/marin/src/marin/rl/alternating/wandb.py` called `join_path(...)`
      with three arguments while the helper only accepts two
  - fix:
    - nested the `join_path` calls and added regression coverage
- Retry 2:
  - `run_id=alt-tpu-002e-v5p-soak5-retry2`
  - same image digest as retry 1
  - result:
    - controller W&B run came up, but controller startup was effectively stuck
      in `wandb.run.log_code(...)` before phase-0 work
  - evidence:
    - local process sample showed time in file hashing / `copyfile` /
      `mmap` from W&B code snapshotting
  - fix:
    - set alternating role-specific `WandbConfig.save_code=False`
    - revalidated tests and pre-commit
- Retry 3:
  - `run_id=alt-tpu-002e-v5p-soak5-retry3`
  - rebuilt image digest:
    - `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:ba49b843c2b780a1615d7303aead9087ce98e379891d48db4c7584fe208f94a8`
  - exact controller launch:

```bash
uv run python experiments/alternating_rl_math500.py controller \
  --run-id alt-tpu-002e-v5p-soak5-retry3 \
  --shared-root gs://marin-us-east5/alternating-rl \
  --image us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:ba49b843c2b780a1615d7303aead9087ce98e379891d48db4c7584fe208f94a8 \
  --tpu-name alt-v5p-probe-001 \
  --tpu-type v5p-8 \
  --zone us-east5-a \
  --num-hosts 1 \
  --local-tensor-parallel-size 4 \
  --capacity-type spot \
  --runtime-version v2-alpha-tpuv5 \
  --steps-per-phase 1 \
  --num-train-steps 5 \
  --train-batch-size 64 \
  --n-prompts 4 \
  --n-generations-per-prompt 16 \
  --eval-examples-per-lesson 8 \
  --max-input-tokens 512 \
  --max-output-tokens 256 \
  --inference-gpu-memory-utilization 0.92
```

- Live status:
  - durable run state exists and currently reports:
    - `status=sampling`
    - `phase_id=0`
    - `policy_version=0`
    - `source_global_step=0`
    - `current_sampling_manifest=gs://marin-us-east5/alternating-rl/alt-tpu-002e-v5p-soak5-retry3/sampling/phase_0000/manifest.json`
  - W&B controller run is live:
    - `https://wandb.ai/marin-community/alternate_rl/runs/alt-tpu-002e-v5p-soak5-retry3-alternating-controller`
  - TPU verification:
    - `marin-alt-sampler` is running on `alt-v5p-probe-001`
- Interpretation:
  - the full-visibility relaunch is now genuinely live on the requested active
    TPU
  - trainer W&B run should appear once phase-0 reaches training

## 2026-03-21 21:35 UTC — Async-style trainer metrics fix for alternating W&B

- Observation from `ALT-TPU-002E retry3`:
  - controller and trainer W&B runs both existed
  - TPU logs proved phase-0 training and export really happened
  - but W&B history still showed only system metrics / zero scalar rows
- Root cause:
  - alternating training was not installing the same RL train-worker hook stack
    used by async RL
  - so it was missing:
    - step timing metrics
    - async-style `train/samples` table
    - throughput / MFU hook wiring
  - and for one-step phases we also needed an explicit per-step W&B commit
    because all scalar logs land on a single step
- Fix landed in repo:
  - added shared hook wiring in:
    - `lib/marin/src/marin/rl/training_observability.py`
  - switched async RL `train_worker.py` to use the shared helper so async and
    alternating now share one metrics path
  - updated alternating training to install the same hook stack
  - updated alternating materialization to write per-batch sample-preview
    sidecars so alternating can log the same `train/samples` table as async RL
- Validation:
  - `uv run pytest tests/rl/test_alternating_training_phase.py -o addopts=`
  - `uv run pytest tests/rl/test_training_observability.py -o addopts=`
  - `uv run python -m py_compile lib/marin/src/marin/rl/training_observability.py lib/marin/src/marin/rl/alternating/training_phase.py lib/marin/src/marin/rl/alternating/materialization.py lib/marin/src/marin/rl/train_worker.py`
- Important caveat:
  - this does **not** change the already-running `alt-tpu-002e-v5p-soak5-retry3`
    TPU job because its image is already fixed in place
  - the next rebuilt image / relaunch is the first run that can verify the new
    async-style trainer metrics in W&B

## 2026-03-21 21:45 UTC — Storage / Arrow / HF export clarification

- Run state update:
  - stopped `alt-tpu-002e-v5p-soak5-retry3` after phase-0 completion because
    the live TPU image did not include the new shared async-style trainer
    metrics fix
  - durable state was explicitly marked failed with:
    - `phase_id=1`
    - `policy_version=1`
    - `source_global_step=1`
  - TPU `alt-v5p-probe-001` remains up; only the alternating run was stopped
- Question answered:
  - can async RL avoid loss by saving to local disk, or by using Arrow instead
    of checkpoints / alternating export?
- Findings:
  - async RL already uses Arrow Flight as the fast live weight-transfer path by
    default:
    - `lib/marin/src/marin/rl/rl_experiment_utils.py`
    - `lib/marin/src/marin/rl/train_worker.py`
    - `lib/marin/src/marin/rl/rollout_worker.py`
    - `lib/marin/src/marin/rl/weight_transfer/arrow_flight.py`
  - Arrow is only a live weight handoff between running trainer and rollout
    workers; it does **not** durably save:
    - optimizer state
    - trainer step / checkpointer state
    - curriculum state needed for restart
  - durable recovery therefore still needs Levanter checkpoints via the normal
    checkpointer path:
    - `lib/levanter/src/levanter/checkpoint.py`
  - local-disk checkpoints are technically possible because the checkpointer
    accepts local paths, but they only help if the same TPU VM is still alive;
    they do **not** solve:
    - spot reclaim
    - TPU recreation
    - multi-host visibility
    - controller-visible durable resume
  - the most plausible future checkpointing improvement is a two-tier scheme:
    - frequent local temporary checkpoints for same-host crash recovery
    - less frequent GCS checkpoints for durable recovery
- Why alternating exports to HF at all:
  - alternating phases hand off through a `policy_dir` artifact that vLLM can
    treat like a pretrained model directory
  - sampling/bootstrap currently expects HF-style metadata such as
    `config.json` and tokenizer files:
    - `lib/marin/src/marin/rl/alternating/sampling_phase.py`
    - `lib/marin/src/marin/rl/environments/inference_ctx/vllm.py`
  - even fast bootstrap still stages metadata from the exported policy
    directory and then injects weights into the engine
- Performance implication:
  - HF export is a real part of the phase tax, not just a file copy
  - current export path does:
    - Levanter checkpoint -> HF-compatible state dict conversion on CPU
    - sharded safetensors write
    - remote upload when the output path is on GCS
  - this behavior is implemented via:
    - `ConvertLmConfig(... use_cpu=True)` in
      `lib/marin/src/marin/rl/alternating/training_phase.py`
    - HF shard/temp-upload logic in
      `lib/levanter/src/levanter/compat/hf_checkpoints.py`
- Architectural conclusion:
  - Arrow is the right fast path for persistent async workers
  - Arrow alone cannot replace alternating cross-phase handoff because once the
    trainer exits there is no live transfer server left
  - if alternating remains only a correctness harness, the important
    optimization target is eliminating full HF export per phase rather than
    trying to make the current export loop production-fast
