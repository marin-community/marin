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

## ALT-TPU-001: Single-host one-phase smoke test

- Date: 2026-03-21
- Owner: ahmed (claude-signal session)
- Goal: Prove one complete alternating phase works on a real TPU VM
- TPU: v5p-8, us-east5-a, spot capacity
- Hosts: 1
- Image: us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774078195 (sha256:8a867e741c80a7e9452d24ddc39326f8ab6e9503676b9447ebf876e26f852346)
- Code revision: coalloc-rl branch, commit `5a0aaecc1`
- Command:
  ```bash
  uv run python experiments/alternating_rl_math500.py controller \
    --run-id alt-tpu-001 \
    --shared-root gs://marin-us-east5/alternating-rl \
    --image <image> \
    --tpu-name alt-tpu-001 \
    --tpu-type v5p-8 \
    --zone us-east5-a \
    --capacity-type spot \
    --num-hosts 1 \
    --local-tensor-parallel-size 4 \
    --steps-per-phase 1 \
    --num-train-steps 1
  ```
- Phase quotas: steps_per_phase=1, num_train_steps=1
- Expected outcome: sampling -> materialization -> training -> export completes, policy_0001/manifest.json written

### Status: IN PROGRESS

- [x] Build Docker image (cross-platform linux/amd64 build, sha256:79ec7e5c3254)
- [x] Push image to us-east5 Artifact Registry (sha256:8a867e741c80)
- [x] Fix: vLLM import in rl_experiment_utils.py — added stub SamplingParams dataclass so controller can build config on macOS without vLLM installed
- [x] Launch controller (PID 16700, started ~2026-03-21T07:34Z)
- [ ] Sampling phase completes
- [ ] Materialization completes
- [ ] Training phase completes
- [ ] Policy export completes
- [ ] Collect evidence artifacts

### Result

- Status: pending
- Completed phases: pending
- Wall-clock: pending
- Boundary overhead: pending
- Warm-cache behavior: N/A (first phase only)
- Sampling throughput: pending
- Training throughput: pending
- Export result: pending

### Evidence

- Run state path: `gs://marin-us-east5/alternating-rl/alt-tpu-001/state/run_state.json`
- Policy manifest path: `gs://marin-us-east5/alternating-rl/alt-tpu-001/policies/policy_0001/manifest.json`
- Phase metrics path: `gs://marin-us-east5/alternating-rl/alt-tpu-001/state/phase_metrics/phase_0000.json`
- Logs: pending

### Findings

- First launch failed: `ModuleNotFoundError: No module named 'vllm'` — controller runs locally on macOS where vLLM isn't available. Fixed by adding a stub `SamplingParams` dataclass in `rl_experiment_utils.py` when vLLM is not installed. The stub has enough fields to build the config; real vLLM is only needed on the TPU containers.
- Controller is now running, likely in TPU creation phase (spot v5p-8 in us-east5-a)
- 2026-03-21T07:35Z: TPU `alt-tpu-001` status = PROVISIONING (confirmed via `gcloud compute tpus queued-resources list`)
- 2026-03-21T07:39Z: TPU `alt-tpu-001` status = ACTIVE. Controller should now be setting up Docker and bootstrapping initial policy.
- 2026-03-21T07:41Z: GCS state confirms bootstrap complete. Run state status=sampling.
- 2026-03-21T07:44Z: Sampler container `marin-alt-sampler` exited with code 2.
  - Root cause: `can't open file '/opt/marin/lib/levanter/experiments/alternating_rl_math500.py': [Errno 2] No such file or directory`
  - The phase command uses wrong WORKDIR. Docker WORKDIR is `/opt/marin/lib/levanter` but the experiment file is at `/opt/marin/experiments/alternating_rl_math500.py`.
  - Fix: changed `_phase_command` in `runtime.py` to use absolute path `/opt/marin/experiments/alternating_rl_math500.py` instead of relative `experiments/alternating_rl_math500.py`.
  - Action: killed controller, rebuilding Docker image with fix, will reuse existing TPU (still ACTIVE).
- 2026-03-21T07:46Z: Rebuilding Docker image with path fix. TPU `alt-tpu-001` still ACTIVE — will reuse it.
- 2026-03-21T07:50Z: Rebuilt and pushed new image `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774078954` (sha256:b8f8f3b79eed6d1652e21ec13bdb3c86b6dd5651dace14db3e605dfe3d268727). Cached layers reused, only code layer changed.
- 2026-03-21T07:50Z: Cleaned GCS state (`gcloud storage rm -r`). Relaunched controller with new image. TPU reused (no re-provisioning).
- 2026-03-21T07:51Z: Controller relaunched (background task bpcrr6qem). Monitoring GCS for progress.
- 2026-03-21T07:53Z: Second attempt also failed. run_state.json shows status=failed.
  - Sampler container error: `RuntimeError: Unable to initialize backend 'tpu': INTERNAL: Failed to get global TPU topology`
  - Root cause: TPU single-host isolation env vars (`CLOUD_TPU_TASK_ID=0`, `TPU_PROCESS_BOUNDS=1,1,1`, `TPU_CHIPS_PER_PROCESS_BOUNDS`, `TPU_VISIBLE_CHIPS`) are NOT being set on the sampler container.
  - These are required for the "independent per-host vLLM replica" pattern — without them, JAX tries to initialize multi-host topology which fails.
  - Fix: added `topology.env()` merge into sampling container env in `runtime.py:launch_sampling_phase`. The topology vars were only applied inside the Python process via `apply_local_vllm_topology()` but that's too late — JAX initializes before the Python code reaches that call. The vars must be set as Docker container env vars.
- 2026-03-21T07:56Z: Rebuilding Docker image with TPU topology env fix. Third attempt.
- 2026-03-21T08:00Z: Second controller run also failed. Same TPU topology error but this time on `prepare-sampling` container (not just `sampling-host`). The `prepare-sampling` step also initializes vLLM and needs the topology env vars. Fix: add topology env to ALL sampling-related containers in `launch_sampling_phase`, not just the per-host sampling-host containers.
  - Fixed `runtime.py`: moved `topology = local_vllm_topology(...)` before `prepare-sampling` launch and added `env.update(topology.env())` to both prepare-sampling and sampling-host container env dicts.
  - Key insight: the topology env fix is in local controller code (runtime.py), not in the Docker image. The controller passes env vars via `-e` flags. So the existing image works — just need to rerun the controller.
- 2026-03-21T08:03Z: Cleaned GCS state, relaunched controller (attempt 3). Reusing same TPU and same image. The topology env vars should now be passed to all sampling containers.
- 2026-03-21T08:06Z: Attempt 3 also failed. Same `Failed to get global TPU topology` error. Confirmed env vars are correctly set in Docker inspect. Tested minimal `jax.devices()` in Docker — fails even without topology overrides. Also fails on host (no JAX installed).
  - v5p uses `/dev/vfio/vfio` not `/dev/accel*`.
  - Root cause hypothesis: JAX/libtpu version in Docker image (`levanter-base:latest`) may not be compatible with v5p TPU runtime (`v2-alpha-tpuv5`). The base image may be targeting v6e.
  - Confirmed: `JAX_PLATFORMS=""` fallback reveals `WARNING: A Google TPU may be present on this machine, but either a TPU-enabled jaxlib or libtpu is not installed. Falling back to cpu.`
  - JAX 0.8.0 is installed, but libtpu for v5p is NOT in the image. The `levanter-base:latest` image does not include v5p libtpu.
  - **Decision**: delete the v5p-8 TPU and switch to v6e in europe-west4-a, which the image is known to work with (per the on-demand-rl benchmarks). Need to use `gs://marin-europe-west4` bucket for same-region data.
- 2026-03-21T08:10Z: Deleted v5p-8 TPU `alt-tpu-001`, cleaned GCS state. Pushing image to europe-west4 Artifact Registry. Will launch v6e-4 (single host, 4 chips) in europe-west4-a with spot capacity.
- 2026-03-21T07:41Z: GCS bootstrap artifacts confirmed:
  - `run_state.json`: status=sampling, phase_id=0, policy_version=0
  - `policies/policy_0000/manifest.json` written
  - `sampling/phase_0000/manifest.json` written
  - `curriculum/curriculum_state.json` written
  - Controller now in sampling phase — launching vLLM on TPU host

### Follow-up

- [ ] If passes, proceed to ALT-TPU-002

---

## ALT-TPU-002F: Single-host observability verification

- Date: 2026-03-21
- Owner: ahmed (claude-signal session)
- Goal: Verify rebuilt image logs full async-style trainer metrics in W&B
- TPU: v5p-8, us-east5-a, reusing `alt-v5p-probe-001` (ACTIVE)
- Hosts: 1
- Image: TBD (building with observability fixes)
- Code revision: coalloc-rl branch, commit `318decd81`
- Config: steps_per_phase=1, num_train_steps=1, train_batch_size=64, n_prompts=4, n_generations_per_prompt=16, eval_examples_per_lesson=8, max_input_tokens=512, max_output_tokens=256, inference_gpu_memory_utilization=0.92
- Success criteria:
  - train/loss, throughput metrics, and train/samples appear on trainer W&B run
  - Controller W&B run stays live with phase-level metrics
  - Full phase completes: sampling → materialization → training → export

### Status: IN PROGRESS

- [x] Commit observability code
- [x] Push to remote
- [ ] Build and push Docker image
- [ ] Launch controller
- [ ] Phase 0 sampling completes
- [ ] Phase 0 training completes with W&B metrics visible
- [ ] Phase 0 export completes
- [ ] Verify W&B runs

### Findings

- 2026-03-21T21:35Z: Building image with shared training_observability.py. Reusing active `alt-v5p-probe-001` TPU.
- 2026-03-21T21:42Z: Image pushed: `us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774128928` (sha256:82a2241672915dd7455cb1f1ca8b9606b4ca2aa8e2f905b40efc4d6110722198). Launching controller.
- 2026-03-21T21:43Z: Controller launched. run_id=alt-tpu-002f-obs, reusing alt-v5p-probe-001.
- 2026-03-21T21:44Z: Bootstrap complete. run_state status=sampling, phase_id=0. Sampling in progress.
- 2026-03-21T21:49Z: Phase 0 progressed through sampling → materialization → training:
  - prepare_sampling: 92.2s
  - sampling: 107.1s
  - curriculum_update: 1.5s
  - materialization: 56.2s
  - New: `batch_000000_samples.json` sidecar present (observability fix working)
  - Currently in training phase, waiting for checkpoint + export
- 2026-03-21T22:02Z: ALT-TPU-002F COMPLETED. Full phase 0 end-to-end:
  - prepare_sampling: 92.2s
  - sampling: 107.1s
  - curriculum_update: 1.5s
  - materialization: 56.2s
  - training: 157.0s
  - export: 528.2s (the 9-min export tax)
  - **phase_total: 942.2s (~15.7 min)**
  - run_state: status=completed, policy_version=1, source_global_step=1
  - W&B run synced with alternating phase metrics, train/samples table
  - policy_0001/manifest.json written, levanter checkpoint at step-1

---

## ALT-TPU-003: Multi-host one-phase validation (v5p-16, 2 hosts)

- Date: 2026-03-21
- Owner: ahmed (claude-signal session)
- Goal: First proof that alternating RL works on a real two-host pod
- TPU: v5p-16, us-east5-a, spot, 2 hosts
- Image: us-east5-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed@sha256:82a2241672915dd7455cb1f1ca8b9606b4ca2aa8e2f905b40efc4d6110722198
- Code revision: coalloc-rl branch, commit `9360dfc53`
- Config: steps_per_phase=1, num_train_steps=1, num_hosts=2, TP=4, train_batch_size=64
- Success criteria:
  - One full multi-host phase completes (sampling on 2 hosts → materialization → multi-host training → export)
  - policy_0001/manifest.json written
  - No sharding or step consistency errors

### Status: IN PROGRESS

- Attempt 1 failed before TPU creation: `train_batch_size=64` not divisible by `per_device_parallelism(16) × num_hosts(2) × TP(4) = 128`. Minimum batch size for 2-host v5p-16 is 128.
- Attempt 2 failed: `gcloud components install alpha` errors on homebrew-managed gcloud. Fixed by making install best-effort (try/except).
- Attempt 3 launched with `train_batch_size=128`, `n_prompts=8`, gcloud fix. Creating spot v5p-16.
- 2026-03-21T22:12Z: TPU `alt-tpu-003` status = WAITING_FOR_RESOURCES. Waiting for spot v5p-16 capacity in us-east5-a.
- 2026-03-21T22:15Z: Cancelled v5p-16 request. Switching to existing v6e-16 TPU `vllm-ray-multihost-v6e16-use1d` in us-east1-d (4 hosts, already ACTIVE). Cleaned all Docker containers on it.
- 2026-03-21T22:17Z: Building image with latest Codex changes included. Pushing to us-east1 registry. v6e-16 has 4 workers (not 2), so this tests 4-host alternating RL.
- 2026-03-21T22:35Z: Image pushed to us-east1: `us-east1-docker.pkg.dev/hai-gcp-models/levanter/levanter-ahmed:alt-rl-1774132138` (sha256:b5b01dc4396904fb0a0088e119fa71b0bea9420bb69b0f6a41042449abd54b64)
- 2026-03-21T22:36Z: Launched ALT-TPU-003 on existing v6e-16 (`vllm-ray-multihost-v6e16-use1d`, us-east1-d, 4 hosts). Config: train_batch_size=256, n_prompts=16, num_hosts=4. GCS: gs://marin-us-east1/alternating-rl/alt-tpu-003-v6e16/
- 2026-03-21T22:39Z: Bootstrap complete. run_state: status=sampling, num_hosts=4, phase_id=0. 4-host sampling in progress on v6e-16.
- 2026-03-21T22:45Z: All 4 hosts completed sampling successfully! Host status files written for hosts 0-3.
  - prepare_sampling: 160.7s, sampling: 165.7s, curriculum_update: 2.0s
- 2026-03-21T22:46Z: Failed during materialization. `max_tokens=768` error. First run left stale `marin-alt-materializer` container on worker 0.
- 2026-03-21T22:50Z: Retry with `--max-output-tokens 512` also showed `max_tokens=768` — stale container from first run was still present (name conflict). Controller didn't clean it before reuse.
- 2026-03-21T22:57Z: Manually cleaned all containers on all 4 workers. Cleaned GCS. Relaunched with fully clean state. Pickled config verified: `curriculum.max_seq_len=1024`.
- 2026-03-21T23:01Z: 4-host sampling succeeded again (66s prepare, 99s sampling). Reached materializing status.
- 2026-03-21T23:09Z: Materializer hung on `TPU backend initialization is taking more than 60.0 seconds. Did you run your code on all TPU hosts?`. Root cause: materializer runs on worker 0 only but JAX on v6e-16 expects all 4 hosts for distributed init. Materializer is a CPU-only task.
  - Fix: added `JAX_PLATFORMS=cpu` to materializer container env in runtime.py. Local controller fix, no image rebuild needed.
- 2026-03-21T23:12Z: Cleaned all containers + GCS. Relaunched with materializer CPU fix.
- 2026-03-21T23:20Z: BREAKTHROUGH — multi-host training running on v6e-16!
  - 4-host sampling: 62s + 98s
  - Materialization: 34.9s (JAX_PLATFORMS=cpu fix worked)
  - Training: 101s so far on full 4-host Levanter mesh
  - First multi-host training ever on alternating RL path
  - Waiting for checkpoint + export to complete phase 0
- 2026-03-21T23:30Z: Training ran and W&B logged but crashed writing phase metrics. Error: `OSError: The specified key does not exist` from `fs.rm()` in `update_phase_metrics`. Root cause: all 4 hosts run the same training code and all try to write/delete the same phase_metrics JSON on GCS. Race condition — one host deletes the file, another tries to delete it and gets 404.
  - Fix needed: guard phase metrics write to process_index==0 only, or make the delete-then-write pattern tolerant of 404.
  - The actual training likely succeeded — the crash is in observability, not the training step itself.
  - Fix: added `_rm_if_exists` helper in state.py to tolerate 404 on delete during concurrent writes. Rebuilding image.
- 2026-03-21T23:37Z: Rebuilt image `alt-rl-1774135803` (sha256:685142ed). Cleaned all containers + GCS. Relaunched.
- 2026-03-21T23:47Z: Metrics race fix worked — training_seconds=135.9s logged successfully. But training container failed with `DEADLINE_EXCEEDED: Barrier timed out. 1/4 tasks reached barrier`. Checkpoint `step-1` exists — training succeeded but post-training barrier sync failed (workers 2,3 timed out).
  - This is a multi-host synchronization issue in the post-training phase, not a training correctness issue.
  - Checkpoint is saved, so export-only recovery should work if we can fix the barrier timeout.
- [ ] TPU provisioned
- [ ] Sampling phase (2 hosts)
- [ ] Materialization
- [ ] Multi-host training
- [ ] Export
- [ ] Verify artifacts

---

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
