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
