# Exp 166 Execution Policy

Operator policy for the fixed contacts-v1 amino-acid augmentation ablation
([MarinFold #166](https://github.com/Open-Athena/MarinFold/issues/166)).
Trials are defined by `experiments/protein/exp166_sweep.py`; this policy governs
placement, regional racing, monitoring, recovery, and durable execution state.
It does not define a hyperparameter search.

## Experiment Contract

```yaml
experiment:
  training_script: experiments/protein/exp166_sweep.py
  trial_selector: TRIAL
  # Six fixed exp117 configurations, each run from scratch and from its
  # corresponding exp117 checkpoint.
  logical_trial_count: 12
  regional_race_width: 3
  single_job_command: >
    TRIAL={trial_id} TPU={tpu_slice} REGION={region}
    uv run python -m experiments.protein.exp166_sweep
  objective:
    wandb_metric: eval/tokenized/contacts-v1-val/loss
    observation: final_step
    direction: minimize

execution:
  state_db: scratch/exp166-execution.sqlite
  wall_time: 8 weeks
  max_inflight_chips: 2048
  observation_interval: 15m
  recovery:
    idle_reslice_timeout: 1h
    startup_relocation_timeout: 1h
    same_target_restart_timeout: 6h
    same_region_relocation_timeout: 24h
    cross_region_restart_timeout: 96h
```

The `TRIALS` mapping in the sweep module is the source of truth for the twelve
logical trial IDs and their structured metadata. Do not recreate that mapping
by parsing run names.

## Identity and Completion

- A logical trial is one fixed configuration plus one initialization mode.
  `TRIAL` identifies it; region and TPU placement do not.
- Each logical trial races in three distinct regions. Region is part of the W&B
  run and checkpoint identity so regional attempts cannot co-write state. These
  regional runs are execution replicas, not additional experimental trials.
- TPU family and slice are placement only. A same-region slice change retains
  the regional W&B run ID and checkpoint path.
- An Iris job is one dispatch attempt. Its name is
  `<wandb_run_id>-<slice>-<unique>` and is never reused.
- A logical trial succeeds when one regional run reaches the configured final
  step, logs the final objective, and finishes its required checkpoint/output
  work. W&B `finished` alone is insufficient if the final step or objective is
  missing.
- The race continues until one regional run is **done** by the success
  definition above. Queued, starting, and actively training sibling dispatches
  remain in the race; health, startup order, step lead, or throughput never
  promotes an early winner.
- Once one regional run is done, mark it as the winner and stop every other
  nonterminal regional dispatch for that logical trial, including siblings
  already training. Record these cancellations as race losses, not failures.
  If completions are observed simultaneously, use the earliest verifiable
  completion timestamp as the winner and retain the other completed run as a
  duplicate execution artifact, not an independent trial result.
- A scratch trial restarts from random initialization after a cross-region
  move. An `exp117-init` trial stages its corresponding exp117 checkpoint
  region-locally, then strictly loads only the model subtree into a fresh
  optimizer and exp166 schedule at step 0. Exp166 intermediate checkpoints do
  not move between regions.
- At most one dispatch may be active for a given `(trial_id, region)`. Violating
  this invariant can make two jobs write the same regional checkpoint.

## Durable SQLite Ledger

Every logical trial, regional run, dispatch, observation, and recovery action
must be recorded in `scratch/exp166-execution.sqlite`. The database is the
durable control-plane record; terminal output and an in-memory plan are not.

At minimum, retain:

- `logical_trials`: structured trial metadata, initialization mode, target final
  step, overall status, winning region, and final objective.
- `regional_runs`: trial ID, region, opaque W&B run ID, checkpoint/output
  identity, latest W&B state and step, final objective, and winner/race status.
- `dispatches`: unique Iris job ID, regional run, TPU family/slice, submission
  command, attempt number, timestamps, Iris states, stop reason, exit status,
  and whether the dispatch ever reached W&B `running`.
- `observations`: timestamped W&B state/step, objective when present,
  `throughput/tokens_per_second`, `throughput/examples_per_second`, MFU when
  available, and the Iris parent/child/task evidence inspected.
- `events`: append-only submission, promotion, cancellation, failure,
  resubmission, relocation, operator-decision, and completion records with a
  concise reason.

Use transactions when changing control state and recording the action that
caused it. Never delete failed or superseded attempts; mark them terminal and
append the replacement. Store commands with variable names or redacted values,
never API keys or tokens.

W&B run IDs are opaque identity keys. Recover epochs, LR, weight decay, batch
size, initialization, region, placement, and other metadata from the structured
dispatch record, W&B config/tags, or the sweep module—not from string parsing.

## TPU and Region Grid

The current batch calibrator derives data and tensor parallelism, so every
allowed slice is eligible for every exp166 batch size. Do not reject a target
because the slice is larger than the global batch; excess chips are assigned to
tensor parallelism.

Eligible hardware is deliberately restricted:

```yaml
tpu_grid:
  v6e:
    chips: [64, 128, 256]
  v5litepod:
    # v5e Iris topology names count chips.
    chips: [64, 128, 256]
  v5p:
    # v5p-N counts cores, or N/2 chips. v5p-64 is the 32-chip floor.
    cores: [64, 128, 256, 512, 1024, 2048]
    allow_larger_if_advertised: true
```

Nothing smaller is allowed. All three families are peers: rank concrete
region/slice targets using observed throughput and current availability, not a
static family preference.

There is no static region allowlist. Discover every region currently advertised
by the primary Marin cluster for an eligible slice and keep the full grid in the
ledger. Each race must use three distinct regions. Diversify initial targets
across regions and, when practical, TPU families so one capacity failure mode
does not stall the entire race.

Data, cache, output, and checkpoint resolution must be region-local. A missing
regional dependency is an execution/staging problem to record and fix; it is
not evidence that the region or TPU family should be permanently removed from
the grid. A stale availability read is missing evidence, not proof of
unavailability.

Continuously re-rank targets using:

1. sustained post-compilation tokens/second for the same batch size;
2. recent successful gang acquisition and startup latency;
3. recent preemption/failure history; and
4. the need to avoid saturating one region while other regions remain
   unexplored.

Use rolling medians for throughput comparisons and exclude compilation,
checkpoint, and evaluation intervals. Preserve the raw observations in SQLite
so placement decisions remain auditable.

## Submission and Regional Racing

Use the primary `marin` cluster. A production submission has this shape:

```bash
source ~/marin.env && uv run iris --cluster marin job run \
    --user "$USERNAME" --no-wait --job-name "{unique_job_name}" \
    --region "{region}" --memory=1GB \
    -e WANDB_API_KEY "$WANDB_API_KEY" \
    -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" \
    -e WANDB_ENTITY "$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" \
    -e TRIAL "{trial_id}" -e TPU "{tpu_slice}" -e REGION "{region}" \
    -- python -m experiments.protein.exp166_sweep
```

Before the first production submission, show an assembled command to the
operator for review. Append `--user "$USERNAME"` to every submission and
resubmission.

For each logical trial:

1. Create three regional-run records and submit one unique dispatch in each
   selected region, subject to the global chip cap.
2. Keep all three regional executions racing until one satisfies the full
   completion contract. Iris parent/child `running`, W&B `running`, a step lead,
   and higher throughput are monitoring signals—not reasons to stop siblings.
3. Recover failed or stalled regional executions independently while no winner
   exists. Healthy siblings continue unaffected.
4. When the first regional run is done, transactionally mark the logical trial
   complete and that regional run the winner, then stop every other nonterminal
   sibling and record each cancellation as `race_lost`.
5. Retain all observations and artifacts from losing or duplicate regional
   executions, but use only the winning run's final objective as the logical
   trial result.

Submitted, running, and retrying dispatches all count toward
`max_inflight_chips`. Do not exceed the cap while replacing a job: stop and
observe the old dispatch as terminal before counting the replacement as
available capacity.

## Monitoring and Liveness

**W&B run state is the favored liveness signal.** `state=running` means the
regional run is training and is the only default basis for calling it active.
`crashed`, `failed`, and `finished` are not active states. Investigate and
recover `crashed` or `failed`; do not relabel a crash as a transient flap
without evidence.

Iris parent and child `running` states are scheduling gates, not proof that
training started. Only the child job's individual tasks running as a complete
coscheduled gang provide meaningful Iris-side startup evidence. Use
`iris job summary` to drill from parent to child to tasks when a specific run
needs deeper debugging. Favor W&B over routinely substituting this lower-level
view.

If W&B is temporarily unreachable, record liveness as `unknown`; do not infer
training solely from Iris. If W&B says `running` but step and throughput have
not advanced, investigate logs, gang task state, and checkpoint/evaluation
activity before acting. Never apply an idle-startup timeout automatically to a
run that W&B still reports as training.

At every observation interval:

- refresh W&B state, latest step, objective values, throughput, and timestamp;
- refresh Iris parent/child status for every nonterminal dispatch;
- verify the one-dispatch-per-`(trial_id, region)` invariant;
- identify newly completed trials, siblings to stop only after a winner,
  recovery deadlines, and chip budget;
- transactionally persist observations before submitting or stopping work.

Heartbeats must report two placement spans:

1. chips, regions, and slices **submitted to Iris in any nonterminal state**;
2. chips, regions, and slices **running per W&B** (`state=running`).

Also report logical trials complete/running/pending, regional races unresolved,
median tokens/second by active target, attempts awaiting recovery, and the next
recovery deadline. Never label all Iris-running parents as active training.

## Failure Recovery and Resubmission

TRC/TPU capacity is preemptible and long no-progress periods are expected.
Recovery changes dispatches, not logical trial identity.

- **Idle reslice:** if a dispatch has been submitted but W&B is not `running`
  for more than `idle_reslice_timeout`, stop it and submit a uniquely named
  replacement on a different eligible slice/region. Never idle-reslice a run
  actively training according to W&B.
- **Startup relocation:** if Iris acquired a gang but training did not reach
  W&B `running` within `startup_relocation_timeout`, inspect startup logs and
  relocate unless a concrete, bounded initialization step is progressing.
- **Same-target restart:** a stall-based retry on the same slice/region uses a
  new Iris job name and follows `same_target_restart_timeout`.
- **Same-region relocation:** after `same_region_relocation_timeout` without
  material progress, stop the old job and try another eligible slice in the
  same region, preserving the regional run/checkpoint identity.
- **Cross-region restart:** after `cross_region_restart_timeout`, or when the
  regional pool is demonstrably exhausted, replace or re-arm a competitor in a
  different region. It receives that region's W&B/checkpoint identity and
  starts from the logical trial's declared initialization.
- **Terminal failure:** retry immediately when the failure is transient and the
  next action is clear; timeout thresholds do not require waiting after a
  terminal state.

There is no in-place Iris resubmit. Every restart or relocation stops the old
Iris job and submits a new uniquely named job. Same-region attempts resume from
the regional checkpoint. Cross-region attempts do not inherit exp166 progress.

A SIGSEGV on a multi-host slice is treated as a preempted gang cosibling:
retry in place rather than diagnosing it as a code fault, absent a specific
reason. Escalate instead of blindly retrying when the crash is reproducible at
the same step, appears on a single-host job, carries a deterministic application
stack, or is accompanied by evidence of a data/configuration fault.

Do not classify OOMs, tokenizer/data-contract errors, invalid topology errors,
or deterministic Python/JAX exceptions as capacity failures. Diagnose them
before resubmission. Repeated failures must remain visible in the ledger even
when a later attempt succeeds.

## Operator Guardrails

- Do not submit production work until the command template and first assembled
  command have been reviewed.
- Never mutate cluster infrastructure as part of trial recovery.
- Never move or copy an exp166 checkpoint across regions.
- Never let a replacement overlap its predecessor on the same regional run.
- Never parse run IDs for metadata.
- Never store secrets in SQLite, logs, commands committed to the repository, or
  W&B config.
- Record every manual override and the evidence that justified it.
