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
  # Upper bound, not a quota. A trial races in min(this, its eligible regions);
  # see Region Locality for why exp117-init trials are eligible in fewer.
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
  wall_time: 2 weeks
  # Soft ceiling to keep the job count manageable, not a hard resource quota.
  # It is scoped to exp166 alone; concurrent experiments are accounted separately.
  max_inflight_chips: 8192
  observation_interval: 15m
  recovery:
    idle_reslice_timeout: 3h
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
- Each logical trial races in as many distinct regions as it is eligible for, up
  to `regional_race_width`. Region is part of the W&B run and checkpoint identity
  so regional attempts cannot co-write state. These regional runs are execution
  replicas, not additional experimental trials.
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
- At most one dispatch may be active for a given `(trial_id, region)`. Violating
  this invariant can make two jobs write the same regional checkpoint.

## Region Locality

**Bytes never cross a region boundary.** This is the reason the experiment races
regional replicas instead of relocating one run: each region holds a complete,
self-sufficient copy of everything its attempts need, so a region can be
abandoned or restarted without moving data.

- A regional run's checkpoints belong to that region and stay there. They are
  never copied, moved, mirrored, or read from another region, in either
  direction, for any reason — including recovery, debugging, and salvaging a
  nearly finished run.
- **Moving a trial to a new region means starting that region's run over from
  step 0**, from the logical trial's declared initialization. A new region never
  inherits progress. Losing the steps is the accepted cost of the rule, not a
  problem to engineer around.
- Region is part of the W&B run identity and the checkpoint path precisely so
  two regions cannot co-write state, and so "resume" can only ever mean "resume
  within the same region."
- Data, tokenized cache, output, and checkpoint resolution all resolve
  region-locally. A missing regional dependency is a staging problem to record
  and fix inside that region; it never justifies a cross-region read.
- A same-region slice or TPU-family change is *not* a move. It retains the
  regional W&B run ID and checkpoint path, and resumes from the regional
  checkpoint.

Initialization inputs follow the same rule and constrain where a trial may run:

- A `scratch` trial needs no prior weights, so it may race in any eligible
  region and restarts from random initialization in each one.
- An `exp117-init` trial needs a region-local exp117 seed. Seeds live under
  `checkpoints/protein/exp166-init/<exp117 run id>/<EXP117_VERSION>/checkpoints/`.
  The run id names the exact source run and its origin region, so a seed's
  provenance is readable off its path, and the `exp166-init` namespace keeps that
  name from colliding with the real exp117 run directory in its home region.
- **An `exp117-init` trial may only be placed in a region that already holds its
  seed.** A missing seed is a setup error for that region and fails the run; it
  is never repaired by reading another region.

Seeding is a **one-time setup step, not part of execution.** Each exp117 run
wrote its final checkpoint to a single region, so before the experiment starts
`scratch/exp166_seed_checkpoints.sh` copies **only the final checkpoint** of each
point into every eligible region, producing byte-identical seeds everywhere. The
exp117 run directories themselves are never modified and stay in their original
regions. Once seeding is done, every eligible region is self-sufficient and no
training job ever transfers weights again — `exp117_checkpoint()` resolves the
seed against the execution region and only verifies that it is present and
complete. The copy machinery has been removed from the sweep module so a
cross-region transfer is structurally impossible rather than merely discouraged.

Re-seeding is warranted only when the point set or `EXP117_VERSION` changes, and
is an explicit operator action.

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
    # Operator decision: excluded from placement pending a throughput
    # measurement for this model. Re-enable by operator instruction only.
    enabled: false
```

Nothing smaller is allowed. Enabled families are peers: rank concrete
region/slice targets using observed throughput and observed scheduling outcomes,
not a static family preference.

There is no static region allowlist. Discover every region currently advertised
by the primary Marin cluster for an eligible slice and keep the full grid in the
ledger. Diversify initial targets across regions and, when practical, TPU
families so one capacity failure mode does not stall the entire race. Race width
is bounded by the eligible regions for that trial; see **Region Locality** for
why `exp117-init` trials are placeable in fewer regions than `scratch` trials.

### Capacity Cannot Be Queried

**No command reports available TRC capacity.** `iris cluster status` shows only
what is already in use; it cannot say what TRC would grant next. `ready=0` means
nobody currently holds one, not that it is unobtainable.

Submitting is the only measurement. Never call a target "available" or
"unavailable" from a status read, and never drop one from the grid because a read
looked empty — only repeated failed acquisitions justify deprioritizing it.
Record what was attempted (`gang in N min`, `pending 6h`, `preempted`), and
spread dispatches so scheduling outcomes do the ranking.

Continuously re-rank targets using:

1. sustained post-compilation tokens/second for the same batch size;
2. recent successful gang acquisition and startup latency;
3. recent preemption/failure history; and
4. the need to avoid saturating one region while other regions remain
   unexplored.

Use rolling medians for throughput comparisons and exclude compilation,
checkpoint, and evaluation intervals. Preserve the raw observations in SQLite
so placement decisions remain auditable.

**Treat every throughput number as perishable.** TRC capacity shifts daily, so a
slice size or region that looked optimal last week may be unavailable or slower
today, and a historical measurement from a previous experiment is a prior, not a
constant. Do not converge on one slice size and stop looking. Keep probing other
eligible sizes and regions as availability moves, and prefer a target that is
actually schedulable now over a nominally faster one that never acquires a gang.

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

1. Create one regional-run record per eligible region, up to
   `regional_race_width`, and submit one unique dispatch in each selected
   region, subject to the global chip cap.
2. Keep every regional execution racing until one satisfies the full completion
   contract. Iris parent/child `running`, W&B `running`, a step lead, and higher
   throughput are monitoring signals—not reasons to stop siblings.
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

`scratch/exp166_heartbeat.py` renders exactly this set and persists the
observations; use it rather than assembling the report ad hoc, which drops
fields.

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
  different region that the trial is eligible for. It receives that region's
  W&B/checkpoint identity and **starts over at step 0** from the logical trial's
  declared initialization. It never inherits the old region's progress, and no
  checkpoint is copied to make the move cheaper.
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
- **Never copy, move, or read a checkpoint across regions** — not an exp166
  checkpoint, not an exp117 source checkpoint, in either direction, for any
  reason. If a region lacks the weights a trial needs, that trial does not run
  there. Relocating means starting over; see **Region Locality**.
- Never let a replacement overlap its predecessor on the same regional run.
- Never parse run IDs for metadata.
- Never store secrets in SQLite, logs, commands committed to the repository, or
  W&B config.
- Record every manual override and the evidence that justified it.
