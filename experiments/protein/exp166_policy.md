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
  # One clock: hours since a regional run's step last advanced in W&B. A run that
  # has never produced a step is stalled since its *first* dispatch was submitted.
  recovery:
    restart_after: 3h    # same region, same slice; resumes from the regional checkpoint
    reslice_after: 12h   # same region, different slice
    relocate_after: 4d   # different region, restarts at step 0
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

The eligible grid is fixed and stated here rather than discovered at runtime.
These region/family pairings are stable; treat a change to them as an operator
decision, not something to infer from a cluster read.

| family | slices (chips) | regions |
|---|---|---|
| `v6e` | 32, 64, 128, 256 | europe-west4, us-east1, us-east5 |
| `v5litepod` (v5e) | 32, 64, 128, 256 | europe-west4, us-west4 |
| `v5p` | 64, 128, 256, 512 | us-east5 (us-central1 unseeded) |

`v5litepod-N` counts chips; `v5p-N` counts cores, so `v5p-64` is 32 chips and
`v5p-512` is 256. v5p is **low priority**: carry a probe or two when v5p capacity
looks reachable, but do not rotate the fleet onto it or displace a v6e/v5e target
that is placing. us-central1 is the other v5p region and stays out of the grid —
placing there needs a full seeding and data-staging pass first, which is an
operator decision, not a recovery action.

**32 chips is the floor.** Below it a bs128 trial needs per-device parallelism
above 4 and the calibrator starts trading throughput for gradient accumulation.
256 is the ceiling.

So four regions are placeable: **europe-west4** serves v6e and v5e, **us-east1**
serves v6e, **us-east5** serves v6e and (low priority) v5p, **us-west4** serves
v5e only.
Enabled families are peers — rank targets by observed throughput and scheduling
outcomes, never by a static family preference. Diversify initial targets across
regions and, when practical, families, so one capacity failure mode cannot stall
a whole race. Race width is bounded by the eligible regions for that trial; see
**Region Locality** for why `exp117-init` trials may be placeable in fewer.

### Capacity Is Unknowable, So Placement Is Active

**No command reports available TRC capacity.** `iris cluster status` shows only
what is already in use; `ready=0` means nobody holds one, not that it is
unobtainable. Submitting is the only measurement.

**The escalation ladder is a safety net, not a strategy.** It reacts to a single
stalled dispatch and knows nothing about where capacity actually is. Maximizing
throughput is a standing, active job, performed every heartbeat by reading what
was *granted* — not what was requested — and moving the fleet accordingly:

- **Follow the grants.** Shift placement toward region/slice pairs that recently
  produced steps and away from those that have not. A new dispatch picks its size
  this way, over the full 32–256 range; batch size never constrains it. Rank on
  grant rate before throughput — a slice that is idle because it never schedules
  is worth less than a slower one that runs.
- **Probe floor: never fewer than two live dispatches in each eligible region.**
  A quiet region is in a lull, not broken — every region here has been
  productive before, and "no grants" is only ever a claim about the last few
  hours. Rebalancing away from a quiet region is expected; emptying it is a bug,
  and so is any negative capacity claim stated without its time window.
- **Race floor: every incomplete trial holds `regional_race_width` live
  dispatches.** Top up whenever the live count falls below that; a race silently
  running narrow is the most expensive failure here.
- **Refuse to ossify.** Placement unchanged across several passes while
  chips-training stays flat is a signal to change something, never evidence that
  the current placement is right.

Rank on ledger evidence: sustained post-compilation tokens/second at the same
batch size, time-to-gang, and recent preemption history. Use rolling medians,
excluding compilation, checkpoint and eval intervals. Every throughput figure is
perishable — a prior, never a constant.

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

### Client Revision Floor

The controller rejects submissions from clients older than a rolling floor:
`marin-iris client is too old (build X; minimum Y)`. In an editable install
`iris.version.client_revision_date()` falls back to the last commit date
touching `lib/iris`, so a branch drifts under the floor without any local
change. Every submission then fails while the stall clock keeps retiring
dispatches, draining the fleet to zero.

Fix by setting `BUILD_DATE` in `lib/iris/src/iris/_build_info.py` to today. The
floor is advisory, so stamping it is legitimate; the alternative is merging a
newer `lib/iris`. Watch for it in the ledger as `submit_failed` dispatches with
the error in `stop_reason`, and re-stamp when the floor next advances past it.

## Monitoring

Two signals, each used for exactly one thing.

**Iris is a gate, nothing more.** A dispatch is either `running` — training
*could* be happening — or terminal, meaning it is not and never will be.
Terminal dispatches get replaced. Nothing else about Iris is actionable: not
preemption counts, not task ages, not parent/child detail. Iris `running` is
never evidence that training is happening, and no Iris state ever justifies
leaving a stalled run alone. Never reason about whether Iris "will eventually"
reschedule something. That is not observable from outside, and it is sometimes
false forever.

**W&B is the truth.** A regional run is training if and only if its W&B state is
`running` *and* its step is advancing. Anything else is stalled, including a
`running` run whose step is frozen.

At every observation interval record, for every regional run, W&B state, step,
objective and throughput, plus the Iris gate state of every nonterminal
dispatch. Heartbeats report both placement spans (chips/regions/slices submitted
to Iris, and chips/regions/slices W&B-running), trials complete/running/pending,
races unresolved, median tokens/second by active target, and every stall with
its escalation. `scratch/exp166_heartbeat.py` renders exactly this set; use it
rather than assembling the report by hand.

### Heartbeat Format

Report to the operator in this shape. Never name scripts, flags, or job IDs —
those are plumbing. Say what happened, not how it was invoked.

```markdown
## exp166 · <UTC time>

**Fleet**  36 dispatches · 4,480 chips submitted → 128 training (1 run)
**Trials** 0 complete · 1 training · 11 pending · 12/12 races unresolved

### Placement
| region | v6e-32 | v6e-64 | v6e-128 | v6e-256 | v5e-32 | v5e-64 | v5e-128 | v5e-256 | chips |

### Training now
| trial | region · slice | progress | eval | tok/s |

### Recovery
| action | trial | region · slice | stall | now |

### Leading runs   (top 3 by eval only)
| trial | progress | eval | vs exp117 |

### Notes
What the operator should pay attention to. Nothing else.
```

The two placement spans live in the **Fleet** line as `submitted → training`, so
the gap between chips requested and chips working is unmissable. Omit an empty
section rather than printing a placeholder, except Fleet and Trials which always
appear.

**Notes: 100 words max, usually empty.** Only what the operator would act on or
be wrong without. Not reasoning, not a restatement of the tables.

## Stall Escalation

One clock governs all recovery: **hours since a regional run's step last
advanced.** A run that has never produced a step is stalled since its **first**
dispatch was submitted, so the same clock covers a job that died and one that
never started.

Measure a never-started run from its first dispatch, never its current one.
Every restart mints a new dispatch, so a current-dispatch clock resets on each
attempt, never reaches `reslice_after`, and retries a single slice size forever —
which is precisely the region that most needs to try a different size.

Cause is irrelevant. Preemption, an Iris hang, and absent capacity are
indistinguishable from outside and escalate identically. Do not diagnose, and do
not wait on a theory about which one it is.

| stall exceeds | action |
|---|---|
| `restart_after` | resubmit on the same slice and region |
| `reslice_after` | resubmit in the same region on a different slice |
| `relocate_after` | abandon the region; re-arm the trial in a different one |

The reslice rotation **descends** through the eligible sizes, wrapping from the
floor back to the ceiling. Ascending makes the largest slice the destination for
every stalled mid-size dispatch, which piles the fleet onto one size regardless
of whether it is being granted.

Every replacement stops the old Iris job and submits a new uniquely-named one;
there is no in-place resubmit. Same-region replacements resume from the regional
checkpoint, so a restart costs at most one `save_interval`. A relocation starts
over at step 0 from the trial's declared initialization and never copies a
checkpoint — region outages run hours to days but rarely longer, which is why
`relocate_after` is measured in days.

A terminal Iris dispatch is replaced immediately rather than waiting out the
clock. The one case that is not capacity: a dispatch failing the same way on its
first attempt in more than one region is a code or data fault — stop and
diagnose instead of escalating. Record every action in `events` with the stall
that triggered it, and never delete superseded attempts.

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
- Recovery and rebalancing within the current chip scale need no approval;
  raising or lowering `max_inflight_chips` is an operator decision.
