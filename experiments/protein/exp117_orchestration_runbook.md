# exp117 adaptive-sweep — orchestration runbook (s02)

> **STATUS (2026-07-18): s02 EXECUTING, rung 0.** Full-scale ramp done; ~40 logical trials launched,
> 10 rung-0 objectives in, cap `max_inflight_chips=2048`. **Two competitive basins** (top 3 within
> 0.01): the low-LR/high-WD corner **`lr3.16e-4/wd1.6/bs128 = 2.7489` (best)** and `lr1e-3/wd1.6/bs128
> = 2.756`, vs the high-LR basin `lr3.16e-3/wd0.2/bs256 = 2.758`. Rung 0 not converged. Operator-
> requested **bs64 catch-up** in progress (see Slice-throughput balancing). The old s01 results
> (best 2.727) are **frozen** in `scratch/exp117-adaptive-sweep-s01.sqlite`; s02 does not read them.

The live procedure for driving the exp117 sweep as the **Orchestrator** under
`.agents/skills/run-adaptive-sweep/`. One pass = reconcile → converge-check → replan →
place → launch/stop/restart → record → report. Sources of truth: policy
`experiments/protein/exp117_policy.md`, trainer `experiments/protein/exp117_sweep.py`
(one `(epochs,lr,wd,batch_size,tpu,region)` point per launch), state DB
`scratch/exp117-adaptive-sweep-s02.sqlite`. This runbook holds only what those two docs and the
skill do **not**: the identity-vs-execution mechanics, the placement-feasibility filter, the
liveness signal, submission mechanics, the DB schema, and the operator's execution decisions.

## Fixed facts (quick glance; policy is authoritative)
- **Objective:** `eval/tokenized/contacts-v1-val/loss`, minimize, read at the run's **final step**.
- **Axes (4):** epochs `[8,16,32]` (the resource ladder, ratios `[1,2,4]`); LR
  `[3.1623e-4,1e-3,3.1623e-3,1e-2]` (log10, half-decade ×√10, hard domain `[3.1623e-5, 1e-2]` — the
  top value `1e-2` IS `domain.max` and was the prior sweep's best point, so no upward extension; the
  low edge may extend one half-decade down); WD `[0.1,0.2,0.4,0.8,1.6]` (log10, ×2, domain
  `[0.025, 6.4]`); **batch_size `[64,128,256]`** (log2, domain `[32,1024]`). Points stored by axis
  VALUE. Per rung: 4·5·3 = **60 configs**; across 3 rungs = **180 logical trials**.
- **Budget:** `max_inflight_chips=2048`, `observation_interval=15m`, `full_exploitation_level=32`
  (rung index 2). `wall_time=8 weeks`.
- **Steps depend on batch** (not a constant): `steps/epoch = round(TRAIN_TOKENS / (batch·8192))`,
  `TRAIN_TOKENS=4_676_753_425`. So 8ep → **71,360 / 35,680 / 17,840** steps at batch 64/128/256, and
  ~**37.4B tokens** per 8ep at every batch. `run_progress = _step / num_train_steps`.

## Identity vs. execution (the batch_size change)
- **Trial identity = `(epochs, lr, wd, batch_size)`.** `batch_size` is now a *searched coordinate*
  and part of the run id — it is NOT a placement/throughput knob. Two runs that differ only in batch
  are different logical trials with different checkpoints.
- **Execution-only = `(tpu, correction_factor)`.** The slice drives the per-device batch fit
  (`per_device_parallelism`, grad-accum) via the calibrated per-family `CORRECTION_FACTORS`
  (`{v5e:0.5, v6e:0.3, v5p:0.45}`; override with `-e CORRECTION_FACTOR`). Global batch is held at the
  trial's `batch_size` on every slice, so **a same-region slice change resumes the same run**.
- **run_id** = `prot-exp117-cv1-s02-{1_5b}-e{ep}-lr{tag}-wd{tag}-bs{bs}-{region}`. Keyed on
  `(point, region)`, never the TPU. Region change ⇒ fresh run from step 0 (no cross-region checkpoint
  copy — forbidden). W&B group `exp117-contacts-v1-tune`.

## Placement feasibility (data + tensor parallelism)
Since 2026-07-19 (PR #7204 `batch_calibration` module + `tensor_parallel_size` wiring in the trainer)
the recipe uses data **and** tensor parallelism, so **every allowed slice is feasible for every
batch** — `validate_sweep_target` no longer rejects on chip count, and the old "chips ≤ batch" filter
is gone. For a `(tpu, batch)` the trainer derives: **`data_axis_size = gcd(chips, batch)`** (chips the
batch shards over) and **`tensor_parallel_size = chips / data_axis_size`** (the rest = the model
axis). `tensor_parallel_size == 1` whenever chips ≤ batch, so all pre-TP placements are byte-identical
(verified via PREVIEW). Chip counts: **v6e-N=N, v5litepod-N=N, v5p-N=N/2**.

**Consequence — small batches can now use the biggest slices.** e.g. `bs64` on `v6e-256` = 64 data ×
4 model, using all 256 chips at ~4× the wall-clock of a 64-chip slice. Placement is no longer
chip-capped by batch; pick slice size by desired throughput vs. TP overhead — **more TP = more
model-axis comms, so keep `tensor_parallel_size` modest for the 1.5B model (≈ ≤ 4–8); don't 16-way
shard a small model**. HBM is the only hard limit (`tpu_batch_config` raises if even `pdp=1` won't
fit). v5p is still the only family > 256 chips (`v5p-1024`=512ch, `v5p-2048`=1024ch). Region
availability from policy `targets.allow`; large slices are not capacity-guaranteed.

**Never gate slice choice on guessed HBM headroom.** The design (`batch_calibration` + `tensor_parallel_size`
wiring) makes every permissible slice fit every batch; `tpu_batch_config` is the SOLE feasibility
authority and raises if one genuinely won't. Do not avoid a smaller slice on a hand-estimate that
per-device batch/activations won't fit, and do not "memory-match" a prior slice — choose by
availability and throughput, and let the tool reject an infeasible placement. Small slices are
often the right call (more available; fine for near-done trials). When a same-region relocation of a
near-done trial keeps failing to gang, step DOWN in slice size toward the grid floor rather than
holding out for a large slice. **Below-floor slices trade away HBM headroom and can OOM a large
batch:** empirically `v6e-4` GANGED but OOM'd `bs256` (per-device batch 64 → "Used 31.26G of 31.25G
hbm, exceeded by 14.85M"); the calibrated floor for `bs256` is `v6e-8` (per-device batch 32). So
`tpu_batch_config`/XLA remains the authority — try small, and on an OOM step back UP one size to the
smallest slice that fits. (A genuinely reproducible OOM on a grid slice is still a corrective halt;
an OOM only from going below the floor is not.)

## Slice-throughput balancing (bs64 catch-up)
Every 8-epoch run is the SAME ~37.4B tokens regardless of batch, so wall-clock ≈ tokens ÷ chips. A
chip-efficient placement (slice chips ≈ batch/2, giving 2-way accumulation) packs more parallel
trials but makes small-batch runs lag: bs64 on 32 chips runs at ~¼ the rate of bs256 on 128 chips
for the same tokens. If per-batch signal is needed sooner (operator-requested 2026-07-18), **upsize
the lagging small-batch runs to their max-feasible slice** (bs64: 32→64 chips) as chips free from
completions, BEFORE launching new grid points. Mechanism = a same-region slice change (v6e-32→v6e-64,
v5litepod-32→v5litepod-64): stop the job, resubmit the same `--job-name`-with-new-slice + same
`(EPOCHS,LR,WD,BATCH_SIZE,REGION)` → the run_id is unchanged so it **resumes from checkpoint** (no
restart-from-0). Prioritize the most-progressed runs (soonest objective). Doing this off *freed*
chips keeps ≤ `max_inflight_chips` and avoids a sudden reallocation. **Since 2026-07-19 tensor
parallelism removes the 64-chip ceiling for bs64** (it can use v6e-128/256 etc. via TP), so this same
upsize mechanism now pushes small batches onto even larger slices — mind the chip budget and TP
overhead (keep `tensor_parallel_size` ≲ 4–8; see Placement feasibility).

## Env (every command)
```
cd /home/exedev/repos/marin-br/eac-plm-exp117
set -a; source ~/marin.env; set +a
export PATH="$HOME/google-cloud-sdk/bin:$HOME/.local/bin:$PATH"
```
Submit: `uv run iris --cluster marin job run --user "$USERNAME" --no-wait --region <REGION>
--memory=1GB --job-name <deterministic> -e WANDB_API_KEY "$WANDB_API_KEY" -e WANDB_ENTITY
"$WANDB_ENTITY" -e WANDB_PROJECT "$WANDB_PROJECT" -e HUGGING_FACE_HUB_TOKEN "$HF_TOKEN" -e EPOCHS ..
-e LR .. -e WD .. -e BATCH_SIZE .. -e TPU .. -e REGION .. -- python -m
experiments.protein.exp117_sweep`. Ambient `HUGGING_FACE_HUB_TOKEN` is STALE — always forward
`$HF_TOKEN`. `--job-name` is deterministic so a resubmit is an idempotent resume. `PREVIEW=yes`
prints identity/steps/batch-fit and submits nothing — sanity-check any new point shape with it.
The lightweight CPU driver acquires the TPU internally via Fray (`ResourceConfig.with_tpu`); one
job = one point.

## One pass
1. **Reconcile.** For each active dispatch: `iris job list --prefix /eczech/<job>` for state; read
   the W&B run (group `exp117-contacts-v1-tune`, id per run_id above) for `run_progress` + final-step
   objective. Write observations to the DB. Classify terminals: succeeded (objective present) /
   OOM (`RESOURCE_EXHAUSTED`/`CompileTimeHbmOom` → **corrective halt**: stop + surface, don't retry)
   / terminal failure (killed, network/HF-404 → **retry immediately on-target**, not gated) /
   preemption (iris auto-resumes; 12h+ gaps are NORMAL under TRC batch).
   - **SIGSEGV ≈ preemption on multi-host slices (learned s01, reframed 2026-07-17):** nearly every
     slice in our grid is multi-host, and there a SIGSEGV (exit 139) is *most likely* a preempted
     gang cosibling — one host in the gang gets preempted and a sibling reports the segfault. Treat
     it as a preemption, NOT a code crash to investigate: **retry immediately on-target** (resume
     checkpoint) and do not let it drive relocation/escalation or read as a systemic defect, however
     many occur. Investigate further ONLY with a specific reason — a single-host slice, a segfault
     that repeats at step 0 before any training, or a real Python traceback in the log.
   - **LIVENESS (learned s01, still true):** the W&B **`state` field is UNRELIABLE** — it flips to
     `crashed` on any heartbeat lapse (preemption/resume/network) while training continues. Ground
     truth = (a) `_step` **advancing** across passes, and (b) a recent `Progress on:train
     <k>kit/<N>kit … loss=..` tqdm line (`iris job logs <job> --tail --max-lines 60 | grep
     'Progress on:train'`). Genuinely stalled ⟺ `_step` frozen across passes AND no recent progress
     line. Track the no-progress start time in the DB so timeouts measure from last observed
     progress, not from the pass.
2. **Converge-check.** `sweep_tools.py check-convergence` over all completed trials (strict one-step
   neighbor dominance *by the best observed point*, one axis at a time; a missing neighbor passes only
   at the axis's hard domain bound). A locally-dominant point never counts as converged while a better
   objective is known anywhere else in that rung's grid — search continues until the best-known config
   is itself neighbor-confirmed. Stop the whole sweep only when all 3 rungs pass.
3. **Replan** (only when a new objective landed, a recovery timeout fired, or chips freed):
   - Cold/flat predictions → fill the lowest unresolved rung (8ep). NEVER seed a higher rung to burn chips.
   - With completed trials → `predict-objectives` (GBR, refit over EVERY completed objective; features
     = normalized grid positions + normalized rung) to rank candidates per rung; prefer predicted
     center + the neighbors needed to test strict dominance. Interleave 16/32ep work without barriers;
     prefer a cheaper unresolved rung's next candidate when it both advances convergence there AND
     would change the upper-rung ranking.
   - Edge extension (autonomous, within hard domain): LR at ×√10 (half-decade; only the LOW edge —
     the top is pinned at `domain.max=1e-2`), WD at ×2, batch at ×2 (log2). Record value+reason in
     `decisions`.
4. **Place.** Feasibility-filter (above), then `sweep_tools.py rank-targets` per selected trial with
   the `recovery` block + only feasible targets; choose within the returned `selection_pool`
   (diversity while exploration>0; highest-throughput feasible target at rung 2). Keep
   submitted+running+retrying chips ≤ 2048.
   - **Exploration placement (rung 0 / early):** deliberately SAMPLE distinct `(region, family,
     size)` targets to build per-target throughput evidence — don't pile onto one slice type. Larger
     slices give faster wall-clock (operator: bias larger at cold start) but preempt more, schedule
     slower, and are less chip-efficient at fixed batch, so **bigger ≠ always better** — MEASURE.
     Prefer proven regions (europe-west4, us-east5 v6e) but include untried ones for coverage.
   - **Region re-probing (per-slice, per-region).** Availability is per slice *and* region —
     correlated but not identical (a `v6e-64` gang says nothing about `v6e-256` or `v5p` there), and a
     stale "avoid" label or a >~1-day-old read is missing evidence, not proof of unavailability. A
     lone-slice probe under-samples: re-probe each region with **≥2 real grid points across ≥2 slice
     sizes** (span families where the region allows more than one; else span sizes). On a region-wide
     availability drop (work stuck pending / silent-hung region-wide), spread such probes across every
     allowed region not already saturated with running trials to re-map capacity, then resume
     exploiting proven targets. Drop a probe that never gangs by `startup_relocation` (3h). Never a
     filler — every probe is a meaningful grid point.
   - **Prefer small slices; use every family and region.** Small **16–32 chip** `v6e`/`v5litepod`
     slices gang far more readily than big ones (fewer hosts to coschedule) — bias to them,
     especially for near-done or stalled trials. **No region or family is off limits** — `v5p`
     (us-east5, us-central1) is easy to neglect; include it. When relocating a stalled job, be
     aggressive: each move must be a slice/region combo **not already tried** for that trial
     (a big-slice retry that keeps failing to gang is wasted) — walk down in size and across
     families/regions until one gangs.
5. **Dispatch.** Launch/stop/restart exact work orders. Record prediction+placement+decision in
   `decisions` BEFORE dispatch. One logical trial → at most one objective; never duplicate a
   `(rung,point)`.
6. **Report** material events (rung convergence, grid extension, OOM/halt, restart/relocation) +
   ~weekly rollup. (Operator: fully autonomous after the first-command review; halt only on the
   corrective conditions.)

## Recovery ladder (tool-computed; honor `rank-targets` `recovery`, don't auto-fire)
Any observed progress resets the regional no-progress clock. Terminal failures are exempt (retry
now) — including a SIGSEGV, which on our multi-host slices is almost always a preempted gang
cosibling, not a code fault.
- **`startup_relocation_timeout` 3h** — never appeared in W&B → same-region **relocation** (new slice).
- **`same_target_restart_timeout` 6h** — W&B-registered, iris still `running`, no progress → **restart
  in place on the SAME target** (stop + resubmit; resumes checkpoint; keeps logical trial, regional
  run, dispatch_id, checkpoint, and chip charge; new iris submission attempt; releases + may requeue
  the slice). Cheapest recourse and the FIRST thing to try for an ambiguous "running but stalled"
  run. **Wait the full 6h** — most stalls self-recover as preemptions, and restarting forces a
  re-queue (hours). A restart resets only its submission's running-stall window, NOT the regional
  clock — so it can't defer relocation forever.
- **`same_region_relocation_timeout` 24h** — still stalled → **move to another slice, same region**.
- **`cross_region_restart_timeout` 96h** — only after a same-region move also stalled → move to
  another region (fresh run from 0).

## DB schema (`scratch/exp117-adaptive-sweep-s02.sqlite`)
- `sweep_meta` — grid/policy snapshot for this instance.
- `trials(trial_id, rung, epochs, learning_rate, weight_decay, batch_size, status, objective,
  cohort)` — status: planned|dispatched|running|succeeded|failed|halted. `batch_size` is part of
  identity, so `trial_id` encodes all four axes.
- `dispatches(dispatch_id, trial_id, regional_run_id, region, tpu_slice, chips, state,
  submission_attempt, iris_job_id, submitted_at)` — a same-target restart increments
  `submission_attempt` on the SAME `dispatch_id`; a relocation is a new dispatch.
- `observations(dispatch_id, observed_at, wandb_run_id, run_progress, objective_value)`.
- `decisions(at, kind, detail)` — audit log (predictions, placements, extensions, restarts).

## Architecture (operator-confirmed)
Run the **inline** model: one session is BOTH Orchestrator and Dispatcher — assemble/submit iris
commands and write the DB directly. Do NOT spawn a separate `dispatch-job` subagent; the structured
handoff is collapsed into direct table writes. Keep the role boundary by discipline (dispatch
mechanics don't drive policy/placement — those follow the tools + policy).

## Loop mechanism
An in-session cron heartbeat (~30 min) fires reconcile-replan passes with full context. Cron is
session-only and expires after 7 days — **re-arm each session**; the DB makes cold resume cheap.

## Resume from cold (new session)
Read this file + policy + skill SKILL.md, `source ~/marin.env`, then run one pass. The s02 DB is the
source of truth for what has been launched; reconcile it against live `iris job list` + W&B before
launching anything (avoid duplicate logical trials). Do not read or mutate the s01 DB.
