# Job Monitoring Loop

Monitor a job continuously and recover on failure. For active jobs, recovery is stop then submit again. This playbook now has two tracks:

- `ray` track: Ray Jobs API (`scripts/ray/cluster.py`, `ray_run.py`)
- `iris` track: Iris Jobs API (`uv run iris ... job ...`)

Cluster-level actions are out of scope for this loop. Do not restart, recreate, or otherwise mutate the cluster (including head-node restart) unless the user gives explicit consent in the current thread, except for the TPU bad-node recovery flow documented below.

## Before Starting

When the user asks to start monitoring, first choose a track and gather required info.

### Track Selection

Ask:

> "Should I monitor this as `ray` or `iris`?"

### Required Info: Ray Track

1. `job_id` - Ray job ID (e.g., `ray-run-held-isoflop_sweep-20260131-051716`)
2. `cluster` - cluster name/alias for `scripts/ray/cluster.py` (e.g., `us-east5-a`, `us-central2`)
3. `experiment` - script path used for resubmission (e.g., `experiments/isoflop_sweep.py`)

### Required Info: Iris Track

1. `job_id` - Iris job ID in canonical format `/<user>/<job>` (e.g., `/dlwh/iris-run-train_tiny_model_tpu-20260302-185630`)
2. `config` - Iris config path (e.g., `lib/iris/examples/marin.yaml`)
3. `resubmit_command` - exact Iris submit command for resubmission; must include `--no-wait`
4. For Marin TPU training jobs, use `--extra marin:tpu` (not `--extra marin:cpu`)
5. For TPU jobs, the resubmit command must request TPU resources with `--tpu <variant>`.
   `--reserve <variant>` only holds capacity; it does not attach TPU devices to the task container.

Example Iris resubmit command:
`uv run iris --config lib/iris/examples/marin.yaml job run --no-wait --extra marin:tpu --tpu v5litepod-16 -- python experiments/tutorials/train_tiny_model_tpu.py`

If any required field is missing, ask for it before proceeding.

## Monitoring Ownership and Duration

- Assign a single monitoring owner when the loop starts.
- Keep this loop running until one of the following:
  - the job reaches a terminal state and the user has acknowledged next action
  - a user-specified stopping point is reached
  - an unrecoverable error is found and reported to the user
- Do not stop early after seeing first loss lines, first eval, or first W&B link.
- Expect monitoring to commonly take 4-5 hours for ferry-scale runs.
- For GPT Codex specifically: unless otherwise directed, do not end your turn just to give a status update; keep monitoring until terminal state or until the user's goal is reached.
- If the user requests continuous monitoring, do not end the turn while monitoring is active; continue until terminal state, a user-specified stopping point, or an unrecoverable error.
- If handoff is needed, transfer ownership explicitly with: track, current `job_id`, latest error/signal, W&B link(s), and resubmission metadata.

## Cadence and Tooling Notes

- Cadence default after startup stabilization is `sleep 570`.
- Startup stabilization sequence (after submit/resubmit):
  - once the job is submitted, sleep `120` and check for immediate failure
  - if still alive, switch to the normal `570` cadence
- Tool-runtime workaround:
  - keep one long-running monitor process/session
  - poll the same session in ~30 second chunks as needed by tool runtime limits
  - repeated no-output polls are expected while waiting for the next 570-second check
- Single monitor process rule:
  - run only one active monitor loop per job to avoid duplicate SSH tunnel and port-binding conflicts

## State File

Write to `monitoring_state.json` in scratchpad/workspace.

### Ray State File

```json
{
  "track": "ray",
  "job_id": "<JOB_ID>",
  "cluster": "<CLUSTER>",
  "experiment": "<EXPERIMENT_PATH>",
  "restart_count": 0
}
```

### Iris State File

```json
{
  "track": "iris",
  "job_id": "<JOB_ID>",
  "config": "<IRIS_CONFIG_PATH>",
  "resubmit_command": "<IRIS_JOB_RUN_COMMAND_WITH_NO_WAIT>",
  "restart_count": 0
}
```

## Loop (Interleaved By Track)

Use one shared loop and branch commands/status handling by `track`.

```
1. SLEEP
   - if just submitted/restarted: sleep 120 once
   - otherwise: sleep 570

2. CHECK LOGS
   - if track=ray:
     uv run scripts/ray/cluster.py --cluster <CLUSTER> job-logs -n 200 -g "loss\|error\|Error\|Traceback\|RESOURCE_EXHAUSTED\|compiler_base.cc:2587\|Program hbm requirement\|Largest program allocations" <JOB_ID>
   - if track=iris:
     uv run iris --config <CONFIG> job logs --since-seconds 900 --include-children <JOB_ID> | rg -i -e "loss|error|traceback|exception|resource_exhausted|oom|compiler_base\.cc:2587|program hbm requirement|largest program allocations|ownerdiederror|dead node|node death|autoscaler unsatisfied resources|no accelerator found|failed_precondition|device or resource busy"

3. CHECK STATUS
   - if track=ray:
     - terminal success: `SUCCEEDED`
     - terminal non-success: `FAILED`, `STOPPED`
     - non-terminal: `PENDING`, `RUNNING`
   - if track=iris:
     - query:
       uv run iris --config <CONFIG> job list --json --prefix <JOB_ID>
     - terminal success: `JOB_STATE_SUCCEEDED`
     - terminal non-success: `JOB_STATE_FAILED`, `JOB_STATE_KILLED`, `JOB_STATE_WORKER_FAILED`, `JOB_STATE_UNSCHEDULABLE`
     - non-terminal: `JOB_STATE_PENDING`, `JOB_STATE_BUILDING`, `JOB_STATE_RUNNING`
     - if `pending_reason` indicates worker scale-up/capacity wait, treat it as scheduler capacity wait:
       - do not run cluster update/recreate/restart actions
       - continue waiting on cadence, or stop+resubmit only if user explicitly asks
   - In both tracks, treat `RUNNING` as controller-level signal only; confirm allocation via expected W&B run when possible.

4. PRINT W&B RUN IDS/LINKS
   - Inspect logs for W&B URL/ID (`wandb: 🚀 View run`, `/runs/<RUN_ID>`).
   - Print run id(s) and full link(s) once per training run.
   - If resubmission creates a new run, print new id/link once.

5. REPORT PROGRESS
   - Report as `~<current>/<exact_max>`.
   - Example: `~4600/5155`.
   - Never present rounded max as exact.
   - If max unknown, report `~<current>/unknown`.

6. EVALUATE
   - If user stop point reached -> summarize and exit.
   - If terminal success -> report completion and exit.
   - If terminal non-success -> go to step 7 unless user says otherwise.
   - If output contains `error`/`Error`/`Traceback` -> go to step 7.
   - Treat TPU/XLA HBM reports as failure even without literal OOM:
     - `Program hbm requirement ...`
     - `Largest program allocations in hbm`
   - If progress stalls across multiple intervals with `OwnerDiedError`, dead node, or unsatisfied resources -> mark `degraded` and notify user.
   - If loss/progress lines are present and status non-terminal -> go to step 1.
   - If no output and status non-terminal -> go to step 1.

7. RECOVER (STOP -> RESUBMIT)
   - Recover only the job (never mutate cluster without explicit consent, except TPU bad-node flow below).
   - If current job is still non-terminal, stop it first:
     - if track=ray:
       uv run scripts/ray/cluster.py --cluster <CLUSTER> stop-job <JOB_ID>
     - if track=iris:
       uv run iris --config <CONFIG> job stop <JOB_ID>
   - Then resubmit:
     - if track=ray:
       uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster <CLUSTER> -- python <EXPERIMENT_PATH>
     - if track=iris:
       <RESUBMIT_COMMAND>

   - Capture `job_id` from output.
     - Ray: from submit output / returned submission id.
     - Iris: line like `Job submitted: /<user>/<job>`.
   - Iris nuance:
     - if `resubmit_command` omits `--job-name`, Iris auto-generates a fresh id each resubmission.
     - if `resubmit_command` uses a fixed `--job-name`, Iris may reuse the same id after terminal completion by replacing the finished job.
   - Update state file: `job_id=<NEW_JOB_ID>`, `restart_count += 1`.
   - Go to step 1.
```

## Cluster Mutation Guardrail

- Default rule: never restart, recreate, or otherwise mutate the cluster without explicit human consent in this thread.
- This includes "cluster update" style actions; do not use them in Iris monitoring by default.

## Fixing Small Bugs

When EVALUATE detects an error, before recovery:

1. Analyze logs:
   - Look for `Traceback`, `Error`, `Exception`.
   - Identify file and line number.
2. If it is a small fix (typo, missing import, wrong variable name):
   - Read the file.
   - Make the fix.
   - Proceed to RECOVER for the active track.
3. If it is complex (architectural, unclear cause, broad investigation):
   - Do not auto-fix.
   - Report to user and exit loop.

Small-fix examples:
- `NameError: name 'foo' is not defined`
- `ImportError: cannot import 'bar'`
- `SyntaxError` from missing comma/bracket/colon
- obvious wrong `KeyError` key

Complex examples:
- OOM errors
- TPU/XLA HBM exhaustion signatures (`Program hbm requirement`, `Largest program allocations in hbm`)
- distributed training failures
- data loading issues
- unclear stack traces spanning multiple files

## TPU Bad-Node Recovery

For TPU-related errors such as:
- `RuntimeError: No accelerator found. Please run on a TPU or GPU.`
- `Failed to cleanup driver after error: INTERNAL: FAILED_PRECONDITION`
- `Device or resource busy`

Treat as bad TPU node:

1. Identify bad TPU worker IP from logs.
2. Find TPU VM name for that IP.
3. Delete bad TPU VM and let cluster replace it.
   - `gcloud compute tpus tpu-vm delete <TPU_VM_NAME> --zone <ZONE> --quiet`
4. Recover the job (stop then resubmit) using the active track.

## Notes

- Sleep must be foreground (max ~10 min due to tool timeout).
- Loop control is at agent level, not bash.
- Track `restart_count` to detect flapping.
- State file allows resume after context reset.
- Ray and Iris status enums are not identical:
  - Ray: `PENDING`, `RUNNING`, `STOPPED`, `SUCCEEDED`, `FAILED`
  - Iris: `JOB_STATE_PENDING`, `JOB_STATE_BUILDING`, `JOB_STATE_RUNNING`, `JOB_STATE_SUCCEEDED`, `JOB_STATE_FAILED`, `JOB_STATE_KILLED`, `JOB_STATE_WORKER_FAILED`, `JOB_STATE_UNSCHEDULABLE`
  - Practical mapping:
    - Ray `STOPPED` ~= Iris `JOB_STATE_KILLED`
    - Iris `JOB_STATE_WORKER_FAILED` and `JOB_STATE_UNSCHEDULABLE` have no direct Ray `JobStatus` equivalent and should be treated as terminal failures.
- Iris `job list --prefix` requires canonical job names (`/<user>/<job>`), not short names.
- Iris monitoring is job-level; cluster updates are not part of normal recovery.
- Loop duration can be hours to days.
- If same error repeats after one fix attempt, do not retry blindly; report to user.

## Optional Todo Pattern

If the agent supports todos:

- [ ] sleep 570
- [ ] check logs for active track (`scripts/ray/cluster.py job-logs` or `iris job logs`)
- [ ] check status for active track (`ray list jobs` via helper or `iris job list --json --prefix`)
- [ ] print relevant logs/status (progress and W&B links)
- [ ] evaluate and take action
- [ ] recreate todo list if loop continues
