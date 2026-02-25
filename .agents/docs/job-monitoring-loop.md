# Job Monitoring Loop

Monitor a Ray job, automatically restarting the **job** on failure.

## Before Starting

When the user asks you to start a monitoring loop, gather the required information first:

1. **job_id** - What is the Ray job ID? (e.g., `ray-run-held-isoflop_sweep-20260131-051716`)
2. **cluster** - Which cluster is it running on? (e.g., `us-east5-a`, `us-central2`)
3. **experiment** - What is the experiment script path? (e.g., `experiments/isoflop_sweep.py`)

Cluster-level actions are out of scope for this loop. Do not restart, recreate, or otherwise mutate the cluster (including head-node restart) unless the user gives explicit consent in the current thread, except for the TPU bad-node recovery flow documented below.

Ask the user for any missing information before proceeding. Example:

> "I need a few details to start the monitoring loop:
> - Job ID?
> - Cluster?
> - Experiment script path?"

Once you have all three, write the state file and begin the loop.

## Monitoring Ownership and Duration

- Assign a single monitoring owner when the loop starts.
- Keep this loop running until one of the following:
  - the job reaches terminal state (`SUCCEEDED`, `FAILED`, or `STOPPED`) and the user has acknowledged next action
  - a user-specified stopping point is reached
  - an unrecoverable error is found and reported to the user
- Do not stop early after seeing first loss lines, first eval, or first W&B link.
- Expect monitoring to commonly take 4-5 hours for ferry-scale runs.
- For GPT Codex specifically: if the user requests continuous monitoring, do not end the turn while monitoring is active; continue until terminal state, a user-specified stopping point, or an unrecoverable error.
- If handoff is needed, transfer ownership explicitly with: current `job_id`, cluster, latest error/signal, and W&B link(s).

## Cadence and Tooling Notes

- Cadence default after startup stabilization is `sleep 570`.
- Startup stabilization sequence (after submit/restart):
  - once the job is submitted, sleep `120` and check for immediate failure
  - if still alive, switch to the normal `570` cadence
- Tool-runtime workaround:
  - keep one long-running monitor process/session
  - poll the same session in ~30 second chunks as needed by tool runtime limits
  - repeated no-output polls are expected while waiting for the next 570-second check
- Single monitor process rule:
  - run only one active monitor loop per job to avoid duplicate SSH tunnel and port-binding conflicts

## State File

Write to a local file (e.g., `monitoring_state.json` in the scratchpad):

```json
{
  "job_id": "<JOB_ID>",
  "cluster": "<CLUSTER>",
  "experiment": "<EXPERIMENT_PATH>",
  "restart_count": 0
}
```

## Loop

```
1. SLEEP
   - if just submitted/restarted: sleep 120 once
   - otherwise: sleep 570

2. CHECK
   uv run scripts/ray/cluster.py --cluster <CLUSTER> job-logs -n 200 -g "loss\|error\|Error\|Traceback\|RESOURCE_EXHAUSTED\|compiler_base.cc:2587\|Program hbm requirement\|Largest program allocations" <JOB_ID>

3. CHECK TERMINAL STATUS (RAY)
   - Determine job completion by Ray terminal status (`SUCCEEDED`/`FAILED`/`STOPPED`), not by train-step counts.

4. PRINT W&B RUN IDS/LINKS
   - Inspect recent logs for W&B run URLs / IDs (for example lines containing `wandb: 🚀 View run` or `/runs/<RUN_ID>`).
   - Print the run id(s) and full W&B run link(s) once per training run.
   - If a restart creates a new training run, print the new run id(s)/link(s) once.

5. REPORT PROGRESS
   - When progress is visible, report as `~<current>/<exact_max>`.
   - Example: `~4600/5155`.
   - Never present rounded values (for example `5.16k`) as exact max.
   - If max is unknown, report `~<current>/unknown`.

6. EVALUATE
   - If a user-specified stopping point has been reached → summarize status and exit loop.
   - If the job is terminal and successful (`SUCCEEDED`) → report completion and exit loop.
   - If the job is terminal and not successful (`FAILED`/`STOPPED`) → go to step 7 unless user says otherwise.
   - If output contains "error" or "Error" or "Traceback" → go to step 7
   - Treat TPU/XLA HBM reports as failures even when "OOM" is not present. In particular, if logs include:
     - `Program hbm requirement ...`
     - `Largest program allocations in hbm`
     then treat this as an OOM/resource-exhaustion event and go to step 7.
   - If progress does not advance for multiple 570-second intervals and logs show signals such as:
     - `OwnerDiedError`
     - dead node / node death
     - autoscaler unsatisfied resources
     then flag the run as `degraded` and notify the user immediately.
   - If output contains "loss" lines → go to step 1
   - If no output (job dead) → go to step 7

7. RESTART
   - This step restarts only the Ray job. Never restart/recreate/mutate the cluster here without explicit human consent in this thread, except for TPU bad-node recovery below.
   uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster <CLUSTER> -- python <EXPERIMENT_PATH>

   - Capture new job_id from output
   - Update state file: job_id = <NEW_JOB_ID>, restart_count += 1
   - Go to step 1
```

## Cluster Mutation Guardrail

- Default rule: never restart, recreate, or otherwise mutate the cluster without explicit human consent in this thread.

## Fixing Small Bugs

When step 6 (EVALUATE) detects an error, before restarting:

1. **Analyze the error** in the logs:
   - Look for `Traceback`, `Error`, `Exception`
   - Identify the file and line number

2. **If it's a small fix** (typo, missing import, wrong variable name):
   - Read the relevant file
   - Make the fix with Edit tool
   - Proceed to step 7 (RESTART)

3. **If it's a complex issue** (architectural, unclear cause, requires investigation):
   - Do NOT attempt to fix automatically
   - Report to user and exit the loop

Examples of small fixes:
- `NameError: name 'foo' is not defined` → typo in variable name
- `ImportError: cannot import 'bar'` → missing or misspelled import
- `SyntaxError` → missing comma, bracket, colon
- `KeyError` → wrong dict key name (if obvious from context)

Examples of complex issues (do not auto-fix):
- OOM errors
- TPU/XLA HBM exhaustion signatures such as `Program hbm requirement` and `Largest program allocations in hbm`
- Distributed training failures
- Data loading issues
- Unclear stack traces spanning multiple files

## TPU Bad-Node Recovery

Special note: for TPU-related errors such as:
- `RuntimeError: No accelerator found. Please run on a TPU or GPU.`
- `Failed to cleanup driver after error: INTERNAL: FAILED_PRECONDITION`
- `Device or resource busy`

Treat this as a bad TPU node:
1. Identify the bad TPU worker IP from logs.
2. Find the TPU VM name for that IP.
3. Delete the bad TPU VM and let the cluster replace it.
   - `gcloud compute tpus tpu-vm delete <TPU_VM_NAME> --zone <ZONE> --quiet`
4. Restart the job.

## Notes

- Sleep must be foreground (max ~10 min due to tool timeout)
- The loop is controlled at the agent level, not bash
- Track restart_count to detect flapping jobs
- State file allows resuming if context resets
- The loop is expected to run for the full job duration (often 4-5 hours for ferries, but potentially days for longer jobs)
- If the same error occurs after a fix attempt, do not retry - report to user
- Never restart/recreate/mutate the cluster (including restarting the head node) without explicit human consent in this thread, except for the TPU bad-node recovery flow above

## Optional Todo Pattern

If the agent supports a todo list, this pattern works well:

- [ ] sleep 570
- [ ] check cluster logs with `cluster.py`
- [ ] print relevant logs/status (including progress and W&B links)
- [ ] evaluate state and take actions
- [ ] if not done, recreate this todo list and keep going
