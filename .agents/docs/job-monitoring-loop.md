# Job Monitoring Loop

Monitor a Ray job, automatically restarting on failure.

## Before Starting

When the user asks you to start a monitoring loop, gather the required information first:

1. **job_id** - What is the Ray job ID? (e.g., `ray-run-held-isoflop_sweep-20260131-051716`)
2. **cluster** - Which cluster is it running on? (e.g., `us-east5-a`, `us-central2`)
3. **experiment** - What is the experiment script path? (e.g., `experiments/isoflop_sweep.py`)

Ask the user for any missing information before proceeding. Example:

> "I need a few details to start the monitoring loop:
>
> - Job ID?
> - Cluster?
> - Experiment script path?"

Once you have all three, write the state file and begin the loop.

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
   sleep 120 (once we know job did not die immediately upon submission, sleep 570)

2. CHECK
   uv run scripts/ray/cluster.py --cluster <CLUSTER> job-logs -n 50 -g "loss\|error" <JOB_ID>

3. EVALUATE
   - If output contains "error" or "Error" or "Traceback" → go to step 4
   - If output contains "loss" lines → go to step 1
   - If no output (job dead) → go to step 4

4. RESTART
   uv run lib/marin/src/marin/run/ray_run.py --no_wait --cluster <CLUSTER> -- python <EXPERIMENT_PATH>

   - Capture new job_id from output
   - Update state file: job_id = <NEW_JOB_ID>, restart_count += 1
   - Go to step 1
```

## Fixing Small Bugs

When step 3 (EVALUATE) detects an error, before restarting:

1. **Analyze the error** in the logs:
   - Look for `Traceback`, `Error`, `Exception`
   - Identify the file and line number

2. **If it's a small fix** (typo, missing import, wrong variable name):
   - Read the relevant file
   - Make the fix with Edit tool
   - Proceed to step 4 (RESTART)

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
- Distributed training failures
- Data loading issues
- Unclear stack traces spanning multiple files

Special note: for TPU related errors such as "RuntimeError: No accelerator found. Please run on a TPU or GPU.", "Failed to cleanup driver after error: INTERNAL: FAILED_PRECONDITION", "Device or resource busy", etc., this points to a bad TPU node. You should identify it by its IP addressa and kill it with `gcloud compute tpus tpu-vm delete`, then restart.

## Notes

- Sleep must be foreground (max ~10 min due to tool timeout)
- The loop is controlled at the agent level, not bash
- Track restart_count to detect flapping jobs
- State file allows resuming if context resets
- If the same error occurs after a fix attempt, do not retry - report to user
