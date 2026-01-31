# Job Monitoring Loop

Monitor a Ray job, automatically restarting on failure.

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
   sleep 570

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

## Notes

- Sleep must be foreground (max ~10 min due to tool timeout)
- The loop is controlled at the agent level, not bash
- Track restart_count to detect flapping jobs
- State file allows resuming if context resets
