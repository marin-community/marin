# Monitor an Iris job

Use `babysit-zephyr` for a Zephyr pipeline.

Before monitoring, require:

- canonical `/<user>/<job>` ID;
- cluster or config path;
- exact resubmit command with `--no-wait` and resource flags;
- current-thread authorization before any stop or resubmit.

Keep one owner and one `scratch/<timestamp>_monitoring_state.json` with the job ID, config, resubmit command, latest signal, and restart count. After submit or recovery, wait 120 seconds once; then check every 570 seconds.

```bash
uv run iris --config <config> job logs --since-seconds 900 <job>
uv run iris --config <config> job list --prefix <job>
uv run iris --config <config> job describe <job>
```

Confirm progress from live logs and, when present, W&B timestamps/steps and checkpoint movement. A controller `RUNNING` state, first loss, evaluation, or W&B link is not completion. A scheduler capacity wait is not a failure and does not authorize cluster changes.

For an authorized recovery:

```bash
uv run iris --config <config> job cancel --exact <job>
<exact resubmit command>
```

Fix only a small obvious `NameError`, `ImportError`, `SyntaxError`, or `KeyError` with a clear file and line. Stop for OOM/HBM, distributed, data-loading, repeated, dead-node, or unclear failures. Never restart or recreate the cluster as job recovery.

Finish only after Iris reports success and expected W&B state, output, and final checkpoint metadata are present. Monitoring ownership ends at verified completion plus acknowledged next action, a requested stop/handoff, or an unrecoverable error.
