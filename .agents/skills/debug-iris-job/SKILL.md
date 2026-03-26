---
name: debug-iris-job
description: Debug a running Iris job using task exec and container inspection. Use when asked to debug, inspect, or troubleshoot a running Iris task.
---

# Skill: Debug Iris Job

Debug a running Iris job by executing commands inside its container. Prefer
existing logs, metrics, CLI commands and RPC calls for monitoring and diagnosis.
Use task exec as a last resort when those are insufficient.

For controller-level issues (stuck scheduling, resource leaks), use **debug-iris-controller**.
For TPU bad-node errors, use **debug-tpu**.

## Task Exec

Use `iris task exec` to run commands inside a running task's container.

### Basic usage

```bash
# Run a one-shot command
iris task exec <TASK_ID> -- ls /app

# Run with a custom timeout (default: 60s)
iris task exec <TASK_ID> --timeout 300 -- python -c "import torch; print(torch.cuda.device_count())"

# Run with no timeout - last resort for long-running commands
iris task exec <TASK_ID> --timeout -1 -- tail -f /var/log/app.log
```

### Running a background command that survives disconnect

The exec session is non-interactive and buffers output. To kick off a long-running
command that continues after the exec returns, use `nohup` + `&` inside a bash wrapper:

```bash
# Start a background process that writes to a file
iris task exec <TASK_ID> -- bash -c "nohup bash -c 'while true; do date >> /tmp/heartbeat.txt; sleep 10; done' > /dev/null 2>&1 &"

# Check on it later
iris task exec <TASK_ID> -- cat /tmp/heartbeat.txt

# Check if the process is still running
iris task exec <TASK_ID> -- pgrep -f heartbeat
```

### Task ID format

Task IDs follow the pattern `/<user>/<job>/<task_index>`, e.g., `/rav/my-job/0`.

### Common debugging commands

```bash
# Check GPU/TPU availability
iris task exec <TASK_ID> -- python -c "import jax; print(jax.devices())"
iris task exec <TASK_ID> -- nvidia-smi

# Check disk usage
iris task exec <TASK_ID> -- df -h

# Check memory
iris task exec <TASK_ID> -- free -h

# Check running processes
iris task exec <TASK_ID> -- ps aux

# Inspect environment
iris task exec <TASK_ID> -- env | sort
```
