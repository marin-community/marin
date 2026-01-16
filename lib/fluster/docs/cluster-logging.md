# Cluster Logging Improvements

## Problem

When Docker jobs fail, it's difficult to see what happened inside the container. The current workflow requires manual `docker logs <container_id>` commands after the fact, and containers are often cleaned up before we can inspect them.

## Proposed Solution

### 1. Add `--verbose` flag to demo_cluster.py

Stream job logs in real-time during test execution:

```python
# In demo_cluster.py
@click.option("--verbose", is_flag=True, help="Stream job logs to stdout")
def main(docker, no_browser, validate_only, test_notebook, verbose):
    ...
    if verbose:
        demo.enable_log_streaming()
```

### 2. Capture logs on job failure in DemoCluster

When a job fails, automatically fetch and display its logs:

```python
# In DemoCluster.wait_for_job() or similar
def wait_for_job(self, job_id: str, timeout: float = 30.0) -> cluster_pb2.JobStatus:
    status = self._client.wait(job_id, timeout=timeout)

    if status.state == cluster_pb2.JOB_STATE_FAILED:
        # Fetch and display logs on failure
        logs = self._fetch_job_logs(job_id)
        print(f"\n=== Job {job_id} FAILED ===")
        for line in logs:
            print(f"  [{line.source}] {line.data}")
        print("=" * 40)

    return status
```

### 3. Add log capture to notebook test runner

In `run_notebook()`, capture job logs when execution fails:

```python
def run_notebook(self) -> bool:
    try:
        ep.preprocess(nb)
        return True
    except Exception as e:
        print(f"Notebook execution failed: {e}")

        # Fetch logs for all running/failed jobs
        self._dump_all_job_logs()
        return False

def _dump_all_job_logs(self):
    """Dump logs for all jobs to help debug failures."""
    jobs = self._list_jobs()
    for job in jobs:
        if job.state in (JOB_STATE_RUNNING, JOB_STATE_FAILED):
            print(f"\n=== Logs for {job.job_id} ({JobState.Name(job.state)}) ===")
            logs = self._fetch_job_logs(job.job_id)
            for line in logs[-50:]:  # Last 50 lines
                print(f"  {line.data}")
```

### 4. Keep containers on failure (don't auto-cleanup)

Modify worker cleanup to preserve failed containers:

```python
# In Worker._execute_job()
if job.status == cluster_pb2.JOB_STATE_FAILED:
    logger.warning(f"Preserving failed container {job.container_id} for debugging")
    # Don't call self._runtime.remove(container_id)
```

### 5. Add `fluster logs <job_id>` CLI command

For manual debugging:

```bash
# Fetch logs for a specific job
uv run python -m fluster.cli logs <job_id>

# Tail logs in real-time
uv run python -m fluster.cli logs <job_id> --follow
```

## Implementation Priority

1. **High**: Add log dump on notebook test failure (immediate debugging need)
2. **Medium**: Add `--verbose` flag to demo_cluster.py
3. **Low**: CLI command for manual log inspection

## Quick Fix for Current Issue

Add this to `demo_cluster.py` `run_notebook()` method:

```python
def run_notebook(self) -> bool:
    # ... existing code ...
    except Exception as e:
        print(f"Notebook execution failed: {e}")

        # Dump Docker container logs for debugging
        import subprocess
        result = subprocess.run(
            ["docker", "ps", "-a", "--format", "{{.ID}}\t{{.Names}}\t{{.Status}}"],
            capture_output=True, text=True
        )
        print("\n=== Docker Containers ===")
        print(result.stdout)

        # Get logs from recent containers
        for line in result.stdout.strip().split("\n")[:5]:
            if line:
                cid = line.split("\t")[0]
                logs = subprocess.run(
                    ["docker", "logs", "--tail", "50", cid],
                    capture_output=True, text=True
                )
                print(f"\n=== Container {cid} ===")
                print(logs.stdout or logs.stderr or "(no output)")

        return False
```
