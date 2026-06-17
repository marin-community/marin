# dev_gpu: concrete contracts

## CLI surface

```
uv run scripts/iris/dev_gpu.py --config <cfg> --name <session> <subcommand>
```

- `allocate --gpu-count 8 [--timeout 900]` — submit + hold a GPU holder pod;
  blocks until Ctrl-C.
- `connect` — `kubectl exec -it` an interactive shell into the reserved pod.
- `status` — print the active session metadata.
- `release` — terminate the holder job and clear the session file.

`--name` defaults to `$USER`; pass it explicitly to avoid collisions between
concurrent sessions (mirrors `dev_tpu.py`'s `--tpu-name`).

## Persisted session state

JSON at `~/.cache/marin/dev_gpu_iris/<session>.json`:

```json
{
  "session_name": "matt",
  "config_file": "/abs/path/to/coreweave.yaml",
  "job_id": "/matt/dev-gpu-matt",
  "gpu_count": 8,
  "kubeconfig_path": "/Users/matt/.kube/coreweave-iris",
  "pod": {
    "namespace": "iris",
    "pod_name": "<resolved>",
    "container": "task"
  }
}
```

## Iris contract (reused, unchanged)

- Submit: `IrisClient.submit(entrypoint=Entrypoint.from_command("python", "-c",
  HOLDER_COMMAND), name=f"dev-gpu-{session}", resources=ResourceSpec(cpu=…,
  memory=…, device=gpu_device("H100", gpu_count)))`.
- Wait: poll `job.tasks()[*].status()` until `TASK_STATE_RUNNING`; fail fast on
  `JOB_STATE_FAILED/KILLED/UNSCHEDULABLE/WORKER_FAILED`.
- Release: `IrisClient.terminate(JobName.from_wire(job_id))`.

## kubectl contract (new)

- Resolve pod:
  `kubectl --kubeconfig <path> -n <ns> get pods -l <task-id-label>=<sanitized>
  -o jsonpath=… ` → first `Running` pod.
- Connect:
  `kubectl --kubeconfig <path> exec -it -n <ns> <pod> -c task -- bash -l`.

## Platform gate

Accept only CoreWeave/Kubernetes-backed clusters
(`config.platform.HasField("coreweave")` and/or a `kubernetes_provider` block).
On GCP/TPU clusters, raise with a message pointing to `dev_tpu.py`.
