---
name: reserve-gpu
description: Reserve Iris-backed CoreWeave H100 or GB200 nodes with dev_gpu.py. Use for interactive GPU debugging, multi-node tests, or reconnecting to a dev GPU session.
---

# Dev GPU

Use a reserved node for short interactive tests; move to a full Iris job when
the code works. `dev_gpu.py` submits a holder job and uses `kubectl exec`; it
does not sync files or provide SSH.

H100 clusters are `cw-rno2a` and `cw-us-east-02a`; GB200 is `cw-us-east-08a`.
Cluster YAML under `lib/iris/config/` is canonical. Inspect backend and queue
before a slow or multi-node allocation:

```bash
uv run iris --cluster=<cluster> rpc controller list-backends
uv run iris --cluster=<cluster> rpc controller get-scheduler-state
```

```bash
export GPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD | tr '/' '-')"
export GPU_CONFIG=lib/iris/config/cw-us-east-02a.yaml
uv run scripts/iris/dev_gpu.py --config "$GPU_CONFIG" --name "$GPU_NAME" allocate
uv run scripts/iris/dev_gpu.py --name "$GPU_NAME" connect
uv run scripts/iris/dev_gpu.py --name "$GPU_NAME" status
uv run scripts/iris/dev_gpu.py --name "$GPU_NAME" release
```

`allocate` blocks until interrupted; release it from another shell if needed.
State is under `~/.cache/marin/dev_gpu_iris/`. The default holds one whole H100;
use `--gpu-variant GB200`, and `--nodes N` only for distributed whole-node tests
(`connect --node N`). If rescheduling changes pod names, release and reallocate.

The holder retains an expensive node until release; verify it is gone after
cleanup errors. Do not restart a controller to fix access. Use the pod's
`MARIN_PREFIX` for durable data and `rigging.filesystem.marin_temp_bucket(...)`
for disposable data. Do not read GCS from CoreWeave without explicit approval
(egress). Keep storage and Kubernetes credentials separate; never print secrets
or dump the pod environment. The task image may need GPU JAX setup with
`cd /app && uv sync --all-packages --extra=gpu`; verify this separately on GB200.
Read `lib/iris/OPS.md` and
`lib/iris/docs/coreweave.md` for access, storage, or stuck-pod issues.
