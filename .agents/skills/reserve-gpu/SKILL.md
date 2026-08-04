---
name: reserve-gpu
description: Reserve Iris-backed CoreWeave H100 or GB200 nodes with dev_gpu.py. Use for interactive GPU debugging, multi-node tests, or reconnecting to a dev GPU session.
---

# Dev GPU

Use a dev GPU for short interactive tests. Move to a full Iris job after the
code works on the reserved node.

`scripts/iris/dev_gpu.py` submits a holder job through Iris and uses
`kubectl exec` internally to open a shell. It does not sync files or provide
SSH access.

## Choose a cluster

- H100: `cw-rno2a` or `cw-us-east-02a`
- GB200: `cw-us-east-08a`

The cluster YAML files under `lib/iris/config/` are canonical. If an allocation
is slow, or before requesting several nodes, inspect the live backend and queue:

```bash
uv run iris --cluster=<cluster> rpc controller list-backends
uv run iris --cluster=<cluster> rpc controller get-scheduler-state
```

## Reserve and connect

Use a unique session name:

```bash
export GPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD | tr '/' '-')"
export GPU_CONFIG=lib/iris/config/cw-us-east-02a.yaml

uv run scripts/iris/dev_gpu.py --config "$GPU_CONFIG" --name "$GPU_NAME" allocate
uv run scripts/iris/dev_gpu.py --name "$GPU_NAME" connect
uv run scripts/iris/dev_gpu.py --name "$GPU_NAME" status
uv run scripts/iris/dev_gpu.py --name "$GPU_NAME" release
```

`allocate` blocks until it is interrupted. An agent running it in the
background must call `release` from another shell. Session state is stored under
`~/.cache/marin/dev_gpu_iris/`.

The default is one whole H100 node. Pass `--gpu-variant GB200` for one GB200
tray. Use `--nodes N` only when the test needs distributed topology; connect to
each pod with `connect --node N`. Multi-node sessions require whole nodes.

Run `uv run scripts/iris/dev_gpu.py <subcommand> --help` for current options
and defaults.

## Sharp edges

- A session holds an expensive node until release. Confirm the holder job is
  gone if cleanup reports an error.
- `connect` uses the pod names saved at allocation. If Iris reschedules a task,
  release and reallocate the session.
- Kubernetes access uses `~/.kube/coreweave-iris`; each cluster YAML selects its
  context. Do not start or restart a controller to fix access without explicit
  user approval.
- The task image may need GPU JAX installed with
  `cd /app && uv sync --all-packages --extra=gpu`. This has been exercised on
  H100; verify it on GB200.
- Use the pod's `MARIN_PREFIX` for durable data and
  `rigging.filesystem.marin_temp_bucket(...)` for disposable data. Do not read
  from GCS on CoreWeave without explicit user approval because it can incur
  egress charges.
- Kubernetes access and object-storage access use different credentials. Task
  pods receive storage credentials from `iris-task-env`; `dev_gpu.py` does not
  copy the checkout's `.marin.yaml`. Never print secret values or dump the pod
  environment.

For cluster access, storage credentials, or stuck pods, read
`lib/iris/OPS.md` and `lib/iris/docs/coreweave.md`.
