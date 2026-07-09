---
name: reserve-gpu
description: Reserve an Iris-backed CoreWeave H100 pod for fast debugging with dev_gpu.py.
---

# Skill: Dev GPU

Use this skill for the standard fast H100 debugging loop without wiring a full training job each time. It is the GPU counterpart to `reserve-tpu`.

`scripts/iris/dev_gpu.py` reserves a CoreWeave H100 pod through Iris, waits for the backing Kubernetes pod to come up, and `kubectl exec -it`s you into it. Marin's H100s are CoreWeave Kubernetes pods, not GCE VMs, so access is `kubectl`, not SSH — there is no `ssh`/`scp` transport and no `~/.ssh/config` alias.

This is a lean tool: `allocate`, `connect`, `status`, `release`. It does not sync files or run remote env setup (no `execute`/`watch`/`setup_env`). The CoreWeave task image is self-contained; the loop is "reserve a node, shell in." Sync those steps in yourself once connected.

## Cost rule

A holder pod sits on an expensive 8×H100 node for the session's lifetime. Release as soon as you are done — `Ctrl-C` the `allocate` terminal, or run `release` from another shell.

## Commands

- `allocate`: submit a holder job, resolve the assigned pod, persist session state, block until release
- `status`: show the active local session metadata
- `connect`: open an interactive shell (`kubectl exec -it … -- bash -l`) into the reserved pod
- `release`: terminate the holder job and remove the local session file

## Prerequisites

1. Place the cluster kubeconfig at the path the config expects. The tool passes `--kubeconfig <platform.coreweave.kubeconfig_path>` and `--context <platform.coreweave.kube_context>` to `kubectl` verbatim and fails fast if the file is absent. All CoreWeave clusters share the kubeconfig `~/.kube/coreweave-iris`; the per-cluster context (e.g. `marin-gpu_US-EAST-02A` for `cw-us-east-02a`) is pinned in the cluster yaml, per `lib/iris/docs/coreweave.md`.

2. Ensure the Iris controller is running for the cluster. On the shared CoreWeave cluster this is usually already true; only start it yourself for a fresh cluster.

3. Use a cluster config whose platform is CoreWeave/Kubernetes. The tool gates on this and rejects GCP/TPU configs with a pointer back to `dev_tpu.py`.

## Command pattern

All invocations share this shape; only the subcommand and its flags change:

```bash
uv run scripts/iris/dev_gpu.py \
  --config lib/iris/config/cw-us-east-02a.yaml \
  --name "$USER-gpu" \
  <subcommand> [flags]
```

Subcommands and distinctive flags:

- `allocate` — reserves a whole `h100-8x` node (`--gpu-count` defaults to `8`) and holds it until `Ctrl-C`. Add `--timeout` (default `900`) to bound the wait for the task to reach `RUNNING`, and `--pod-timeout` (default `120`) to bound the wait for the backing pod. Only `--gpu-count 8` is validated; a sub-node value schedules as a fractional share (`nvidia-smi -L` then shows fewer GPUs) but fragments the 8-GPU InfiniBand gang pool, so prefer the whole node.
- `status` — show the active session (job id, config, GPU count, resolved pod).
- `connect` — interactive shell into the pod. It first checks job liveness with the controller (failing fast if the job is gone), then `kubectl exec -it`s into container `task`.
- `release` — terminate the holder job and clear the session file. Pass `--force` to drop local state even when the terminate call fails (then confirm the job is gone with `iris job list`).

## GPU JAX inside the pod

The `iris-task` image ships a CPU-only `uv` environment at `/app`, so bare `python` has no JAX and `uv run python` falls back to a CPU device. To get GPU JAX (`jax[cuda13]`):

```bash
cd /app && uv sync --all-packages --extra=gpu
```

`--all-packages` is required: the `gpu` extra is defined on the sub-packages (`marin-levanter` / `marin-core`), not the root project. This is the GPU analog of `dev_tpu.py`'s `--extra=tpu`. Verify the hardware with `nvidia-smi -L` (expect 8×H100 80GB on a whole node).

## Observability

Use normal Iris tooling to inspect the backing cluster and holder job:

```bash
uv run iris --config=lib/iris/config/cw-us-east-02a.yaml job list --prefix /$USER/dev-gpu
uv run iris --config=lib/iris/config/cw-us-east-02a.yaml job logs /$USER/dev-gpu-<name>
```

Inspect the pod directly with the same kubeconfig + context the tool uses:

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris --context marin-gpu_US-EAST-02A \
  --namespace iris get pods -l iris.task_id=<sanitized-task-id>
```

## Session behavior

- Local session state lives under `~/.cache/marin/dev_gpu_iris/`.
- If the `allocate` terminal dies unexpectedly, run `release` to terminate the holder job and clear the stale state file.
- A failed `allocate` cleans up after itself: the holder job is terminated and the local state file is removed only once the job is confirmed gone, so a failed terminate never orphans an expensive pod with no local record of its job id.
- `connect` execs into the pod resolved at allocation time. If Iris rescheduled the task onto a new pod while the job stayed active, `connect` fails — re-allocate.

## Agent Usage

Always pass `--name` to avoid collisions with other agents:

```bash
export GPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD | tr '/' '-')"
uv run scripts/iris/dev_gpu.py --config lib/iris/config/cw-us-east-02a.yaml --name "$GPU_NAME" allocate
```

## Cleanup

Normal cleanup is `Ctrl-C` in the `allocate` terminal. To clean up from another shell, run the `release` subcommand (add `--force` only if the job is already dead and `release` keeps erroring).
