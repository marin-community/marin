---
name: reserve-gpu
description: Reserve one or more Iris-backed CoreWeave GPU nodes (H100 or GB200) for fast debugging with dev_gpu.py.
---

# Skill: Dev GPU

Use this skill for the standard fast GPU debugging loop without wiring a full training job each time. It is the GPU counterpart to `reserve-tpu`.

`scripts/iris/dev_gpu.py` reserves CoreWeave GPU nodes through Iris, waits for the backing Kubernetes pods to come up, and `kubectl exec -it`s you into one. A whole node is an 8-GPU `h100-8x` box or a 4-GPU `gb200-4x` NVL72 tray, chosen with `--gpu-variant`. Marin's GPUs are CoreWeave Kubernetes pods, not GCE VMs, so access is `kubectl`, not SSH — there is no `ssh`/`scp` transport and no `~/.ssh/config` alias.

This is a lean tool: `allocate`, `connect`, `status`, `release`. It does not sync files or run remote env setup (no `execute`/`watch`/`setup_env`). The CoreWeave task image is self-contained; the loop is "reserve a node, shell in." Sync those steps in yourself once connected.

## Cost rule

A holder pod sits on an expensive whole node — 8×H100 or 4×GB200 — for the session's lifetime, and a `--nodes N` session holds N of them. Release as soon as you are done — `Ctrl-C` the `allocate` terminal, or run `release` from another shell.

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

- `allocate` — reserves a whole `h100-8x` node (`--gpus-per-node` defaults to `8`) and holds it until `Ctrl-C`. Add `--timeout` (default `900`) to bound the wait for the tasks to reach `RUNNING`, and `--pod-timeout` (default `120`) to bound the wait for the backing pods. Only `--gpus-per-node 8` is validated; a sub-node value schedules as a fractional share (`nvidia-smi -L` then shows fewer GPUs) but fragments the 8-GPU InfiniBand gang pool, so prefer the whole node. `--nodes N` reserves N whole nodes in one session; `--gpu-variant GB200` reserves `gb200-4x` trays instead, where a whole node is 4 GPUs.
- `status` — show the active session (job id, config, node count, GPUs per node, and every resolved pod).
- `connect` — interactive shell into a pod. It first checks job liveness with the controller (failing fast if the job is gone), then `kubectl exec -it`s into container `task`. On a multi-node session pass `--node N` (default `0`) to pick which pod, in `status` order.
- `release` — terminate the holder job and clear the session file. Pass `--force` to drop local state even when the terminate call fails (then confirm the job is gone with `iris job list`).

## Multi-node sessions

`--nodes N` submits one Iris job with N gang-scheduled tasks, so the whole session lands
at once or not at all. Each task is one pod on one whole node, and the pods are numbered
in task order: `connect --node 0`, `connect --node 1`, and so on.

The gang's placement follows the variant. GB200 (`gb200-4x`) gangs of up to 16 nodes bind
hard to a single `ds.coreweave.com/nvlink.domain`, so every node shares one rack's NVLink
fabric — the reason to reserve GB200 nodes together rather than one at a time. H100 and
other variants get soft `leafgroup` InfiniBand colocation instead. `allocate` prints the
level it used as `Coscheduling: …`, and you can confirm the placement from the node labels:

```bash
kubectl --kubeconfig ~/.kube/coreweave-iris --context marin-us-east-08a_US-EAST-08A \
  get nodes -o custom-columns=NAME:.metadata.name,DOMAIN:'.metadata.labels.ds\.coreweave\.com/nvlink\.domain'
```

Multi-node sessions must reserve whole nodes, so `--gpus-per-node` has to stay at the
variant's per-node count (GB200: 4, H100: 8). A fractional pod would let two nodes of the
session land on the same machine, and `allocate` rejects it before submitting.

## GPU JAX inside the pod

The `iris-task` image ships a CPU-only `uv` environment at `/app`, so bare `python` has no JAX and `uv run python` falls back to a CPU device. To get GPU JAX (`jax[cuda13]`):

```bash
cd /app && uv sync --all-packages --extra=gpu
```

`--all-packages` is required: the `gpu` extra is defined on the sub-packages (`marin-levanter` / `marin-core`), not the root project. This is the GPU analog of `dev_tpu.py`'s `--extra=tpu`.

This recipe is only exercised on H100. GB200 trays are aarch64 Grace hosts and pull a different CUDA wheel set, so treat the sync as untested there.

Either way, `nvidia-smi -L` confirms what the pod actually got: 8×H100 80GB, or 4×GB200 on a `gb200-4x` tray.

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
- A failed `allocate` attempts cleanup on its way out: it terminates the holder job and drops the local state file only if that terminate call was accepted. An accepted terminate is not proof the pod is gone — confirm with `iris job list` when it matters. If the call fails, the state file survives, so you always keep a local record of the job id and can retry with `release`.
- `connect` execs into the pods resolved at allocation time. If Iris rescheduled a task onto a new pod while the job stayed active, `connect` fails for that node — re-allocate.

## Agent Usage

Always pass `--name` to avoid collisions with other agents:

```bash
export GPU_NAME="${USER}-$(git rev-parse --abbrev-ref HEAD | tr '/' '-')"
uv run scripts/iris/dev_gpu.py --config lib/iris/config/cw-us-east-02a.yaml --name "$GPU_NAME" allocate
```

`allocate` blocks until Ctrl-C, so an agent has to launch it detached — and Ctrl-C cannot reach it there. A background job started from a non-interactive shell inherits `SIGINT` as ignored, so `kill -INT` does nothing and Python never raises `KeyboardInterrupt`. Release with the `release` subcommand from a second shell instead; the held `allocate` notices the dead job within 30 s, cleans up, and exits.

Its stdout is block-buffered into a redirected log, so the session summary only appears once the process exits. Read `~/.cache/marin/dev_gpu_iris/<name>.json` for live session state.

## Cleanup

Normal cleanup is `Ctrl-C` in the `allocate` terminal, which only works when `allocate` is running in the foreground of a real terminal. To clean up from another shell, or from any agent-launched session, run the `release` subcommand (add `--force` only if the job is already dead and `release` keeps erroring).
