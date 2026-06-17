# dev_gpu: research notes

## Prior art

`scripts/iris/dev_tpu.py` + the `reserve-tpu` skill are the direct model. It
submits a holder job (sleep-forever entrypoint, `:197`) to reserve a TPU worker,
resolves the GCP node, and exposes `allocate`/`connect`/`execute`/`watch`/
`setup_env`/`status`/`release`. Transport is `gcloud compute (tpus tpu-vm)
ssh/scp`; node resolution is GCP IP/TPU-name lookups; a hard
`_require_gcp_platform` gate (`:299`) blocks non-GCP clusters. The design is sound
but single-backend by intent — we mirror that philosophy rather than generalize.

## In-repo references that shaped the design

- `lib/iris/src/iris/cluster/types.py:508` — `gpu_device(variant, count)` already
  exists alongside `tpu_device`; the GPU resource spec is free.
- `lib/iris/config/coreweave.yaml` — CoreWeave is k8s-backed (`platform.coreweave`,
  `kubernetes_provider`, namespace `iris`). Scale group `h100-8x`:
  `device_type: gpu, device_variant: H100, device_count: 8`,
  `instance_type: gd-8xh100ib-i128`.
- `lib/iris/src/iris/cluster/backends/k8s/tasks.py` — `_pod_name(task_id,
  attempt_id)` (`:241`), container named `task`, pods labeled with the sanitized
  task-id (`_LABEL_TASK_ID`); `exec_in_container` (`:1493`) does `kubectl exec`.
- `lib/iris/src/iris/cli/task.py:28` — `iris task exec` is the existing
  cross-backend exec, but **one-shot, no TTY** (captures stdout/stderr, has a
  timeout) → cannot back an interactive `connect`. Hence shelling out to
  `kubectl exec -it` directly.

## Decisions from brainstorming

1. **Name/scope → `dev_gpu.py`, k8s-specific.** Mirrors `dev_tpu.py`'s
   one-tool-one-backend honesty. Considered and rejected: `dev_gpu.py` with
   backend dispatch (GCP-VM + k8s) — the GCP-GPU path is low-value since Marin's
   H100s aren't on GCP, and the abstraction isn't earned yet.
2. **MVP scope → lean: `allocate`/`connect`/`status`/`release`.** Proves the core
   kubectl path end to end with minimal code. Deferred: file sync (`kubectl cp` /
   tar-over-exec), `execute`, `watch`, `setup_env` — added once the bones work.
3. **Transport → Iris for scheduling, `kubectl` for access.** Faithful to how
   `dev_tpu.py` uses Iris only to submit/hold/resolve, then shells out to gcloud.

## Live validation (2026-06-17, cw-us-east-02a / marin-gpu)

First real run on the production H100 fleet. Resolves the design's main open question.

- **`allocate` + `connect` work end to end.** `kubectl exec -it` against the Iris
  task pod gives a real interactive TTY shell — the key unknown. `nvidia-smi -L`
  inside the pod showed all **8× H100 80GB**. `--gpu-count 8` scheduled cleanly.
- **Failed-allocate cleanup verified.** An earlier run failed at pod resolution
  (see kubeconfig note below); the `allocate` cleanup `finally` terminated the
  holder job (`job list` showed `killed — Terminated by user`), so no H100 node
  leaked. The state-leak fix and `--force` path are sound in practice.
- **Kubeconfig must exist at the configured path.** The tool passes
  `--kubeconfig <platform.coreweave.kubeconfig_path>` verbatim; if that file is
  absent kubectl errors and pod resolution times out. We deliberately kept this
  strict (fail-fast on the documented setup) rather than adding a fallback to
  kubectl's default resolution — operators place creds at
  `~/.kube/coreweave-iris-gpu` per `lib/iris/docs/coreweave.md`.
- **Pod env is CPU-JAX by default (image territory, not a tool bug).** The
  `iris-task` image's `/app` uv venv is synced with the `cpu` extra; bare `python`
  has no JAX, and `uv run python` falls back to CpuDevice. GPU JAX (`jax[cuda13]`)
  needs `cd /app && uv sync --all-packages --extra=gpu` in the pod — `--all-packages`
  is required because the `gpu` extra is defined on the sub-packages
  (marin-levanter / marin-core), not the root project. This is exactly what a
  future `setup_env` step should automate — the GPU analog of `dev_tpu.py`'s
  `uv sync --all-packages --extra=tpu` is `--extra=gpu`.
- **Sub-node `--gpu-count` confirmed (fractional share).** `--gpu-count 1`
  schedules: it routes to the `h100-8x` group (matched by device type+variant,
  not count — `lib/iris/AGENTS.md:146`), and the pod requests
  `nvidia.com/gpu: 1` (+ `rdma/ib: 1` under host_network). `nvidia-smi -L` inside
  the pod showed **1 H100, not 8** — a fractional share of an 8-GPU node, not a
  whole box. Caveat: a 1-GPU squatter prevents that node from satisfying a full
  8-GPU InfiniBand gang, so sub-node requests fragment the pool — which is why
  the tool defaults to 8.
