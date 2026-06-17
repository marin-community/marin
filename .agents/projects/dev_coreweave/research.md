# dev_coreweave: research notes

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

1. **Name/scope → `dev_coreweave.py`, k8s-specific.** Mirrors `dev_tpu.py`'s
   one-tool-one-backend honesty. Considered and rejected: `dev_gpu.py` with
   backend dispatch (GCP-VM + k8s) — the GCP-GPU path is low-value since Marin's
   H100s aren't on GCP, and the abstraction isn't earned yet.
2. **MVP scope → lean: `allocate`/`connect`/`status`/`release`.** Proves the core
   kubectl path end to end with minimal code. Deferred: file sync (`kubectl cp` /
   tar-over-exec), `execute`, `watch`, `setup_env` — added once the bones work.
3. **Transport → Iris for scheduling, `kubectl` for access.** Faithful to how
   `dev_tpu.py` uses Iris only to submit/hold/resolve, then shells out to gcloud.
