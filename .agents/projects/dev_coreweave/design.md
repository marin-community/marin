# dev_coreweave: ad-hoc CoreWeave H100 dev pods

_Why are we doing this? What's the benefit?_

`scripts/iris/dev_tpu.py` (the `reserve-tpu` skill) lets a developer reserve a
TPU worker through Iris and `ssh` in for ad-hoc dev work without wiring up a full
training job. There is no GPU equivalent, even though GPU workloads now run on
CoreWeave H100s. This adds `scripts/iris/dev_coreweave.py`: reserve a
CoreWeave H100 pod through Iris and `kubectl exec -it` into it. It unblocks the
same fast edit/debug loop on the hardware where the GPU work actually runs.

## Challenges

_What's hard?_

Marin's H100s are **CoreWeave Kubernetes pods**, not GCE VMs. `dev_tpu.py` is
GCP/TPU-coupled end to end — a hard `_require_gcp_platform` gate
(`scripts/iris/dev_tpu.py:299`), `gcloud compute (tpus tpu-vm) ssh/scp` transport,
and GCP IP-based node resolution — so none of its transport carries over. A pod
is not an SSH-able host. The one cross-backend exec primitive Iris already has,
`iris task exec` → controller `exec_in_container`
(`lib/iris/src/iris/cli/task.py:28`, `.../backends/k8s/tasks.py:1493`), is
**one-shot with no TTY**, so it cannot back an interactive shell. Interactive
access has to shell out to `kubectl exec -it` directly.

## Costs / Risks

- New developer-facing tool to maintain alongside `dev_tpu.py`, with deliberate
  duplication of the holder/session/wait scaffolding (no shared module yet).
- `kubectl exec -it` against an Iris-managed task pod is unverifiable without a
  live H100 node; the prototype carries that as a manual first check.
- A holder pod sits on an expensive 8×H100 node for the session's lifetime — same
  cost profile as `dev_tpu.py`, but the unit is pricier.

## Design

_How are we doing this?_

Same split as `dev_tpu.py`: **Iris schedules/holds/resolves the allocation;
`kubectl` provides interactive access.** Lean MVP — `allocate`, `connect`,
`status`, `release`. No file sync: the CoreWeave task image
(`ghcr.io/marin-community/iris-task:latest`, workingDir `/app`, per
`lib/iris/config/coreweave.yaml`) is self-contained, so the loop is "reserve a
node, shell in."

- **allocate** — submit a holder job whose entrypoint is the existing
  `HOLDER_COMMAND` sleep (`scripts/iris/dev_tpu.py:197`), with
  `ResourceSpec(device=gpu_device("H100", count))`
  (`lib/iris/src/iris/cluster/types.py:508`). `--gpu-count` defaults to `8` (a
  whole `h100-8x` node; the pool is bare-metal 8×H100). Block until the task is
  `RUNNING`, resolve the pod, persist session state, hold until Ctrl-C.
- **pod resolution** — find the backing pod in namespace `iris`
  (`kubernetes_provider.namespace`) by label selector on the sanitized task-id
  (`_LABEL_TASK_ID`, `.../backends/k8s/tasks.py`), container `task`. Fallback:
  the deterministic `_pod_name(task_id, attempt_id)` scheme
  (`.../backends/k8s/tasks.py:241`).
- **connect** —
  `kubectl --kubeconfig <path> exec -it -n iris <pod> -c task -- bash -l`. The
  kubeconfig comes from `platform.coreweave.kubeconfig_path`
  (e.g. `~/.kube/coreweave-iris`); operators already hold it since
  `iris cluster start/status` runs from the laptop.
- **status / release** — print the session JSON; `client.terminate(job)` + delete
  the session file (Iris GC reaps the pod).

The lifted scaffolding (click group, `Context`, session dataclasses +
`load`/`save`, `controller_client`, `HOLDER_COMMAND`, wait-for-running loop) comes
straight from `dev_tpu.py`; only transport (gcloud→kubectl) and node resolution
(GCP IP→pod label) change. A future refactor could factor the shared bones into a
common module once a second consumer exists — deferred (YAGNI).

## Testing

_Agents make mistakes — how do we catch them?_

Behavior-focused unit tests on the pure pieces, kubectl/Iris boundaries faked (no
live cluster, per `TESTING.md`): pod label-selector construction, `connect`
kubectl arg-building, session-state JSON round-trip, and the platform gate
(accepts CoreWeave config, rejects GCP/TPU with a pointer to `dev_tpu.py`).
End-to-end against a real H100 pod — does `kubectl exec -it` give a working
interactive shell? — is a **manual** validation step, the first thing to run on
the prototype.

## Open Questions

- **Pod resolution under retry:** label selector vs. deterministic `_pod_name` —
  which survives an attempt/retry bump cleanly? Prefer the label selector (no need
  to reconstruct `attempt_id`); confirm during implementation.
- **Label-selector truncation collision (known, low risk):** the `iris.task_id`
  label value is sanitized and truncated to 63 chars, so two very long, same-prefix
  task ids could share a selector and `parse_running_pod` could pick the wrong pod.
  Unreachable for this tool's short `dev-cw-<user>` names; if it ever matters, also
  match `iris.task_hash` (the k8s backend's collision-resistant label) rather than
  re-deriving the hash here.
- **Sub-node GPU requests (resolved):** `--gpu-count 1` schedules as a fractional
  share — the pod gets exactly N GPUs (`nvidia-smi -L` showed 1, not 8), routed to
  `h100-8x` by device type+variant. Works, but a sub-node squatter fragments the
  8-GPU gang pool, so the default stays 8. See research.md "Live validation".
- **Shared module:** is this prototype the right moment to extract the
  `dev_tpu.py`/`dev_coreweave.py` common core, or wait until the kubectl path is
  proven?
