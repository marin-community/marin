---
name: triage-tpu-worker-failure
description: Repeated TPU task failures on one worker — wedged container holding the iommu group vs a genuinely bad node.
---

# Runbook: Triage repeated TPU task failures (wedged container vs a genuinely bad node)

**When you're here:** Every attempt of a task (or every task that lands on one
worker) dies with `RuntimeError: No accelerator found. Please run on a TPU or
GPU.`, `Couldn't open iommu group`, `FAILED_PRECONDITION`, or `Device or
resource busy`. The error string looks identical to the "bad node" recovery
recipe in OPS.md — but that recipe deletes the node, and for the common case
that destroys a healthy slice without fixing anything.

**TL;DR:**
- These error strings have **two** causes that look identical: a *wedged
  iris-managed container* still holding the TPU's vfio/iommu group on an
  otherwise healthy node, or a *genuinely bad node*.
- Rule out the wedge **first**: SSH the worker VM, compare `docker ps
  --filter label=iris.managed=true` against the controller's task list for that
  worker. More containers than tasks ⇒ orphan ⇒ `sudo docker kill` it, slice
  recovers immediately, no node delete.
- Only fall through to deleting the node when the container count matches and the
  node is genuinely sick. Delete the **specific** node only.

## Before you touch anything

- **Deleting a node is the destructive step and the most common wrong move
  here.** Do not delete until you have confirmed the container count on the VM
  matches the controller's task list (i.e. there is no orphan to kill). A delete
  on a wedged-but-healthy node wastes a live slice and leaves the wedge to recur.
- **Only ever delete the one specific bad node.** If multiple nodes fail at the
  same time, or the same node fails again after a delete + resubmit, stop and
  escalate to the user — that is a fleet event (stockout, preemption storm, or a
  controller bug), not a single bad host. See OPS.md "TPU Bad-Node Recovery"
  (lib/iris/OPS.md:260) for the same guardrail.
- Baseline to capture before any action: the failing `worker_id` and its task
  `attempt_id`/error from the controller (`iris job bug-report /user/job`), and
  the zone. You will map the worker IP → VM name from these.

## Diagnose

Pull one failing attempt's `worker_id` and `error` (`iris job bug-report
/user/job`). The worker id is the VM name with a `-worker-N` suffix; strip the
suffix to get the VM name. Then walk the branches.

**1. Is the controller's reaper even running?** If tasks are stuck in ASSIGNED
with stale `last_heartbeat_ms` rather than actively looping new attempts, the
worker-failure reaper thread may be blocked on a hung `gcloud ... tpu-vm delete`
subprocess — see Known Bug #2 (lib/iris/OPS.md:224). `py-spy dump` the
controller, look for `subprocess.run` → `terminate` on the reaper, kill the
stuck gcloud process. That is a different runbook-less situation; if you see it,
clear it before continuing here.

**2. Wedged container vs bad node — the disambiguating check.** SSH the worker
VM with the impersonated controller service account and run `sudo docker ps
--filter label=iris.managed=true`. (SSH + docker-inspection command shapes:
`2026-04-21-iris-tpu-wedge-and-holder-poll.md` §"How OPS.md could have shortened
this", and OPS.md "Connecting", lib/iris/OPS.md:230. `sudo` is mandatory —
docker.sock is root-owned, else you get "permission denied … Docker daemon
socket".) Compare the container count to what the controller lists running on
that worker (its task list from the bug-report / worker-state RPC):

- **More `iris.managed=true` containers than the controller lists tasks** ⇒
  **wedged-orphan branch.** One container is `Up` that the controller already
  considers gone; it is still holding the iommu group, so every new task on the
  VM fails identically. Confirm by grepping the workerd log for the signature —
  `sudo docker logs iris-worker 2>&1 | grep "on_stop callback"`: many
  `on_stop callback for task-<id> did not complete` warnings = cleanup was
  skipped = orphan. Attribute it with `docker inspect <id> --format
  '{{.Config.Labels}}'`; if `iris.worker_id` matches this worker it was born on
  this process. Go to Resolve §A.

- **Container count matches the controller's task list, and the failures
  continue** ⇒ **bad-node branch.** No orphan to kill — the hardware/runtime is
  genuinely sick. Go to Resolve §B.

**3. Is it actually a fleet event, not this node?** Before deleting, sanity-check
that this is one host and not a stockout/preemption wave: see
`docs/dev-guide/tpu_observability.md` "Troubleshooting Commands" (rejected
demand / preemption / node-termination queries) for ground truth. If
terminations are spiking across zones, do not delete — escalate.

## Resolve

### A. Wedged orphan on a healthy node (least destructive — do this first)

`sudo docker kill <container-id>` the wedged iris-managed container(s) on the VM
(same SSH invocation as Diagnose §2). The wedge is a plain orphan — SIGKILL
reaches it — so killing it releases the iommu group and the worker starts
accepting tasks immediately. No node delete, no resubmit. Do **not** delete the
VM; the node is healthy.

### B. Genuinely bad node

Only after §2 confirmed the container count matches: map worker IP → VM name and
delete the **one** node, then resubmit the job. Command syntax lives in OPS.md
"TPU Bad-Node Recovery" (lib/iris/OPS.md:260-272) — `gcloud compute tpus tpu-vm
delete <NAME> --zone <ZONE> --quiet`. Keep the guardrail: one node only; if it
recurs or spreads, escalate.

## Verify

- **Wedge (§A):** re-run the `docker ps --filter label=iris.managed=true` count
  — it should now match the controller's task list. Watch the next attempt of
  the affected task land on that worker and reach RUNNING (not another
  WORKER_FAILED). The slice should resume without a node delete.
- **Bad node (§B):** confirm the resubmitted job's tasks schedule onto a
  different (or freshly created) node and clear the build/init phase. The deleted
  node should disappear from `gcloud compute tpus tpu-vm list`.
- For both: tail the controller's attempt history (`iris job bug-report`) and
  confirm the error string stops recurring on that worker_id — process "came
  back" is not enough; you need a clean RUNNING attempt.

## Why this happens

The trap is that a wedged container and a bad node emit the **same** error. On
2026-04-21 a v5p-preemptible worker looped `Couldn't open iommu group` /
`No accelerator found` while the controller listed only 2 tasks but `docker ps`
showed 3 iris-managed containers. Root cause: `ManagedThread._safe_target` never
fired `on_stop` on the natural-return path
(`lib/iris/src/iris/managed_thread.py:85-115`), so the task container's
`docker kill` + `docker rm` were skipped. A container whose PID 1 wedged in the
vfio teardown stayed `Up`, kept the iommu group, and poisoned every later task
on the host. The controller already considered that task gone — so to the
bad-node recipe the node "looks bad", but deleting it throws away a healthy slice
**and does not fix the wedge** (the next slice can wedge the same way). The fix
was the missing `docker kill`, which `sudo docker kill` issues by hand. Two
companion bugs in the same incident — ASSIGNED→WORKER_FAILED carrying no health
signal (`transitions.py:1878`) and the poll loop shipping virtual reservation
holders to real workers (`transitions.py:3116`) — let the loop run at full
scheduler velocity; both are fixed, but the manual `docker kill` remains the live
mitigation for an existing orphan.

See `.agents/ops/2026-04-21-iris-tpu-wedge-and-holder-poll.md` for the full
diagnosis path and the worker-VM inspection recipe.

## See also

- lib/iris/OPS.md "TPU Bad-Node Recovery" (lib/iris/OPS.md:260) — IP→VM mapping
  and the `tpu-vm delete` syntax for branch §B, plus the one-node guardrail.
- lib/iris/OPS.md "Connecting" / "GCP Resources" (lib/iris/OPS.md:230-258) —
  SSH-tunnel and VM-listing command reference.
- lib/iris/OPS.md "Known Bugs" #2 (lib/iris/OPS.md:224) — the reaper-stall on a
  hung gcloud subprocess (Diagnose §1).
- `docs/dev-guide/tpu_observability.md` "Troubleshooting Commands" — stockout /
  preemption / node-termination ground truth to rule out a fleet event before
  deleting.
- `.agents/ops/2026-04-21-iris-tpu-wedge-and-holder-poll.md` — the postmortem
  that feeds this runbook.
- Skills that route these error strings here: `debug`, `babysit-job`,
  `scan-logs`.
