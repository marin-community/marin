---
date: 2026-07-25
system: iris
severity: degraded
resolution: fixed
pr: none
issue: none
---

# Iris ARM log-shipper CrashLoopBackOff

## TL;DR

- Iris task containers kept running on `cw-us-east-08a`, but their `log-shipper`
  sidecars failed with `exec .venv/bin/python: exec format error`.
- The affected GB200 nodes were ARM64. The controller image used by the sidecar,
  `iris-controller:d9690bb649`, contained only a Linux AMD64 manifest.
- Iris/finelog had no durable output for the affected tasks even though their
  kubelet-local `task` container logs were non-empty.
- Kubernetes controller restarts and the scheduled image workflow were changed
  to publish the controller image for AMD64 and ARM64.
- All four `cw-*` controllers were restarted in place on multi-architecture image
  `a88653b976`, starting with CI. Sixteen new ARM64 task Pods then ran the log
  shipper with zero restarts and wrote logs to Iris/finelog.
- Existing task Pods retained their immutable `d9690bb649` sidecar specification
  and were not restarted. Their historical local logs were not backfilled.

## Original problem report

The `log-shipper` container in
`iris-held-glm52-kernelgym-3200-2026-29a226e3-0-c54fdeb8e044d5c5`
in the `iris` namespace on `cw-us-east-08a` showed
`CrashLoopBackOff` with 88 restarts. The user asked what triggered it and
whether finelog/Iris had the workload logs.

## Investigation path

1. The Pod and its previous container logs were inspected first. The main
   `task` container was running and emitted Ray startup output, while
   `log-shipper` exited 255 with
   `exec .venv/bin/python: exec format error`.

2. The Pod ran on ARM64 node `s62pys64`, an `gb200-4x` instance. Its shipper
   used `ghcr.io/marin-community/iris-controller:d9690bb649`; the registry
   manifest exposed only `linux/amd64`. Older controller tags
   `6544e1251d` and `a8e3e05514` exposed both architectures and ran healthy
   shippers.

3. The cluster-wide scope was measured before mutation. Of 67 Pods with a log
   shipper, 44 had restarted and 20 were in `CrashLoopBackOff`; the affected
   Pods all used the AMD64-only controller image.

4. Iris logs for
   `/held/glm52-kernelgym-3200-20260719-1654-s1-r13/vllm` returned no lines.
   Direct `kubectl logs ... -c task` returned real output. This ruled out an
   empty workload log and established that the broken shipper prevented durable
   ingestion.

5. The Pod builder confirmed the architecture coupling:
   `lib/iris/src/iris/cluster/backends/k8s/tasks.py:589-622` deliberately runs
   the controller image as a restartable init sidecar because the base task
   image does not initially contain Iris.

6. The restart image builder in `lib/iris/src/iris/cli/cluster.py:223-244`
   built Kubernetes task images for AMD64 and ARM64 but built controller images
   only for AMD64. The scheduled workflow had the same assumption in
   `.github/workflows/ops-docker-images.yaml`.

7. A regression test reproduced the mismatch, then passed after both Kubernetes
   image targets used `linux/amd64,linux/arm64`. The Iris CLI test slice passed
   97 tests, and the repository changed-file checks passed.

8. Controller-only restarts ran sequentially on `cw-us-west-04a`,
   `cw-us-east-02a`, `cw-rno2a`, and `cw-us-east-08a`. Each restart took a
   checkpoint, deployed `a88653b976`, reached healthy status, and committed its
   rollout record.

9. After the final rollout at 00:29 UTC on July 26, 16 naturally submitted task
   Pods started on ARM64 GB200 nodes with controller manifest
   `sha256:b115168978f732ba6a415974a3c4f92ab9293a4387c9c65529a527d91ccde65d`.
   Every checked shipper was ready with zero restarts. Iris returned current
   finelog output for
   `/rav/rav-qbdrv-on/grug-train-20260726-001018`.

## User course corrections

- The investigation considered moving the shipper to the task image or another
  uv-compatible image. The user selected an ARM64 controller build as the
  simplest fix. This preserved the existing guarantee that the shipper image
  contains the Iris package before task dependency installation.
- The user required the rollout to begin with CI in `cw-us-west-04a`, then
  proceed through the remaining `cw-*` configurations. The rollout followed
  that order and stopped for a health check after every cluster.
- The user explicitly required a controller restart, not a full cluster
  restart. Only `iris --cluster=<name> cluster controller restart` was used;
  task Pods and cluster infrastructure were not restarted.

## Root cause

The log shipper and controller shared one image, but the image publishing paths
treated the controller as controller-node-only. Kubernetes task Pods construct
the sidecar from that image at
`lib/iris/src/iris/cluster/backends/k8s/tasks.py:605-622`, so its platform set
must cover task-node architectures too. The restart builder and scheduled
workflow violated that invariant by publishing the controller image only for
AMD64.

The AMD64 executable could be pulled onto ARM64 GB200 nodes because the tag had
no ARM64 variant. The kernel rejected `.venv/bin/python` before the shipper
could start, so no task lines reached finelog even though the task container
continued to run and kubelet retained local logs.

## Fix

`lib/iris/src/iris/cli/cluster.py:64-65,223-244` now uses the shared
`KUBERNETES_IMAGE_PLATFORMS` value for both controller and task images.
`.github/workflows/ops-docker-images.yaml:46-49,64-78,125-140` now builds a
native ARM64 controller variant and includes it in the published manifest.
`lib/iris/tests/cli/test_cluster_image_builds.py` protects the platform
selection, and `lib/iris/OPS.md:68-74` documents why Kubernetes controller
images are multi-architecture.

The live repair published
`ghcr.io/marin-community/iris-controller:a88653b976` for AMD64 and ARM64 and
performed controller-only restarts:

- `cw-us-west-04a`: checkpoint
  `s3://marin-us-west-04a/iris/cw-us-west-04a/state/controller-state/1785023914224`
- `cw-us-east-02a`: checkpoint
  `s3://marin-us-east-02a/iris/cw-us-east-02a/state/controller-state/1785025304755`
- `cw-rno2a`: checkpoint
  `s3://marin-us-east-02a/iris/cw-rno2a/state/controller-state/1785025471690`
- `cw-us-east-08a`: checkpoint
  `s3://marin-us-east-02a/iris/cw-us-east-08a/state/controller-state/1785025633521`

## How OPS.md could have shortened this

The Controller Restart section previously stated that controller images were
AMD64-only and task images alone needed ARM64. It now states that Kubernetes
task Pods run the controller image as the log-shipper sidecar, so both images
must be published for AMD64 and ARM64. This makes the image-platform consequence
visible before any future restart or publishing change.

## Artifacts

- `lib/iris/tests/cli/test_cluster_image_builds.py`
- `lib/iris/OPS.md`
- `ghcr.io/marin-community/iris-controller:a88653b976`
- Controller manifest:
  `sha256:b115168978f732ba6a415974a3c4f92ab9293a4387c9c65529a527d91ccde65d`
