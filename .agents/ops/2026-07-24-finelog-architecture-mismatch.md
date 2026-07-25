---
date: 2026-07-24
system: finelog
severity: degraded
resolution: fixed
pr: https://github.com/marin-community/marin/pull/7583
issue: none
---

# TL;DR

- The first `cw-us-east-08a` finelog replacement landed on an arm64 node with
  an amd64 image and exited with `exec format error`.
- Previous deployments worked because Kubernetes happened to place them on
  amd64 nodes; the image had never been multi-architecture.
- The immediate recovery restored the previous image. Finelog health, a live
  query, the Iris controller, and all 40 running tasks remained healthy.
- The final fix pins finelog pods and image builds to amd64. All five finelog
  services completed rollout with zero restarts and successful health and SQL
  probes.

# Original problem report

Roll all finelog servers to the SQLite-syncing build without interrupting
running Iris tasks. CoreWeave clusters were assumed to have amd64 nodes for
control services.

# Investigation path

1. The first replacement pod on `cw-us-east-08a` exited immediately with
   `exec format error`.
2. The image manifest contained only `linux/amd64`, while the pod had scheduled
   on an arm64 node.
3. The deploy builder and scheduled image workflow had always built on amd64.
   Successful older rollouts had landed on amd64 nodes by chance.
4. A multi-architecture index was built as an immediate mitigation. Registry
   inspection confirmed both child manifests.
5. The rollout later established an amd64 node selector for stateful finelog
   services, making arm64 control-service images unnecessary.
6. The image build default returned to `linux/amd64`; Iris worker and task
   images stayed multi-architecture because arm64 GPU nodes run workloads.

# User course corrections

- The investigation initially treated mixed-architecture scheduling as a
  requirement for multi-architecture control images. The user clarified that
  every cluster has amd64 nodes and asked to taint or select those nodes for
  finelog and Iris controllers.
- The user later confirmed that finelog and controller builds no longer need
  arm64 once the scheduling constraint is present. Workload images retained
  arm64 support.

# Root cause

`lib/finelog/deploy/k8s/02-deployment.yaml.tmpl` did not constrain finelog to
amd64 even though `lib/finelog/src/finelog/deploy/build.py` produced an amd64
image. Mixed-architecture CoreWeave scheduling could therefore assign the pod
to an incompatible node.

# Fix

The Kubernetes Deployment now selects `kubernetes.io/arch: amd64`. Finelog
builds default to `linux/amd64`, matching the placement constraint. Iris
controller images use the same control-node policy, while Iris worker and task
images retain `linux/amd64,linux/arm64`.

The failed `cw-us-east-08a` replacement was rolled back before continuing. The
remaining services were rolled only after the scheduling constraint and health
guards were in place.

# How OPS.md could have shortened this

Add a control-service rollout preflight to `lib/iris/OPS.md`: compare the image
platform with the selected node architecture before restarting a Kubernetes
Deployment. This catches architecture mismatches before a pod is replaced.

# Artifacts

- Pull request: https://github.com/marin-community/marin/pull/7583
- Related rollback log:
  `.agents/ops/2026-07-24-finelog-schema-rollback.md`
