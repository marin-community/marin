---
date: 2026-07-24
system: finelog
severity: degraded
resolution: mitigated
pr: https://github.com/marin-community/marin/pull/7583
issue: https://github.com/marin-community/marin/issues/605
---

# TL;DR

- The first multi-architecture finelog rollout on `cw-us-east-02a` became
  Ready and passed `/health`.
- A rollout guard incorrectly expected Kubernetes `imageID` to report the
  platform child digest. CoreWeave containerd reported the OCI index digest,
  so the guard initiated a rollback.
- Active writers had already registered a `COLUMN_TYPE_MAP` schema in the
  shared SQLite catalog. The old image could not parse that enum value and
  crash-looped after rollback.
- The service was recovered by rolling forward to the new image. `SELECT 1`
  succeeded, the pod remained Ready with zero restarts, and running Iris task
  Pods remained Ready with zero restarts.
- All remaining finelog and Iris rollouts stopped after the failed rollback.

# Original problem report

Roll all finelog servers to the new SQLite-syncing image and raise Kubernetes
finelog compute from a 500m CPU request / 2 CPU limit to a 2 CPU request /
8 CPU limit. Roll back and stop if any live issue occurs.

# Investigation path

1. The `cw-us-east-02a` Deployment was updated to OCI index
   `sha256:ac86fc3312211893dae1fd3bcbe2b4a8266b374006992742cb3f315ae6a73bbe`
   with 2 requested CPUs, 8 limit CPUs, 16 GiB requested memory, and 32 GiB
   limit memory.
2. Kubernetes reported the new pod Ready, and `finelog deploy up` passed its
   `/health` check.
3. The external guard compared the pod's `imageID` with the amd64 child digest.
   Containerd instead returned the index digest, which is valid but caused the
   guard to call `kubectl rollout undo`.
4. The restored image
   `sha256:af6e1daff6020aea9da3b0eceafb1740d280ed51e8f22742c01ae16451630d9d`
   exited with `unknown column type name "COLUMN_TYPE_MAP" in catalog JSON`.
   The new server had accepted live schema registrations that the old enum
   parser did not understand.
5. Reapplying the desired manifest initially exposed a separate strategic
   merge problem: the rollback restored TCP probes while the desired manifest
   uses HTTP probes, and `kubectl apply` retained both mutually exclusive
   handlers.
6. The obsolete TCP probe handlers were atomically moved to HTTP handlers, and
   the desired manifest was reapplied.
7. The recovered pod was Ready with zero restarts. The 250 GiB PVC remained
   Bound, a live `SELECT 1` succeeded, recent logs contained no panic or fatal
   error, and every running Iris-managed pod was Ready with zero restarts.

# Root cause

The rollout itself was healthy. The incident combined three independent
compatibility assumptions:

1. A multi-architecture runtime may report the OCI index digest as `imageID`;
   it does not have to expose the selected platform manifest digest there.
2. The new finelog catalog format is forward-only once a live client registers
   `COLUMN_TYPE_MAP`; the old binary cannot reopen that catalog.
3. `kubectl rollout undo` and a later client-side apply can leave mutually
   exclusive probe-handler map keys in the strategic-merge result.

# Resolution

`cw-us-east-02a` was recovered on the new multi-architecture image. No database
files were replaced or deleted. The remaining finelog rollouts and the planned
`marin-dev` Iris rollout were stopped.

# Follow-up

- Treat finelog catalog migrations as forward-only unless the rollback
  procedure also restores a verified pre-rollout SQLite snapshot.
- Validate a multi-architecture image by resolving the index for the node
  architecture through the registry, not by assuming Kubernetes `imageID`
  exposes the child manifest.
- Make probe-type changes safe across `rollout undo`, either by using a replace
  operation or by explicitly removing the obsolete handler before apply.

# Artifacts

- Pull request: https://github.com/marin-community/marin/pull/7583
- Finelog index:
  `sha256:ac86fc3312211893dae1fd3bcbe2b4a8266b374006992742cb3f315ae6a73bbe`
- Recovered pod: `finelog-cw-use02a-5c99f98fd7-8ktk8`
