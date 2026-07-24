---
date: 2026-07-24
system: iris
severity: degraded
resolution: fixed
pr: none
issue: none
---

# TL;DR

- A `cw-us-east-02a` replacement controller became Ready but later raised `sqlite3.DatabaseError: database disk image is malformed`.
- Checkpoint restore had copied a new main database into an existing directory, allowing stale SQLite WAL/SHM sidecars from the node-local disk to survive.
- Restore now stages and validates both databases, replaces the entire directory, and identifies state with a persisted checkpoint epoch instead of filesystem mtimes.
- Controller restart now resolves operator-side signing secrets before checkpointing, building, or changing rollout state.
- Image `70224363d5` restored checkpoint `1784862840412`; the controller stayed healthy while sampled task Pods retained their UIDs and zero restart counts.

# Original problem report

The `cw-us-east-02a` rollout produced a Kubernetes-Ready controller, but
`FederationSync` later logged `sqlite3.DatabaseError: database disk image is
malformed`. The operator also needed to know why a node-local database could
have a newer mtime than the backup and whether a database sentinel could prove
which checkpoint it contained.

# Investigation path

1. The replacement controller moved from node `g530eda` to `g5bc6da`. Readiness
   initially passed, but federation reads exposed malformed SQLite state. The
   rollout was returned to image `615c141497` and its pre-deploy checkpoint.

2. Filesystem mtime selection was examined first. SQLite WAL/SHM files can be
   newer than the main file, and mtimes across node-local disks are not a
   reliable ancestry signal even with synchronized clocks.

3. The restore path was compared with rollback. `download_checkpoint_to_local`
   synchronized into the existing database directory, while rollback removed
   the directory first. A regression test reproduced stale
   `controller.sqlite3-wal`, `controller.sqlite3-shm`, and auth sidecars
   surviving a restore.

4. Restore was changed to download into a sibling staging directory, decompress
   there, run `PRAGMA quick_check(1)`, and atomically replace the local database
   directory. Startup now probes both SQLite files before deciding to reuse
   node-local state.

5. A `last_checkpoint_epoch_ms` value was added to the `meta` table. It is
   written after a successful upload and stamped from the checkpoint directory
   during restore, providing a logical ancestry check independent of mtimes.

6. The first retry built image `54f7f8ef8d` and created checkpoint
   `1784857156461`, then stopped before changing the Deployment because the
   operator environment lacked `google.cloud.secretmanager`. This exposed that
   controller signing-key resolution happened after checkpoint and image build.

7. Restart preflight moved signing-key resolution ahead of rollout-record
   reads, controller discovery, checkpointing, builds, and Kubernetes writes.
   The resolved value is cached for the later Secret projection so preflight
   and deployment cannot observe different values.

8. The next retry built and deployed image `70224363d5`. Startup logged
   `Synced and validated checkpoint
   s3://marin-us-east-02a/iris/cw-us-east-02a/state/controller-state/1784862840412`
   and a clean `wal_checkpoint(TRUNCATE)`.

9. Post-rollout reads returned the same checkpoint epoch from `meta`, and row
   counts advanced from 2,137 jobs / 6,988 tasks / 7,680 task attempts to
   2,139 / 6,990 / 7,682. Eight sampled long-running Pods retained their
   pre-rollout UIDs and remained Ready with zero restarts across two checks.

# User course corrections

- The investigation was redirected from trusting local mtimes to validating
  actual database contents and the last task attempt. That exposed the need for
  a persisted checkpoint sentinel and read-backed health checks.
- The user requested both a SQLite probe and another controlled restart. This
  converted the proposed validation into live evidence before additional
  CoreWeave rollouts.
- When secret resolution failed late, the user requested moving it before
  rollout work. This prevented future dependency or access errors from creating
  checkpoints, builds, or pending rollout records.
- The user required rollback on any live issue and preservation of running
  tasks. The first malformed rollout was rolled back; subsequent attempts
  captured Pod UIDs and restart counts before and after.

# Root cause

`lib/iris/src/iris/cluster/controller/checkpoint.py:500` restored checkpoint
files into a directory that could already contain SQLite sidecars from a
different database generation. Replacing the main database without removing
those files could produce a database that passed process-level readiness but
failed on later table reads.

The restart workflow also had a separate unsafe ordering problem:
operator-side signing-key resolution occurred inside
`K8sControllerProvider.start_controller`, after expensive and stateful rollout
steps. A missing optional resolver dependency therefore failed too late even
though the running controller was not changed.

# Fix

`lib/iris/src/iris/cluster/controller/checkpoint.py` now restores through a
validated staging directory and swaps the whole directory. It records and
checks `last_checkpoint_epoch_ms`, while
`lib/iris/src/iris/cluster/controller/main.py` reuses local state only when
both SQLite files pass integrity checks and the sentinel matches the selected
remote checkpoint.

`lib/iris/src/iris/cluster/controller/dashboard.py` now backs `/health` with
reads from controller and federation tables and returns HTTP 503 on database
errors.

`lib/iris/src/iris/cli/cluster.py:1203` now runs provider preflight before any
remote rollout-state read. `lib/iris/src/iris/cluster/platforms/k8s/controller.py:380`
resolves and caches the signing-key environment without mutating Kubernetes;
the subsequent start projects that exact cached value.

# How OPS.md could have shortened this

- In **Restarting / rolling the controller**, add that preflight runs before
  checkpoint and build, and that a preflight failure guarantees no rollout
  mutation. This distinguishes local dependency failures from deployment
  failures.
- In **Controller storage and rollback**, document
  `SELECT key, value FROM meta WHERE key = 'last_checkpoint_epoch_ms'` and
  compare it with the numeric checkpoint directory. This is the reusable
  ancestry check for node-local state.
- In **Post-restart verification**, require a database-backed `/health` result,
  a representative `task_attempts` read, and before/after Pod UID and restart
  checks. Kubernetes readiness alone did not detect this class of corruption.

# Artifacts

- Pull request: https://github.com/marin-community/marin/pull/7583
- `s3://marin-us-east-02a/iris/cw-us-east-02a/state/controller-state/1784862840412`
- Controller image `ghcr.io/marin-community/iris-controller:70224363d5`
