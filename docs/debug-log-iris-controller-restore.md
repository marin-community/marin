# Iris controller restore corruption

## Initial status

On 2026-07-24, the `cw-us-east-02a` replacement controller moved from node
`g530eda` to `g5bc6da`. Kubernetes readiness and the controller status command
reported healthy, but `FederationSync` raised `sqlite3.DatabaseError: database
disk image is malformed`. The controller was rolled back to the pre-deploy
checkpoint and previous image.

## Investigation

The first hypothesis was that filesystem modification times had selected an
older node-local database over the remote checkpoint. This was plausible
because SQLite WAL files can have newer mtimes than the main database and node
clocks are compared indirectly.

The restore path exposed a more direct failure mode:
`download_checkpoint_to_local` synchronized checkpoint files into the existing
database directory. A node-local `controller.sqlite3-wal`,
`controller.sqlite3-shm`, or auth sidecar could therefore survive while the
main database was replaced. A regression test reproduced the stale sidecars
remaining after restore. The rollback path did not have this flaw because it
deleted the database directory before downloading its checkpoint.

## Fix

Checkpoint restore now synchronizes into a sibling staging directory,
decompresses the files, runs SQLite `PRAGMA quick_check`, and replaces the
entire local database directory only after validation. Startup also validates
both databases and uses a `last_checkpoint_epoch_ms` metadata value instead of
filesystem mtimes. The value records checkpoint ancestry after a successful
upload and is stamped into a restored database. A local database is reused only
when it is healthy and its marker matches the selected remote checkpoint.

The controller health endpoint now performs database-backed reads from the
metadata, task-attempt, federation changelog, and federation sync-state tables.
It returns HTTP 503 on a database error.

## Validation

The stale-sidecar regression failed before the restore change and passed after
it. The focused checkpoint, rollout, and health suite passed 96 tests. The full
Iris suite passed 2,783 tests with 3 skipped when run against a native extension
built from the current Rust source.

The first `cw-us-east-02a` retry created checkpoint
`1784857156461` and published controller image `54f7f8ef8d`, but stopped before
updating the controller Deployment. A local dependency sync had removed the
optional GCP Secret Manager client, so manifest rendering could not resolve the
controller signing key. The rollback helper failed at the same pre-deploy
boundary. The existing `615c141497` controller remained Ready with zero
restarts, and its database queries continued to succeed. The local controller
extra was then restored; no second live attempt was made.

The next live validation should compare task-attempt state before and after
restart, confirm the checkpoint marker returned by `/health`, and leave the
controller running long enough for federation sync to execute.
