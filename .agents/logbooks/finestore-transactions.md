# FineStore transactions and cache rollout

## Scope

Build FineStore format v2 around an immutable manifest and one conditional `HEAD` commit token. Add bounded multi-table transactions, snapshot reads, and manifest-driven background compaction. Move evaluation contracts into an import-light package and migrate the XLA directory cache plus suitable persistent key-value caches onto the generic storage API.

## Evidence index

- Design: [FineStore transactions](https://loom.oa.dev/s/qyinm7he/artifacts/finestore-transactions)
- Rollout: [FineStore targeted build-out](https://loom.oa.dev/s/qyinm7he/artifacts/finestore-buildout)
- Weaver backlog: #248, #249, #250, #251, #252, #253

## Baseline

- Date: 2026-08-12
- Base commit: `0b07ad3d7233b639ae6db5489472a2ebac4410a2`
- FineStore v1 discovers Parquet shards by directory listing and publishes each table independently.
- XLA cache synchronization publishes batches as ZIP files under `.rigging-archives-v1`.
- Evaluation record types live in `finestore.eval`, coupling a generic store to one application.

## Constraints and invariants

- A committed `HEAD` version is the visibility boundary for every table and object in a transaction.
- Readers pin one manifest for their lifetime.
- Data, schemas, and manifests are immutable. Only `HEAD` is conditionally mutable.
- Compaction changes logical visibility through a manifest commit and does not delete files that an older reader may still reference.
- Writable remote stores must support conditional writes. Generic fsspec backends without compare-and-swap are read-only.
- Finelog keeps its SQLite catalog and existing background maintenance model; its durable sequence is the analogous commit token.

## Log

### 2026-08-12 — rollout started

- Rebased the implementation branch onto `origin/main` at `0b07ad3d7233b639ae6db5489472a2ebac4410a2`.
- Created Weaver backlog items for format v2, transactions and compaction, evaluation separation, XLA file sets, key-value caches, and validation.
- Started the call-site and package-boundary inventory before changing the storage format.

### 2026-08-12 — format and adapters implemented

- Added local, GCS, and S3 conditional-object writes and made one `HEAD` version the archive commit token.
- Replaced listing-based reads with pinned full manifests. Multi-table transactions, ordinary flushes, sealing, metadata changes, and compaction all publish through the same compare-and-swap path.
- Added threshold-driven background logical compaction. Source shards remain immutable and are not physically collected while older read views can name them.
- Added a one-shot sealed v1-to-v2 migration that publishes existing Parquet objects without copying them, plus a fleet command for evaluation archives.
- Moved evaluation contracts and the evaluation archive adapter into `marin-evalstore` and updated the writers, dashboard, migrations, packaging, and CI dependency graph.
- Replaced the XLA ZIP mirror with bounded FineStore file-set transactions. Moved the Cutlass and Pallas persistent byte caches from one object per key to the FineStore blob table.

Validation so far:

- `uv run --package marin-finestore pytest lib/finestore/tests -q`: 53 passed.
- `uv run --package marin-iris pytest lib/iris/tests/test_jax_init.py -q`: 35 passed.
- Evaluation archive, dashboard, Harbor, and migration tests: 55 passed before the format-migration addition.
- Cutlass cache tests: 11 passed; the broader affected Levanter selection reached 80 passed and 12 skipped after fixing its initial cache assertions.

### 2026-08-12 — rollout validation

- Added primary-key sorting and a snapshot-pinned streaming merge so file-set warmup does not materialize all cached payloads. Remote key-value writes coalesce into bounded transactions.
- `./infra/pre-commit.py --changed-files --fix`: lint, formatting, license headers, pyrefly, file, AST, TOML/YAML, whitespace, and Markdown gates passed.
- FineStore package: 56 passed.
- Evaluation archive/dashboard/Harbor/migration selection: 55 passed.
- Rigging content-hash/conditional-object/distributed-lock selection: 16 passed.
- Iris JAX cache selection: 35 passed.
- Levanter Cutlass and Pallas cache selection: 83 passed, 12 skipped.
- Package release and dependency-graph selection: 92 passed.
- Both `marin-finestore` and `marin-evalstore` built as source distributions and pure-Python wheels.
- The repository-wide affected-test runner expanded to all 1,444 root tests after the workspace dependency change. Its workers became unhealthy in `test_evaldash_local_store` and were stopped after repeated hard timeouts. A serial `-x` run isolated the first failure to an unrelated Zephyr test where `psutil.Process()` could not observe the current sandbox PID (`psutil.NoSuchProcess: pid=5`); it reached 57 passes first. Dedicated live Iris, Levanter Torch, and Levanter TPU suites remain CI-only.

### 2026-08-12 — self-review

- Removed two implementation-coupled tests during the pre-PR test-quality pass and corrected stale generation/listing/deletion prose. The final FineStore suite is 54 passing tests.
- `uv lock --check` and `git diff --check` pass after the final cleanup.

### 2026-08-13 — advisory review follow-up

- Replaced the historical reader alias at every call site with `ReadView` and removed the compatibility class.
- Enforced the transaction payload bound, represented seal changes as one explicit state transition, and shared the versioned row merge between reads and compaction.
- Scoped background cache executors to each cache instance and made close wait for its queued writes without a timeout.
- Made the samples-v4 backup verify existing metadata and shard sizes before dropping the active table. Captured the S3 endpoint when constructing a conditional object and removed ambiguous path normalization.
- `./infra/pre-commit.py --changed-files --fix`: all gates passed.
- FineStore: 54 passed; evaluation archive/Harbor/migration selection: 47 passed; Iris JAX cache: 35 passed; Cutlass/Pallas cache: 16 passed; Rigging conditional-write/lock: 9 passed.
