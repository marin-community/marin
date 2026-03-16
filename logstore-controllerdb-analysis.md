# LogStore and ControllerDB Implementation Analysis

## 1. LogStore

**Defined at:** `lib/iris/src/iris/cluster/log_store.py:65`

### SQL Schema
```sql
CREATE TABLE IF NOT EXISTS logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    source TEXT NOT NULL,
    data TEXT NOT NULL,
    epoch_ms INTEGER NOT NULL,
    level INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_key ON logs(key, id);
```

The schema is embedded as a module-level `_SCHEMA` string at `log_store.py:28-38`. It is **also** duplicated in the ControllerDB migration `0001_init.sql:165-183` (identical `logs` table + index).

### Constructor
```python
def __init__(self, log_dir: Path | None = None, *, db_path: Path | None = None, max_records: int = _MAX_RECORDS):
```
Three modes: explicit `db_path`, `log_dir` (creates `logs.db` inside), or temp dir. Creates **two** SQLite connections (`_write_conn`, `_read_conn`) in WAL mode for concurrent read/write.

### Key Methods
| Method | Line | Purpose |
|--------|------|---------|
| `append(key, entries)` | :112 | Insert log entries for a single key |
| `append_batch(items)` | :125 | Insert entries from multiple keys in one transaction |
| `get_logs(key, ...)` | :152 | Fetch logs for a single key with filtering (since_ms, cursor, substring, min_level, tail) |
| `get_logs_by_prefix(prefix, ...)` | :210 | Fetch logs for all keys matching prefix, supports `shallow` to exclude nested |
| `has_logs(key)` | :282 | Check if any logs exist for key |
| `clear(key)` | :290 | Delete all logs for key |
| `close()` | :295 | Close both connections + cleanup temp dir |
| `cursor(key)` | :309 | Return a `LogCursor` for incremental reads |
| `_evict_if_needed()` | :313 | Cap total rows at `max_records`, evict oldest in batches |

### Related Classes
- `LogCursor` (`:344`) — stateful incremental reader for a key
- `LogStoreHandler` (`:364`) — `logging.Handler` that writes Python log records into LogStore
- `LogReadResult` (`:60`) — dataclass with `entries: list[LogEntry]` and `cursor: int`

### Threading Model
Separate write and read locks (`Lock`, not `RLock`). Two separate SQLite connections so WAL concurrency works: readers never block the writer.

---

## 2. ControllerDB

**Defined at:** `lib/iris/src/iris/cluster/controller/db.py:1000`

### SQL Schema
Managed via migration files in `lib/iris/src/iris/cluster/controller/migrations/`:

| Migration | Tables/Changes |
|-----------|---------------|
| `0001_init.sql` | `schema_migrations`, `meta`, `users`, `jobs`, `tasks`, `task_attempts`, `workers`, `worker_attributes`, `worker_task_history`, `worker_resource_history`, `endpoints`, `dispatch_queue`, `txn_log`, `txn_actions`, `scaling_groups`, `tracked_workers`, `reservation_claims`, **`logs`** + indexes + triggers |
| `0002_read_indexes.sql` | Additional read indexes |
| `0003_normalize_scaling_groups.sql` | Scaling group normalization |
| `0004_api_keys.sql` | API key table |
| `0004_worker_indexes.sql` | Worker indexes |
| `0005_task_profiles.sql` | Task profiles table |
| `0006_jwt_signing_key.sql` | JWT signing key |

Note: `0001_init.sql` includes a `logs` table (lines 165-172) and `idx_logs_key` index (line 183) identical to the LogStore schema.

### Constructor
```python
def __init__(self, db_path: Path):
    self._db_path = db_path
    self._db_path.parent.mkdir(parents=True, exist_ok=True)
    self._lock = RLock()  # single reentrant lock
    self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
    self._conn.row_factory = sqlite3.Row
    self._configure(self._conn)
    self.apply_migrations()
```
Single connection, single `RLock`. WAL mode + `PRAGMA foreign_keys = ON`.

### Key Methods
| Method | Line | Purpose |
|--------|------|---------|
| `close()` | :1023 | Close connection |
| `transaction()` | :1027 | Context manager yielding `TransactionCursor`, `BEGIN IMMEDIATE` |
| `fetchall(query, params)` | :1041 | Raw read query |
| `fetchone(query, params)` | :1045 | Raw read single row |
| `execute(query, params)` | :1049 | Execute within a transaction |
| `snapshot()` | :1053 | Returns `QuerySnapshot` for read-only access |
| `apply_migrations()` | :1065 | Apply pending `.sql` files from `migrations/` dir |
| `backup_to(destination)` | :1130 | Hot backup using SQLite backup API |
| `replace_from(source)` | :1141 | Replace DB file from local or remote path (fsspec), reopen connection, re-run migrations |
| `ensure_user(...)` | :1100 | Create user if not exists |
| `set_user_role(...)` | :1107 | Update user role |
| `get_user_role(...)` | :1111 | Get user role |
| `next_sequence(key, cur)` | :1121 | Atomic counter in `meta` table |
| `delete_endpoint(...)` | :1166 | Delete and return endpoint |
| `delete_endpoints(...)` | :1178 | Bulk delete endpoints |

### Related Classes (same file)
- `TransactionCursor` — wraps `sqlite3.Cursor` with `insert()`, `update()`, `delete()`, `execute()`, `executemany()` helpers
- `QuerySnapshot` (`:353`) — read-only snapshot context manager using `BEGIN` + rollback, with `select()`, `count()`, `raw()` methods
- `Row` — generic named-attribute wrapper for query results
- `Table` / `Column` / `_SqlPredicate` — typed SQL DSL for building queries
- Dataclasses: `Worker`, `Job`, `Task`, `TaskAttempt`, `Endpoint`, etc.

---

## 3. LogStore Instantiation and Usage

### Production instantiation
| File | Line | Context |
|------|------|---------|
| `lib/iris/src/iris/cluster/controller/controller.py` | :764 | `self._log_store = LogStore(db_path=config.local_state_dir / "logs.sqlite3")` |
| `lib/iris/src/iris/cluster/worker/worker.py` | :172 | `self._log_store = LogStore()` (temp dir) |

### Production usage (source files)
| File | Usage |
|------|-------|
| `controller/controller.py` | Passed to `ControllerTransitions` and `ControllerServiceImpl` |
| `controller/transitions.py` | `self._log_store` — appends task/process logs |
| `controller/service.py` | Reads logs via `get_logs`, `get_logs_by_prefix` for RPC responses |
| `cluster/process_status.py` | Uses `LogStoreHandler` |
| `worker/worker.py` | Creates LogStore, passes to task attempts |
| `worker/task_attempt.py` | Writes task stdout/stderr to LogStore |
| `worker/service.py` | Reads from LogStore for worker-side log queries |

### Test instantiation
| File | Line |
|------|------|
| `test_transitions.py` | :65, :3072 |
| `test_logs.py` | :29, :117, :122, :167, :171, :351 |
| `test_api_keys.py` | :41 |
| `test_dashboard.py` | :135 |
| `test_scheduler.py` | :450 |
| `test_job.py` | :47 |
| `test_heartbeat.py` | :29, :123, :163 |
| `test_service.py` | :165 |
| `test_reservation.py` | :78 |

---

## 4. ControllerDB Instantiation and Usage

### Production instantiation
| File | Line | Context |
|------|------|---------|
| `controller/controller.py` | :763 | `self._db = ControllerDB(db_path=config.local_state_dir / "controller.sqlite3")` |
| `controller/main.py` | :113 | `db = ControllerDB(db_path=db_path)` (CLI entry, passed to Controller) |
| `cluster/local_cluster.py` | :179 | `db = ControllerDB(db_path=...)` (local dev cluster) |

### Production usage (source files)
All controller subsystems use it: `controller.py`, `transitions.py`, `service.py`, `query.py`, `scaling_group.py`, `autoscaler.py`, `auth.py`, `checkpoint.py`, `config.py`.

### Test instantiation
| File | Line |
|------|------|
| `test_query.py` | :16 |
| `test_auth.py` | :521 |
| `test_checkpoint.py` | :94, :115, :135, :165, :172, :194, :208 |
| `test_transitions.py` | :64, :3071 |
| `test_service.py` | :164 |
| `test_task_profiles.py` | :20 |
| `test_api_keys.py` | :34 |
| `test_job.py` | :46 |
| `test_heartbeat.py` | :28, :122, :162 |
| `test_scheduler.py` | :449 |
| `test_db.py` | :19 |
| `test_dashboard.py` | :134 |
| `test_reservation.py` | :77 |

---

## 5. Checkpoint Backup/Restore

**Module:** `lib/iris/src/iris/cluster/controller/checkpoint.py`

### Write flow (`write_checkpoint`)
1. `db.backup_to(tmp_path)` — uses SQLite's C-level backup API (hot backup, no lock contention)
2. `_fsspec_copy(tmp_path, remote_timestamped)` — upload `checkpoint-{epoch_ms}.sqlite3`
3. `_fsspec_copy(tmp_path, remote_latest)` — upload `latest.sqlite3` (overwritten each time)
4. Delete local temp file

### Restore flow (`download_checkpoint_to_local`)
1. Check if `checkpoint_path` or `{remote}/controller-state/latest.sqlite3` exists via fsspec
2. Download to `{local_path}.download.tmp`
3. Atomic rename to `local_path`
4. Return `True`

### Startup logic (`main.py:103-113`)
```python
if db_path.exists():
    logger.info("Local DB exists, skipping remote restore")
else:
    restored = download_checkpoint_to_local(remote_state_dir, db_path, checkpoint_path)
db = ControllerDB(db_path=db_path)
```
Download happens **before** ControllerDB is created. The `replace_from` method on ControllerDB is an alternative path (replaces open connection in-place) but is not used in the main startup path.

### `replace_from` (`db.py:1141`)
```python
def replace_from(self, source: str | Path) -> None:
    with self._lock:
        tmp_path = self._db_path.with_suffix(".tmp")
        with fsspec.core.open(str(source), "rb") as src, open(tmp_path, "wb") as dst:
            dst.write(src.read())
        self._conn.close()
        tmp_path.rename(self._db_path)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._configure(self._conn)
    self.apply_migrations()
```
Downloads to temp, closes connection, atomic rename, reopens, re-runs migrations. Only safe at startup before concurrent access.

---

## 6. Tests

### LogStore tests
- `lib/iris/tests/cluster/controller/test_logs.py` — 16 tests: tail, cursor, prefix, shallow, eviction, persistence, concurrency, substring filter escaping

### ControllerDB tests
- `lib/iris/tests/cluster/controller/test_db.py` — 14 tests: TransactionCursor CRUD (insert, update, delete), rollback, lastrowid, raw queries, composite predicates
- `lib/iris/tests/cluster/controller/test_checkpoint.py` — 8 tests: write/download roundtrip, temp file cleanup, skip-if-local-exists, fresh start, periodic checkpoint

### Indirect testing
LogStore and ControllerDB are heavily exercised by integration-style tests:
- `test_transitions.py` (~3000+ lines, ~100+ tests)
- `test_service.py`, `test_heartbeat.py`, `test_scheduler.py`, `test_job.py`, `test_reservation.py`, `test_dashboard.py`, `test_api_keys.py`

---

## 7. Relationship Between LogStore and ControllerDB

**Completely separate classes in separate files:**
- `LogStore` → `lib/iris/src/iris/cluster/log_store.py`
- `ControllerDB` → `lib/iris/src/iris/cluster/controller/db.py`

**Separate SQLite databases at runtime:**
- Controller: `{local_state_dir}/controller.sqlite3` (ControllerDB)
- Logs: `{local_state_dir}/logs.sqlite3` (LogStore)
- Worker: LogStore with temp dir (no persistent path)

**Schema duplication:** The `logs` table schema appears in both:
1. `log_store.py:28-38` as `_SCHEMA` (used by LogStore to self-init)
2. `0001_init.sql:165-183` in ControllerDB migrations (creates an unused `logs` table inside `controller.sqlite3`)

This duplication is an artifact — the `logs` table in ControllerDB's migration is vestigial. LogStore manages its own separate database file.

**No code dependency between them:** LogStore does not import from `controller/db.py`. ControllerDB does not import from `log_store.py`. They are peers, both used by `Controller` and `ControllerTransitions`.

**Threading approaches differ:**
- LogStore: two `Lock` instances + two connections (write conn, read conn)
- ControllerDB: one `RLock` + one connection

**No shared backup/checkpoint:** `checkpoint.py` only backs up ControllerDB. LogStore has no backup mechanism — logs are ephemeral (capped at 5M records with eviction).
