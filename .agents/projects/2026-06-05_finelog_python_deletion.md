# Finelog Python deletion + in-process Rust server (issue #6197)

Gate satisfied: PR #6135 (Rust rewrite) landed on `main` as `f072159b2f`.

## Goal

Delete the dead Python finelog `server/` (ASGI) + `store/` (DuckDB) now that the
Rust `finelog-server` is the sole production implementation. Keep `client/`,
`deploy/`, `proto/`, `scripts/`. Prune shared-module dead code. Make
`./infra/pre-commit.py --all-files` + `uv run pyrefly` clean and tests pass.

## Hard prerequisite (the reason this is not a pure deletion)

Iris's controller (`_start_local_log_server`) and 9 iris test files import
`finelog.server.LogServiceImpl` / `finelog.store.DuckDBLogStore` in-process and
round-trip logs through them. They must be cut over first.

Chosen mechanism (per user): a small **PyO3 surface on the Rust server** that
Python can start/stop in-process — modelled on `rust/dupekit`. Scope: ONE
combined PR.

## Plan

1. **Rust PyO3 crate** `rust/finelog-py` (additive; no change to `rust/finelog`):
   - `[lib] name="_native" crate-type=["cdylib"]`, deps: `finelog` (path),
     `pyo3` (extension-module, abi3-py311), `axum` 0.7, `tokio`.
   - `EmbeddedServer(log_dir=None, remote_log_dir="", host="127.0.0.1", port=0,
     debug_admin=False)`: owns a multi-thread tokio runtime, binds listener
     (port 0 → ephemeral), spawns `axum::serve` with graceful shutdown, drains
     the store on stop. Exposes `.address`, `.port`, `.stop()`, context manager,
     `Drop`.
   - jemalloc stays binary-only (`main.rs`), so the cdylib does not set a global
     allocator.

2. **Packaging (mirror dupekit)**:
   - `rust/finelog-py/pyproject.toml` (maturin; `marin-finelog-native`; module
     `finelog_native._native`; `python-source="."`).
   - `finelog_native/__init__.py` (re-export + graceful ImportError stub) +
     `__init__.pyi`.
   - Add `finelog-py` to `rust/Cargo.toml` workspace members.
   - Add `marin-finelog-native >= 0.1.0.dev0` to consuming deps (iris/root).
   - Extend `scripts/rust_mode.py` to toggle finelog-native path source too.
   - CI: `finelog-native-release-wheels.yaml` (mirror dupekit), wire unit job.

3. **Iris cutover**:
   - `controller.py`: `_start_local_log_server` → start `EmbeddedServer`, use its
     address. Drop finelog.server/store imports. `close()` → `.stop()`.
   - `dashboard.py`: `log_service` type → `LogServiceProxy` (prod already passes
     a proxy). Drop `from finelog.server import LogServiceImpl`.
   - Test fixtures (9 files): shared `embedded_finelog` fixture (skips when the
     native ext is absent) + `LogClient`; rewrite direct
     `log_service.fetch_logs/push_logs` calls to go over RPC.

4. **Delete + prune**:
   - Remove `finelog/server/`, `finelog/store/`, their tests; `tests/parity/`
     python backend → rust-only (drop the cutover test).
   - Dead-code pass on `schema.py`, `policy.py`, `errors.py`, `types.py`, `rpc/`.
   - `bench_dashboard_queries.py` uses `finelog.store.catalog.Catalog` — port or
     retire.

5. Green `./infra/pre-commit.py --all-files`, `uv run pyrefly`, tests.

## De-risk order

Build the PyO3 extension + smoke test (start server, /health, LogClient
round-trip) BEFORE the iris/test refactor — if it can't serve in-process, the
approach is moot.

## Outcome (as shipped)

Deviations from the plan above, decided during implementation:

- **Crate location**: the PyO3 cdylib lives at `rust/finelog/pyext` (package
  `finelog-native`, `[lib] name="_native"`), co-located with the Rust server
  source rather than a sibling `rust/finelog-py`.
- **Packaging**: `marin-finelog` became a native maturin wheel bundling the
  pure-Python `client/deploy/proto` AND the `finelog._native` ext — modelled on
  `marin-dupekit`. It left the uv workspace (`members`/`sources`); root + iris
  pin `marin-finelog >= 0.2.0.dev0`; `scripts/rust_mode.py dev` toggles an
  editable source for both dupekit and finelog. New `lib/finelog/build_package.py`
  + `.github/workflows/finelog-release-wheels.yaml` publish platform wheels. The
  iris Dockerfile pulls the wheel instead of building finelog from source.
  Follow-up issue tracks splitting out a `marin-finelog-server` package so plain
  client consumers don't pull the native ext.
- **On-disk, not in-memory**: the Rust engine's in-memory mode (`log_dir=None`)
  spawns no maintenance task, so its RAM buffer never flushes to a readable
  segment — written logs are never queryable. The controller's local log server
  and every test fixture therefore use an on-disk `log_dir`
  (`local_state_dir/log-server`), which also gives log persistence.
- **Benches**: the two migration-era python-vs-rust A/B benches
  (`benchmarks/query_bench.py`, `scripts/bench_dashboard_queries.py`) were
  deleted — their python arm spawned the deleted engine and the comparison is
  now vacuous. `dashboard/scripts/demo.py` was ported to the in-process
  `EmbeddedServer`.

**Bootstrap (user-owned)**: `uv.lock`, the iris image build, and the probes lock
stay red until the first native `marin-finelog` wheel is published to PyPI.
