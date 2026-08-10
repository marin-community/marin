# Research — DuckDB SQL service on Iris

Framing: an Iris-hosted service (in the spirit of `finelog`) exposing a minimal
dashboard where a user pastes a SQL query, the service runs it in **DuckDB** on a
full **v6e** machine (all CPUs, all RAM), and returns results. Keep it simple.

All paths under repo root.

## 1. finelog as a service template

finelog is now a **Rust** binary (Python server/store deleted — see §5); it runs
as a standalone GCP VM / k8s Deployment, and the Iris controller embeds it
in-process via PyO3. It is **not** Iris-hosted today.

- Binary entrypoint: `lib/finelog/rust/src/main.rs:57` (clap CLI `--port/--log-dir`,
  opens `Store`, `build_app`, `axum::serve` `:97-101`; one port serves `/health` +
  RPC + SPA).
- Router assembly: `lib/finelog/rust/src/server/app.rs:111` `build_app` — `/health`
  `:114`, SPA routes before connect `.fallback_service` for RPC POSTs `:124-126`.
- **SQL surface (model for us):** `StatsService` —
  `lib/finelog/rust/proto/finelog_stats.proto:173-179`, `rpc Query(QueryRequest)
  returns (QueryResponse)` `:176`; `QueryRequest { string sql = 1; }` `:117-121`.
  Impl `lib/finelog/rust/src/server/stats_service.rs`.
- Query/exec layer is **DataFusion, not DuckDB**: `lib/finelog/rust/src/query/mod.rs`
  — `make_ctx()` `:147-156` builds a read-only `SessionContext`, `run_query_over`
  `:258-291` registers providers, runs user SQL verbatim, collects Arrow batches.
  `datafusion = "53"` at `lib/finelog/rust/Cargo.toml:76`.
- Dashboard: Vue SPA `lib/finelog/dashboard/`, served by Rust
  `lib/finelog/rust/src/server/spa.rs`.
- Deploy: Python click CLI `lib/finelog/src/finelog/deploy/cli.py:97` →
  `deploy up/down/restart/status/logs`; GCP-VM or k8s backend. k8s spec
  `lib/finelog/deploy/k8s/02-deployment.yaml.tmpl` (resources `:55-61`).
- Config: `lib/finelog/src/finelog/deploy/config.py`; example
  `lib/finelog/config/marin.yaml` (`name/port/image/remote_log_dir/deployment.gcp`).
- Client SQL call: `cli.py:254` `query_cmd` → `client.query(sql, max_rows=...)`.

## 2. Running a service on Iris + resource requests

- Launch: `iris job run -- <cmd>`. CLI `lib/iris/src/iris/cli/job.py`. Long-running
  server = a job whose command blocks; use `--no-wait` (`lib/iris/README.md:309`).
- Resource spec: `ResourceSpec` dataclass `lib/iris/src/iris/cluster/types.py:529`
  — `cpu: float`, `memory: str|int`, `disk`, `device: DeviceConfig|None`.
  `.to_proto()` `:546`. Accelerator CPU floor `MIN_ACCELERATOR_CPU_MILLICORES =
  4000` `:543`.
- Builder from flags: `build_resources(tpu, cpu, memory, disk)` `job.py:281`; with
  `--tpu`, `spec.device = tpu_device(primary)` `:296-300`.
- Guardrails: `validate_extra_resources` `job.py:361` — `--tpu`/`--gpu` and mem
  ≥ 4 GB require `--enable-extra-resources` `:393`.
- Expose HTTP port: Iris endpoint registry `register_endpoint`/`unregister_endpoint`
  `lib/iris/src/iris/cluster/client/protocol.py:68-76`; controller reverse proxy
  `lib/iris/src/iris/cluster/controller/endpoint_proxy.py:70`
  (`/proxy/{endpoint_name}/{sub_path}`); resolution `endpoints.py:63`. finelog is
  registered this way `endpoints.py:47`.
- Task self-view of resources: `TaskResources.from_environment()`
  `lib/iris/src/iris/env_resources.py:58` reads `IRIS_TASK_RESOURCES`, else falls
  back to host cgroup/`/proc` (`:108+`). A task on a whole host sees all CPUs/RAM.
- Heavier alternative: Iris Actor system `lib/iris/README.md:74-100` — overkill for
  a single dashboard.

## 3. DuckDB in the repo

- finelog declares it deploy-side only: `lib/finelog/pyproject.toml:10-16`
  (`duckdb>=1.0.0`, "used by the deploy CLI to query pulled parquet segments
  in-process"); imported `lib/finelog/src/finelog/deploy/cli.py:25`.
- Production finelog query path is DataFusion (Rust); no reusable embedded-DuckDB
  service exists today.
- Other DuckDB usage: `scripts/ops/storage/dashboard/server.py`,
  `scripts/ops/storage/generate_report.py` (+ render/delete), `scripts/ops/cross_region.py`,
  dep in `scripts/ops/storage/pyproject.toml`.

## 4. Dashboard / web UI patterns

Repo idiom = Starlette ASGI + Connect-RPC + Vue SPA (Python services), or axum+SPA
(Rust). For "paste SQL, run it", copy the iris worker dashboard.

- Smallest: `lib/iris/src/iris/cluster/worker/dashboard.py` — `_create_app()`
  returns a Starlette app `:37-41`, mounts RPC + `/health` `:56` + `/` HTML `:60`.
- Full: `lib/iris/src/iris/cluster/controller/dashboard.py:410+` — Starlette HTML
  shell (`html_shell("controller")` `:606`) + multiple Connect-RPC sub-apps.
- Shared helpers: `lib/iris/src/iris/cluster/dashboard_common.py` — `html_shell()`
  `:166`, `static_files_mount()` `:128`, `favicon_route()` `:152`,
  `@public`/`@requires_auth` `:33-39`, `on_shutdown` `:45`.
- Cleanest shape for us: one Starlette route returns an HTML form; a POST endpoint
  runs the query — mirrors finelog's StatsService.Query but Python + DuckDB.

## 5. Related design docs

- **`stats_service/`** (design/research/spec) — closest prior art: SQL-queryable
  typed-table service co-hosted in finelog; raw `Table.query(sql)`. Note: its
  `duckdb_store.py:79` references are **stale** (deleted, see next).
- **`2026-06-05_finelog_python_deletion.md`** — deleted finelog Python ASGI server +
  DuckDB store; explains DuckDB→DataFusion and why no Python finelog server remains.
- `20260331_iris_sql_redesign.md` / `iris-sql-store.md` / `20260310_iris_sql_canonical.md`
  — controller SQLite stores-layer redesign.
- `20260315_iris_controller_query_design.md` — controller query RPC design.
- **`iris_endpoint_proxy/`** — the `/proxy/<name>/` reverse proxy that exposes a
  dashboard through the controller. Key if the service runs as an Iris job.
- Dashboard UI framework decisions: `20260128_iris_dashboard_analyze.md`,
  `20260128_iris_preact_decision.md`, `20260311_iris_vue_refactor.md`.

## 6. v6e machine specifics

- TPU topology: `lib/iris/src/iris/cluster/tpu_topology.py:62-70` — `v6e-1`(1
  chip/host), `v6e-4`(4/host), `v6e-8`(8 chips/host = single host on GCP), `v6e-16`…
- "Whole machine" = `ResourceSpec(cpu=<all>, memory=<all>, device=tpu_device("v6e-8"))`
  via `--tpu v6e-8 --cpu N --memory NGB --enable-extra-resources`.
- **No hardcoded GCP v6e host vCPU/RAM constant in the repo** — host CPU/RAM
  discovered at runtime by `TaskResources.from_environment()`. GCP `ct6e-standard-8t`
  (v6e-8 host) is ~180 vCPU / ~1.4 TB RAM, per GCP docs not the repo.

## Flags / surprises

- **DuckDB ≠ finelog's engine.** finelog is now Rust + DataFusion; only live DuckDB
  is a deploy-CLI helper. DuckDB-in-Python here is **built fresh**, not reused.
- **finelog is not Iris-hosted today** — GCP VM / k8s. "Iris-hosted finelog-like
  service via `iris job run`" is a new deployment shape; endpoint-proxy makes the
  dashboard reachable.
- **No repo constant for v6e host CPU/RAM** — "all CPUs/RAM" is runtime-discovered.
- **TPU jobs need `--enable-extra-resources`** + CPU floor applies.
- **DuckDB does not use the TPU.** Requesting a v6e host buys its large CPU/RAM
  envelope; the TPU chips sit idle. Worth an explicit note in the design.
