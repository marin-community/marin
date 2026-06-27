# Research — buoy (a better viewer for wandb runs)

Persisted digest for the `buoy` design. File:line refs are pinned to the state of `main` at design time (rebased onto `05a1d4d47`).

## Framing

New Iris service that serves a web app. User selects a wandb run → service downloads **all** run data (metric history, config, artifacts, code) → stores it on GCS in a refetchable cache → app plots metrics and renders xprof profiles when present. The narrow, honest goal: **wandb's UI can't render xprof profiles inline — buoy makes that a link** (in-app metric plots are a convenience over the same cache). wandb stays the source of truth; the GCS copy is a 30-day TTL cache, not a durable archive — so this is *not* about vendor-independence or offline durability.

## 1. Iris services (long-running, HTTP-serving)

- **Serving stack = Starlette (ASGI)** + Connect RPC, single port.
  - Controller dashboard: `lib/iris/src/iris/cluster/controller/dashboard.py:40` (Starlette import), class `ControllerDashboard` at `:384-543`, route mounting `:488-516`, SPA shell serving `:545-548`. Auth + static asset helper imported from `dashboard_common` at `:57`.
  - Connect RPC routes mounted at `/iris.cluster.ControllerService/*` (`dashboard.py:513`). SPA fetches data via RPC; Python only serves the HTML shell + compiled assets.
- **Entrypoint convention** (from iris Dockerfile): controller `iris.cluster.controller.main serve --host 0.0.0.0 --port 10000`; worker `iris.cluster.worker.main serve --host ... --port 10001`.
- **Job/service submission**: `iris job run [options] -- [entrypoint]`. Launcher pattern (controller resolution + job launch) in `lib/marin/src/marin/inference/quick_serve_cli.py:80-97`; in-job service boot in `lib/marin/src/marin/inference/quick_serve.py:24` (`iris_ctx` usage).
- **Service architecture / endpoint registration**: `docs/design/marin-service-architecture.md` (§Resolution, §System services). Services register as Iris endpoints, reachable via controller proxy (no SSH tunnel).

## 2. Existing web apps / dashboards (prior art to copy)

- **Iris controller dashboard** — Vue.js SPA at `lib/iris/dashboard/`, built via `npm run build` → `./dist`, served by Starlette. Reference for auth/routing/static-mount + RPC coexistence.
- **Finelog server** — Vue.js SPA at `lib/finelog/dashboard/`, served by a **Rust** binary (`finelog-server`, `spa.rs`), port 10001. Same Vue + Connect-RPC shape, different backend language.
- **Takeaway**: standard = Starlette + Vue.js SPA + Connect RPC. For a Python service, copy the controller dashboard shape.

## 3. wandb usage today

- **Entity**: repo scripts use `marin-community` (e.g. `scripts/training/time_to_train/parse_wandb_runs.py:11`). **User-provided creds use `stanford-mercury`** — needs reconciliation (open question).
- **wandb API patterns already in repo**:
  - `wandb.Api()` ctor: `scripts/training/time_to_train/parse_wandb_runs.py:115`.
  - `run.history(keys=[...])` `:87`, `run.summary` `:92`, `run.config` `:84`, `run.createdAt`/`run.heartbeatAt` `:78,:81`.
  - Artifact fetch: `api.artifact(ref, type=...)` in `lib/marin/src/marin/profiling/ingest.py:72-78`.
  - **Existing profile downloaders**: `download_wandb_profile_artifact()` `ingest.py:54-79`; `download_profile_dir_for_run()` `ingest.py:82-100` (downloads entire run dirs). This is the closest existing "download from wandb → local/GCS" code — reuse/generalize it.
- **Full-fidelity history download** (wandb public API, user-pointed): `Run.download_history_exports()` downloads complete metric history as **parquet** files (`DownloadHistoryResult`); raises `IncompleteRunHistoryError` if wandb hasn't exported yet. `Run.scan_history()` iterates **all** un-sampled rows (vs `history()` which samples ~500 points — too lossy for a faithful mirror). Mirror strategy: prefer `download_history_exports()`, fall back to `scan_history()`. Files/code via `run.files()` (paginated; includes uploaded source/`wandb-metadata.json`); `logged_artifacts()` / `used_artifacts()` for artifacts; `config`/`summary`/`metadata` properties for the rest. Ref: https://docs.wandb.ai/models/ref/python/public-api/run
- **Levanter tracker** (the writer side): base `lib/levanter/src/levanter/tracker/tracker.py:17-85`; `WandbTracker` `lib/levanter/src/levanter/tracker/wandb.py:40-79` (logs metrics, hyperparams, artifacts).

## 4. xprof / profiling

- Module `lib/marin/src/marin/profiling/`:
  - `xplane.py:84-96` — locate `.xplane.pb` in artifacts; `:99-120` — export xprof tables (overview_page, kernel_stats, op_profile, memory_profile, …).
  - `ingest.py:54-79` — download profile from wandb with metadata.
  - Classes: `XPlaneTimeline` `xplane.py:60-68`, `XPlaneTableExport` `xplane.py:71-77`, `ProfileSummary` (from schema).
  - CLI: `cli.py:33-68` `summarize` command (`--xplane-file`, `--profile-dir`, `--run-target`).
- **No existing xprof *serving* code.** Profiles today are analyzed via CLI or downloaded as artifacts. Rendering xprof in a browser is net-new (open question on approach: TensorBoard profile plugin vs. pre-exported tables vs. embedding).

## 5. GCS conventions

- Path helpers in `lib/rigging/src/rigging/filesystem.py`:
  - `marin_prefix()` → `data_config().resolved_root()`.
  - `marin_temp_bucket(ttl_days, prefix)` → e.g. `gs://marin-us-central1/tmp/ttl=14d/{prefix}`.
- Regional buckets config-driven via `data_config().region_buckets` (`marin-us-central1`, `marin-eu-west4`, `marin-us-east1`, …).
- Access: `gcsfs.GCSFileSystem()` (`scripts/datakit/generate_tier2_skewed.py:57`); fsspec `url_to_fs` (`scripts/ops/storage/render_report.py`).
- Observed artifact layout: `gs://marin-{region}/data/{category}/{run-id}/`.

## 6. Plotting

- **Plotly is the standard** for interactive charts: `lib/marin/src/marin/scaling_laws/scaling_plots.py:17-27` (graph_objects + `plotly.io`, `pio.templates.default = "plotly_white"`), `create_isoflop_plot()` `:72`. Figures serializable to HTML/JSON for embedding.
- Matplotlib present but mostly for static/design-doc plots (`docs/design/plot_plateau_detection.py:1`).

## 7. Service registration with Iris (how an independent service is reached via the controller proxy)

Two registration paths; **finelog is the example to copy**, the controller dashboard is *not* (it's built into the controller process, special-cased).

- **Cluster-config / system-endpoint model (finelog).** Deploy as a plain k8s `ClusterIP` Service (`lib/finelog/deploy/k8s/03-service.yaml.tmpl`) + Deployment. The cluster config carries an `endpoints: dict[str, EndpointSpec]` map (`lib/iris/src/iris/cluster/config.py:582`); the controller resolves each `uri` (e.g. `k8s://buoy.iris` + `{"port": ...}`) at startup (`controller/main.py:43-67`, `endpoints.py:_resolve_k8s :161-185`) and registers it into `_system_endpoints` (`controller/controller.py:512-516`). Reached through the endpoint proxy at `/proxy/<name-with-dots>/...` — finelog's `/system/log-server` → `https://iris.oa.dev/proxy/system.log-server/` (proxy replaces `/`→`.`; `lib/finelog/OPS.md:3-32`, `controller/endpoint_proxy.py:1-50`). Persistent, always-on.
- **Runtime task-registration model (quick_serve).** A job entrypoint calls `ctx.registry.register(name, address, metadata)` (`NamespacedEndpointRegistry`, `lib/iris/src/iris/client/client.py:342-389`) → `RegisterEndpoint` RPC (`rpc/controller.proto:359-370`, handler `controller/service.py:1903-1951`). Worked example: `lib/marin/src/marin/inference/quick_serve.py:209-303` (`iris_ctx()`, `ctx.get_port`, `ctx.registry.register/unregister`, `proxy_path(...)`). Reached at `/proxy/{namespace}.{name}/`. Lighter — no cluster-config edit or k8s manifests — but tied to a job's lifetime.

For buoy (a persistent shared viewer) the finelog/system-endpoint model fits best; the quick_serve model is the lighter alternative if we want it job-scoped.

## Canonical example / test fixture (run with a profile)

- Run: `GM2560-MAY-D2560-B8-R1-E8M1-PALLASCEV8192-RING-FA4SGD-XENTAB-N1-cw-20260627-021250`, state `finished`, project `marin-community/marin_moe`.
  URL: https://wandb.ai/marin-community/marin_moe/runs/GM2560-MAY-D2560-B8-R1-E8M1-PALLASCEV8192-RING-FA4SGD-XENTAB-N1-cw-20260627-021250
- **Profile is a logged wandb _artifact_, not a loose run file.** Artifact `…-profiler:v0`, **type `jax_profile`**, ~275 MB, contents `plugins/profile/<ts>/<host>.xplane.pb`.
- Implications:
  - The mirror must pull the profile via `run.logged_artifacts()` (filter `type == "jax_profile"`), not just `run.files()`.
  - `plugins/profile/<ts>/<host>.xplane.pb` **is already a TensorBoard/xprof logdir layout** — so `xprof --logdir <downloaded-artifact-root>` works with no restructuring.
  - Entity here is `marin-community`, confirming the provided key must be able to read `marin-community` (reconciles the `stanford-mercury` vs `marin-community` entity question — buoy stays entity-configurable).
  - Profiles are big (~275 MB each) → reinforces on-demand single-run fetch + local-cache-before-xprof.

## Spike — local validation (2026-06-27)

Ran a throwaway prototype against the canonical run with the provided key. All core risks validated; three design assumptions corrected.

- **Environment**: repo `.venv` has `wandb 0.26.0` (with `download_history_exports` + `scan_history`), `starlette`, `httpx`, `uvicorn`, `pyarrow`, `plotly`, `pandas`. `xprof`/`tensorboard` were **not** installed — installed `xprof==2.22.3` (standalone OpenXLA profiler) into a scratch venv.
- **wandb access**: the `stanford-mercury` key reads `marin-community/marin_moe` fine (0.2s to fetch the run). Confirms entity-configurable + key cross-entity access.
- **Profile artifact**: `…-profiler:v0` (type `jax_profile`, 288 MB) downloaded in 6.5s; tree is `plugins/profile/2026_06_27_02_16_11/{g73b892.xplane.pb, g73b892.trace.json.gz, perfetto_trace.json.gz}` — a valid xprof/TB logdir. The run also logs a **329 MB `code` artifact** → "all logged artifacts" ≈ 600 MB/run (informs artifact-scope decision).
- **History (corrected)**: `download_history_exports(dir, require_complete_history=False)` returned `paths=[], contains_live_data=True` for this *finished* run — **no parquet export exists**; wandb exports lazily. So `scan_history()` is the **primary** path in practice. Signature is `(download_dir, require_complete_history=True)` (no `root=` kwarg). Rows are highly heterogeneous (adjacent rows 1302 vs 348 keys). `scan_history` → pandas union (1315 cols) → numeric coercion → **1286 scalar metric columns** → `history.parquet`; per-metric series extract cleanly with null gaps (`optim/learning_rate` starts at step 2). This run logged only 12 steps.
- **xprof behind proxy (the central risk) — PASSES, premise corrected**: standalone `xprof` has **no `--path_prefix`** flag. It doesn't need one: `index.html` uses relative asset URLs + a JS `<base href>` computed from `document.location.pathname` (splits on `/data/plugin/profile`). A dumb forwarding Starlette reverse proxy serving xprof under `/proxy/system.buoy/xprof/{run_key}/…` (no body rewriting) returned **200 for every computed URL** (index, `styles.css`, `bundle.js`, `zone.js`, `runs` JSON → `["2026_06_27_02_16_11"]`, nested `data/plugin/profile/hosts`). Requirement: iframe `src` must target `…/data/plugin/profile/` so base-href triggers. **Caveat**: xprof's trace viewer frames `https://ui.perfetto.dev` and loads charts from `gstatic.com` (per its CSP) → viewer's browser needs public internet; not fully offline. Not yet checked in a real browser: that the trace viewer + op-profile tabs visually render (the manual integration step).

## 8. Branch state

- `rav/fake-wandb` is currently at the same commit as `main` (`05a1d4d47`). **No implementation started yet.**

## Surprises / unclear

- Two web dashboards exist but one is Rust (finelog) and one Vue/Starlette (iris controller) — the Python prior art to copy is the iris controller dashboard.
- No browser xprof rendering anywhere — this is the riskiest net-new piece.
- Entity discrepancy (`marin-community` vs `stanford-mercury`).
- "Known location" decided: TTL temp bucket `marin_temp_bucket(30, "buoy")` → `gs://marin-{region}/tmp/ttl=30d/buoy/{entity}/{project}/{run_id}/`. Auto-cleans after 30d; pairs with re-fetch-on-view (stale caches re-download).

## Decisions (from interrogation)

- **Name**: **buoy**. Package lives at **`lib/buoy/`** (peer to `lib/finelog`), self-contained — depends on `iris` (registration), `rigging` (`marin_temp_bucket`), `wandb`, `starlette`, and the `xprof` binary; **does not depend on `marin`** (it talks to the wandb public API directly and runs the real xprof binary, so it doesn't need `marin.profiling`).
- **Entity**: configurable — view any entity the injected key can read (default surfaced in UI). Reconciles `marin-community` vs `stanford-mercury`.
- **GCS cache**: `marin_temp_bucket(30, "buoy")/{entity}/{project}/{run_id}/` (refetchable cache; wandb is source of truth).
- **xprof**: embed the *real* xprof UI (TensorBoard profile plugin / `xprof`) via iframe; serve the downloaded profile in whatever way is simplest (local cache or GCS logdir).
- **Trigger / scope**: on-demand, single run. User picks a run → buoy caches that run → view it.
- **Live runs**: re-fetch on view — if a cached run was still running, re-pull latest history/artifacts when opened.
- **Iris registration**: finelog-style system endpoint (`/proxy/system.buoy/`) as primary; quick_serve-style runtime `ctx.registry.register` as the lighter alternative.
