# buoy Agent Notes

A web viewer for wandb runs, hosted as an Iris job. wandb is the source of
truth; buoy mirrors a run's metrics/config/profile into a refetchable GCS cache
on demand and serves metric plots + the embedded xprof UI behind the controller
proxy. Start with the shared `/AGENTS.md`; buoy-specific notes below.

Design + history: `.agents/projects/buoy/design.md` and the validated single-file
prototype in `.agents/projects/buoy/poc/`.

## Layout

- `src/buoy/config.py` — `BuoyConfig.from_env()`: cache root, xprof bin, caps.
- `src/buoy/cache.py` — fsspec-addressed run cache (GCS or local). `manifest.json`
  is written LAST as the commit marker. All fsspec access is funneled here.
- `src/buoy/mirror.py` — `mirror_run` (streams history → unified-schema parquet
  shards, mirrors config/summary/profile) and `MirrorManager` (background +
  per-run coalescing + pollable status).
- `src/buoy/xprof.py` — per-run xprof subprocess lifecycle (lazy launch from a
  LOCAL logdir copy, last-request eviction, cap → 503).
- `src/buoy/app.py` — Starlette JSON API + xprof reverse proxy + `/wrap` frame +
  static SPA.
- `src/buoy/serve.py` — Iris job entrypoint (install xprof, serve, register).
- `src/buoy/launch.py` — `buoy` CLI: submit the service job to a cluster.
- `src/buoy/static/` — no-build SPA (plotly.js via CDN).

## Hard requirements (learned from the POC, enforced here)

- **Async mirror.** The controller proxy caps requests at 30s; a cold mirror of
  a large run exceeds that. `POST /api/mirror` returns 202; the SPA polls
  `/api/mirror_status`. Never mirror synchronously inside a request.
- **Memory-bounded history.** A run can be ~10^5 steps × ~400 keys. Stream
  `scan_history` in pages to parquet shards; never materialize it whole.
- **`/tmp` is noexec** in the task container — install the xprof venv under the
  workdir (`serve.install_xprof`), not `/tmp`.
- **Iframe isolation.** xprof writes `window.parent` history/location; embed it
  via `/wrap` (same-origin throwaway frame) so it can't clobber the SPA URL.

## Dev

```bash
cd lib/buoy && uv run --group dev python -m pytest -q   # local cache via tmp_path; no GCS/wandb
buoy --cluster marin                                    # submit the service job
```
