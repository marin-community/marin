# buoy POC

A single-file prototype (`buoy_poc.py`) that validates the [buoy design](../design.md)
end to end against a real wandb run. **Not the real service** — it mirrors to a local
dir instead of GCS, mirrors synchronously, and skips auth/registration. It faithfully
exercises the parts the design was unsure about:

- pull a wandb run's history (`scan_history`) + config + the `jax_profile` artifact
- normalize heterogeneous history rows into one parquet frame
- plot metrics with plotly
- launch `xprof` per run and **reverse-proxy it into an iframe behind a path prefix**
  (the design's central risk — confirmed working with a dumb forwarding proxy, no
  `--path_prefix`, because xprof self-prefixes via a JS-computed `<base href>`)

## Run it

Needs the repo `.venv` (wandb, starlette, uvicorn, httpx, pandas, pyarrow, plotly) plus
the `xprof` binary. `xprof` pulls its own deps, so install it into a scratch venv and
point the POC at it:

```bash
uv venv /tmp/xprof-venv --python 3.11
uv pip install --python /tmp/xprof-venv/bin/python xprof

export WANDB_API_KEY=...                      # a key that can read the entity below
export BUOY_XPROF_BIN=/tmp/xprof-venv/bin/xprof
.venv/bin/python .agents/projects/buoy/poc/buoy_poc.py   # -> http://127.0.0.1:8800
```

Open http://127.0.0.1:8800 and load the canonical run with a profile:

- entity `marin-community`, project `marin_moe`
- run `GM2560-MAY-D2560-B8-R1-E8M1-PALLASCEV8192-RING-FA4SGD-XENTAB-N1-cw-20260627-021250`

The page mirrors the run (~288 MB profile download, a few seconds), then shows the
metric plots and the embedded xprof profile UI.

## Env vars

| var | meaning |
|---|---|
| `WANDB_API_KEY` | required; wandb auth |
| `BUOY_XPROF_BIN` | path to the `xprof` executable (else taken from `PATH`) |
| `BUOY_POC_CACHE` | local cache dir (default `/tmp/buoy-poc-cache`) |

## Known POC shortcuts (handled properly in the real design/spec)

- local-dir cache, not GCS `marin_temp_bucket`
- synchronous mirror (real one is async + a `mirror_status` poll, to fit the 30s proxy timeout)
- no per-run lock / single-replica story, no LRU eviction of xprof subprocesses
- mirrors only the profile + config (not all artifacts/files)
- xprof's trace viewer needs the browser to reach `ui.perfetto.dev` + `gstatic.com`
