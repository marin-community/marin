#!/usr/bin/env python
"""buoy POC — a single-file prototype that validates the buoy design end to end.

This is NOT the real service. It cuts every corner that doesn't affect the design
questions: it mirrors to a LOCAL dir instead of GCS, mirrors synchronously instead of
via a background job, and skips auth/registration. What it faithfully exercises:

  * pull a wandb run's history (scan_history) + config + the jax_profile artifact
  * normalize heterogeneous history rows into one parquet frame
  * plot metrics with plotly
  * launch `xprof` per run and reverse-proxy it so the real xprof UI embeds in an
    iframe behind a path prefix (the design's central risk)

Run:
    pip install xprof            # into some venv; or use the repo .venv + a scratch venv
    export WANDB_API_KEY=...
    # if xprof isn't on PATH, point at it explicitly:
    #   export BUOY_XPROF_BIN=/path/to/venv/bin/xprof
    python buoy_poc.py           # serves http://127.0.0.1:8800

Then open http://127.0.0.1:8800 and load e.g.
    entity=marin-community project=marin_moe
    run=GM2560-MAY-D2560-B8-R1-E8M1-PALLASCEV8192-RING-FA4SGD-XENTAB-N1-cw-20260627-021250
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass, field

import httpx
import pandas as pd
import plotly.graph_objects as go
import pyarrow as pa
import pyarrow.parquet as pq
import uvicorn
import wandb
from plotly.io import to_html
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

CACHE_ROOT = os.environ.get("BUOY_POC_CACHE", "/tmp/buoy-poc-cache")
XPROF_BIN = os.environ.get("BUOY_XPROF_BIN") or shutil.which("xprof")
PROFILE_ARTIFACT_TYPE = "jax_profile"
# metrics surfaced in the run view (those present are plotted, one chart each)
PLOT_METRICS = [
    "train/cross_entropy_loss",
    "train/loss",
    "optim/learning_rate",
    "throughput/hook_time",
    "throughput/loading_time",
    "run_progress",
]


# --------------------------------------------------------------------------- mirror

def run_key(entity: str, project: str, run_id: str) -> str:
    return f"{entity}/{project}/{run_id}"


def run_root(entity: str, project: str, run_id: str) -> str:
    return os.path.join(CACHE_ROOT, entity, project, run_id)


def normalize_history(rows: list[dict]) -> pd.DataFrame:
    """Union heterogeneous history rows; keep _step + numeric scalar columns only."""
    df = pd.DataFrame(rows)
    if "_step" not in df.columns:
        df["_step"] = range(len(df))
    keep = ["_step"]
    for c in df.columns:
        if c == "_step" or (c.startswith("_") and c not in ("_runtime", "_timestamp")):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        if s.notna().any():
            df[c] = s
            keep.append(c)
    out = df[keep].copy()
    out["_step"] = pd.to_numeric(out["_step"], errors="coerce")
    return out.dropna(subset=["_step"]).sort_values("_step").reset_index(drop=True)


def mirror_run(entity: str, project: str, run_id: str, *, refresh: bool = False) -> dict:
    """Synchronously mirror a run into the local cache; return its manifest dict.

    Mirrors history.parquet + config.json + the jax_profile artifact, writing
    manifest.json LAST as the commit marker (presence == fully cached).
    """
    root = run_root(entity, project, run_id)
    manifest_path = os.path.join(root, "manifest.json")
    if os.path.exists(manifest_path) and not refresh:
        with open(manifest_path) as f:
            m = json.load(f)
        if m.get("state") != "running":
            return m

    os.makedirs(root, exist_ok=True)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")

    # history: try parquet exports, else scan_history (the common path)
    history_source = "scan"
    rows: list[dict] = []
    try:
        res = run.download_history_exports(os.path.join(root, "hist"), require_complete_history=False)
        if getattr(res, "paths", None):
            history_source = "exports"
            for p in res.paths:
                rows.extend(pq.read_table(p).to_pylist())
    except Exception:  # noqa: BLE001 - POC: any export issue just falls through to scan
        pass
    if not rows:
        rows = list(run.scan_history())
    hist = normalize_history(rows)
    pq.write_table(pa.Table.from_pandas(hist, preserve_index=False), os.path.join(root, "history.parquet"))

    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(dict(run.config), f, default=str)

    # profile: the jax_profile artifact root is already an xprof logdir
    profile = None
    for art in run.logged_artifacts():
        if art.type == PROFILE_ARTIFACT_TYPE:
            d = art.download(root=os.path.join(root, "artifacts", art.name.replace(":", "_")))
            profile = {"artifact_name": art.name, "logdir": d, "size_bytes": int(getattr(art, "size", 0) or 0)}
            break

    manifest = {
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "display_name": run.name,
        "state": run.state,
        "url": run.url,
        "mirrored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metric_keys": [c for c in hist.columns if c != "_step"],
        "history_source": history_source,
        "profile": profile,
    }
    with open(manifest_path, "w") as f:  # written LAST = commit marker
        json.dump(manifest, f)
    return manifest


# ----------------------------------------------------------------------------- xprof

@dataclass
class _Xprof:
    proc: subprocess.Popen
    port: int
    last_used: float


@dataclass
class XprofManager:
    procs: dict[str, _Xprof] = field(default_factory=dict)

    def ensure(self, key: str, logdir: str) -> int:
        x = self.procs.get(key)
        if x and x.proc.poll() is None:
            x.last_used = time.time()
            return x.port
        if not XPROF_BIN:
            raise RuntimeError("xprof binary not found; set BUOY_XPROF_BIN or `pip install xprof`")
        port = _free_port()
        proc = subprocess.Popen(
            [XPROF_BIN, "--logdir", logdir, "--port", str(port)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        self.procs[key] = _Xprof(proc, port, time.time())
        _wait_port(port)
        return port


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.2)
    raise RuntimeError(f"xprof did not bind on :{port}")


XPROF = XprofManager()


# ------------------------------------------------------------------------------ app

def _plots_html(root: str, manifest: dict) -> str:
    hist = pq.read_table(os.path.join(root, "history.parquet")).to_pandas()
    blocks, first = [], True
    for key in PLOT_METRICS:
        if key not in hist.columns:
            continue
        s = hist[["_step", key]].dropna()
        if s.empty:
            continue
        fig = go.Figure(go.Scatter(x=s["_step"], y=s[key], mode="lines+markers", name=key))
        fig.update_layout(title=key, height=300, margin=dict(l=40, r=20, t=40, b=30), template="plotly_white")
        blocks.append(to_html(fig, include_plotlyjs="cdn" if first else False, full_html=False))
        first = False
    if not blocks:
        return f"<p>no plottable metrics among {PLOT_METRICS} ({len(manifest['metric_keys'])} total)</p>"
    return "".join(blocks)


async def index(request: Request) -> HTMLResponse:
    e = request.query_params.get("entity", "marin-community")
    p = request.query_params.get("project", "marin_moe")
    r = request.query_params.get("run", "")
    form = f"""
    <form method=get style="font-family:sans-serif;margin:1em">
      entity <input name=entity value="{e}" size=20>
      project <input name=project value="{p}" size=14>
      run <input name=run value="{r}" size=70>
      <button>load</button>
    </form>"""
    if not r:
        return HTMLResponse(f"<h1>buoy POC</h1>{form}<p>enter a run to mirror + view</p>")

    manifest = await asyncio.to_thread(mirror_run, e, p, r)
    root = run_root(e, p, r)
    plots = _plots_html(root, manifest)
    prof_html = "<p><i>no jax_profile artifact on this run</i></p>"
    if manifest.get("profile"):
        src = f"/xprof/{e}/{p}/{r}/data/plugin/profile/"
        prof_html = f'<iframe src="{src}" style="width:100%;height:80vh;border:1px solid #ccc"></iframe>'
    return HTMLResponse(f"""
    <h1>buoy POC — {manifest['display_name']}</h1>{form}
    <p>state=<b>{manifest['state']}</b> · history_source=<b>{manifest['history_source']}</b> ·
       {len(manifest['metric_keys'])} metrics · <a href="{manifest['url']}">wandb</a></p>
    <h2>metrics</h2>{plots}
    <h2>xprof profile</h2>{prof_html}
    """)


async def xprof_proxy(request: Request) -> Response:
    e, p, r = request.path_params["entity"], request.path_params["project"], request.path_params["run_id"]
    sub = request.path_params["sub"]
    manifest_path = os.path.join(run_root(e, p, r), "manifest.json")
    if not os.path.exists(manifest_path):
        return Response("not mirrored", status_code=404)
    with open(manifest_path) as f:
        manifest = json.load(f)
    if not manifest.get("profile"):
        return Response("run has no profile", status_code=409)
    port = await asyncio.to_thread(XPROF.ensure, run_key(e, p, r), manifest["profile"]["logdir"])
    url = "/" + sub + (("?" + request.url.query) if request.url.query else "")
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as client:
        upstream = await client.get(url)
    drop = {"content-encoding", "content-length", "transfer-encoding"}
    headers = {k: v for k, v in upstream.headers.items() if k.lower() not in drop}
    return Response(upstream.content, status_code=upstream.status_code, headers=headers,
                    media_type=upstream.headers.get("content-type"))


async def api_mirror(request: Request) -> JSONResponse:
    body = await request.json()
    m = await asyncio.to_thread(mirror_run, body["entity"], body["project"], body["run_id"],
                                refresh=body.get("refresh", False))
    return JSONResponse(m)


app = Starlette(routes=[
    Route("/", index),
    Route("/api/mirror", api_mirror, methods=["POST"]),
    Route("/xprof/{entity}/{project}/{run_id}/{sub:path}", xprof_proxy),
])


if __name__ == "__main__":
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("set WANDB_API_KEY")
    print(f"buoy POC on http://127.0.0.1:8800  (cache={CACHE_ROOT}, xprof={XPROF_BIN})")
    uvicorn.run(app, host="127.0.0.1", port=8800)
