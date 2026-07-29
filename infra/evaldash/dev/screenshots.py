# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Boot the local dashboard, screenshot each view with Playwright, then shut down.

A single foreground process: uvicorn runs in a background thread bound to loopback, Playwright drives
a headless Chromium against it, and the process exits when done (no detached server to reap). Use it
to eyeball the SPA end-to-end without a standing server.

    PYTHONPATH=infra/evaldash/src:lib/marin/src:lib/rigging/src \
    uv run --with starlette --with uvicorn --with sqlalchemy --with pyarrow --with pydantic \
      --with fsspec --with playwright \
      python infra/evaldash/dev/screenshots.py <out_dir> [records_dir]

``records_dir`` defaults to a freshly generated fixture set. The built SPA must exist
(``npm --prefix infra/evaldash/dashboard run build``); its dist is read from EVALDASH_DASHBOARD_DIST
or the repo layout.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
from pathlib import Path

os.environ.setdefault("EVALDASH_STORE", "local")

import fixtures
import server
import uvicorn
from marin.evaluation.records import list_records
from playwright.sync_api import sync_playwright

HOST = "127.0.0.1"
PORT = int(os.environ.get("EVALDASH_SCREENSHOT_PORT", "8137"))
VIEWPORT = {"width": 1480, "height": 1000}

# (filename, path, wait-for-selector-or-text). Paths are SPA routes; the server's catch-all serves
# index.html for each so deep links render. Selectors gate the screenshot on the view being drawn.
SHOTS = [
    ("01-leaderboard.png", "/", "text=Leaderboard"),
    ("02-runs.png", "/runs", "text=Runs"),
    ("03-run-detail.png", "/runs/snowball-2026.07.20-mmlu", "text=Metrics"),
    ("04-sample-mcq.png", "/runs/snowball-2026.07.20-mmlu/samples?task=mmlu&i=1", "text=Choices"),
    ("05-sample-agentic.png", "/runs/snowball-2026.07.20-aime/samples?task=aime&i=0", "text=Trajectory"),
    ("06-run-failed.png", "/runs/tootsie-8b-2026.07.20-aime", "text=Error"),
    ("07-debug.png", "/debug", "text=Prefixes scanned"),
]


def _serve(app) -> tuple[uvicorn.Server, threading.Thread]:
    config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning", lifespan="on")
    uv_server = uvicorn.Server(config)
    thread = threading.Thread(target=uv_server.run, daemon=True)
    thread.start()
    for _ in range(100):
        if uv_server.started:
            break
        time.sleep(0.1)
    return uv_server, thread


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/evaldash-shots")
    out_dir.mkdir(parents=True, exist_ok=True)

    records_dir = sys.argv[2] if len(sys.argv) > 2 else None
    if not records_dir:
        records_dir = tempfile.mkdtemp(prefix="evaldash-fixtures-")
        fixtures.build_fixtures(records_dir)
    os.environ["RECORDS_PREFIXES"] = records_dir

    store = server.MemoryRecordStore()
    store.refresh(list_records(records_dir))
    app = server.create_app(store, server._dashboard_dist(), server.NullClusterGateway())

    uv_server, thread = _serve(app)
    base = f"http://{HOST}:{PORT}"
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for scheme in ("light", "dark"):
                context = browser.new_context(viewport=VIEWPORT, color_scheme=scheme)
                page = context.new_page()
                for name, path, wait_for in SHOTS:
                    if scheme == "dark" and not name.startswith(("01", "03", "05")):
                        continue  # a representative subset in dark; full set in light
                    page.goto(f"{base}{path}", wait_until="networkidle")
                    try:
                        page.wait_for_selector(wait_for, timeout=4000)
                    except Exception:
                        print(f"  warn: selector {wait_for!r} not found for {path}")
                    # The theme is a `.dark` class on <html> (keyed CSS variables); set it directly.
                    page.evaluate(f"document.documentElement.classList.toggle('dark', {str(scheme == 'dark').lower()})")
                    page.wait_for_timeout(300)
                    dest = out_dir / (name if scheme == "light" else name.replace(".png", "-dark.png"))
                    page.screenshot(path=str(dest), full_page=True)
                    print(f"  wrote {dest}")
                context.close()
            browser.close()
    finally:
        uv_server.should_exit = True
        thread.join(timeout=5)
    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
