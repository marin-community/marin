# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Snowball family tree applet from its reviewed graph and frontend."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"

DIST.mkdir(exist_ok=True)
data = json.loads((ROOT / "graph-data.json").read_text())
serialized = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
html = (ROOT / "index.html").read_text().replace("__DATA__", serialized)
(DIST / "index.html").write_text(html)
for asset in ["app.js", "style.css"]:
    shutil.copyfile(ROOT / asset, DIST / asset)
(DIST / "evidence").mkdir(exist_ok=True)
for evidence in (ROOT / "evidence").glob("*.json"):
    shutil.copyfile(evidence, DIST / "evidence" / evidence.name)
