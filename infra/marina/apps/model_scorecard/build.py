# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package the scorecard: embed the checked-in scorecard.json into the page and copy assets."""

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
ASSETS = ["app.js", "style.css"]

data = json.loads((ROOT / "scorecard.json").read_text())
for model in data["models"]:
    if list(model["cells"]) != data["evals"]:
        raise ValueError(f"{model['name']} cells do not match the eval columns; rerun build_data.py")

if DIST.exists():
    shutil.rmtree(DIST)
DIST.mkdir()
serialized = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
(DIST / "index.html").write_text((ROOT / "index.html").read_text().replace("__DATA__", serialized))
for asset in ASSETS:
    shutil.copyfile(ROOT / asset, DIST / asset)
