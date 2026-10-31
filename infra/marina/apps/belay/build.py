# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Belay page: copy the frontend and Marina's shared tokens and parts into ``dist/``."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
SHARED = ROOT.parents[1] / "web"

if DIST.exists():
    shutil.rmtree(DIST)
DIST.mkdir()
for name in ("index.html", "app.js", "style.css"):
    shutil.copyfile(ROOT / "web" / name, DIST / name)
for name in ("tokens.css", "parts.css"):
    shutil.copyfile(SHARED / name, DIST / name)
