# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Belay pages: copy the frontend into ``dist/``. ``web/static.html`` is only for ``demo export-public``."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"

if DIST.exists():
    shutil.rmtree(DIST)
DIST.mkdir()
for name in (
    "index.html",
    "boot.js",
    "common.js",
    "front.js",
    "status.js",
    "mfu.js",
    "mfu.json",
    "scaling.json",
    "speedrun.json",
    "admin.js",
    "fonts.css",
    "belay.css",
):
    shutil.copyfile(ROOT / "web" / name, DIST / name)
