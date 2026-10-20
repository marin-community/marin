# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Copy the curriculum browser's static source into its release package."""

import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
dist = root / "dist"
dist.mkdir(exist_ok=True)
for name in ("index.html", "style.css", "app.js"):
    shutil.copyfile(root / name, dist / name)
