# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Package the curriculum browser's maintained static sources."""

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ("index.html", "app.js", "style.css")


def main() -> None:
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    for asset in ASSETS:
        shutil.copyfile(ROOT / asset, output / asset)


if __name__ == "__main__":
    main()
