# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Belay pages: copy the frontend into ``dist/``. ``web/static.html`` is only for ``demo export-public``.

Every asset URL in the HTML carries ``?v=__BELAY_VERSION__``; the copy replaces it with a digest of the frontend, so
a browser never pairs a cached script with newer JSON after a release. ``common.js`` reads the same version from the
``belay-version`` meta tag for the files it fetches.
"""

import hashlib
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DIST = ROOT / "dist"
VERSION_MARK = "__BELAY_VERSION__"
PAGE_FILES = (
    "boot.js",
    "common.js",
    "front.js",
    "status.js",
    "mfu.js",
    "mfu.json",
    "scaling.json",
    "speedrun.json",
    "tracks.json",
    "admin.js",
    "fonts.css",
    "belay.css",
)


def web_version() -> str:
    """A short digest of every frontend file, which changes whenever any of them does."""
    digest = hashlib.sha256()
    for name in sorted((*PAGE_FILES, "index.html", "static.html")):
        digest.update(name.encode())
        digest.update((WEB / name).read_bytes())
    return digest.hexdigest()[:12]


def write_page(source: Path, target: Path, version: str) -> None:
    text = source.read_text()
    assert VERSION_MARK in text, f"{source.name} has no {VERSION_MARK} asset URLs"
    target.write_text(text.replace(VERSION_MARK, version))


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    for name in PAGE_FILES:
        shutil.copyfile(WEB / name, DIST / name)
    write_page(WEB / "index.html", DIST / "index.html", web_version())


if __name__ == "__main__":
    main()
