# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the Belay pages: copy the frontend into ``dist/``.

Every asset URL in the HTML carries ``?v=__BELAY_VERSION__``; the copy replaces it with a digest of the frontend, so
a browser never pairs a cached script with newer JSON after a release. ``common.js`` reads the same version from the
``belay-version`` meta tag for the files it fetches.
"""

import hashlib
import shutil
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DIST = ROOT / "dist"
VERSION_MARK = "__BELAY_VERSION__"
ROOT_MARK = "__BELAY_ROOT__"
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
    for name in sorted((*PAGE_FILES, "index.html")):
        digest.update(name.encode())
        digest.update((WEB / name).read_bytes())
    return digest.hexdigest()[:12]


def applet_root() -> str:
    """The path Marina serves the app under: ``/<name>/`` for a named app, else ``/a/<applet id>/``. Asset URLs are
    absolute under it, so a page opened as ``/status/`` (trailing slash) still loads them."""
    manifest = tomllib.loads((ROOT / "applet.toml").read_text())
    return f"/{manifest['name']}/" if manifest.get("name") else f"/a/{manifest['applet_id']}/"


def write_page(source: Path, target: Path, version: str, root: str) -> None:
    text = source.read_text()
    assert VERSION_MARK in text and ROOT_MARK in text, f"{source.name} has no {ROOT_MARK}/{VERSION_MARK} asset URLs"
    target.write_text(text.replace(VERSION_MARK, version).replace(ROOT_MARK, root))


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    for name in PAGE_FILES:
        shutil.copyfile(WEB / name, DIST / name)
    write_page(WEB / "index.html", DIST / "index.html", web_version(), applet_root())


if __name__ == "__main__":
    main()
