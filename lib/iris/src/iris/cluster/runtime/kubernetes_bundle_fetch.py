# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Init-container script for staging bundles and workdir files in Kubernetes pods.

This module is read at import time by the Kubernetes runtime and embedded as an
inline Python script in the init-container command. It runs inside the task
image, so only stdlib imports are used.
"""

import io
import os
import zipfile
from pathlib import Path
from urllib.request import urlopen

bundle_id = os.environ.get("IRIS_BUNDLE_ID", "")
dest = Path(os.environ.get("IRIS_WORKDIR", "/app")).resolve()
dest.mkdir(parents=True, exist_ok=True)

if bundle_id:
    controller_url = os.environ["IRIS_CONTROLLER_URL"]
    url = f"{controller_url}/bundles/{bundle_id}.zip"
    archive = urlopen(url, timeout=120).read()
    with zipfile.ZipFile(io.BytesIO(archive), "r") as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"Zip slip detected: {member.filename}")
        zf.extractall(dest)

files_src = os.environ.get("IRIS_WORKDIR_FILES_SRC", "")
if files_src:
    src = Path(files_src)
    if src.exists():
        for path in src.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(src)
            target = (dest / rel).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"Path traversal detected: {rel}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
