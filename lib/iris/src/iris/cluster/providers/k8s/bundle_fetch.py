# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone bundle fetch script for Kubernetes init containers.

Downloads and extracts a bundle zip from the Iris controller, with
retry logic and zip-slip protection. Uses only stdlib so the init
container needs no extra dependencies.
"""

import hashlib
import os
import shutil
import sys
import time
import urllib.request
import zipfile


def fetch_bundle(controller_url: str, bundle_id: str, workdir: str) -> None:
    """Download a bundle zip from the controller and extract it into workdir."""
    url = f"{controller_url}/bundles/{bundle_id}.zip"
    zip_path = os.path.join(workdir, ".bundle.zip")

    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
                sha_header = resp.getheader("X-Bundle-SHA256")

            if sha_header:
                actual = hashlib.sha256(data).hexdigest()
                if actual != sha_header:
                    raise ValueError(f"SHA-256 mismatch: expected {sha_header}, got {actual}")

            with open(zip_path, "wb") as f:
                f.write(data)
            break
        except Exception as e:
            if attempt == 2:
                raise
            wait = 2**attempt
            print(f"Bundle fetch attempt {attempt + 1} failed: {e}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)

    workdir_norm = os.path.normpath(workdir)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = os.path.normpath(os.path.join(workdir, info.filename))
            if not target.startswith(workdir_norm + os.sep) and target != workdir_norm:
                raise ValueError(f"Zip-slip detected: {info.filename}")
            zf.extract(info, workdir)

    os.remove(zip_path)


def copy_workdir_files(src_dir: str, workdir: str) -> None:
    """Copy staged workdir files from ConfigMap mount into workdir."""
    if not os.path.isdir(src_dir):
        return
    for root, _dirs, files in os.walk(src_dir):
        for fname in files:
            src_path = os.path.join(root, fname)
            rel = os.path.relpath(src_path, src_dir)
            dst_path = os.path.join(workdir, rel)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)


if __name__ == "__main__":
    workdir = os.environ.get("IRIS_WORKDIR", "/app")
    os.makedirs(workdir, exist_ok=True)

    bundle_id = os.environ.get("IRIS_BUNDLE_ID", "")
    controller_url = os.environ.get("IRIS_CONTROLLER_URL", "")
    if bundle_id and controller_url:
        fetch_bundle(controller_url, bundle_id, workdir)

    src_dir = os.environ.get("IRIS_WORKDIR_FILES_SRC", "")
    if src_dir:
        copy_workdir_files(src_dir, workdir)

    print("Workdir staging complete.")
