# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone bundle fetch script for Kubernetes init containers.

Downloads and extracts a bundle zip from the Iris controller, with
retry logic and zip-slip protection. Also fetches externalized workdir
files (blob refs) from the controller's blob endpoint. Uses only stdlib
so the init container needs no extra dependencies.
"""

import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import zipfile


def _download_verified(url: str, expected_sha: str | None, label: str) -> bytes:
    """Download bytes from ``url`` with retries and optional SHA-256 verification.

    Retries up to three times with exponential backoff. When ``expected_sha`` is
    given it is compared against the payload; otherwise the controller's
    ``X-Bundle-SHA256`` response header is used when present. Re-raises the last
    error if every attempt fails.
    """
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = resp.read()
                sha = expected_sha if expected_sha is not None else resp.getheader("X-Bundle-SHA256")
            if sha:
                actual = hashlib.sha256(data).hexdigest()
                if actual != sha:
                    raise ValueError(f"SHA-256 mismatch for {label}: expected {sha}, got {actual}")
            return data
        except Exception as e:
            if attempt == 2:
                raise
            wait = 2**attempt
            print(f"{label} attempt {attempt + 1} failed: {e}; retrying in {wait}s", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"unreachable: {label} download loop exited without result")


def fetch_bundle(controller_url: str, bundle_id: str, workdir: str) -> None:
    """Download a bundle zip from the controller and extract it into workdir."""
    url = f"{controller_url}/bundles/{bundle_id}.zip"
    zip_path = os.path.join(workdir, ".bundle.zip")

    data = _download_verified(url, expected_sha=None, label=f"Bundle fetch {bundle_id}")
    with open(zip_path, "wb") as f:
        f.write(data)

    workdir_norm = os.path.normpath(workdir)
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            target = os.path.normpath(os.path.join(workdir, info.filename))
            if not target.startswith(workdir_norm + os.sep) and target != workdir_norm:
                raise ValueError(f"Zip-slip detected: {info.filename}")
            zf.extract(info, workdir)

    os.remove(zip_path)


def fetch_workdir_blob_refs(controller_url: str, blob_refs_json: str, workdir: str) -> None:
    """Download externalized workdir files from the controller's blob endpoint."""
    refs = json.loads(blob_refs_json)
    workdir_norm = os.path.normpath(workdir)
    for name, blob_id in refs.items():
        url = f"{controller_url}/blobs/{blob_id}"
        dst_path = os.path.normpath(os.path.join(workdir, name))
        # Contain writes to workdir: reject ``../`` escapes and absolute keys
        # that ``os.path.join`` would otherwise honor and drop ``workdir``.
        if not dst_path.startswith(workdir_norm + os.sep) and dst_path != workdir_norm:
            raise ValueError(f"Path traversal in workdir blob ref: {name}")
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)

        data = _download_verified(url, expected_sha=blob_id, label=f"Blob fetch {name}")
        with open(dst_path, "wb") as f:
            f.write(data)
        print(f"Fetched blob ref {name} ({len(data)} bytes) from {blob_id[:12]}")


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

    blob_refs_json = os.environ.get("IRIS_WORKDIR_BLOB_REFS", "")
    if blob_refs_json and controller_url:
        fetch_workdir_blob_refs(controller_url, blob_refs_json, workdir)

    print("Workdir staging complete.")
