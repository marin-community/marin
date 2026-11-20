# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-run code and data hashes with per-batch commit receipts."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from rigging.filesystem.storage_path import StoragePath

REQUIRED_PROVENANCE_FILES = (
    "experiments/glm_tool_serving/scale_production.py",
    "experiments/glm_tool_serving/ortet_production.py",
    "experiments/glm_tool_serving/provenance.py",
    "experiments/glm_tool_serving/fleet.py",
    "experiments/glm_tool_serving/assets.py",
    "experiments/tool_normalization/environments.json",
    "experiments/tool_normalization/opencode-tools.json",
    "experiments/glm_tool_serving/production_data/rule-approval-v4.json",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(app_root: Path, root: Path, catalog_sha256: str, fallback_git_head: str | None) -> dict:
    """Write root/provenance.json before the batch loop; returns the document."""
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    missing = []
    for relative in REQUIRED_PROVENANCE_FILES:
        path = app_root / relative
        if path.exists():
            files[relative] = _file_sha256(path)
        else:
            missing.append(relative)
    for pattern in ("experiments/tool_normalization/*.py", "experiments/tool_normalization/*.json"):
        for path in sorted(app_root.glob(pattern)):
            files[path.relative_to(app_root).as_posix()] = _file_sha256(path)
    head = None
    source = "unavailable"
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=app_root, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip():
            head, source = result.stdout.strip(), "git"
    except (OSError, subprocess.SubprocessError):
        pass
    if head is None and fallback_git_head:
        head, source = fallback_git_head, "env"
    document = {
        "git_head": head,
        "git_head_source": source,
        "files_sha256": files,
        "missing_required_files": missing,
        "catalog_sha256": catalog_sha256,
        "generated_at": time.time(),
        "note": (
            "per-file hashes are authoritative; git_head is a label (pod workspaces may lack .git); "
            "missing_required_files entries recorded explicitly when a required file is absent"
        ),
    }
    (root / "provenance.json").write_text(json.dumps(document, indent=2, sort_keys=True))
    return document


def emit_receipt(prefix: str, name: str, archive: Path, catalog_sha256: str, cursor: dict, provenance: dict) -> None:
    """Emit receipts/<batch>.json after the cursor commit.

    Auxiliary by design: a failed receipt write must not fail the replica after its
    batch is already committed, so failures are logged and swallowed here.
    """
    try:
        receipt = {
            "batch": name,
            "archive_uri": str(StoragePath(prefix) / "batches" / archive.name),
            "archive_sha256": _file_sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "catalog_sha256": catalog_sha256,
            "post_commit_cursor": cursor,
            "provenance": provenance,
            "at": time.time(),
        }
        with (StoragePath(prefix) / "receipts" / (name + ".json")).open("w") as stream:
            json.dump(receipt, stream)
        print(json.dumps({"phase": "receipt_emitted", "batch": name, "sha256": receipt["archive_sha256"]}), flush=True)
    except Exception as error:
        print(json.dumps({"phase": "receipt_write_failed", "batch": name, "error": str(error)[:300]}), flush=True)
