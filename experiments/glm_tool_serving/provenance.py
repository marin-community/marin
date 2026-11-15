"""Per-run provenance and per-batch commit receipts.

Added 2026-11-15 at the independent indexer's proposal (fleet-owner reviewed):
each run writes provenance.json (code + data hashes) into its workspace before the
batch loop, archives include it, and every committed batch emits
receipts/<batch>.json with the archive URI, SHA-256, and post-commit cursor.
Purely additive: historical archives predate receipts, and the indexer falls back
to scanning when a receipt is absent (including crashes after the cursor commit).
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

PROVENANCE_FILES = (
    "experiments/tool_normalization/driver.py",
    "experiments/tool_normalization/model.py",
    "experiments/tool_normalization/pipeline.py",
    "experiments/tool_normalization/preflight.py",
    "experiments/tool_normalization/review_real.py",
    "experiments/tool_normalization/contract_pilot.py",
    "experiments/tool_normalization/environments.json",
    "experiments/tool_normalization/opencode-tools.json",
    "experiments/glm_tool_serving/scale_production.py",
    "experiments/glm_tool_serving/ortet_production.py",
    "experiments/glm_tool_serving/provenance.py",
    "experiments/glm_tool_serving/production_data/rule-approval-v3.json",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_provenance(app_root: Path, root: Path, catalog_sha256: str) -> dict:
    """Write root/provenance.json before the batch loop; returns the document."""
    root.mkdir(parents=True, exist_ok=True)
    files = {}
    for relative in PROVENANCE_FILES:
        path = app_root / relative
        if path.exists():
            files[relative] = _file_sha256(path)
    head = None
    source = "unavailable"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=app_root, capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            head, source = result.stdout.strip(), "git"
    except (OSError, subprocess.SubprocessError):
        pass
    if head is None and os.environ.get("PROVENANCE_GIT_HEAD"):
        head, source = os.environ["PROVENANCE_GIT_HEAD"], "env"
    document = {
        "git_head": head,
        "git_head_source": source,
        "files_sha256": files,
        "catalog_sha256": catalog_sha256,
        "generated_at": time.time(),
        "note": "per-file hashes are authoritative; git_head is a label (pod workspaces may lack .git)",
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
            "archive_uri": f"{prefix}/batches/{archive.name}",
            "archive_sha256": _file_sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "catalog_sha256": catalog_sha256,
            "post_commit_cursor": cursor,
            "provenance": provenance,
            "at": time.time(),
        }
        import fsspec

        with fsspec.open(f"{prefix}/receipts/{name}.json", "w") as stream:
            json.dump(receipt, stream)
        print(json.dumps({"phase": "receipt_emitted", "batch": name, "sha256": receipt["archive_sha256"]}), flush=True)
    except Exception as error:
        print(json.dumps({"phase": "receipt_write_failed", "batch": name, "error": str(error)[:300]}), flush=True)
