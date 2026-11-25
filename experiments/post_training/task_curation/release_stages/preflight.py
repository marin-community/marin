# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-stage preflight receipts (DESIGN.md §6, BRIEF requirement 3).

Before any cluster submission every stage gets: a type-check of its module, a dry
run of the lowered graph, and a small-slice run on real inputs through the real
code path. The result is a sealed :class:`PreflightReceipt` under the release
prefix; the submission guard refuses a stage whose receipt is missing, failed, or
stale for the current landed source SHA. Preflight executes through the
``atlas-release`` entry point, which supplies each stage's slice runner.
"""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.release_stages.manifest import ArtifactRef

PREFLIGHT_RECEIPT_NAME = "preflight-receipt.json"
PREFLIGHT_KIND = "release-stage-preflight"


class PreflightReceipt(BaseModel):
    """One stage's preflight record; sealed write-once under the release prefix."""

    stage: str
    mode: str
    repo_sha: str
    input_digests: dict[str, str]
    ok: bool
    error: str | None = None
    checked_at: str
    slice_evidence: dict = {}

    def digest(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()


def repo_head(repo_root: Path) -> str:
    """The checkout's HEAD sha, or the submitter-attested revision (the bundle mode)."""
    from experiments.post_training.task_curation.release_stages.guard import (  # noqa: PLC0415 — avoids the guard/preflight cycle at module load
        attested_revision,
    )

    attested = attested_revision()
    if attested is not None:
        return attested
    result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def type_check_module(module: str, repo_root: Path) -> str:
    """Run the repo's type checker over one stage module; returns its output."""
    result = subprocess.run(
        ["uvx", "--from", "pyrefly>=1.0.0,<1.1.0", "pyrefly", "check", module, "--baseline", ".pyrefly-baseline.json"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"type check failed for {module}:\n{result.stdout[-2000:]}{result.stderr[-500:]}")
    return result.stdout.strip()


def input_digests(inputs: dict) -> dict[str, str]:
    """Stable digests over a stage's typed inputs (ArtifactRefs bind their sha256)."""
    document = json.dumps(inputs, sort_keys=True, default=str)
    return {"inputs": hashlib.sha256(document.encode()).hexdigest()}


def receipt_path(output_root: str, stage: str, name: str = PREFLIGHT_RECEIPT_NAME) -> StoragePath:
    return StoragePath(f"{output_root}/preflight/{stage}/{name}")


def write_receipt(output_root: str, receipt: PreflightReceipt, name: str = PREFLIGHT_RECEIPT_NAME) -> ArtifactRef:
    """Seal the receipt write-once; a rerun must use a fresh release prefix."""
    raw = (json.dumps(json.loads(receipt.model_dump_json()), indent=2, sort_keys=True) + "\n").encode()
    path = receipt_path(output_root, receipt.stage, name)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(
                f"preflight receipt for {receipt.stage} already exists with different content at {path};"
                " rerun preflight after resolving the failure, not over it"
            )
    else:
        path.write_bytes(raw, auto_mkdir=True)
    if path.read_bytes() != raw:
        raise ValueError(f"preflight receipt readback mismatch at {path}")
    return ArtifactRef(uri=str(path), sha256=hashlib.sha256(raw).hexdigest())


def read_receipt(output_root: str, stage: str) -> PreflightReceipt | None:
    path = receipt_path(output_root, stage)
    if not path.exists():
        return None
    return PreflightReceipt.model_validate_json(path.read_text())


def receipt_is_fresh(receipt: PreflightReceipt | None, landed_sha: str, digests: dict[str, str]) -> bool:
    """The guard's freshness rule: ok, sealed at the landed sha, inputs unchanged."""
    if receipt is None or not receipt.ok:
        return False
    if receipt.repo_sha != landed_sha:
        return False
    return receipt.input_digests == digests


def require_fresh_preflight(output_root: str, stage: str, landed_sha: str, digests: dict[str, str]) -> PreflightReceipt:
    """Refuse a stage whose preflight receipt is missing, failed, or stale."""
    receipt = read_receipt(output_root, stage)
    if not receipt_is_fresh(receipt, landed_sha, digests):
        state = "missing" if receipt is None else ("failed" if not receipt.ok else "stale")
        raise ValueError(
            f"stage {stage} has no fresh preflight receipt ({state}) for landed sha {landed_sha};"
            " run 'atlas-release preflight' before submitting"
        )
    return receipt


def run_preflight(
    stage: str,
    module: str,
    inputs: dict,
    *,
    output_root: str,
    repo_root: Path,
    slice_runner,
) -> PreflightReceipt:
    """Type-check the module, run the small-slice runner, and seal the receipt.

    ``slice_runner`` is a callable that executes the stage's real code path on a
    bounded slice of real inputs (per-stage: a single census source, a bounded
    text-scan slice, ...). Its returned evidence is recorded on the receipt.
    """
    repo_sha = repo_head(repo_root)
    digests = input_digests(inputs)
    now = datetime.now(tz=UTC).isoformat(timespec="seconds")
    try:
        type_check_module(module, repo_root)
        evidence = slice_runner()
        receipt = PreflightReceipt(
            stage=stage,
            mode="preflight",
            repo_sha=repo_sha,
            input_digests=digests,
            ok=True,
            checked_at=now,
            slice_evidence=evidence if isinstance(evidence, dict) else {"result": str(evidence)[:500]},
        )
    except Exception as error:
        receipt = PreflightReceipt(
            stage=stage,
            mode="preflight",
            repo_sha=repo_sha,
            input_digests=digests,
            ok=False,
            error=str(error)[:2000],
            checked_at=now,
        )
    write_receipt(output_root, receipt)
    if not receipt.ok:
        raise ValueError(f"preflight failed for {stage}: {receipt.error}")
    return receipt
