"""Dependency-free evidence checks shared by live composite probe drivers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import sandbox_provider


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture_case_artifacts(
    case_id: str, root: Path
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Retain a fail-closed receipt before consuming live trial artifacts."""
    root = Path(root)
    receipt_path = root / "probe-case-evidence.json"
    paths = {
        "harbor_result": root / "result.json",
        "taskcompendium_result": root / "verifier/taskcompendium-result.json",
        "candidate_provider": root / "daytona-environment.json",
        "exception": root / "exception.txt",
    }
    receipt: dict[str, Any] = {
        "schema_version": "capability-composite-probe-case-v1",
        "case_id": case_id,
        "state": "capturing",
        "artifacts": {
            name: {
                "path": str(path.relative_to(root)),
                "sha256": _sha256(path),
            }
            for name, path in paths.items()
            if path.is_file()
        },
    }

    def finish(state: str, reason: str | None = None) -> None:
        receipt["state"] = state
        if reason is not None:
            receipt["reason"] = reason
        temporary = receipt_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        temporary.replace(receipt_path)

    artifact = paths["taskcompendium_result"]
    if not artifact.is_file():
        finish("infra_error", "missing_taskcompendium_result")
        raise RuntimeError(
            f"{case_id} trial lacks verifier/taskcompendium-result.json; "
            f"retained {receipt_path.name}"
        )
    provider_path = paths["candidate_provider"]
    if not provider_path.is_file():
        finish("infra_error", "missing_candidate_provider_record")
        raise RuntimeError(
            f"{case_id} trial lacks daytona-environment.json; "
            f"retained {receipt_path.name}"
        )
    try:
        result = json.loads(artifact.read_text())
        provider = json.loads(provider_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        finish("infra_error", "malformed_trial_artifact")
        raise RuntimeError(
            f"{case_id} has a malformed trial artifact; retained {receipt_path.name}"
        ) from error
    if not isinstance(result, dict) or not isinstance(provider, dict):
        finish("infra_error", "non_object_trial_artifact")
        raise TypeError(
            f"{case_id} has a non-object trial artifact; retained {receipt_path.name}"
        )
    finish("captured")
    return result, provider, receipt_path


def validate_case_result(
    case_id: str, result: dict[str, Any], provider: dict[str, Any]
) -> list[dict[str, Any]]:
    """Fail with the semantic/transport cause before checking success evidence."""
    if provider.get("network_block_all") is not True:
        raise RuntimeError(f"{case_id} candidate environment allowed network")
    status = result.get("status")
    if status != "graded" or result.get("reward") is None:
        detail = result.get("detail")
        error = detail.get("error") if isinstance(detail, dict) else None
        raise RuntimeError(
            f"{case_id} grading outcome {status or 'missing'} with null reward: "
            f"{error or 'no structured error'}"
        )
    detail = result.get("detail")
    machine = detail.get("machine_results") if isinstance(detail, dict) else None
    if (
        not isinstance(machine, list)
        or not machine
        or any(
            not isinstance(entry, dict)
            or entry.get("detail", {}).get("verifier_isolation")
            not in sandbox_provider.NETWORK_BLOCKED_ISOLATION
            for entry in machine
        )
    ):
        raise RuntimeError(f"{case_id} lacks network-blocked verifier evidence")
    return machine
