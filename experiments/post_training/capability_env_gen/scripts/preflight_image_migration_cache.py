"""Read-only provider cache proof for an image-only verifier migration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capability_pipeline.daytona_policy import VERIFIER_BOOTSTRAP
from capability_pipeline.daytona_snapshot import validate_snapshot_recipe


def _dt():
    """Load the maintained SDK helper without importing the Harbor runtime."""
    path = Path(os.environ["CAPABILITY_DAYTONA_TOOLS"]) / "dt.py"
    spec = importlib.util.spec_from_file_location("migration_daytona_helper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("maintained Daytona helper is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_cache(
    client: Any,
    *,
    image: str,
    expected_dockerfile_sha256: str,
) -> dict[str, Any]:
    """Bind the currently visible provider object to the exact verifier recipe."""
    dockerfile = f"FROM {image}\nUSER root\n{VERIFIER_BOOTSTRAP}"
    dockerfile_sha256 = hashlib.sha256(dockerfile.encode()).hexdigest()
    if dockerfile_sha256 != expected_dockerfile_sha256:
        raise RuntimeError(
            "local verifier recipe differs from the reviewed migration receipt"
        )
    snapshot_name = f"cap-verifier-{dockerfile_sha256[:20]}"
    snapshot = client.snapshot.get(snapshot_name)
    recipe = validate_snapshot_recipe(
        snapshot,
        expected_name=snapshot_name,
        expected_dockerfile=dockerfile,
    )
    state = getattr(snapshot, "state", None)
    state = getattr(state, "value", state)
    if str(state).lower() not in {"active", "snapshotstate.active"}:
        raise RuntimeError("reviewed verifier snapshot is not active")
    return {
        "schema_version": "capability-image-migration-cache-preflight-v1",
        "state": "passed",
        "checked_at": datetime.now(UTC).isoformat(),
        "image": image,
        "snapshot_id": str(getattr(snapshot, "id", "")),
        "snapshot_state": str(state),
        **recipe,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--expected-dockerfile-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        receipt = validate_cache(
            _dt().client(),
            image=args.image,
            expected_dockerfile_sha256=args.expected_dockerfile_sha256,
        )
    except Exception as error:  # noqa: BLE001 - persist class, never provider text
        receipt = {
            "schema_version": "capability-image-migration-cache-preflight-v1",
            "state": "failed",
            "checked_at": datetime.now(UTC).isoformat(),
            "image": args.image,
            "expected_dockerfile_sha256": args.expected_dockerfile_sha256,
            "error_type": type(error).__name__,
        }
        args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return 1
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
