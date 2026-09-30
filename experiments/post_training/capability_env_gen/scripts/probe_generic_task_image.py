#!/usr/bin/env python3
"""Trusted generic digest cold-boot worker; task gates run afterward."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from capability_pipeline import sandbox_provider
from capability_pipeline.generic_image_cold_pull import cold_pull


def _create_snapshot(client, name: str, recipe: str, resources: dict) -> None:
    from daytona import CreateSnapshotParams, Image, Resources

    with tempfile.TemporaryDirectory(prefix="generic-image-recipe-") as temporary:
        dockerfile = Path(temporary) / "Dockerfile"
        dockerfile.write_text(recipe)
        client.snapshot.create(
            CreateSnapshotParams(
                name=name, image=Image.from_dockerfile(str(dockerfile)),
                resources=Resources(cpu=resources["cpu"], memory=resources["memory"], disk=resources["disk"]),
            ),
            timeout=3600,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--capture-tools", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--builder-session-id", action="append", default=[])
    parser.add_argument("--publication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("cold-pull receipt already exists")
    sys.path.insert(0, str(args.capture_tools.resolve()))
    import dtx  # type: ignore[import-not-found]

    provider_tools = dtx
    if sandbox_provider.provider() == sandbox_provider.SILO:
        # dtx.client() is Daytona-only; silo's staged dt.py client() reads the
        # same Daytona parameter objects dtx.create and _create_snapshot pass.
        import dt  # type: ignore[import-not-found]

        provider_tools = SimpleNamespace(client=dt.client, create=dtx.create, sh=dtx.sh)

    receipt = cold_pull(
        plan_path=args.plan, workspace=args.workspace, capture_tools=args.capture_tools,
        approval_path=args.approval, builder_session_ids=set(args.builder_session_id),
        publication_path=args.publication, dtx=provider_tools, create_snapshot=_create_snapshot,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"state": receipt["state"], "role": receipt["role"], "receipt": str(args.output)}))
    return 0 if receipt["state"] == "passed_pending_task_gates" else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - provider text may contain secrets.
        # Type, class (transient/content/harness), HTTP status and module names only.
        from capability_pipeline.generic_image_capture import error_summary

        print(json.dumps({"state": "failed", **error_summary(error)}, sort_keys=True))
        raise SystemExit(1) from None
