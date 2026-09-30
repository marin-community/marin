#!/usr/bin/env python3
"""Apply reviewed OCI pointers after isolated publication and cold pulls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from capability_pipeline.generic_image_migration import migrate_image_pointers


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--frozen-workspace", type=Path, required=True)
    parser.add_argument("--capture-tools", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--builder-session-id", action="append", default=[])
    parser.add_argument("--publication", nargs=2, action="append", metavar=("ROLE", "PATH"), required=True)
    parser.add_argument("--cold-pull", nargs=2, action="append", metavar=("ROLE", "PATH"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    publication = dict(args.publication)
    cold = dict(args.cold_pull)
    if len(publication) != len(args.publication) or len(cold) != len(args.cold_pull):
        raise ValueError("duplicate role evidence")
    receipt = migrate_image_pointers(
        task=args.task, plan_path=args.plan, frozen_workspace=args.frozen_workspace,
        capture_tools=args.capture_tools, approval_path=args.approval,
        builder_session_ids=set(args.builder_session_id),
        publication_paths={key: Path(value) for key, value in publication.items()},
        cold_pull_paths={key: Path(value) for key, value in cold.items()},
        output=args.output,
    )
    print(json.dumps({"state": receipt["state"], "receipt": str(args.output / "migration.json")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - avoid logging provider text.
        print(json.dumps({"state": "failed", "error_type": type(error).__name__}))
        raise SystemExit(1) from None
