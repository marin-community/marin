"""Verify that Iris will transport exactly the required submission leaf."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


def check_stage(workspace, stage, files, default_exclude):
    workspace, stage = Path(workspace).resolve(), Path(stage).resolve()
    if not stage.is_relative_to(workspace) or stage == workspace:
        raise ValueError("submission stage must be inside the Iris workspace")
    actual = {Path(path).absolute() for path in files}
    expected = set()
    for path in stage.rglob("*"):
        if path.is_symlink():
            raise ValueError("submission stage contains a symlink")
        if path.is_file() and not default_exclude.search(
            path.relative_to(workspace).as_posix()
        ):
            expected.add(path)
    if stage / "worker.sh" not in expected:
        raise ValueError("submission stage lacks worker.sh")
    if expected - actual:
        missing = sorted(
            path.relative_to(workspace).as_posix() for path in expected - actual
        )
        raise ValueError(f"Iris bundle omits required stage files: {missing[:10]}")
    if any(
        path.is_relative_to(stage.parent) and not path.is_relative_to(stage)
        for path in actual
    ):
        raise ValueError("Iris bundle includes another submission leaf")
    inventory = {
        path.relative_to(workspace).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(expected)
    }
    return {
        "stage": stage.relative_to(workspace).as_posix(),
        "stage_files": len(expected),
        "stage_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()
        ).hexdigest(),
    }


def main():
    from iris.cluster.client.bundle import DEFAULT_EXCLUDE, collect_workspace_files

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()
    workspace, stage = Path(args.workspace).resolve(), Path(args.stage).resolve()
    relative = stage.relative_to(workspace).as_posix()
    exclude = (
        re.compile("|".join(f"(?:{value})" for value in args.exclude))
        if args.exclude
        else None
    )
    files = collect_workspace_files(
        workspace, exclude=exclude, extra_includes=[relative + "/**/*"]
    )
    print(json.dumps(check_stage(workspace, stage, files, DEFAULT_EXCLUDE)))


if __name__ == "__main__":
    main()
