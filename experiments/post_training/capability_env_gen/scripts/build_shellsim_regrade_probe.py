#!/usr/bin/env python3
"""Build metadata-only authored ShellSim fixed-grading probe inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from capability_pipeline.regrade import create_plan_bundle
from scripts.build_runtime_probe import DEFAULT_IMAGE, EXECUTABLE_IMAGE, TOKEN, build


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root: Path, taskcompendium_source: Path, daytona_helper: Path, *, executable: bool) -> dict:
    if root.exists() or root.is_symlink():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    fixture = root / "fixture"
    build(fixture, image=EXECUTABLE_IMAGE if executable else DEFAULT_IMAGE,
          executable=executable, environment="shellsim")
    source = root / "evaluation-input"
    (source / "bundle").parent.mkdir(parents=True)
    shutil.copytree(fixture / "task", source / "bundle")
    shutil.copytree(fixture / "harbor", source / "package")
    tools = source / "tools"
    tools.mkdir()
    shutil.copy2(daytona_helper, tools / "dt.py")
    cases = [
        {
            "id": "positive", "class": "positive", "response": TOKEN,
            "expect": {"status": "graded", "reward_min": 1.0, "reward_max": 1.0},
        },
        {
            "id": "negative", "class": "negative", "response": "WRONG",
            "expect": {"status": "graded", "reward_min": 0.0, "reward_max": 0.0},
        },
    ]
    if executable:
        for case in cases:
            case["workspace"] = "controls/seed"
            case["commands"] = [
                "cat replay-note.txt >/dev/null",
                f"printf '%s\\n' {TOKEN if case['id'] == 'positive' else 'WRONG'} > answer.txt",
            ]
        seed = source / "bundle/controls/seed"
        seed.mkdir(parents=True)
        (seed / "replay-note.txt").write_text("Fixed authored ShellSim workspace.\n")
    (source / "bundle/controls.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    inputs = {
        path.relative_to(source).as_posix(): {"path": path.relative_to(source).as_posix()}
        for path in source.rglob("*") if path.is_file()
    }
    (source / "plan.json").write_text(json.dumps({"inputs": inputs}, indent=2) + "\n")
    files = {
        path.relative_to(source).as_posix(): _sha(path)
        for path in source.rglob("*") if path.is_file()
    }
    (source / "manifest.json").write_text(json.dumps({
        "schema_version": "capability-runtime-evaluation-bundle-v1",
        "plan_sha256": files["plan.json"],
        "files": files,
        "directories": sorted(path.relative_to(source).as_posix() for path in source.rglob("*") if path.is_dir()),
    }, indent=2) + "\n")
    planned = create_plan_bundle(source, taskcompendium_source, root / "regrade-plan", parallelism=2)
    (root / "probe-input.json").write_text(json.dumps({
        "schema_version": "capability-shellsim-regrade-probe-input-v1",
        "surface": "script" if executable else "native",
        "evaluation_manifest_sha256": _sha(source / "manifest.json"),
        "regrade_plan_sha256": planned["plan_sha256"],
        "regrade_manifest_sha256": planned["manifest_sha256"],
    }, indent=2) + "\n")
    return planned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--taskcompendium-source", required=True, type=Path)
    parser.add_argument("--daytona-helper", required=True, type=Path)
    parser.add_argument("--executable", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.out, args.taskcompendium_source, args.daytona_helper, executable=args.executable), indent=2))


if __name__ == "__main__":
    main()
