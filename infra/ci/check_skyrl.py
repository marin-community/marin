# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Check vendored SkyRL's independent lock, Marin imports, and built wheel."""

import argparse
import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

from infra.ci.check_skyrl_marin import check_project

SOURCE = Path(__file__).resolve().parents[2] / "vendor/skyrl"
BUNDLE_MANIFEST = "cloud/iris/runtime_bundle_files.txt"
WHEEL_RUNTIME_ROOTS = {"cloud", "marinskyrl", "skyrl_train", "skyrl_gym", "chat_templates", "open_mopd_versions.py"}


def wheel_source_files(source: Path) -> dict[str, Path]:
    """Map the current package files and assets to their wheel member paths."""
    files = {}
    for path in source.rglob("*"):
        name = path.relative_to(source)
        if {".git", ".venv", "__pycache__"}.intersection(name.parts) or path.suffix in {".pyc", ".pyo"}:
            continue
        member = name.as_posix().removeprefix("skyrl-train/").removeprefix("skyrl-gym/")
        if member.split("/")[0] in WHEEL_RUNTIME_ROOTS and path.is_file():
            path.resolve().relative_to(source.resolve())
            files[member] = path
    return files


def audit_wheel(wheel: Path, source: Path) -> list[str]:
    """Check that the wheel contains the current checked-out source and assets."""
    with zipfile.ZipFile(wheel) as archive:
        members = sorted(archive.namelist())
        if BUNDLE_MANIFEST not in members:
            raise ValueError("Wheel is missing the launcher bundle manifest")
        expected = wheel_source_files(source)
        for packaged, path in expected.items():
            if packaged not in members or archive.read(packaged) != path.read_bytes():
                raise ValueError(f"Wheel member missing or changed: {packaged}")
        unexpected = {
            name for name in members if name.split("/")[0] in WHEEL_RUNTIME_ROOTS and not name.endswith("/")
        } - expected.keys()
        if unexpected:
            raise ValueError(f"Wheel contains runtime files outside the checked-out source: {sorted(unexpected)}")
    return members


def check_vendor(source: Path) -> None:
    """Check the independent lock, isolated Marin contract, and wheel contents."""
    subprocess.run(["uv", "lock", "--project", str(source), "--check"], check=True)
    check_project(source)
    with tempfile.TemporaryDirectory(prefix="skyrl-wheel-") as temporary:
        output = Path(temporary)
        subprocess.run(["uv", "build", "--wheel", "--project", str(source), "--out-dir", str(output)], check=True)
        [wheel] = output.glob("*.whl")
        members = audit_wheel(wheel, source)
        print(json.dumps({"members": len(members)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    args = parser.parse_args()
    check_vendor(args.source.resolve())


if __name__ == "__main__":
    main()
