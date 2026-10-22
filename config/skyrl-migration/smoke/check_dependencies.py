# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Verify the installed GPU profile against SkyRL's frozen lock, including its uv overrides."""

import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

from packaging.requirements import Requirement

PAYLOAD = Path(__file__).resolve().parent


def main() -> None:
    receipt = json.loads((PAYLOAD / "receipt.json").read_text())
    extras = [argument for extra in receipt["runtime_extras"] for argument in ("--extra", extra)]
    exported = subprocess.run(
        [
            "uv",
            "export",
            "--frozen",
            "--no-dev",
            "--no-emit-project",
            "--no-hashes",
            "--no-header",
            "--project",
            str(PAYLOAD / "dependencies"),
            *extras,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    checked = []
    for line in exported.stdout.splitlines():
        if not line or line.lstrip().startswith("#"):
            continue
        requirement = Requirement(line)
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        distribution = importlib.metadata.distribution(requirement.name)
        assert requirement.specifier.contains(distribution.version, prereleases=True), (line, distribution.version)
        if requirement.url is not None:
            direct = json.loads(distribution.read_text("direct_url.json"))
            if requirement.url.startswith("git+"):
                revision = requirement.url.rsplit("@", 1)[1].split("#")[0]
                assert direct["vcs_info"]["commit_id"] == revision, (line, direct)
            else:
                assert direct["url"] == requirement.url, (line, direct)
        checked.append({"name": requirement.name, "version": distribution.version})
    # pip does not interpret uv overrides. Keep its independent metadata report rather than claiming it is clean.
    metadata = subprocess.run(
        ["uv", "pip", "check", "--python", sys.executable],
        capture_output=True,
        text=True,
        check=False,
    )
    print(metadata.stdout + metadata.stderr, flush=True)
    result = {
        "locked_packages": checked,
        "pip_check_returncode": metadata.returncode,
        "pip_check": metadata.stdout + metadata.stderr,
    }
    output = Path(os.environ["IRIS_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "dependency-check.json").write_text(json.dumps(result, indent=2) + "\n")
    profile = "+".join(receipt["runtime_extras"])
    print(f"Verified {len(checked)} installed packages against the frozen {profile} profile", flush=True)


if __name__ == "__main__":
    main()
