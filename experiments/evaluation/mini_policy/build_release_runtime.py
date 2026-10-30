# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the grading wheels once and retain their byte hashes for dataset export."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

from rigging.filesystem.buckets import filesystem_for

from scripts.python_libs_package import patched_tree

DEST = "s3://marin-us-east-02a/marin/experiments/mini-eval-policy/dataset-release-20261030/runtime-v2"


def main():
    root = Path(__file__).resolve().parents[3]
    output = Path(os.environ["IRIS_OUTPUT_DIR"]) / "runtime"
    output.mkdir(parents=True, exist_ok=True)
    with patched_tree("0.2.0+mini.20261030"):
        for project in ("lib/rigging", "lib/finestore"):
            subprocess.run(
                ["uv", "build", "--project", str(root / project), "--wheel", "--out-dir", str(output)], check=True
            )
    subprocess.run(
        ["uv", "build", "--project", str(root / "vendor/evalchemy"), "--wheel", "--out-dir", str(output)], check=True
    )
    command = [
        "uv",
        "export",
        "--project",
        str(root / "vendor/evalchemy"),
        "--frozen",
        "--no-dev",
        "--no-emit-project",
        "--no-hashes",
        "--extra",
        "ifbench",
        "--extra",
        "mrcr",
        "--extra",
        "nupa5k-loose",
        "--no-emit-package",
        "marin-rigging",
        "--no-emit-package",
        "marin-finestore",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    (output / "requirements.txt").write_text(result.stdout)
    manifest = []
    for path in sorted(output.iterdir()):
        content = path.read_bytes()
        manifest.append({"file": path.name, "sha256": hashlib.sha256(content).hexdigest(), "size": len(content)})
        fs, key = filesystem_for(DEST + "/" + path.name)
        fs.pipe_file(key, content)
    fs, key = filesystem_for(DEST + "/manifest.json")
    fs.pipe_file(key, json.dumps(manifest, indent=2).encode())
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
