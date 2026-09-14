# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build verifier images for explicitly selected rows of the pinned R2E sample."""

import argparse
import gzip
import json
import subprocess
from pathlib import Path

import msgspec

from taskcompendium.importers.r2egym import source_image
from taskcompendium.models import ContainerRuntime, ImageOverlay

PACKAGE = Path(__file__).resolve().parents[1]


def build(indices: list[int], output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    fixtures = PACKAGE / "tests/fixtures/r2egym"
    rows = dict(enumerate(json.loads(gzip.decompress((fixtures / "rows.json.gz").read_bytes()))))
    broadened = json.loads(gzip.decompress((fixtures / "broadened_rows.json.gz").read_bytes()))
    source_indices = json.loads((fixtures / "broadened_rows_provenance.json").read_text())["rows"]
    rows.update(zip(source_indices, broadened, strict=True))
    if len(set(indices)) != len(indices) or any(index not in rows for index in indices):
        raise ValueError(f"Select unique source rows from {sorted(rows)}")
    runtimes = {}
    directory = PACKAGE / "src/taskcompendium/harbor"
    for index in indices:
        row = rows[index]
        image = source_image(row)
        if image is None:
            raise ValueError(f"No pinned image for R2E row {index}")
        commit = row["commit_hash"]
        tag = f"taskcompendium-r2e-{commit}:validation"
        subprocess.run(
            [
                "docker",
                "build",
                "--platform",
                "linux/amd64",
                "--build-arg",
                f"R2E_SOURCE_IMAGE={image}",
                "--tag",
                tag,
                "--file",
                str(directory / "r2e_runtime.Dockerfile"),
                str(directory),
            ],
            check=True,
        )
        digest = subprocess.check_output(["docker", "image", "inspect", tag, "--format", "{{.Id}}"], text=True).strip()
        runtimes[commit] = ContainerRuntime(
            digest,
            timeout=300,
            workspace=ImageOverlay((".venv",)),
            supervisor_python="/usr/local/bin/python3",
        )
        print(json.dumps({"row": index, "source_image": image, "verifier_image": digest}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(msgspec.to_builtins(runtimes), indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", type=int, nargs="+", required=True, help="Pinned source row indices: 0 through 19, 500, or 550"
    )
    parser.add_argument("--output", type=Path, required=True, help="New runtime mapping JSON for build_poc.py")
    arguments = parser.parse_args()
    build(arguments.rows, arguments.output)
