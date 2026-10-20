# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare complete container exports without suppressing filesystem differences."""

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

from experiments.post_training.task_curation.calibforge_fixtures import FIXTURE_ARCHIVE, SETUP_SCRIPT


def filesystem_manifest(container: str) -> dict:
    """Hash exported filesystem entries, including ownership, modes and timestamps."""
    result = {}
    with subprocess.Popen(["docker", "export", container], stdout=subprocess.PIPE) as process:
        assert process.stdout is not None
        with tarfile.open(fileobj=process.stdout, mode="r|") as archive:
            for member in archive:
                digest = None
                if member.isfile():
                    content = archive.extractfile(member)
                    assert content is not None
                    digest = hashlib.file_digest(content, "sha256").hexdigest()
                result[member.name] = {
                    "type": member.type.decode(),
                    "mode": member.mode,
                    "uid": member.uid,
                    "gid": member.gid,
                    "mtime": member.mtime,
                    "link": member.linkname,
                    "sha256": digest,
                }
        if process.wait() != 0:
            raise RuntimeError(f"Container export failed: {container}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--bundles", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    records = {r["task_id"]: r for r in json.loads((args.bundles / "manifest.json").read_text())["tasks"]}
    outcomes = []
    failed_references = []
    for case in json.loads(args.references.read_text())["tasks"]:
        if case["status"] != "built":
            failed_references.append(case)
            continue
        record = records[case["task_id"]]
        reference = subprocess.check_output(
            ["docker", "create", "--platform", "linux/amd64", case["reference_image"], "sleep", "infinity"], text=True
        ).strip()
        candidate = None
        try:
            candidate = subprocess.check_output(
                ["docker", "create", "--platform", "linux/amd64", record["base_image"], "sleep", "infinity"], text=True
            ).strip()
            subprocess.run(["docker", "start", reference, candidate], check=True, capture_output=True)
            subprocess.run(["docker", "exec", candidate, "test", "!", "-e", "/setup_files"], check=True)
            subprocess.run(["docker", "exec", candidate, "mkdir", "-p", "/setup_files"], check=True)
            for filename in (FIXTURE_ARCHIVE, SETUP_SCRIPT):
                subprocess.run(
                    [
                        "docker",
                        "cp",
                        str(args.bundles / "tasks" / case["task_id"] / filename),
                        candidate + ":/setup_files/",
                    ],
                    check=True,
                )
            subprocess.run(
                ["docker", "exec", candidate, "sh", f"/setup_files/{SETUP_SCRIPT}"],
                check=True,
                capture_output=True,
            )
            left = filesystem_manifest(reference)
            right = filesystem_manifest(candidate)
            differences = {
                name: {"reference": left.get(name), "candidate": right.get(name)}
                for name in sorted(left.keys() | right.keys())
                if left.get(name) != right.get(name)
            }
            outcomes.append(
                {
                    "task_id": case["task_id"],
                    "reference_image": case["reference_image"],
                    "base_image": record["base_image"],
                    "reference_entries": len(left),
                    "candidate_entries": len(right),
                    "differences": differences,
                }
            )
            print(json.dumps({"task_id": case["task_id"], "different_entries": len(differences)}), flush=True)
        finally:
            subprocess.run(
                ["docker", "rm", "-f", reference, *([candidate] if candidate else [])], check=True, capture_output=True
            )
    args.output.write_text(
        json.dumps(
            {
                "admission": "not_run",
                "normalizations": [],
                "tasks": outcomes,
                "failed_native_references": failed_references,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
