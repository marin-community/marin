# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze public /app and /data fixtures from qualified native CalibForge builds."""

import argparse
import gzip
import hashlib
import io
import json
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

from experiments.post_training.task_curation.calibforge_fixtures import FIXTURE_ARCHIVE, SETUP_SCRIPT


def captured_payload(reference_image: str, roots: tuple[str, ...] = ("app",)) -> tuple[bytes, list[dict]]:
    """Capture only regular public fixture files and directories, with original metadata."""
    if not reference_image.startswith("sha256:"):
        raise ValueError("Capture requires an immutable local reference image ID")
    container = subprocess.check_output(
        ["docker", "create", "--platform", "linux/amd64", reference_image], text=True
    ).strip()
    entries = []
    output = io.BytesIO()
    try:
        with tarfile.open(fileobj=output, mode="w") as merged:
            for root in roots:
                if root not in {"app", "data"}:
                    raise ValueError(f"Unqualified capture root: {root}")
                payload = subprocess.check_output(["docker", "cp", container + ":/" + root, "-"])
                with tarfile.open(fileobj=io.BytesIO(payload)) as archive:
                    for member in archive:
                        path = PurePosixPath(member.name)
                        if (
                            not path.parts
                            or path.parts[0] != root
                            or ".." in path.parts
                            or not (member.isfile() or member.isdir())
                        ):
                            raise ValueError(f"Unsupported public fixture member: {member.name}")
                        content = None
                        if member.isfile():
                            stream = archive.extractfile(member)
                            assert stream is not None
                            content = stream.read()
                        merged.addfile(member, io.BytesIO(content) if content is not None else None)
                        entries.append(
                            {
                                "path": member.name,
                                "type": member.type.decode(),
                                "mode": member.mode,
                                "uid": member.uid,
                                "gid": member.gid,
                                "mtime": member.mtime,
                                "sha256": hashlib.sha256(content).hexdigest() if content is not None else None,
                            }
                        )
    finally:
        subprocess.run(["docker", "rm", container], check=True, capture_output=True)
    return gzip.compress(output.getvalue(), mtime=0), entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--generators", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    shutil.copytree(args.previous, args.output)
    manifest = json.loads((args.previous / "manifest.json").read_text())
    records = {row["task_id"]: row for row in manifest["tasks"]}
    generators = {row["task_id"]: row for row in json.loads((args.generators / "manifest.json").read_text())["tasks"]}
    references = json.loads(args.references.read_text())
    if references["build_network"] != "none":
        raise ValueError("Fixture capture requires offline native reference builds")
    failed_references = []
    captured = 0
    for reference in references["tasks"]:
        if reference["status"] != "built":
            failed_references.append(reference)
            continue
        identity = reference["task_id"]
        record = generators[identity]
        if "@sha256:" not in record["base_image"] or record["workdir"] != "/app":
            raise ValueError(f"Capture needs a pinned base and /app workspace: {identity}")
        roots = tuple(reference["public_fixture_roots"])
        payload, entries = captured_payload(reference["reference_image"], roots)
        generator_script = (args.generators / "tasks" / identity / SETUP_SCRIPT).read_text()
        removed_scripts = [
            shlex.split(line[1:-1])[-1]
            for line in generator_script.splitlines()
            if line.startswith("(cd ") and " && rm -f " in line
        ]
        script = "\n".join(
            [
                "#!/bin/sh",
                "set -eu",
                *[f'test ! -d /{root} || test -z "$(ls -A /{root})"' for root in roots],
                f"tar -xzf /setup_files/{FIXTURE_ARCHIVE} -C /",
                f"rm -f /setup_files/{FIXTURE_ARCHIVE} /setup_files/{SETUP_SCRIPT}",
                "rmdir /setup_files",
                "",
            ]
        )
        destination = args.output / "tasks" / identity
        destination.mkdir(parents=True, exist_ok=True)
        (destination / FIXTURE_ARCHIVE).write_bytes(payload)
        (destination / SETUP_SCRIPT).write_text(script)
        records[identity] = {
            **record,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "setup_sha256": hashlib.sha256(script.encode()).hexdigest(),
            "capture": {
                "reference_image": reference["reference_image"],
                "scope": ["/" + root for root in roots],
                "entries": entries,
                "removed_script_paths": removed_scripts,
                "source_setup_sha256": record["setup_sha256"],
                "reference_manifest_sha256": hashlib.sha256(args.references.read_bytes()).hexdigest(),
            },
        }
        captured += 1
    manifest.update(
        tasks=list(records.values()),
        failed_native_references=failed_references,
        compiled_count=sum(row["status"].startswith("compiled") for row in records.values()),
        previous_manifest_sha256=hashlib.sha256((args.previous / "manifest.json").read_bytes()).hexdigest(),
    )
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"captured": captured, "compiled": manifest["compiled_count"], "admission": "not_run"}))


if __name__ == "__main__":
    main()
