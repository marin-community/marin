# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Inventory approved CalibForge environments with the existing OpenSWE BuildKit parser."""

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def environment_inventory(source: Path, approved: Path, parser_image: str) -> dict:
    """Parse the complete approved population without executing its build commands."""
    paths = pq.read_table(approved, columns=["path"]).column("path").to_pylist()
    if len(paths) != len(set(paths)):
        raise ValueError("Approved task IDs are not unique")
    image = json.loads(subprocess.check_output(["docker", "image", "inspect", parser_image], text=True))[0]
    records = []
    base_counts: Counter[str] = Counter()
    instruction_counts: Counter[str] = Counter()
    with subprocess.Popen(
        ["docker", "run", "--rm", "--interactive", "--network", "none", image["Id"]],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as parser:
        assert parser.stdin is not None and parser.stdout is not None
        for identity in paths:
            context = source / identity / "environment"
            dockerfile = (context / "Dockerfile").read_text()
            parser.stdin.write(json.dumps({"id": identity, "dockerfile": dockerfile}) + "\n")
            parser.stdin.flush()
            line = parser.stdout.readline()
            if not line:
                raise RuntimeError(f"BuildKit parser stopped at {identity}")
            parsed = json.loads(line)
            if parsed["id"] != identity:
                raise ValueError(f"Invalid BuildKit response for {identity}")
            instructions = parsed["instructions"]
            commands = [instruction["command"] for instruction in instructions]
            bases = [item["args"][0] for item in instructions if item["command"] == "from"]
            base_counts.update(bases)
            instruction_counts.update(set(commands))
            first_copy = next((i for i, command in enumerate(commands) if command in {"copy", "add"}), len(commands))
            files = {}
            for path in sorted(context.rglob("*")):
                if path.is_symlink():
                    files[str(path.relative_to(context))] = {"link": str(path.readlink())}
                elif path.is_file():
                    files[str(path.relative_to(context))] = {
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "mode": oct(path.stat().st_mode & 0o7777),
                    }
            records.append(
                {
                    "task_id": identity,
                    "dockerfile_sha256": hashlib.sha256(dockerfile.encode()).hexdigest(),
                    "instructions": instructions,
                    "base_images": bases,
                    "has_run_after_copy": "run" in commands[first_copy + 1 :],
                    "context_files": files,
                }
            )
        parser.stdin.close()
        if parser.wait() != 0:
            raise RuntimeError("BuildKit inventory parser failed")
    return {
        "approved_sha256": hashlib.sha256(approved.read_bytes()).hexdigest(),
        "parser_image_id": image["Id"],
        "task_count": len(records),
        "base_counts": dict(base_counts),
        "instruction_task_counts": dict(instruction_counts),
        "tasks": records,
        "admission": "not_run",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--approved", type=Path, required=True)
    parser.add_argument("--parser-image", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = environment_inventory(args.source, args.approved, args.parser_image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "tasks"}))


if __name__ == "__main__":
    main()
