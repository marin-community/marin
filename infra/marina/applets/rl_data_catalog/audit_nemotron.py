# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build bundled source counts from complete cached Nemotron blend files."""

import argparse
import hashlib
import json
import pprint
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from .server.nemotron_records import SWE_GYM_SOURCE, SWE_REBENCH_SOURCE, SWE_SELECTIONS, count_records, record_selection


class Digest(Protocol):
    def update(self, data: bytes) -> None: ...


def records(path: Path, digest: Digest) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        for line in stream:
            digest.update(line)
            yield json.loads(line)


def audit_blend(path: Path, gym_instances: set[str]) -> dict[str, Any]:
    # The upstream composition declares exactly two SWE origins. Resolve Gym
    # directly by instance membership and attribute the complement to rebench.
    swe_instances = {}
    with path.open("rb") as stream:
        for line in stream:
            row = json.loads(line)
            if record_selection(row) in SWE_SELECTIONS:
                source_id = row["metadata"]["instance_id"]
                swe_instances[source_id] = SWE_GYM_SOURCE if source_id in gym_instances else SWE_REBENCH_SOURCE
    digest = hashlib.sha256()
    groups = count_records(records(path, digest), swe_instances)
    return {"total": sum(group["count"] for group in groups), "sha256": digest.hexdigest(), "groups": groups}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Complete JSONL directory named by HF revision SHA")
    parser.add_argument(
        "--swe-gym-identifiers", type=Path, required=True, help="JSON with revision and rows.instance_id"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gym = json.loads(args.swe_gym_identifiers.read_text())
    gym_instances = {row["instance_id"] for row in gym["rows"]}
    paths = [path for path in sorted(args.snapshot.glob("*.jsonl")) if path.stem in {"rlvr1", "rlvr2", "mopd"}]
    if not paths:
        raise ValueError("Snapshot contains no supported complete blend JSONL files")
    manifest = {
        "revision": args.snapshot.name,
        "swe_gym_revision": gym["revision"],
        "swe_attribution": (
            "SWE-Gym records match instance_id membership in SWE-Gym/SWE-Gym; "
            "remaining SWE records are attributed to SWE-rebench-V2 by the blend card exhaustive two-source composition."
        ),
        "blends": {path.stem: audit_blend(path, gym_instances) for path in paths},
    }
    body = "# Copyright The Marin Authors\n# SPDX-License-Identifier: Apache-2.0\n\n"
    body += '"""Complete-file counts for the pinned Nemotron blend revision."""\n\n'
    body += "NEMOTRON_COUNTS = " + pprint.pformat(manifest, width=115, sort_dicts=False) + "\n"
    args.output.write_text(body)


if __name__ == "__main__":
    main()
