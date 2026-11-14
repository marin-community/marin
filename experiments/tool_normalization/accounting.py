# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build source/token accounting and an index of producer-verified candidate exports."""

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

ACCEPTED_STATUSES = {"converted_adapter_checked", "header_normalized_no_calls"}


def summarize(root: Path) -> dict:
    with sqlite3.connect(f"file:{root / 'queue.sqlite3'}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT id,source_file,status,detail FROM rows ORDER BY id").fetchall()
    sources = defaultdict(Counter)
    candidates = []
    for identity, source_file, status, detail in rows:
        payload = Path(source_file).read_bytes()
        source = json.loads(payload)
        if source["id"] != identity:
            raise ValueError("Source identity changed")
        counts = sources[source["source"]]
        counts[f"{status}_records"] += 1
        counts[f"{status}_source_tokens"] += source["tokens"]
        if status not in ACCEPTED_STATUSES:
            continue
        directory = root / "records" / identity
        result = json.loads(detail)
        if result != json.loads((directory / "result.json").read_text()):
            raise ValueError("Queue and export disposition disagree")
        reference = json.loads((directory / "source-reference.json").read_text())
        if reference["sha256"] != hashlib.sha256(payload).hexdigest():
            raise ValueError("Accepted source changed")
        manifest_file = directory / "export/manifest.json"
        manifest = json.loads(manifest_file.read_text())
        if result["verification"]["tokens"] != manifest["tokens"]:
            raise ValueError("Export token count differs from producer verification")
        for name, expected in manifest["files"].items():
            if hashlib.sha256((directory / "export" / name).read_bytes()).hexdigest() != expected:
                raise ValueError("Accepted export payload changed")
        counts["candidate_tokens"] += manifest["tokens"]
        counts["supervised_tokens"] += manifest["supervised_tokens"]
        candidates.append(
            {
                "id": identity,
                "source": source["source"],
                "status": status,
                "manifest": str(manifest_file.resolve()),
                "manifest_sha256": hashlib.sha256(manifest_file.read_bytes()).hexdigest(),
            }
        )
    report = {
        "records": len(rows),
        "producer_verified_exports": len(candidates),
        "by_source": dict(sources),
        "scope": "Candidate index from producer verification; held and partial exports excluded; no original-task claim",
    }
    for name, text in (
        ("candidate-index.jsonl", "".join(json.dumps(row) + "\n" for row in candidates)),
        ("accounting.json", json.dumps(report, indent=2) + "\n"),
    ):
        target = root / name
        temporary = target.with_suffix(".tmp")
        temporary.write_text(text)
        temporary.replace(target)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))


if __name__ == "__main__":
    main()
