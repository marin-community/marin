# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Combine prepared expert datasets, removing old padding before rebatching."""

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter

from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.cruxeval_transfer.rendering import END_TURN_ID


def combine_rows(sources: list[tuple[bytes, dict]]) -> tuple[bytes, int]:
    output = io.BytesIO()
    total_rows = 0
    context = sources[0][1]["max_sequence_tokens"]
    batch = sources[0][1]["batch_size"]
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as target:
        for compressed, manifest in sources:
            if manifest["max_sequence_tokens"] != context or manifest["batch_size"] != batch:
                raise ValueError("Incompatible prepared datasets")
            supervised = 0
            count = 0
            with gzip.GzipFile(fileobj=io.BytesIO(compressed), mode="rb") as stream:
                for index, line in enumerate(stream):
                    row = json.loads(line)
                    if len(row["input_ids"]) != context or len(row["loss_weights"]) != context:
                        raise ValueError("Unexpected row shape")
                    count += 1
                    if index >= manifest["training_rows"]:
                        if any(row["loss_weights"]):
                            raise ValueError("Old padding contains supervised tokens")
                        continue
                    supervised += sum(row["loss_weights"])
                    total_rows += 1
                    target.write(line)
            if count != manifest["training_rows"] + manifest["padding_rows"]:
                raise ValueError("Prepared row count differs from manifest")
            if supervised != manifest["supervised_tokens"]:
                raise ValueError("Prepared supervision differs from manifest")
        padding = -total_rows % batch
        empty = json.dumps({"input_ids": [END_TURN_ID] * context, "loss_weights": [0] * context}).encode() + b"\n"
        for _ in range(padding):
            target.write(empty)
    return output.getvalue(), padding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.source) != 2:
        raise ValueError("Expected the two expert datasets")
    output = StoragePath(args.output)
    if output.exists():
        raise ValueError("Use a new immutable prefix")
    manifests, provenance, sources, audits, probes = [], [], [], [], []
    for uri in args.source:
        root = StoragePath(uri)
        manifest_bytes = (root / "manifest.json").read_bytes()
        manifest = json.loads(manifest_bytes)
        values = {}
        for name in ("train-prebuilt.jsonl.gz", "token-audit.jsonl", "row-probes.jsonl"):
            values[name] = (root / name).read_bytes()
            if hashlib.sha256(values[name]).hexdigest() != manifest["output_hashes"][name]:
                raise ValueError(f"Prepared file changed: {uri}/{name}")
        manifests.append(manifest)
        provenance.append(
            {"uri": uri, "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "manifest": manifest}
        )
        sources.append((values["train-prebuilt.jsonl.gz"], manifest))
        audits.extend(json.loads(line) for line in values["token-audit.jsonl"].splitlines())
        probes.extend(json.loads(line) for line in values["row-probes.jsonl"].splitlines())
    if len({row["id"] for row in audits}) != len(audits):
        raise ValueError("Duplicate selected response IDs across sources")
    if len({manifest["student_revision"] for manifest in manifests}) != 1:
        raise ValueError("Different student tokenizers")
    compressed, padding = combine_rows(sources)
    files = {"train-prebuilt.jsonl.gz": compressed}
    for name, rows in (("token-audit.jsonl", audits), ("row-probes.jsonl", probes)):
        files[name] = "".join(json.dumps(row) + "\n" for row in rows).encode()
    report = {
        "sources": provenance,
        "student": manifests[0]["student"],
        "student_revision": manifests[0]["student_revision"],
        "test_exposed": True,
        "evaluation_documents": 3206,
        "training_rows": len(audits),
        "padding_rows": padding,
        "batch_size": manifests[0]["batch_size"],
        "max_sequence_tokens": manifests[0]["max_sequence_tokens"],
        "steps_per_epoch": (len(audits) + padding) // manifests[0]["batch_size"],
        "supervised_tokens": sum(row["supervised_tokens"] for row in audits),
        "input_tokens": sum(row["tokens"] for row in audits),
        "direction_counts": dict(Counter(row["task"] for row in audits)),
        "output_hashes": {name: hashlib.sha256(value).hexdigest() for name, value in files.items()},
    }
    output.mkdirs()
    for name, value in files.items():
        (output / name).write_bytes(value)
        if hashlib.sha256((output / name).read_bytes()).hexdigest() != report["output_hashes"][name]:
            raise ValueError(f"Upload hash mismatch: {name}")
    (output / "manifest.json").write_text(json.dumps(report, indent=2))
    print("MIXED_DATA_PREPARED " + json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
