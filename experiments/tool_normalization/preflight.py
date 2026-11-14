# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Account for complete source records before model dispatch, without inference."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import jsonschema

from experiments.tool_normalization.candidate import (
    NATIVE_SCHEMAS,
    assembled_messages,
    calls_for_candidate,
    has_source_calls,
    source_messages,
)
from experiments.tool_normalization.historical_projection import historical_projection
from experiments.tool_normalization.native_seed import native_seed
from experiments.tool_normalization.scope import EXCLUDED_SOURCES


def classify(source: dict, packages: list[str]) -> dict:
    if source["source"] in EXCLUDED_SOURCES:
        return {"status": "excluded_source"}
    if source["tokens"] > 262144:
        return {"status": "source_length_exceeded"}
    try:
        original = source_messages(source["text"])
        seed = native_seed(original, packages)
        if seed is None:
            return {"status": "glm_mapping_residue" if has_source_calls(original) else "structure_unresolved"}
        candidate, observations, _ = historical_projection(original, seed)
        calls = calls_for_candidate(original, candidate, NATIVE_SCHEMAS)
        assembled_messages(original, candidate, observations)
        return {"status": "mapped_calls" if calls or has_source_calls(original) else "tool_free", "calls": len(calls)}
    except (ValueError, jsonschema.ValidationError) as error:
        return {"status": "structure_unresolved", "reason": str(error)}


def run(manifest: Path, config: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    cfg = json.loads(config.read_text())
    environment = json.loads(Path(__file__).with_name("environments.json").read_text())[cfg["execution"]["image"]]
    counts = Counter()
    tokens = Counter()
    with manifest.open() as inputs, (output / "rows.jsonl").open("w") as records:
        for line in inputs:
            row = json.loads(line)
            payload = Path(row["source_file"]).read_bytes()
            source = json.loads(payload)
            if source["id"] != row["id"]:
                raise ValueError("Manifest/source identity mismatch")
            if hashlib.sha256(source["text"].encode()).hexdigest() != source["text_sha256"]:
                raise ValueError("Source text hash mismatch")
            result = classify(source, environment["packages"])
            record = {**row, "source_sha256": hashlib.sha256(payload).hexdigest(), "tokens": source["tokens"], **result}
            records.write(json.dumps(record) + "\n")
            counts[result["status"]] += 1
            tokens[result["status"]] += source["tokens"]
            if sum(counts.values()) % 1000 == 0:
                print(json.dumps({"rows": sum(counts.values()), "counts": counts}), flush=True)
    report = {
        "counts": counts,
        "tokens": tokens,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "scope": "Structural mapping preflight only; export length, masks and GLM review still required",
    }
    (output / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest, args.config, args.output)


if __name__ == "__main__":
    main()
