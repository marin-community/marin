# /// script
# requires-python = ">=3.12"
# dependencies = ["ijson==3.6.0"]
# ///
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stream selected metrics to the pinned reducer without materializing raw exports."""

import argparse
import json
import subprocess
from pathlib import Path

import ijson
from ijson.common import ObjectBuilder


def collect(source: Path, reducer: Path, framework: str) -> bytes:
    wanted = set(json.loads(subprocess.check_output(["node", str(reducer), "--keys"])))
    phases = {"metrics": {}, "warmup_metrics": {}}
    with source.open("rb") as stream:
        events = iter(ijson.parse(stream, use_float=True))
        for prefix, event, key in events:
            if prefix not in phases or event != "map_key" or key not in wanted:
                continue
            builder = ObjectBuilder()
            _, event, value = next(events)
            builder.event(event, value)
            depth = int(event in {"start_map", "start_array"})
            while depth:
                _, event, value = next(events)
                builder.event(event, value)
                depth += int(event in {"start_map", "start_array"}) - int(event in {"end_map", "end_array"})
            phases[prefix][key] = builder.value
    profile = {"isl": [], "osl": []}
    with source.with_name("profile_export.jsonl").open() as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            metadata = record.get("metadata", {})
            if metadata.get("benchmark_phase", "profiling") != "profiling" or metadata.get("was_cancelled"):
                continue
            for key, metric in [("isl", "input_sequence_length"), ("osl", "output_sequence_length")]:
                value = record["metrics"].get(metric)
                if isinstance(value, dict):
                    value = value.get("value")
                if isinstance(value, (float, int)):
                    profile[key].append(value)
    return subprocess.check_output(
        ["node", str(reducer), framework], input=json.dumps(phases | {"profile": profile}, allow_nan=False).encode()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--framework", required=True)
    args = parser.parse_args()
    reducer = Path(__file__).resolve().parent.parent / ".build/reduce-series.mjs"
    if not reducer.is_file():
        raise FileNotFoundError("Build Knotmeter before ingestion: npm ci && npm run build")
    print(collect(args.source, reducer, args.framework).decode())


if __name__ == "__main__":
    main()
