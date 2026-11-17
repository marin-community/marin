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
from collections.abc import Iterator
from contextlib import contextmanager
from io import TextIOWrapper
from pathlib import Path
from typing import IO, Any

import ijson
from ijson.common import ObjectBuilder


def skip_value(events: Iterator[tuple[str, str, Any]]) -> None:
    _, event, _ = next(events)
    depth = int(event in {"start_map", "start_array"})
    while depth:
        _, event, _ = next(events)
        depth += int(event in {"start_map", "start_array"}) - int(event in {"end_map", "end_array"})


def chart_metric(events: Iterator[tuple[str, str, Any]], prefix: str) -> dict[str, Any]:
    """Retain only fields read by the pinned reducer, including all original slices."""
    allowed = {
        prefix: {"series"},
        f"{prefix}.series.item": {"endpoint_url", "labels", "timeslices"},
        f"{prefix}.series.item.timeslices.item": {"start_ns", "end_ns", "avg", "rate", "sum", "count"},
    }
    builder = ObjectBuilder()
    _, event, value = next(events)
    builder.event(event, value)
    depth = int(event in {"start_map", "start_array"})
    while depth:
        path, event, value = next(events)
        if event == "map_key" and path in allowed and value not in allowed[path]:
            skip_value(events)
            continue
        builder.event(event, value)
        depth += int(event in {"start_map", "start_array"}) - int(event in {"end_map", "end_array"})
    value = builder.value
    assert isinstance(value, dict)
    return value


@contextmanager
def export_stream(source: Path) -> Iterator[IO[bytes]]:
    if source.suffix != ".zst":
        with source.open("rb") as stream:
            yield stream
        return
    with subprocess.Popen(["zstd", "--decompress", "--stdout", str(source)], stdout=subprocess.PIPE) as process:
        assert process.stdout is not None
        with process.stdout as stream:
            yield stream
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)


def collect(source: Path, reducer: Path, framework: str, endpoint: str | None) -> bytes:
    wanted = set(json.loads(subprocess.check_output(["node", str(reducer), "--keys"])))
    phases = {"metrics": {}, "warmup_metrics": {}}
    with export_stream(source) as stream:
        events = iter(ijson.parse(stream, use_float=True))
        for prefix, event, key in events:
            if prefix not in phases or event != "map_key" or key not in wanted:
                continue
            metric = chart_metric(events, f"{prefix}.{key}")
            if endpoint is not None:
                metric["series"] = [s for s in metric.get("series", []) if s.get("endpoint_url") == endpoint]
            phases[prefix][key] = metric
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
    with subprocess.Popen(["node", str(reducer), framework], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        assert process.stdin is not None and process.stdout is not None
        with TextIOWrapper(process.stdin, encoding="utf-8") as stream:
            json.dump(phases | {"profile": profile}, stream, allow_nan=False)
        phases.clear()
        output = process.stdout.read()
        if process.wait() != 0:
            raise subprocess.CalledProcessError(process.returncode, process.args)
        return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--framework", required=True)
    parser.add_argument("--endpoint", help="Use the recorded worker endpoint when the router mirrors its counters")
    args = parser.parse_args()
    reducer = Path(__file__).resolve().parent.parent / ".build/reduce-series.mjs"
    if not reducer.is_file():
        raise FileNotFoundError("Build Knotmeter before ingestion: npm ci && npm run build")
    print(collect(args.source, reducer, args.framework, args.endpoint).decode())


if __name__ == "__main__":
    main()
