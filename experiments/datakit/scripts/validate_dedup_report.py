# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate a dedup report against its exact artifact counters and embedded data."""

import argparse
import json
import math
from typing import Any

from marin.execution.artifact import read_record
from rigging.filesystem import StoragePath

COUNTER_PREFIX = "dedup/fuzzy/document"
DATA_PREFIX = "const D = "
DATA_SUFFIX = ";\nconst fmt"


def _artifact_result(path: str) -> dict[str, Any]:
    record = read_record(path)
    if record is None or not isinstance(record.result, dict):
        raise FileNotFoundError(f"No artifact result at {path}")
    return record.result


def _embedded_data(html: str) -> dict[str, Any]:
    start = html.index(DATA_PREFIX) + len(DATA_PREFIX)
    end = html.index(DATA_SUFFIX, start)
    result = json.loads(html[start:end])
    if not isinstance(result, dict):
        raise TypeError("Embedded report data is not an object")
    return result


def validate_report(dedup_path: str, report_path: str, expected_docs: int | None) -> dict[str, Any]:
    """Assert exact report/data consistency and return the validated headline stats."""
    dedup = _artifact_result(dedup_path)
    report = _artifact_result(report_path)
    html_path = report["html_path"]
    html = StoragePath(html_path).read_text()
    data = _embedded_data(html)

    counters = dedup["counters"]
    cluster_members = int(counters.get(f"{COUNTER_PREFIX}/cluster_members", 0))
    clusters = int(counters.get(f"{COUNTER_PREFIX}/canonicals", 0))
    singletons = int(counters.get(f"{COUNTER_PREFIX}/singletons_skipped", 0))
    transitive_kept = int(counters.get(f"{COUNTER_PREFIX}/transitive_members_kept", 0))
    duplicates_to_drop = cluster_members - clusters
    total_docs = cluster_members + singletons + transitive_kept
    expected_stats: dict[str, int | float] = {
        "cluster_members": cluster_members,
        "clusters": clusters,
        "duplicates_to_drop": duplicates_to_drop,
        "singletons_skipped": singletons,
        "dup_rate": duplicates_to_drop / total_docs if total_docs else 0.0,
        "n_sources": len(dedup["sources"]),
    }
    ngram_kind = dedup["params"].get("ngram_kind", "char")
    if ngram_kind == "word":
        expected_stats["transitive_members_kept"] = transitive_kept
    if report["stats"] != expected_stats:
        raise AssertionError(f"Report artifact stats differ: actual={report['stats']}, expected={expected_stats}")
    if data["stats"] != expected_stats:
        raise AssertionError(f"Embedded stats differ: actual={data['stats']}, expected={expected_stats}")
    if expected_docs is not None and total_docs != expected_docs:
        raise AssertionError(f"Report accounts for {total_docs} documents, expected {expected_docs}")
    if duplicates_to_drop < 0:
        raise AssertionError(f"Negative duplicates_to_drop: {duplicates_to_drop}")
    if not math.isfinite(data["stats"]["dup_rate"]):
        raise AssertionError(f"Non-finite duplicate rate: {data['stats']['dup_rate']}")

    sources = data["sources"]
    if len(sources) != len(dedup["sources"]):
        raise AssertionError(f"Embedded source count is {len(sources)}, expected {len(dedup['sources'])}")
    if {source["source_main_dir"] for source in sources} != set(dedup["sources"]):
        raise AssertionError("Embedded source directories differ from the dedup artifact")
    for source in sources:
        if not 0 <= source["sampled_clusters"] <= source["sampled_members"] <= data["sample_limit"]:
            raise AssertionError(f"Invalid per-source sample counts: {source}")

    sampled_members = sum(source["sampled_members"] for source in sources)
    histogram_members = sum(row["size"] * row["clusters"] for row in data["cluster_size_hist"])
    if histogram_members != sampled_members:
        raise AssertionError(
            f"Cluster histogram accounts for {histogram_members} sampled members, expected {sampled_members}"
        )
    if "__DATA__" in html or "__TITLE__" in html:
        raise AssertionError("Report contains an unrendered template placeholder")
    if "<title>Datakit dedup</title>" not in html:
        raise AssertionError("Report title is missing")

    return {
        "valid": True,
        "dedup_path": dedup_path,
        "report_path": report_path,
        "html_path": html_path,
        "total_docs": total_docs,
        "sampled_members": sampled_members,
        "stats": expected_stats,
        "params": data["params"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dedup", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--expected-docs", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = validate_report(args.dedup, args.report, args.expected_docs)
    payload = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        StoragePath(args.output).write_text(payload)
    print(payload)


if __name__ == "__main__":
    main()
