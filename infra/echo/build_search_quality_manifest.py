#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the deduplicated Echo search-quality replay manifest."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import search_config


def normalized_query(query: str) -> str:
    return " ".join(query.casefold().split())


def json_objects(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.lstrip().startswith("{")]


def observed_cases(path: Path) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = {}
    for entry in json_objects(path):
        if entry.get("mode") != "federated":
            continue
        query = str(entry["query"])
        groups.setdefault(normalized_query(query), []).append(entry)

    cases = []
    for normalized, entries in sorted(groups.items()):
        domain_counts = Counter(tuple(entry["domains"]) for entry in entries)
        domains = list(domain_counts.most_common(1)[0][0])
        identifier = hashlib.sha256(normalized.encode()).hexdigest()[:12]
        cases.append(
            {
                "id": f"observed-{identifier}",
                "query": entries[-1]["query"],
                "domains": domains,
                "source": "observed",
                "occurrences": len(entries),
                "observed_domain_sets": [
                    {"domains": list(domain_set), "occurrences": count}
                    for domain_set, count in domain_counts.most_common()
                ],
            }
        )
    return cases


def benchmark_cases(path: Path) -> list[dict[str, object]]:
    cases = []
    for entry in json_objects(path):
        domains = entry["domains"] or list(search_config.DEFAULT_SEARCH_DOMAINS)
        cases.append(
            {
                "id": f"benchmark-{entry['id']}",
                "query": entry["query"],
                "domains": domains,
                "source": "benchmark",
                "occurrences": 1,
                "benchmark_split": entry["split"],
                "benchmark_intent": entry["intent"],
            }
        )
    return cases


def build_manifest(query_log: Path, benchmark: Path, extra_queries: list[str]) -> list[dict[str, object]]:
    cases = observed_cases(query_log)
    cases.extend(
        {
            "id": f"feedback-{hashlib.sha256(normalized_query(query).encode()).hexdigest()[:12]}",
            "query": query,
            "domains": list(search_config.DEFAULT_SEARCH_DOMAINS),
            "source": "feedback-only",
            "occurrences": 1,
        }
        for query in extra_queries
    )
    cases.extend(benchmark_cases(benchmark))
    normalized = [normalized_query(str(case["query"])) for case in cases]
    if len(normalized) != len(set(normalized)):
        duplicates = sorted(query for query, count in Counter(normalized).items() if count > 1)
        raise ValueError(f"duplicate queries across manifest sources: {', '.join(duplicates)}")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_log", type=Path, help="sanitized Cloud Logging Weaver artifact")
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--extra-query", action="append", default=[], help="feedback query absent from the log slice")
    args = parser.parse_args()
    for case in build_manifest(args.query_log, args.benchmark, args.extra_query):
        print(json.dumps(case, sort_keys=True))


if __name__ == "__main__":
    main()
