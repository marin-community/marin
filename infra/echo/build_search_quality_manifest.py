#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the deduplicated Echo search-quality replay manifest."""

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import search_config

ManifestSource = Literal["observed", "feedback-only", "benchmark"]


@dataclass(frozen=True)
class ObservedDomainSet:
    domains: tuple[search_config.SearchDomain, ...]
    occurrences: int


@dataclass(frozen=True)
class ManifestCase:
    id: str
    query: str
    domains: tuple[search_config.SearchDomain, ...]
    source: ManifestSource
    occurrences: int = 1
    observed_domain_sets: tuple[ObservedDomainSet, ...] = ()
    benchmark_split: str | None = None
    benchmark_intent: str | None = None

    @classmethod
    def from_json(cls, value: object) -> "ManifestCase":
        if not isinstance(value, dict):
            raise ValueError("replay case must be an object")
        identifier = value.get("id")
        query = value.get("query")
        source = value.get("source")
        occurrences = value.get("occurrences", 1)
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("id must be a nonblank string")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"{identifier}: query must be a nonblank string")
        if source not in ("observed", "feedback-only", "benchmark"):
            raise ValueError(f"{identifier}: unknown source {source!r}")
        if not isinstance(occurrences, int) or occurrences < 1:
            raise ValueError(f"{identifier}: occurrences must be a positive integer")
        return cls(
            identifier,
            query.strip(),
            checked_domains(value.get("domains"), identifier),
            cast(ManifestSource, source),
            occurrences,
        )

    def json_value(self) -> dict[str, object]:
        value: dict[str, object] = {
            "id": self.id,
            "query": self.query,
            "domains": list(self.domains),
            "source": self.source,
            "occurrences": self.occurrences,
        }
        if self.observed_domain_sets:
            value["observed_domain_sets"] = [
                {"domains": list(domain_set.domains), "occurrences": domain_set.occurrences}
                for domain_set in self.observed_domain_sets
            ]
        if self.benchmark_split is not None:
            value["benchmark_split"] = self.benchmark_split
        if self.benchmark_intent is not None:
            value["benchmark_intent"] = self.benchmark_intent
        return value


def checked_domains(value: object, identifier: str) -> tuple[search_config.SearchDomain, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{identifier}: domains must be a nonempty list")
    for domain in value:
        if domain not in search_config.SEARCH_DOMAINS:
            raise ValueError(f"{identifier}: unknown domain {domain!r}")
    return tuple(cast(search_config.SearchDomain, domain) for domain in value)


def json_objects(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.lstrip().startswith("{")]


def observed_cases(path: Path) -> list[ManifestCase]:
    groups: dict[str, list[dict[str, object]]] = {}
    for entry in json_objects(path):
        if entry.get("mode") != "federated":
            continue
        query = str(entry["query"])
        groups.setdefault(search_config.normalize_query(query), []).append(entry)

    cases = []
    for normalized, entries in sorted(groups.items()):
        domain_counts = Counter(checked_domains(entry["domains"], "observed query") for entry in entries)
        domains = domain_counts.most_common(1)[0][0]
        identifier = hashlib.sha256(normalized.encode()).hexdigest()[:12]
        cases.append(
            ManifestCase(
                id=f"observed-{identifier}",
                query=str(entries[-1]["query"]),
                domains=domains,
                source="observed",
                occurrences=len(entries),
                observed_domain_sets=tuple(
                    ObservedDomainSet(domain_set, count) for domain_set, count in domain_counts.most_common()
                ),
            )
        )
    return cases


def benchmark_cases(path: Path) -> list[ManifestCase]:
    cases = []
    for entry in json_objects(path):
        raw_domains = entry["domains"] or list(search_config.DEFAULT_SEARCH_DOMAINS)
        identifier = f"benchmark-{entry['id']}"
        cases.append(
            ManifestCase(
                id=identifier,
                query=str(entry["query"]),
                domains=checked_domains(raw_domains, identifier),
                source="benchmark",
                benchmark_split=str(entry["split"]),
                benchmark_intent=str(entry["intent"]),
            )
        )
    return cases


def build_manifest(query_log: Path, benchmark: Path, extra_queries: list[str]) -> list[ManifestCase]:
    cases = observed_cases(query_log)
    cases.extend(
        ManifestCase(
            id=f"feedback-{hashlib.sha256(search_config.normalize_query(query).encode()).hexdigest()[:12]}",
            query=query,
            domains=search_config.DEFAULT_SEARCH_DOMAINS,
            source="feedback-only",
        )
        for query in extra_queries
    )
    cases.extend(benchmark_cases(benchmark))
    normalized = [search_config.normalize_query(case.query) for case in cases]
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
        print(json.dumps(case.json_value(), sort_keys=True))


if __name__ == "__main__":
    main()
