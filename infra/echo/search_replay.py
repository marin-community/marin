#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Replay an Echo query manifest through normal federated search.

Each successful request is durably recorded by Echo. The output JSONL maps the
manifest case to the resulting search execution ID and is safe to resume.
"""

import argparse
import json
import time
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

import cli as echo_cli
import search_config

MAX_WORKERS = 4
RATE_LIMIT_RETRY_DELAYS = (15, 30, 60, 120)
REPLAY_REQUEST_TIMEOUT = 180


@dataclass(frozen=True)
class ReplayCase:
    id: str
    query: str
    domains: tuple[search_config.SearchDomain, ...]
    source: str
    occurrences: int


@dataclass(frozen=True)
class ReplayResult:
    id: str
    execution_id: int
    returned_count: int


def checked_case(value: object) -> ReplayCase:
    if not isinstance(value, dict):
        raise ValueError("replay case must be an object")
    identifier = value.get("id")
    query = value.get("query")
    source = value.get("source")
    domains = value.get("domains")
    occurrences = value.get("occurrences", 1)
    if not isinstance(identifier, str) or not identifier:
        raise ValueError("id must be a nonblank string")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"{identifier}: query must be a nonblank string")
    if not isinstance(source, str) or not source:
        raise ValueError(f"{identifier}: source must be a nonblank string")
    if not isinstance(domains, list) or not domains:
        raise ValueError(f"{identifier}: domains must be a nonempty list")
    if not isinstance(occurrences, int) or occurrences < 1:
        raise ValueError(f"{identifier}: occurrences must be a positive integer")
    for domain in domains:
        if domain not in search_config.SEARCH_DOMAINS:
            raise ValueError(f"{identifier}: unknown domain {domain!r}")
    return ReplayCase(identifier, query.strip(), tuple(domains), source, occurrences)


def load_cases(path: Path) -> list[ReplayCase]:
    cases: list[ReplayCase] = []
    seen_ids: set[str] = set()
    seen_queries: set[str] = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = checked_case(json.loads(line))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        normalized_query = " ".join(case.query.casefold().split())
        if case.id in seen_ids:
            raise ValueError(f"{path}:{line_number}: duplicate id {case.id!r}")
        if normalized_query in seen_queries:
            raise ValueError(f"{path}:{line_number}: duplicate normalized query {case.query!r}")
        seen_ids.add(case.id)
        seen_queries.add(normalized_query)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} contains no replay cases")
    return cases


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()}


def reconcile_history(cases: Sequence[ReplayCase], history: Path, output: Path) -> int:
    """Recover completed cases whose response was lost after server persistence."""
    cases_by_query = {" ".join(case.query.casefold().split()): case for case in cases}
    completed = completed_ids(output)
    recovered: dict[str, ReplayResult] = {}
    for line in history.read_text().splitlines():
        if not line.strip():
            continue
        execution = json.loads(line)
        case = cases_by_query.get(execution["normalized_query"])
        if case is None or case.id in completed or case.id in recovered:
            continue
        if tuple(execution["domains"]) != case.domains:
            continue
        recovered[case.id] = ReplayResult(case.id, execution["id"], execution["returned_count"])
    with output.open("a") as stream:
        for case in cases:
            if case.id not in recovered:
                continue
            stream.write(json.dumps(recovered[case.id].__dict__, sort_keys=True) + "\n")
    return len(recovered)


def replay_one(case: ReplayCase, limit: int) -> ReplayResult:
    response = None
    for attempt in range(len(RATE_LIMIT_RETRY_DELAYS) + 1):
        try:
            response = echo_cli.request_response(
                "GET",
                "/federated-search",
                params={"q": case.query, "domain": list(case.domains), "limit": limit},
                timeout=REPLAY_REQUEST_TIMEOUT,
            )
            break
        except SystemExit as error:
            if "-> 429:" not in str(error) or attempt == len(RATE_LIMIT_RETRY_DELAYS):
                raise
            time.sleep(RATE_LIMIT_RETRY_DELAYS[attempt])
    assert response is not None
    execution_id = response.headers.get("X-Echo-Search-Execution-ID")
    if execution_id is None:
        raise RuntimeError(f"{case.id}: Echo response omitted the execution ID")
    results = echo_cli.response_objects(response.json())
    return ReplayResult(case.id, int(execution_id), len(results))


def replay(cases: Sequence[ReplayCase], output: Path, limit: int, workers: int) -> None:
    completed = completed_ids(output)
    pending = iter(case for case in cases if case.id not in completed)
    with output.open("a") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[ReplayResult], ReplayCase] = {}
        for case in pending:
            futures[pool.submit(replay_one, case, limit)] = case
            if len(futures) == workers:
                break
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                case = futures.pop(future)
                try:
                    result = future.result()
                except (Exception, SystemExit):
                    for other in futures:
                        other.cancel()
                    raise
                stream.write(json.dumps(result.__dict__, sort_keys=True) + "\n")
                stream.flush()
                completed.add(case.id)
                print(f"{len(completed)}/{len(cases)} {case.id} -> execution {result.execution_id}")
                next_case = next(pending, None)
                if next_case is not None:
                    futures[pool.submit(replay_one, next_case, limit)] = next_case


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--limit", type=int, default=10, choices=range(1, search_config.MAX_SEARCH_LIMIT + 1))
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, choices=range(1, MAX_WORKERS + 1))
    parser.add_argument("--history", type=Path, help="exported history used to recover server-completed requests")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = load_cases(args.manifest)
    if args.history is not None:
        recovered = reconcile_history(cases, args.history, args.output)
        print(f"recovered {recovered} completed cases from durable history")
    replay(cases, args.output, args.limit, args.workers)


if __name__ == "__main__":
    main()
