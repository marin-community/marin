#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect and score Echo search results against graded relevance judgments."""

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import cli as echo_cli
import search_config
from search_result import SearchResult

BenchmarkSplit = Literal["dev", "test"]


@dataclass(frozen=True)
class RelevanceJudgment:
    domain: search_config.SearchDomain
    target: str
    grade: int


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    query: str
    domains: tuple[search_config.SearchDomain, ...]
    intent: str
    split: BenchmarkSplit
    relevant: tuple[RelevanceJudgment, ...]


@dataclass(frozen=True)
class CapturedSearch:
    id: str
    latency_ms: float
    results: tuple[SearchResult, ...]


@dataclass(frozen=True)
class QueryMetrics:
    reciprocal_rank: float
    ndcg: float
    hit: bool
    recalled: int
    relevant: int


def checked_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonblank string")
    return value


def checked_domain(value: object) -> search_config.SearchDomain:
    if value not in search_config.SEARCH_DOMAINS:
        raise ValueError(f"unknown search domain {value!r}")
    return value


def checked_split(value: object) -> BenchmarkSplit:
    if value not in ("dev", "test"):
        raise ValueError(f"unknown benchmark split {value!r}")
    return value


def benchmark_case(value: object) -> BenchmarkCase:
    if not isinstance(value, dict):
        raise ValueError("benchmark case must be an object")
    domains_value = value.get("domains")
    relevant_value = value.get("relevant")
    if not isinstance(domains_value, list):
        raise ValueError("domains must be a list")
    if not isinstance(relevant_value, list):
        raise ValueError("relevant must be a list")
    relevant = []
    for item in relevant_value:
        if not isinstance(item, dict):
            raise ValueError("relevance judgment must be an object")
        grade = item.get("grade")
        if not isinstance(grade, int) or not 1 <= grade <= 3:
            raise ValueError(f"relevance grade must be an integer from 1 through 3, got {grade!r}")
        relevant.append(
            RelevanceJudgment(
                domain=checked_domain(item.get("domain")),
                target=checked_string(item.get("target"), "relevance target"),
                grade=grade,
            )
        )
    return BenchmarkCase(
        id=checked_string(value.get("id"), "id"),
        query=checked_string(value.get("query"), "query"),
        domains=tuple(checked_domain(domain) for domain in domains_value),
        intent=checked_string(value.get("intent"), "intent"),
        split=checked_split(value.get("split")),
        relevant=tuple(relevant),
    )


def load_benchmark(path: Path) -> list[BenchmarkCase]:
    cases = []
    seen = set()
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = benchmark_case(json.loads(line))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        if case.id in seen:
            raise ValueError(f"{path}:{line_number}: duplicate benchmark id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} contains no benchmark cases")
    return cases


def captured_search(value: object) -> CapturedSearch:
    if not isinstance(value, dict):
        raise ValueError("captured search must be an object")
    results = value.get("results")
    if not isinstance(results, list):
        raise ValueError("captured results must be a list")
    latency_ms = value.get("latency_ms")
    if not isinstance(latency_ms, int | float) or latency_ms < 0:
        raise ValueError("latency_ms must be nonnegative")
    return CapturedSearch(
        id=checked_string(value.get("id"), "id"),
        latency_ms=float(latency_ms),
        results=tuple(SearchResult.from_json(result) for result in results),
    )


def load_captured(path: Path) -> dict[str, CapturedSearch]:
    captures = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            capture = captured_search(json.loads(line))
        except (json.JSONDecodeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number}: {error}") from error
        if capture.id in captures:
            raise ValueError(f"{path}:{line_number}: duplicate captured id {capture.id!r}")
        captures[capture.id] = capture
    return captures


def github_artifact(url: str) -> str | None:
    parts = urlsplit(url)
    path_parts = parts.path.rstrip("/").split("/")
    for kind in ("pull", "issues"):
        if kind not in path_parts:
            continue
        index = path_parts.index(kind)
        if index + 1 < len(path_parts) and path_parts[index + 1].isdigit():
            domain = "pr" if kind == "pull" else "issue"
            return f"{domain}:{'/'.join(path_parts[: index + 2])}"
    return None


def judgment_key(judgment: RelevanceJudgment) -> str:
    target = judgment.target
    if judgment.domain == "file":
        return f"file:{target.removeprefix('file:')}"
    if judgment.domain == "wiki":
        if target.startswith("wiki:"):
            return target
        identifier = urlsplit(target).path.rstrip("/").rsplit("/", 1)[-1]
        return f"wiki:{identifier}"
    if judgment.domain in ("pr", "issue"):
        artifact = github_artifact(target)
        if artifact is None:
            raise ValueError(f"relevance target is not a GitHub PR or issue URL: {target}")
        return artifact
    return f"discord:{target.rstrip('/')}"


def result_key(result: SearchResult) -> str:
    if result.domain in ("file", "wiki"):
        return result.id
    if result.domain in ("pr", "issue"):
        return github_artifact(result.url) or f"{result.domain}:{result.url.rstrip('/')}"
    return f"discord:{result.url.rstrip('/')}"


def query_metrics(case: BenchmarkCase, results: Sequence[SearchResult], limit: int) -> QueryMetrics:
    judgments = {judgment_key(judgment): judgment.grade for judgment in case.relevant}
    seen = set()
    gains = []
    first_rank = 0
    recalled = 0
    for rank, result in enumerate(results[:limit], start=1):
        key = result_key(result)
        grade = judgments.get(key, 0) if key not in seen else 0
        seen.add(key)
        gains.append(grade)
        if grade:
            recalled += 1
            if not first_rank:
                first_rank = rank
    ideal = sorted(judgments.values(), reverse=True)[:limit]
    dcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(gains, start=1))
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
    return QueryMetrics(
        reciprocal_rank=1 / first_rank if first_rank else 0.0,
        ndcg=dcg / idcg if idcg else 0.0,
        hit=bool(first_rank),
        recalled=recalled,
        relevant=len(judgments),
    )


def mean(values: Iterable[float]) -> float:
    collected = list(values)
    return statistics.fmean(collected) if collected else 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(index, 0)]


def evaluation(
    cases: Sequence[BenchmarkCase],
    captures: dict[str, CapturedSearch],
    limit: int,
) -> dict[str, object]:
    missing = [case.id for case in cases if case.id not in captures]
    if missing:
        raise ValueError(f"missing captured searches: {', '.join(missing)}")
    answerable = [case for case in cases if case.relevant]
    no_answer = [case for case in cases if not case.relevant]
    metrics = [query_metrics(case, captures[case.id].results, limit) for case in answerable]
    no_answer_correct = sum(not captures[case.id].results for case in no_answer)
    latencies = [captures[case.id].latency_ms for case in cases]
    return {
        "queries": len(cases),
        "answerable_queries": len(answerable),
        "no_answer_queries": len(no_answer),
        f"mrr@{limit}": mean(metric.reciprocal_rank for metric in metrics),
        f"ndcg@{limit}": mean(metric.ndcg for metric in metrics),
        f"hit_rate@{limit}": mean(float(metric.hit) for metric in metrics),
        f"judgment_recall@{limit}": (
            sum(metric.recalled for metric in metrics) / sum(metric.relevant for metric in metrics) if metrics else 0.0
        ),
        "no_answer_accuracy": no_answer_correct / len(no_answer) if no_answer else 0.0,
        "mean_results": mean(len(captures[case.id].results) for case in cases),
        "latency_ms_p50": percentile(latencies, 0.50),
        "latency_ms_p95": percentile(latencies, 0.95),
    }


def select_cases(cases: Sequence[BenchmarkCase], split: str) -> list[BenchmarkCase]:
    return [case for case in cases if split == "all" or case.split == split]


def collect_one(case: BenchmarkCase, limit: int) -> tuple[BenchmarkCase, CapturedSearch]:
    start = time.perf_counter()
    value = echo_cli.response_objects(
        echo_cli.request(
            "GET",
            "/federated-search",
            params={
                "q": case.query,
                "domain": list(case.domains or search_config.DEFAULT_SEARCH_DOMAINS),
                "limit": limit,
            },
        )
    )
    elapsed = (time.perf_counter() - start) * 1_000
    return case, CapturedSearch(case.id, elapsed, tuple(SearchResult.from_json(result) for result in value))


def write_captures(path: Path, captures: Sequence[CapturedSearch]) -> None:
    with path.open("w") as output:
        for capture in captures:
            value = {
                "id": capture.id,
                "latency_ms": round(capture.latency_ms, 3),
                "results": [
                    {
                        "id": result.id,
                        "domain": result.domain,
                        "title": result.title,
                        "subtitle": result.subtitle,
                        "url": result.url,
                        "snippet": result.snippet,
                        "score": result.score,
                        "distance": result.distance,
                        "lexical_score": result.lexical_score,
                        "references": [
                            {"line": reference.line, "text": reference.text, "url": reference.url}
                            for reference in result.references
                        ],
                    }
                    for result in capture.results
                ],
            }
            output.write(json.dumps(value, sort_keys=True) + "\n")


def collect(args: argparse.Namespace) -> None:
    cases = select_cases(load_benchmark(args.benchmark), args.split)
    captures: dict[str, CapturedSearch] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(collect_one, case, args.limit): case for case in cases}
        for completed, future in enumerate(as_completed(pending), start=1):
            case, capture = future.result()
            captures[case.id] = capture
            print(
                f"[{completed}/{len(cases)}] {case.id}: {len(capture.results)} results "
                f"in {capture.latency_ms:.0f} ms",
                file=sys.stderr,
            )
    write_captures(args.output, [captures[case.id] for case in cases])


def evaluate(args: argparse.Namespace) -> None:
    cases = select_cases(load_benchmark(args.benchmark), args.split)
    print(json.dumps(evaluation(cases, load_captured(args.results), args.limit), indent=2, sort_keys=True))


def bounded_workers(value: str) -> int:
    workers = int(value)
    if not 1 <= workers <= 16:
        raise argparse.ArgumentTypeError("must be between 1 and 16")
    return workers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    collect_parser = commands.add_parser("collect", help="capture live Echo results as JSONL")
    collect_parser.add_argument("benchmark", type=Path)
    collect_parser.add_argument("output", type=Path)
    collect_parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    collect_parser.add_argument("--limit", type=echo_cli.bounded_limit, default=10)
    collect_parser.add_argument("--workers", type=bounded_workers, default=4)
    collect_parser.set_defaults(func=collect)

    evaluate_parser = commands.add_parser("evaluate", help="score captured Echo result JSONL")
    evaluate_parser.add_argument("benchmark", type=Path)
    evaluate_parser.add_argument("results", type=Path)
    evaluate_parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    evaluate_parser.add_argument("--limit", type=echo_cli.bounded_limit, default=10)
    evaluate_parser.set_defaults(func=evaluate)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
