# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect a bounded public GitHub commit inventory for the RSI experiment.

Inventory files contain answers. Do not mount them in an agent container.
This CLI collects metadata only. It does not establish executable task quality.
"""

import argparse
import base64
import hashlib
import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

from experiments.post_training.russell_rsi.sources import SourceSnapshot

HISTORIC_REPOSITORIES = ("spartan-array/spartan", "rjpower/piccolo", "marin-community/marin")
SEARCH_LIMIT = 1000


@dataclass
class CommitRecord:
    """Answer-side commit metadata and repository provenance."""

    sha: str
    tree_sha: str
    author: dict[str, str]
    author_login: str | None
    committer: dict[str, str]
    message: str
    parents: list[str]
    repositories: list[str]
    discovery_queries: list[str]
    family: str = ""
    split: str = ""


@dataclass
class QueryRecord:
    query: str
    total_count: int
    retrieved: int
    incomplete_results: bool
    status: str


@dataclass
class Inventory:
    commits: dict[str, CommitRecord] = field(default_factory=dict)
    repositories: dict[str, dict[str, Any]] = field(default_factory=dict)
    queries: list[QueryRecord] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)


def source_snapshot(
    api: "GitHub", specification: dict[str, Any], inventory: Inventory, max_bytes: int = 2_000_000
) -> SourceSnapshot:
    """Read explicit source paths and revision licenses without a repository clone."""
    repository = specification["repository"]
    commit_sha = specification["commit_sha"]
    record = inventory.commits[commit_sha]
    if repository not in record.repositories or not record.split or not record.family:
        raise ValueError("Snapshot requires an assigned inventory record")
    repository_endpoint = f"repos/{quote(repository, safe='/')}"
    metadata = api.get(f"{repository_endpoint}/commits/{quote(commit_sha, safe='')}")
    if len(metadata["files"]) >= 300:
        raise ValueError("Commit file metadata can be truncated at 300 files")
    if len(metadata["parents"]) != 1:
        raise ValueError("Snapshots require one parent")
    parent_sha = metadata["parents"][0]["sha"]
    if record.parents != [parent_sha]:
        raise ValueError("Snapshot parent differs from inventory provenance")
    additions = {item["filename"] for item in metadata["files"] if item["status"] == "added"}
    license_paths = tuple(specification["license_paths"])
    if not license_paths:
        raise ValueError("A revision license is necessary")
    paths = sorted(set(specification["source_paths"]) | set(license_paths))
    if any(
        item["status"] in {"renamed", "removed"}
        and (item["filename"] in paths or item.get("previous_filename") in paths)
        for item in metadata["files"]
    ):
        raise ValueError("Snapshot export does not support renamed or removed source paths")
    if len(paths) > 100:
        raise ValueError("Snapshot file budget exceeded")
    for name in paths:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or ".git" in path.parts:
            raise ValueError(f"Unsafe snapshot path: {name}")
        if (
            {"test", "tests", "testing"}.intersection(path.parts)
            or path.name.startswith("test_")
            or path.name.endswith("_test.py")
            or path.name == "conftest.py"
        ):
            raise ValueError("Changed tests cannot be reference source")
    trees = []
    size = 0
    for sha in (parent_sha, commit_sha):
        files = {}
        for name in paths:
            if sha == parent_sha and name in additions and name not in license_paths:
                continue
            content = api.get(f"{repository_endpoint}/contents/{quote(name, safe='/')}?" + urlencode({"ref": sha}))
            if content.get("type") != "file" or content.get("encoding") != "base64":
                raise ValueError(f"Not a regular text file: {name}")
            size += content["size"]
            if size > max_bytes:
                raise ValueError("Snapshot byte budget exceeded")
            files[name] = base64.b64decode(content["content"]).decode("utf-8")
        trees.append(files)
    return SourceSnapshot(
        repository=repository,
        parent_sha=parent_sha,
        commit_sha=commit_sha,
        parent_files=trees[0],
        reference_files=trees[1],
        license_paths=license_paths,
        split=record.split,
    )


class RequestBudgetExhausted(RuntimeError):
    """Stop discovery when the API read budget is empty."""


class GitHub:
    """Run cached GitHub API reads through the local gh credentials."""

    def __init__(self, cache: Path, max_requests: int):
        self.cache = cache
        self.max_requests = max_requests
        self.requests = 0
        self.cache_hits = 0
        self.last_search = 0.0
        cache.mkdir(parents=True, exist_ok=True)

    def get(self, endpoint: str) -> Any:
        path = self.cache / (hashlib.sha256(endpoint.encode()).hexdigest() + ".json")
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())
        if self.requests >= self.max_requests:
            raise RequestBudgetExhausted("GitHub request budget exhausted")
        if endpoint.startswith("search/"):
            time.sleep(max(0, 2.1 - (time.monotonic() - self.last_search)))
            self.last_search = time.monotonic()
        self.requests += 1
        result = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True, timeout=60, check=True)
        value = json.loads(result.stdout)
        path.write_text(json.dumps(value))
        return value


def add_commit(inventory: Inventory, item: dict[str, Any], query: str) -> None:
    sha = item["sha"]
    repo = item["repository"]["full_name"]
    inventory.repositories.setdefault(repo, {"full_name": repo, "metadata_status": "not_fetched"})
    if sha in inventory.commits:
        record = inventory.commits[sha]
        record.repositories = sorted(set([*record.repositories, repo]))
        record.discovery_queries = sorted(set([*record.discovery_queries, query]))
        return
    commit = item["commit"]
    inventory.commits[sha] = CommitRecord(
        sha=sha,
        tree_sha=commit["tree"]["sha"],
        author=commit["author"],
        author_login=(item.get("author") or {}).get("login"),
        committer=commit["committer"],
        message=commit["message"],
        parents=[parent["sha"] for parent in item["parents"]],
        repositories=[repo],
        discovery_queries=[query],
    )


def search_commits(api: GitHub, inventory: Inventory, query: str, start: date, end: date, max_pages: int) -> None:
    """Divide search windows at the API limit and record truncated leaves."""
    scoped = f"{query} author-date:{start.isoformat()}..{end.isoformat()}"
    endpoint = "search/commits?" + urlencode({"q": scoped, "per_page": 100, "page": 1})
    try:
        first = api.get(endpoint)
    except RequestBudgetExhausted:
        inventory.queries.append(QueryRecord(scoped, 0, 0, False, "budget_exhausted"))
        raise
    except subprocess.SubprocessError as error:
        inventory.errors.append({"query": scoped, "error": str(error)})
        return
    total = first["total_count"]
    if total > min(SEARCH_LIMIT, max_pages * 100) and start < end:
        inventory.queries.append(QueryRecord(scoped, total, 0, first["incomplete_results"], "partitioned"))
        midpoint = start + (end - start) // 2
        search_commits(api, inventory, query, start, midpoint, max_pages)
        search_commits(api, inventory, query, midpoint + timedelta(days=1), end, max_pages)
        return
    retrieved = 0
    incomplete = first["incomplete_results"]
    for page in range(1, min(max_pages, (min(total, SEARCH_LIMIT) + 99) // 100) + 1):
        if page == 1:
            data = first
        else:
            try:
                data = api.get("search/commits?" + urlencode({"q": scoped, "per_page": 100, "page": page}))
            except RequestBudgetExhausted:
                inventory.queries.append(QueryRecord(scoped, total, retrieved, incomplete, "budget_exhausted"))
                raise
            except subprocess.SubprocessError as error:
                inventory.errors.append({"query": scoped, "error": str(error)})
                break
        incomplete = incomplete or data["incomplete_results"]
        for item in data["items"]:
            add_commit(inventory, item, scoped)
        retrieved += len(data["items"])
    status = "complete" if retrieved >= total and not incomplete else "truncated"
    inventory.queries.append(QueryRecord(scoped, total, retrieved, incomplete, status))


def assign_splits(inventory: Inventory, seed: str) -> None:
    """Group observed repository ancestry, shared SHAs, and identical trees."""
    roots = {repo: repo for repo in inventory.repositories}

    def root(repo: str) -> str:
        while roots[repo] != repo:
            repo = roots[repo]
        return repo

    for repo, metadata in inventory.repositories.items():
        for relation in ("parent", "source"):
            ancestor = (metadata.get(relation) or {}).get("full_name")
            if ancestor:
                roots.setdefault(ancestor, ancestor)
                left, right = sorted([root(repo), root(ancestor)])
                roots[right] = left

    trees: dict[str, str] = {}
    for commit in inventory.commits.values():
        repos = commit.repositories
        if commit.tree_sha in trees:
            repos = [*repos, trees[commit.tree_sha]]
        trees[commit.tree_sha] = repos[0]
        for repo in repos[1:]:
            left, right = sorted([root(repos[0]), root(repo)])
            roots[right] = left
    for commit in inventory.commits.values():
        family = root(commit.repositories[0])
        bucket = int(hashlib.sha256(f"{seed}:{family}".encode()).hexdigest(), 16) % 10
        commit.family = family
        commit.split = "test" if bucket == 0 else "dev" if bucket == 1 else "train"


def read_inventory(path: Path) -> Inventory:
    """Read a frozen inventory and its repository metadata."""
    summary = json.loads(path.with_name("summary.json").read_text())
    commits = [CommitRecord(**json.loads(line)) for line in path.read_text().splitlines() if line.strip()]
    return Inventory(
        commits={record.sha: record for record in commits},
        repositories=summary["repositories"],
        queries=[QueryRecord(**query) for query in summary["queries"]],
        errors=summary["errors"],
    )


def freeze_source_splits(
    inventory: Inventory, repositories: set[str], seed: str, sealed_test_families: set[str]
) -> dict[str, Any]:
    """Freeze eligible families before student results, with development and test data."""
    families = {record.family for record in inventory.commits.values() if repositories.intersection(record.repositories)}
    if len(families) < 3 or not sealed_test_families.issubset(families):
        raise ValueError("Freeze requires three eligible families and valid sealed test families")
    ranked = sorted(
        families - sealed_test_families, key=lambda family: hashlib.sha256(f"{seed}:{family}".encode()).digest()
    )
    assignments = {family: "test" for family in sealed_test_families}
    if not sealed_test_families:
        assignments[ranked.pop(0)] = "test"
    assignments[ranked.pop(0)] = "dev"
    assignments.update({family: "train" for family in ranked})
    for record in inventory.commits.values():
        if record.family in assignments:
            record.split = assignments[record.family]
    return {
        "algorithm": "seed-ranked-eligible-families-v1",
        "seed": seed,
        "eligible_repositories": sorted(repositories),
        "sealed_test_families": sorted(sealed_test_families),
        "assignments": assignments,
        "freeze_basis": "Python source, dependency and license feasibility, before student results",
    }


def extend_source_splits(inventory: Inventory, manifest: dict[str, Any], repositories: set[str]) -> dict[str, Any]:
    """Add eligible training families without moving a frozen family."""
    assignments = dict(manifest["assignments"])
    for record in inventory.commits.values():
        if repositories.intersection(record.repositories):
            assignments.setdefault(record.family, "train")
    for record in inventory.commits.values():
        if record.family in assignments:
            record.split = assignments[record.family]
    return {
        **manifest,
        "eligible_repositories": sorted(set(manifest["eligible_repositories"]) | repositories),
        "assignments": assignments,
        "extension_policy": "New eligible families enter train. Existing family assignments stay frozen.",
    }


def write_inventory(inventory: Inventory, output: Path, api: GitHub, config: dict[str, Any]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = [asdict(record) for _, record in sorted(inventory.commits.items())]
    with (output / "inventory.jsonl").open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "scope": "Bounded public GitHub indexed commit metadata, not all internet commits",
        "answer_artifact": True,
        "config": config,
        "unique_commits": len(rows),
        "repository_count": len(inventory.repositories),
        "years": dict(sorted(Counter(row["author"]["date"][:4] for row in rows).items())),
        "splits": dict(Counter(row["split"] for row in rows)),
        "requests": api.requests,
        "cache_hits": api.cache_hits,
        "queries": [asdict(query) for query in inventory.queries],
        "repositories": inventory.repositories,
        "errors": inventory.errors,
        "candidate_status": "Unverified. No parent-fails/commit-passes claim.",
        "licenses": "Repository metadata is current. Verify license at each selected commit before use.",
        "family_policy": (
            "Observed repository ancestry, shared SHAs and trees. Missing ancestry can leave forks separated."
        ),
        "status": (
            "partial"
            if inventory.errors or any(q.status != "complete" for q in inventory.queries if q.status != "partitioned")
            else "complete"
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--author", default="rjpower")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2008, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--max-requests", type=int, default=160)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--split-seed", default="russell-rsi-v1")
    parser.add_argument("--repo", action="append", default=[])
    parser.add_argument("--global-search", action="store_true")
    parser.add_argument("--snapshot-spec", type=Path)
    parser.add_argument("--inventory", type=Path)
    args = parser.parse_args()
    api = GitHub(args.output / "api-cache", args.max_requests)
    if args.snapshot_spec:
        inventory = read_inventory(args.inventory or args.output / "inventory.jsonl")
        specifications = json.loads(args.snapshot_spec.read_text())
        snapshots = [source_snapshot(api, specification, inventory) for specification in specifications]
        with (args.output / "snapshots.jsonl").open("w") as stream:
            for snapshot in snapshots:
                stream.write(snapshot.model_dump_json() + "\n")
        snapshot_summary = {
            "scope": "Explicit source paths only. Other commit files are outside each snapshot.",
            "answer_artifact": True,
            "specification": str(args.snapshot_spec),
            "inventory": str(args.inventory or args.output / "inventory.jsonl"),
            "split_seed": args.split_seed,
            "snapshots": len(snapshots),
            "requests": api.requests,
            "cache_hits": api.cache_hits,
            "source_bytes": sum(
                len(source.encode())
                for snapshot in snapshots
                for files in (snapshot.parent_files, snapshot.reference_files)
                for source in files.values()
            ),
            "qualification": "Source snapshots only. Isolated parent/reference acceptance is necessary.",
        }
        (args.output / "snapshots.summary.json").write_text(json.dumps(snapshot_summary, indent=2) + "\n")
        print(json.dumps({"snapshots": len(snapshots), "output": str(args.output)}))
        return
    inventory = Inventory()
    started = time.time()
    scopes: list[str] = []
    try:
        page = 1
        while True:
            repos = api.get(f"users/{quote(args.author, safe='')}/repos?per_page=100&type=owner&page={page}")
            for repo in repos:
                inventory.repositories[repo["full_name"]] = repo
            if len(repos) < 100:
                break
            page += 1
        scopes = sorted(set(inventory.repositories) | set(HISTORIC_REPOSITORIES) | set(args.repo))
        for repo in scopes:
            search_commits(api, inventory, f"author:{args.author} repo:{repo}", args.start, args.end, args.max_pages)
        if args.global_search:
            for year in range(args.start.year, args.end.year + 1):
                search_commits(
                    api,
                    inventory,
                    f"author:{args.author}",
                    max(args.start, date(year, 1, 1)),
                    min(args.end, date(year, 12, 31)),
                    args.max_pages,
                )
        for repo, metadata in list(inventory.repositories.items()):
            if "license" in metadata and (not metadata.get("fork") or "source" in metadata):
                continue
            try:
                inventory.repositories[repo] = api.get(f"repos/{quote(repo, safe='/')}")
            except subprocess.SubprocessError as error:
                inventory.errors.append({"repository": repo, "error": str(error)})
    except RequestBudgetExhausted as error:
        inventory.errors.append({"scope": "discovery", "error": str(error)})
    finally:
        assign_splits(inventory, args.split_seed)
        config = {key: str(value) if isinstance(value, (Path, date)) else value for key, value in vars(args).items()}
        config["elapsed_seconds"] = round(time.time() - started, 2)
        config["requested_repositories"] = scopes
        write_inventory(inventory, args.output, api, config)
    print(json.dumps({"output": str(args.output), "commits": len(inventory.commits), "errors": len(inventory.errors)}))


if __name__ == "__main__":
    main()
