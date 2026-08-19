# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Count the fuzzy-dedup funnel for every source of a verified run.

The production verification artifact lost its per-comparison counters when the
root failed before metadata finalization, so the removal rate of each source is
not recorded anywhere. This reads the Parquet footers of the candidate and
verified trees, and the canonical flag of the candidate tree, and writes one row
per source: how many documents a source holds, how many the candidate stage put
in a cluster, how many of those are removals on offer, and how many the verifier
took.

Run it in region, because it opens every shard of both attribute trees::

    uv run iris --cluster=cw-us-east-02a job run --no-wait \
        --priority interactive --cpu 32 --memory 64g \
        -- python experiments/datakit/scripts/fuzzy_verification_funnel.py \
            --candidates s3://.../datakit/dedup_709f5997 \
            --verified s3://.../datakit/verify_fuzzy_dups_c757e4f0 \
            --out s3://.../user/rav/projects/fuzzy-verify-review/funnel/funnel.json
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import pyarrow.parquet as pq
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

logger = logging.getLogger(__name__)

DEFAULT_THREADS = 64


@dataclass(frozen=True)
class SourceCounts:
    source_key: str
    source_tag: str
    normalized_documents: int
    candidate_members: int
    candidate_canonicals: int
    verified_markers: int
    candidate_shards: int
    verified_shards: int

    @property
    def removals_on_offer(self) -> int:
        """Non-canonical cluster members: the most the verifier could remove."""
        return self.candidate_members - self.candidate_canonicals

    def as_row(self) -> dict:
        offered = self.removals_on_offer
        return {
            "source_key": self.source_key,
            "source_tag": self.source_tag,
            "normalized_documents": self.normalized_documents,
            "candidate_members": self.candidate_members,
            "candidate_canonicals": self.candidate_canonicals,
            "removals_on_offer": offered,
            "verified_markers": self.verified_markers,
            "candidate_shards": self.candidate_shards,
            "verified_shards": self.verified_shards,
            "clustered_fraction": _ratio(self.candidate_members, self.normalized_documents),
            "drop_rate": _ratio(self.verified_markers, self.normalized_documents),
            "acceptance_of_offered": _ratio(self.verified_markers, offered),
        }


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _row_count(fs, path: str) -> int:
    with fs.open(path, "rb") as handle:
        return pq.ParquetFile(handle).metadata.num_rows


def _candidate_counts(fs, path: str) -> tuple[int, int]:
    """Return (rows, canonicals) of one candidate shard."""
    with fs.open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        rows = parquet.metadata.num_rows
        if rows == 0:
            return 0, 0
        table = parquet.read(columns=["is_cluster_canonical"])
    canonicals = sum(1 for value in table.column("is_cluster_canonical").to_pylist() if value)
    return rows, canonicals


def _parquet_paths(fs, directory: str) -> list[str]:
    if not fs.exists(directory):
        return []
    return sorted(path for path in fs.ls(directory, detail=False) if str(path).endswith(".parquet"))


@dataclass(frozen=True)
class ShardTask:
    """One shard to count, tagged with the tree it belongs to."""

    source_tag: str
    kind: str
    path: str


def _count_shard(fs, task: ShardTask) -> tuple[str, str, int, int]:
    if task.kind == "candidate":
        rows, canonicals = _candidate_counts(fs, task.path)
        return task.source_tag, task.kind, rows, canonicals
    return task.source_tag, task.kind, _row_count(fs, task.path), 0


def count_all(jobs: list[dict], threads: int) -> list[SourceCounts]:
    """Count every shard of every source through one flat thread pool.

    Per-source pools leave most threads idle on the many small sources, and the
    shard counts are heavily skewed: one source holds 12,818 shards and another
    holds four.
    """
    fs, _ = url_to_fs(jobs[0]["candidate_dir"])
    tasks: list[ShardTask] = []
    shard_counts: dict[str, dict[str, int]] = {}
    for job in jobs:
        for kind, directory in (
            ("normalized", job["normalized_dir"]),
            ("candidate", job["candidate_dir"]),
            ("verified", job["verified_dir"]),
        ):
            _, root = url_to_fs(directory)
            paths = _parquet_paths(fs, root)
            shard_counts.setdefault(job["source_tag"], {})[kind] = len(paths)
            tasks.extend(ShardTask(job["source_tag"], kind, path) for path in paths)
    logger.info("Listing found %d shards across %d sources", len(tasks), len(jobs))

    totals: dict[str, dict[str, int]] = {job["source_tag"]: {} for job in jobs}
    done = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for source_tag, kind, rows, canonicals in pool.map(lambda task: _count_shard(fs, task), tasks):
            entry = totals[source_tag]
            entry[kind] = entry.get(kind, 0) + rows
            if kind == "candidate":
                entry["canonicals"] = entry.get("canonicals", 0) + canonicals
            done += 1
            if done % 5_000 == 0:
                logger.info("Counted %d/%d shards in %.0fs", done, len(tasks), time.monotonic() - started)

    return [
        SourceCounts(
            source_key=job["source_key"],
            source_tag=job["source_tag"],
            normalized_documents=totals[job["source_tag"]].get("normalized", 0),
            candidate_members=totals[job["source_tag"]].get("candidate", 0),
            candidate_canonicals=totals[job["source_tag"]].get("canonicals", 0),
            verified_markers=totals[job["source_tag"]].get("verified", 0),
            candidate_shards=shard_counts[job["source_tag"]]["candidate"],
            verified_shards=shard_counts[job["source_tag"]]["verified"],
        )
        for job in jobs
    ]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, help="Fuzzy candidate artifact root")
    parser.add_argument("--verified", required=True, help="Verified fuzzy-duplicate artifact root")
    parser.add_argument("--prefix", required=True, help="Storage root the artifact paths resolve against")
    parser.add_argument("--out", required=True)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--sources", default="all", help="'all' or a comma-separated list of source tags")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    candidates = json.loads(StoragePath(prefix_join(args.candidates, ".artifact.json")).read_bytes())["result"]
    verified = json.loads(StoragePath(prefix_join(args.verified, ".artifact.json")).read_bytes())["result"]
    wanted = None if args.sources == "all" else {tag.strip() for tag in args.sources.split(",") if tag.strip()}

    # The two artifacts do not share every source key: the Focus Crawl is keyed
    # under its pre-#8111 extraction in the candidate tree. The source tag is
    # not a join key either, because that one differing key sorts to a
    # different position and shifts the tags of the sources before it.
    candidate_sources = dict(candidates["sources"])
    unmatched_candidates = sorted(set(candidate_sources) - set(verified["sources"]))
    unmatched_verified = sorted(set(verified["sources"]) - set(candidate_sources))
    if len(unmatched_candidates) != len(unmatched_verified):
        raise ValueError(
            f"cannot pair source keys: candidate_only={unmatched_candidates!r}, verified_only={unmatched_verified!r}"
        )
    if len(unmatched_candidates) > 1:
        raise ValueError(f"more than one unpaired source key: {unmatched_verified!r}")
    for verified_key, candidate_key in zip(unmatched_verified, unmatched_candidates, strict=True):
        logger.info("Pairing renamed source %r with candidate %r", verified_key, candidate_key)
        candidate_sources[verified_key] = candidate_sources[candidate_key]

    jobs = []
    for source_key, entry in sorted(verified["sources"].items()):
        if wanted is not None and entry["source_tag"] not in wanted:
            continue
        candidate_entry = candidate_sources.get(source_key)
        if candidate_entry is None:
            raise KeyError(f"candidate artifact has no shard tree for {entry['source_tag']!r} ({source_key!r})")
        jobs.append(
            {
                "source_key": source_key,
                "source_tag": entry["source_tag"],
                "normalized_dir": prefix_join(args.prefix, source_key),
                "candidate_dir": prefix_join(args.prefix, candidate_entry["attr_dir"]),
                "verified_dir": prefix_join(args.prefix, entry["attr_dir"]),
            }
        )

    logger.info("Counting %d sources with %d threads", len(jobs), args.threads)
    started = time.monotonic()
    counts = count_all(jobs, args.threads)
    rows = [entry.as_row() for entry in counts]
    for entry in counts:
        logger.info(
            "%s normalized=%d members=%d offered=%d markers=%d drop=%.4f%% accept=%.2f%%",
            entry.source_tag,
            entry.normalized_documents,
            entry.candidate_members,
            entry.removals_on_offer,
            entry.verified_markers,
            100 * _ratio(entry.verified_markers, entry.normalized_documents),
            100 * _ratio(entry.verified_markers, entry.removals_on_offer),
        )

    totals = {
        field: sum(row[field] for row in rows)
        for field in (
            "normalized_documents",
            "candidate_members",
            "candidate_canonicals",
            "removals_on_offer",
            "verified_markers",
        )
    }
    totals["drop_rate"] = _ratio(totals["verified_markers"], totals["normalized_documents"])
    totals["acceptance_of_offered"] = _ratio(totals["verified_markers"], totals["removals_on_offer"])
    payload = {
        "candidates": args.candidates,
        "verified": args.verified,
        "elapsed_seconds": time.monotonic() - started,
        "totals": totals,
        "sources": sorted(rows, key=lambda row: -row["removals_on_offer"]),
    }
    StoragePath(args.out).write_bytes(json.dumps(payload, indent=1).encode())
    logger.info("Totals: %s", json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
