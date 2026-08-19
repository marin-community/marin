# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Sample complete candidate clusters and review every ordered pair of them.

One pass over the candidate attribute tree selects clusters by a residue of
``dup_cluster_id``. Cluster IDs are uniform hashes, so the residue is a uniform
sample of clusters and it keeps every member of a selected cluster. That
completeness is what the review needs: a truncated cluster hides the partner
that would remove a document.

``--source-tags`` keeps only the clusters that touch named sources, which is
how a code-only review is built without changing the sampling frame. Members
outside those sources stay in the cluster.

The job then fetches the member text from the co-partitioned normalized tree,
scores every ordered pair, replays the production comparison budget, and writes
a compact review to storage. Run it in region: it opens every candidate shard.

    uv run iris --cluster=cw-us-east-02a job run --no-wait \
        --priority interactive --cpu 32 --memory 128GB --enable-extra-resources \
        -- python experiments/datakit/scripts/fuzzy_verification_cluster_sample.py \
            --candidates s3://.../datakit/dedup_709f5997 \
            --verified s3://.../datakit/verify_fuzzy_dups_c757e4f0 \
            --prefix s3://.../marin --modulus 16384 \
            --source-tags source_090,source_279 \
            --out s3://.../user/rav/projects/fuzzy-verify-review/code_sample
"""

import argparse
import bisect
import json
import logging
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import pyarrow.parquet as pq
from marin.processing.classification.deduplication.fuzzy_verification import FuzzyVerificationParams
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.datakit.reports.fuzzy_verification_nxn import (
    RULES,
    ClusterDocument,
    best_partner,
    classify_document,
    measure_novelty,
    review_cluster,
    unified_diff,
)

logger = logging.getLogger(__name__)

CANDIDATE_COLUMNS = ["id", "dup_cluster_id", "is_cluster_canonical"]
DEFAULT_THREADS = 128
DEFAULT_MODULUS = 16_384
CONTAINMENT_BANDS = ((1.0, "1.00"), (0.99, "0.99-1.00"), (0.95, "0.95-0.99"), (0.90, "0.90-0.95"), (0.80, "0.80-0.90"))


def _band(containment: float) -> str:
    for threshold, label in CONTAINMENT_BANDS:
        if containment >= threshold:
            return label
    return "<0.80"


def _scan_candidate_shard(fs, path: str, modulus: int, residue: int) -> list[tuple]:
    """Return the sampled members of one candidate shard."""
    source_tag = path.rsplit("/", 2)[-2]
    shard = path.rsplit("/", 1)[-1]
    with fs.open(path, "rb") as handle:
        table = pq.ParquetFile(handle).read(columns=CANDIDATE_COLUMNS)
    if table.num_rows == 0:
        return []
    ids = table.column("id").to_pylist()
    clusters = table.column("dup_cluster_id").to_pylist()
    canonical = table.column("is_cluster_canonical").to_pylist()
    return [
        (str(clusters[index]), ids[index], bool(canonical[index]), source_tag, shard)
        for index in range(table.num_rows)
        if int(clusters[index]) % modulus == residue
    ]


def _row_groups_for_ids(parquet: pq.ParquetFile, wanted: list[str]) -> list[int]:
    """Row groups whose sorted ``id`` range can hold one of ``wanted``.

    Normalized shards are written in ascending ID order, so a row group's
    statistics bound the IDs it holds. A sparse sample touches few groups, and
    the text column of a skipped group is never decompressed.
    """
    schema = parquet.schema_arrow
    if "id" not in schema.names:
        return list(range(parquet.num_row_groups))
    column = schema.names.index("id")
    selected = []
    for index in range(parquet.num_row_groups):
        statistics = parquet.metadata.row_group(index).column(column).statistics
        if statistics is None or not statistics.has_min_max:
            return list(range(parquet.num_row_groups))
        position = bisect.bisect_left(wanted, str(statistics.min))
        if position < len(wanted) and wanted[position] <= str(statistics.max):
            selected.append(index)
    return selected


def _fetch_shard_text(fs, path: str, wanted: set[str]) -> dict[str, str]:
    """Read the text of ``wanted`` IDs from one normalized shard."""
    ordered = sorted(wanted)
    with fs.open(path, "rb") as handle:
        parquet = pq.ParquetFile(handle)
        found: dict[str, str] = {}
        for index in _row_groups_for_ids(parquet, ordered):
            table = parquet.read_row_group(index, columns=["id", "text"])
            for row_id, text in zip(table.column("id").to_pylist(), table.column("text").to_pylist(), strict=True):
                if row_id in wanted:
                    found[row_id] = text or ""
    return found


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(len(ordered) * fraction), len(ordered) - 1)]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--verified", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--modulus", type=int, default=DEFAULT_MODULUS)
    parser.add_argument("--residue", type=int, default=0)
    parser.add_argument("--source-tags", default="all", help="'all' or comma-separated tags a cluster must touch")
    parser.add_argument("--max-cluster-size", type=int, default=64, help="Skip clusters above this member count")
    parser.add_argument("--max-clusters", type=int, default=20_000)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--examples-per-band", type=int, default=15)
    parser.add_argument("--diff-lines", type=int, default=80)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    params = FuzzyVerificationParams()
    started = time.monotonic()

    candidates = json.loads(StoragePath(prefix_join(args.candidates, ".artifact.json")).read_bytes())["result"]
    verified = json.loads(StoragePath(prefix_join(args.verified, ".artifact.json")).read_bytes())["result"]
    normalized_by_tag = {entry["source_tag"]: key for key, entry in verified["sources"].items()}
    verified_dir_by_tag = {entry["source_tag"]: entry["attr_dir"] for entry in verified["sources"].values()}
    candidate_dirs = [str(entry["attr_dir"]) for entry in candidates["sources"].values()]

    fs, _ = url_to_fs(args.prefix)
    _, candidate_root = url_to_fs(prefix_join(args.prefix, candidate_dirs[0]).rsplit("/", 1)[0])
    paths = sorted(path for path in fs.find(candidate_root) if str(path).endswith(".parquet"))
    logger.info("Scanning %d candidate shards at modulus %d", len(paths), args.modulus)

    members: dict[str, list[tuple]] = defaultdict(list)
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for done, sampled in enumerate(
            pool.map(lambda path: _scan_candidate_shard(fs, path, args.modulus, args.residue), paths), start=1
        ):
            for cluster_id, doc_id, is_canonical, source_tag, shard in sampled:
                members[cluster_id].append((doc_id, is_canonical, source_tag, shard))
            if done % 20_000 == 0:
                logger.info(
                    "Scanned %d/%d shards, %d clusters, %.0fs",
                    done,
                    len(paths),
                    len(members),
                    time.monotonic() - started,
                )
    logger.info("Sample holds %d clusters and %d members", len(members), sum(len(v) for v in members.values()))

    wanted_tags = None if args.source_tags == "all" else {tag.strip() for tag in args.source_tags.split(",")}
    selected = {}
    skipped: Counter = Counter()
    for cluster_id, entries in members.items():
        if len(entries) < 2:
            skipped["single_member"] += 1
            continue
        if len(entries) > args.max_cluster_size:
            skipped["over_max_size"] += 1
            continue
        if wanted_tags is not None and not any(entry[2] in wanted_tags for entry in entries):
            skipped["no_target_source"] += 1
            continue
        selected[cluster_id] = entries
    rng = random.Random(args.seed)
    if len(selected) > args.max_clusters:
        keep = set(rng.sample(sorted(selected), args.max_clusters))
        skipped["over_max_clusters"] = len(selected) - len(keep)
        selected = {cluster_id: entries for cluster_id, entries in selected.items() if cluster_id in keep}
    logger.info("Reviewing %d clusters; skipped %s", len(selected), dict(skipped))

    needed: dict[tuple[str, str], set[str]] = defaultdict(set)
    for entries in selected.values():
        for doc_id, _canonical, source_tag, shard in entries:
            needed[(source_tag, shard)].add(doc_id)
    logger.info("Fetching %d documents from %d normalized shards", sum(len(v) for v in needed.values()), len(needed))

    def fetch(item):
        (source_tag, shard), ids = item
        path = prefix_join(args.prefix, normalized_by_tag[source_tag], shard)
        _, root = url_to_fs(path)
        return (source_tag, shard), _fetch_shard_text(fs, root, ids)

    texts: dict[tuple[str, str], dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for done, (key, found) in enumerate(pool.map(fetch, list(needed.items())), start=1):
            texts[key] = found
            if done % 2_000 == 0:
                logger.info("Fetched %d/%d shards, %.0fs", done, len(needed), time.monotonic() - started)

    marker_ids: dict[str, set[str]] = {}

    def read_markers(item):
        (source_tag, shard), ids = item
        path = prefix_join(args.prefix, verified_dir_by_tag[source_tag], shard)
        _, root = url_to_fs(path)
        if not fs.exists(root):
            return source_tag, shard, set()
        with fs.open(root, "rb") as handle:
            table = pq.ParquetFile(handle).read(columns=["id"])
        return source_tag, shard, {row for row in table.column("id").to_pylist() if row in ids}

    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        for source_tag, _shard, found in pool.map(read_markers, list(needed.items())):
            marker_ids.setdefault(source_tag, set()).update(found)

    reviews = []
    missing_text = 0
    for cluster_id, entries in selected.items():
        documents = []
        for doc_id, is_canonical, source_tag, shard in entries:
            text = texts.get((source_tag, shard), {}).get(doc_id)
            if text is None:
                missing_text += 1
                continue
            documents.append(
                ClusterDocument(
                    source_name=normalized_by_tag[source_tag],
                    id=doc_id,
                    text=text,
                    # The bounded replay's local-representative nomination
                    # needs LSH buckets, which live in the MinHash tree rather
                    # than the candidate tree. Leaving them empty disables that
                    # branch, so the replay under-counts local matches. Those
                    # need equal normalized token sequences, so the loss is
                    # small, and the production markers are reported next to
                    # the replay as a check.
                    buckets=(),
                    is_cluster_canonical=is_canonical,
                    dropped=doc_id in marker_ids.get(source_tag, ()),
                )
            )
        if len(documents) < 2:
            skipped["incomplete_after_fetch"] += 1
            continue
        reviews.append(review_cluster(cluster_id, documents, params))
    logger.info("Reviewed %d clusters; %d members had no text", len(reviews), missing_text)

    document_count = sum(len(review.documents) for review in reviews)
    bounded = sum(len(review.bounded) for review in reviews)
    summary = {
        "sample": {"modulus": args.modulus, "residue": args.residue, "source_tags": args.source_tags},
        "clusters": len(reviews),
        "documents": document_count,
        "ordered_pairs": sum(len(review.pairs) for review in reviews),
        "production_markers": sum(1 for review in reviews for document in review.documents if document.dropped),
        "bounded_replay_removed": bounded,
        "skipped": dict(skipped),
        "rules": {
            name: {
                "removed": sum(len(review.removable[name]) for review in reviews),
                "removed_fraction": sum(len(review.removable[name]) for review in reviews) / max(document_count, 1),
            }
            for name in RULES
        },
        "causes": {},
        "by_source": {},
        "novelty_bands": {},
    }

    causes: Counter = Counter()
    per_source: dict[str, Counter] = defaultdict(Counter)
    novelty_rows: list[dict] = []
    for review in reviews:
        for index, document in enumerate(review.documents):
            cause = classify_document(review, index)
            causes[cause] += 1
            per_source[document.source_name][cause] += 1
            per_source[document.source_name]["documents"] += 1
            if cause not in ("rule_blocked", "budget_blocked"):
                continue
            pair = best_partner(review, index)
            if pair is None:
                continue
            representative = review.documents[pair.representative_index]
            novelty, novel_lines = measure_novelty(document.text, representative.text)
            novelty_rows.append(
                {
                    "cluster_id": review.cluster_id,
                    "cause": cause,
                    "band": _band(pair.member_containment),
                    "member_source": document.source_name,
                    "member_id": document.id,
                    "representative_source": representative.source_name,
                    "representative_id": representative.id,
                    "member_containment": pair.member_containment,
                    "jaccard": pair.jaccard,
                    "member_unique_ngrams": pair.member_unique_ngrams,
                    "member_chars": pair.member_chars,
                    "representative_chars": pair.representative_chars,
                    "saturated": pair.saturated,
                    "under_tokenized": pair.under_tokenized,
                    "novelty": asdict(novelty),
                    "novel_lines_sample": novel_lines[:12],
                    "_member_text": document.text,
                    "_representative_text": representative.text,
                }
            )
    summary["causes"] = dict(causes)
    summary["by_source"] = {name: dict(counter) for name, counter in sorted(per_source.items())}

    by_band: dict[str, list[dict]] = defaultdict(list)
    for row in novelty_rows:
        by_band[row["band"]].append(row)
    for band, group in sorted(by_band.items()):
        line_ratios = [row["novelty"]["novel_line_ratio"] for row in group]
        summary["novelty_bands"][band] = {
            "documents": len(group),
            "median_novel_line_ratio": _percentile(line_ratios, 0.5),
            "p90_novel_line_ratio": _percentile(line_ratios, 0.9),
            "median_novel_token_ratio": _percentile([row["novelty"]["novel_token_ratio"] for row in group], 0.5),
            "zero_substantive_novel_lines": sum(1 for row in group if row["novelty"]["novel_substantive_lines"] == 0),
            "at_most_two_substantive_novel_lines": sum(
                1 for row in group if row["novelty"]["novel_substantive_lines"] <= 2
            ),
        }

    examples = []
    for _, group in sorted(by_band.items()):
        for row in rng.sample(group, min(args.examples_per_band, len(group))):
            example = {key: value for key, value in row.items() if not key.startswith("_")}
            example["diff"] = unified_diff(row["_member_text"], row["_representative_text"], args.diff_lines)
            examples.append(example)

    summary["elapsed_seconds"] = time.monotonic() - started
    StoragePath(prefix_join(args.out, "summary.json")).write_bytes(json.dumps(summary, indent=1).encode())
    StoragePath(prefix_join(args.out, "examples.json")).write_bytes(
        json.dumps(examples, indent=1, ensure_ascii=False).encode()
    )
    StoragePath(prefix_join(args.out, "novelty.jsonl")).write_bytes(
        b"\n".join(
            json.dumps({key: value for key, value in row.items() if not key.startswith("_")}).encode()
            for row in novelty_rows
        )
    )
    logger.info("Summary: %s", json.dumps({k: v for k, v in summary.items() if k != "by_source"}, indent=1))


if __name__ == "__main__":
    main()
