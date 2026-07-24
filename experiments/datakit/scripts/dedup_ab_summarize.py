# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Summarize every score and comparison row from an issue #6854 audit."""

import argparse
import json
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from typing import Any

import pyarrow.parquet as pq
from rigging.filesystem import StoragePath

from experiments.datakit.scripts.dedup_ab_audit import DedupAuditData

FRACTION_BINS = 20


def _fraction_bin(value: float) -> str:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"Expected a finite fraction in [0, 1], got {value}")
    index = min(FRACTION_BINS - 1, int(value * FRACTION_BINS))
    lower = index / FRACTION_BINS
    upper = (index + 1) / FRACTION_BINS
    return f"{lower:.2f}-{upper:.2f}"


def _length_bin(chars: int) -> str:
    if chars < 0:
        raise ValueError(f"Expected non-negative characters, got {chars}")
    if chars == 0:
        return "0"
    exponent = chars.bit_length() - 1
    return f"2^{exponent}-2^{exponent + 1}"


def _ordered(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): counter[key] for key in sorted(counter, key=str)}


def summarize_scores(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all score rows without retaining text or row dictionaries."""
    counts: Counter[tuple[str, str, Any]] = Counter()
    clusters: Counter[tuple[str, str]] = Counter()
    drop_sources: Counter[tuple[str, str]] = Counter()
    drop_source_pairs: Counter[tuple[str, str, str]] = Counter()
    total = 0
    for record in records:
        total += 1
        variant = record["variant"]
        role = record["role"]
        counts[variant, "roles", role] += 1
        clusters[variant, record["cluster_id"]] += 1
        if role != "drop":
            continue

        source = record["source_main_dir"]
        canonical_source = record["canonical_source_main_dir"]
        evidence = record["evidence_class"]
        drop_sources[variant, source] += 1
        drop_source_pairs[variant, source, canonical_source] += 1
        counts[variant, "evidence", evidence] += 1
        counts[variant, "exact_raw_text", bool(record["exact_raw_text"])] += 1
        counts[variant, "exact_clean_text", bool(record["exact_clean_text"])] += 1
        counts[variant, "cross_source", bool(record["cross_source"])] += 1
        counts[variant, "member_is_longer", bool(record["member_is_longer"])] += 1
        counts[
            variant,
            "either_text_truncated_for_minhash",
            bool(record["member_text_truncated_for_minhash"] or record["canonical_text_truncated_for_minhash"]),
        ] += 1
        for name in (
            "char_5gram_jaccard",
            "word_5gram_jaccard",
            "word_5gram_canonical_containment",
            "word_5gram_member_containment",
            "length_ratio",
        ):
            counts[variant, f"{name}_bins", _fraction_bin(float(record[name]))] += 1
        counts[variant, "member_raw_chars_bins", _length_bin(int(record["raw_chars"]))] += 1
        counts[variant, "canonical_raw_chars_bins", _length_bin(int(record["canonical_raw_chars"]))] += 1
        counts[variant, "baseline_shared_buckets", int(record["baseline_shared_buckets"])] += 1
        counts[variant, "treatment_shared_buckets", int(record["treatment_shared_buckets"])] += 1

    variants: dict[str, Any] = {}
    for variant in ("baseline", "treatment"):
        cluster_sizes = Counter(size for (cluster_variant, _), size in clusters.items() if cluster_variant == variant)
        fields = sorted({field for count_variant, field, _ in counts if count_variant == variant})
        variants[variant] = {
            **{
                field: _ordered(
                    Counter({value: count for (v, f, value), count in counts.items() if v == variant and f == field})
                )
                for field in fields
            },
            "clusters": sum(1 for cluster_variant, _ in clusters if cluster_variant == variant),
            "cluster_size_histogram": _ordered(cluster_sizes),
            "drop_sources": _ordered(
                Counter(
                    {
                        source: count
                        for (source_variant, source), count in drop_sources.items()
                        if source_variant == variant
                    }
                )
            ),
            "drop_source_pairs": _ordered(
                Counter(
                    {
                        f"{source} -> {canonical_source}": count
                        for (source_variant, source, canonical_source), count in drop_source_pairs.items()
                        if source_variant == variant
                    }
                )
            ),
        }
    return {"score_rows": total, "variants": variants}


def summarize_comparisons(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate all A/B occurrence categories and baseline propagation distances."""
    categories: Counter[str] = Counter()
    attributions: Counter[str] = Counter()
    distances: Counter[int] = Counter()
    evidence_pairs: Counter[str] = Counter()
    total = 0
    for record in records:
        total += 1
        categories[record["category"]] += 1
        attributions[record["baseline_only_attribution"]] += 1
        distance = int(record["baseline_graph_distance"])
        if distance >= 0:
            distances[distance] += 1
        evidence_pairs[f"{record['baseline_evidence_class']} -> {record['treatment_evidence_class']}"] += 1
    return {
        "comparison_rows": total,
        "categories": _ordered(categories),
        "baseline_only_attributions": _ordered(attributions),
        "baseline_graph_distances": _ordered(distances),
        "evidence_pairs": _ordered(evidence_pairs),
    }


def _records(directory: str, columns: list[str]) -> Iterator[dict[str, Any]]:
    paths = sorted(str(path) for path in StoragePath(f"{directory.rstrip('/')}/*.parquet").glob())
    if not paths:
        raise FileNotFoundError(f"No Parquet files under {directory}")
    for path in paths:
        with StoragePath(path).open("rb") as handle:
            for batch in pq.ParquetFile(handle).iter_batches(columns=columns):
                yield from batch.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    audit = DedupAuditData.model_validate_json(StoragePath(args.audit).read_text())
    score_columns = [
        "variant",
        "role",
        "evidence_class",
        "source_main_dir",
        "canonical_source_main_dir",
        "cluster_id",
        "cross_source",
        "raw_chars",
        "canonical_raw_chars",
        "member_is_longer",
        "member_text_truncated_for_minhash",
        "canonical_text_truncated_for_minhash",
        "exact_raw_text",
        "exact_clean_text",
        "char_5gram_jaccard",
        "word_5gram_jaccard",
        "word_5gram_canonical_containment",
        "word_5gram_member_containment",
        "length_ratio",
        "baseline_shared_buckets",
        "treatment_shared_buckets",
    ]
    comparison_columns = [
        "category",
        "baseline_only_attribution",
        "baseline_graph_distance",
        "baseline_evidence_class",
        "treatment_evidence_class",
    ]
    result = {
        "version": "v1",
        "audit": args.audit,
        **summarize_scores(_records(audit.scores_dir, score_columns)),
        **summarize_comparisons(_records(audit.comparisons_dir, comparison_columns)),
    }
    StoragePath(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
