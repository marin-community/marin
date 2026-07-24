# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exhaustively audit baseline and treatment fuzzy-dedup outputs.

Every marker from either arm is joined to the co-partitioned normalized text
and both arms' MinHash buckets. Each non-canonical marker is then compared with
its arm's canonical using exact character- and word-5-gram sets. The output
retains stable source/shard/document references so every ambiguous case can be
reopened against the full text without copying the corpus.
"""

import argparse
import bisect
import hashlib
import json
import logging
import re
from collections.abc import Iterator
from itertools import chain, pairwise
from typing import Any, Literal

import dupekit
import pyarrow as pa
import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.artifact import read_record
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

NGRAM_SIZE = 5
TEXT_CAP_CHARS = 500_000
PREVIEW_CHARS = 2_000
EXPECTED_BASELINE_VERSION = "v2"
EXPECTED_TREATMENT_VERSION = "v3"
EXPECTED_BASELINE_NGRAM = "char"
EXPECTED_TREATMENT_NGRAM = "word"
CC_SHARD_PATTERN = re.compile(r"part-(\d+)")

logger = logging.getLogger(__name__)


class DedupAuditData(BaseModel):
    """Paths and exact counters produced by one exhaustive A/B audit."""

    version: str = "v3"
    baseline_dedup: str
    treatment_dedup: str
    baseline_minhash: str
    treatment_minhash: str
    scores_dir: str
    graph_distances_dir: str
    comparisons_dir: str
    counters: dict[str, int | float]


def _artifact_result(path: str) -> dict[str, Any]:
    record = read_record(path)
    if record is None or not isinstance(record.result, dict):
        raise FileNotFoundError(f"No artifact result at {path}")
    return record.result


def _minhash_kind(artifact: dict[str, Any]) -> str:
    params = artifact["params"]
    return str(params.get("ngram_kind", EXPECTED_BASELINE_NGRAM))


def _validate_arm(
    *,
    variant: str,
    dedup: dict[str, Any],
    minhash: dict[str, Any],
    expected_version: str,
    expected_ngram: str,
) -> None:
    inputs = minhash.get("inputs")
    if not isinstance(inputs, list) or not inputs:
        raise ValueError(f"{variant} MinHash collection has no inputs")
    for index, item in enumerate(inputs):
        if item.get("version") != expected_version:
            raise ValueError(
                f"{variant} inputs[{index}] has version {item.get('version')!r}, expected {expected_version!r}"
            )
        if _minhash_kind(item) != expected_ngram:
            raise ValueError(
                f"{variant} inputs[{index}] has ngram_kind={_minhash_kind(item)!r}, expected {expected_ngram!r}"
            )
        if item["params"].get("text_cap_chars") != TEXT_CAP_CHARS:
            raise ValueError(
                f"{variant} inputs[{index}] has text_cap_chars={item['params'].get('text_cap_chars')!r}, "
                f"expected {TEXT_CAP_CHARS}"
            )
        if item["params"].get("ngram_size") != NGRAM_SIZE:
            raise ValueError(
                f"{variant} inputs[{index}] has ngram_size={item['params'].get('ngram_size')!r}, "
                f"expected {NGRAM_SIZE}"
            )
    if set(dedup["sources"]) != {item["source_main_dir"] for item in inputs}:
        raise ValueError(f"{variant} dedup and MinHash source sets differ")


def _source_artifacts(collection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item["source_main_dir"]: item for item in collection["inputs"]}


def _shards(directory: str) -> dict[str, str]:
    return {str(path).rsplit("/", 1)[-1]: str(path) for path in StoragePath(f"{directory.rstrip('/')}/*.parquet").glob()}


def _source_shard_entries(
    baseline_dedup: dict[str, Any],
    treatment_dedup: dict[str, Any],
    baseline_minhash: dict[str, Any],
    treatment_minhash: dict[str, Any],
) -> list[dict[str, str]]:
    baseline_sources = set(baseline_dedup["sources"])
    treatment_sources = set(treatment_dedup["sources"])
    if baseline_sources != treatment_sources:
        raise ValueError(
            f"A/B source mismatch: baseline-only={sorted(baseline_sources - treatment_sources)}, "
            f"treatment-only={sorted(treatment_sources - baseline_sources)}"
        )

    baseline_minhash_sources = _source_artifacts(baseline_minhash)
    treatment_minhash_sources = _source_artifacts(treatment_minhash)
    entries: list[dict[str, str]] = []
    for source_main_dir in sorted(baseline_sources):
        directories = {
            "normalized": source_main_dir,
            "baseline_marker": baseline_dedup["sources"][source_main_dir]["attr_dir"],
            "treatment_marker": treatment_dedup["sources"][source_main_dir]["attr_dir"],
            "baseline_minhash": baseline_minhash_sources[source_main_dir]["attr_dir"],
            "treatment_minhash": treatment_minhash_sources[source_main_dir]["attr_dir"],
        }
        shard_maps = {name: _shards(directory) for name, directory in directories.items()}
        expected = set(shard_maps["normalized"])
        for name, shards in shard_maps.items():
            if set(shards) != expected:
                raise ValueError(
                    f"Shard mismatch for {source_main_dir} ({name}): "
                    f"missing={sorted(expected - set(shards))}, extra={sorted(set(shards) - expected)}"
                )
        for basename in sorted(expected):
            entries.append(
                {
                    "source_main_dir": source_main_dir,
                    "basename": basename,
                    **{f"{name}_path": shards[basename] for name, shards in shard_maps.items()},
                }
            )
    return entries


def _read_table(path: str, columns: list[str]) -> pa.Table:
    with StoragePath(path).open("rb") as handle:
        return pq.ParquetFile(handle).read(columns=columns)


def _cc_shards(directory: str) -> dict[int, str]:
    result: dict[int, str] = {}
    for path in StoragePath(f"{directory.rstrip('/')}/*.parquet").glob():
        path_str = str(path)
        basename = path_str.rsplit("/", 1)[-1].removesuffix(".parquet")
        match = CC_SHARD_PATTERN.fullmatch(basename.split("-of-", 1)[0])
        if match is None:
            raise ValueError(f"Unrecognized connected-components shard name: {path_str}")
        index = int(match.group(1))
        if index in result:
            raise ValueError(f"Duplicate connected-components shard index {index} under {directory}")
        result[index] = path_str
    return result


def _cc_distance_entries(dedup_path: str, source_main_dirs: set[str]) -> list[dict[str, Any]]:
    iterations: list[dict[int, str]] = []
    expected_shards: set[int] | None = None
    iteration = 0
    while True:
        shards = _cc_shards(f"{dedup_path.rstrip('/')}/metadata/cc/it_{iteration}")
        if not shards:
            break
        if expected_shards is None:
            expected_shards = set(shards)
        elif set(shards) != expected_shards:
            raise ValueError(
                f"Incomplete connected-components iteration {iteration}: "
                f"missing={sorted(expected_shards - set(shards))}, extra={sorted(set(shards) - expected_shards)}"
            )
        iterations.append(shards)
        iteration += 1
    if not iterations or expected_shards is None:
        raise FileNotFoundError(f"No complete connected-components iterations under {dedup_path}/metadata/cc")

    source_by_tag = {f"source_{index:03d}": source for index, source in enumerate(sorted(source_main_dirs))}
    logger.info(
        "Found %d complete baseline CC iterations across %d shards",
        len(iterations),
        len(expected_shards),
    )
    return [
        {
            "shard_index": shard_index,
            "iteration_paths": [iteration[shard_index] for iteration in iterations],
            "source_by_tag": source_by_tag,
        }
        for shard_index in sorted(expected_shards)
    ]


def _graph_distance_records(entry: dict[str, Any]) -> Iterator[dict[str, Any]]:
    final_table = _read_table(
        entry["iteration_paths"][-1],
        ["record_id", "id_norm", "adjacency_list", "component_id", "changed"],
    )
    final_changes = sum(bool(changed) for changed in final_table["changed"].to_pylist())
    if final_changes:
        raise AssertionError(
            f"Final connected-components shard {entry['shard_index']} still has {final_changes} changed nodes"
        )
    final_nodes: dict[str, str] = {}
    for record_id, id_norm, adjacency, component_id in zip(
        final_table["record_id"].to_pylist(),
        final_table["id_norm"].to_pylist(),
        final_table["adjacency_list"].to_pylist(),
        final_table["component_id"].to_pylist(),
        strict=True,
    ):
        if len(adjacency) == 1 and adjacency[0] == id_norm:
            continue
        final_nodes[record_id] = component_id

    distances: dict[str, int] = {}
    for iteration, path in enumerate(entry["iteration_paths"]):
        table = _read_table(path, ["record_id", "component_id"])
        for record_id, component_id in zip(
            table["record_id"].to_pylist(),
            table["component_id"].to_pylist(),
            strict=True,
        ):
            if record_id not in distances and final_nodes.get(record_id) == component_id:
                distances[record_id] = iteration
        if len(distances) == len(final_nodes):
            break
    if distances.keys() != final_nodes.keys():
        missing = sorted(final_nodes.keys() - distances.keys())
        raise AssertionError(
            f"Could not determine graph distance for {len(missing)} nodes in CC shard {entry['shard_index']}: "
            f"{missing[:10]}"
        )

    for record_id, distance in distances.items():
        source_tag, doc_id = record_id.split("|", 1)
        source_main_dir = entry["source_by_tag"][source_tag]
        occurrence_key = json.dumps([source_main_dir, doc_id], separators=(",", ":"))
        counters.pipeline.update_counter("audit/graph_distance/markers", 1)
        counters.pipeline.update_counter(f"audit/graph_distance/distance_{distance}", 1)
        yield {
            "occurrence_key": occurrence_key,
            "graph_distance": distance,
        }


def _marker_map(path: str) -> dict[str, dict[str, Any]]:
    with StoragePath(path).open("rb") as handle:
        parquet_file = pq.ParquetFile(handle)
        if parquet_file.metadata.num_rows == 0:
            return {}
        table = parquet_file.read(columns=["id", "attributes"])
    ids = table["id"].to_pylist()
    if len(set(ids)) != len(ids):
        raise AssertionError(f"Duplicate marker IDs in {path}")
    return dict(zip(ids, table["attributes"].to_pylist(), strict=True))


def _clean_texts(texts: list[str]) -> list[str]:
    truncated = [text[:TEXT_CAP_CHARS] for text in texts]
    batch = pa.RecordBatch.from_pydict({"text": truncated})
    result = dupekit.transform(
        batch,
        [dupekit.Transformation.CleanText(input_col="text", output_col="clean_text")],
    )
    return result["clean_text"].to_pylist()


def _variant_record(
    *,
    variant: str,
    entry: dict[str, str],
    doc_id: str,
    text: str,
    clean_text: str,
    marker: dict[str, Any],
    baseline_buckets: list[str],
    treatment_buckets: list[str],
) -> dict[str, Any]:
    occurrence_key = json.dumps([entry["source_main_dir"], doc_id], separators=(",", ":"))
    return {
        "pair_key": f"{variant}|{marker['dup_cluster_id']}",
        "occurrence_key": occurrence_key,
        "variant": variant,
        "source_main_dir": entry["source_main_dir"],
        "basename": entry["basename"],
        "id": doc_id,
        "cluster_id": marker["dup_cluster_id"],
        "is_canonical": bool(marker["is_cluster_canonical"]),
        "raw_chars": len(text),
        "raw_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "clean_text": clean_text,
        "text_prefix": text[:PREVIEW_CHARS],
        "text_suffix": text[-PREVIEW_CHARS:],
        "baseline_buckets": baseline_buckets,
        "treatment_buckets": treatment_buckets,
    }


def _joined_marker_records(entry: dict[str, str]) -> Iterator[dict[str, Any]]:
    baseline_markers = _marker_map(entry["baseline_marker_path"])
    treatment_markers = _marker_map(entry["treatment_marker_path"])
    flagged_ids = sorted(baseline_markers.keys() | treatment_markers.keys())
    if not flagged_ids:
        return

    normalized = _read_table(entry["normalized_path"], ["id", "text"])
    baseline_minhash = _read_table(entry["baseline_minhash_path"], ["id", "buckets"])
    treatment_minhash = _read_table(entry["treatment_minhash_path"], ["id", "buckets"])
    normalized_ids = normalized["id"].to_pylist()
    if any(left >= right for left, right in pairwise(normalized_ids)):
        raise AssertionError(f"Normalized IDs are not strictly sorted in {entry['normalized_path']}")
    if baseline_minhash["id"].to_pylist() != normalized_ids:
        raise AssertionError(f"Baseline MinHash IDs do not match normalized IDs in {entry['normalized_path']}")
    if treatment_minhash["id"].to_pylist() != normalized_ids:
        raise AssertionError(f"Treatment MinHash IDs do not match normalized IDs in {entry['normalized_path']}")

    indices: list[int] = []
    texts: list[str] = []
    for doc_id in flagged_ids:
        index = bisect.bisect_left(normalized_ids, doc_id)
        if index == len(normalized_ids) or normalized_ids[index] != doc_id:
            raise AssertionError(f"Marker ID {doc_id} is absent from {entry['normalized_path']}")
        indices.append(index)
        texts.append(normalized["text"][index].as_py())
    clean_texts = _clean_texts(texts)

    for doc_id, index, text, clean_text in zip(flagged_ids, indices, texts, clean_texts, strict=True):
        baseline_buckets = baseline_minhash["buckets"][index].as_py()
        treatment_buckets = treatment_minhash["buckets"][index].as_py()
        if doc_id in baseline_markers:
            counters.pipeline.update_counter("audit/markers/baseline", 1)
            yield _variant_record(
                variant="baseline",
                entry=entry,
                doc_id=doc_id,
                text=text,
                clean_text=clean_text,
                marker=baseline_markers[doc_id],
                baseline_buckets=baseline_buckets,
                treatment_buckets=treatment_buckets,
            )
        if doc_id in treatment_markers:
            counters.pipeline.update_counter("audit/markers/treatment", 1)
            yield _variant_record(
                variant="treatment",
                entry=entry,
                doc_id=doc_id,
                text=text,
                clean_text=clean_text,
                marker=treatment_markers[doc_id],
                baseline_buckets=baseline_buckets,
                treatment_buckets=treatment_buckets,
            )


def _shingles(text: str, kind: Literal["char", "word"]) -> set[str]:
    if kind == "char":
        units = list(text)
        if len(units) < NGRAM_SIZE:
            return {text}
        return {"".join(units[index : index + NGRAM_SIZE]) for index in range(len(units) - NGRAM_SIZE + 1)}

    words = text.split()
    if len(words) < NGRAM_SIZE:
        return {text}
    return {" ".join(words[index : index + NGRAM_SIZE]) for index in range(len(words) - NGRAM_SIZE + 1)}


def _set_metrics(left: set[str], right: set[str]) -> tuple[float, float, float]:
    intersection = len(left & right)
    union = len(left) + len(right) - intersection
    jaccard = intersection / union if union else 1.0
    left_containment = intersection / len(left) if left else 1.0
    right_containment = intersection / len(right) if right else 1.0
    return jaccard, left_containment, right_containment


def _evidence_class(
    *,
    exact_raw_text: bool,
    word_jaccard: float,
    canonical_word_containment: float,
    member_word_containment: float,
) -> str:
    if exact_raw_text:
        return "strong_duplicate"
    if word_jaccard <= 0.05 and max(canonical_word_containment, member_word_containment) <= 0.15:
        return "strong_false_positive"
    return "ambiguous"


def _score_cluster(pair_key: str, records: Iterator[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    canonical = next(records)
    if not canonical["is_canonical"]:
        raise AssertionError(f"Cluster {pair_key} has no leading canonical")

    canonical_char = _shingles(canonical["clean_text"], "char")
    canonical_word = _shingles(canonical["clean_text"], "word")
    seen_canonical = False
    for record in chain([canonical], records):
        if record["is_canonical"]:
            if seen_canonical:
                raise AssertionError(f"Cluster {pair_key} has more than one canonical")
            seen_canonical = True

        member_char = _shingles(record["clean_text"], "char")
        member_word = _shingles(record["clean_text"], "word")
        char_jaccard, canonical_char_containment, member_char_containment = _set_metrics(
            canonical_char,
            member_char,
        )
        word_jaccard, canonical_word_containment, member_word_containment = _set_metrics(
            canonical_word,
            member_word,
        )
        max_chars = max(len(canonical["clean_text"]), len(record["clean_text"]))
        length_ratio = min(len(canonical["clean_text"]), len(record["clean_text"])) / max_chars if max_chars else 1.0
        member_clean_text_contained = record["clean_text"] in canonical["clean_text"]
        exact_raw_text = record["raw_sha256"] == canonical["raw_sha256"]
        evidence_class = (
            "canonical"
            if record["is_canonical"]
            else _evidence_class(
                exact_raw_text=exact_raw_text,
                word_jaccard=word_jaccard,
                canonical_word_containment=canonical_word_containment,
                member_word_containment=member_word_containment,
            )
        )
        if not record["is_canonical"]:
            counters.pipeline.update_counter(f"audit/drops/{record['variant']}", 1)
            counters.pipeline.update_counter(f"audit/evidence/{record['variant']}/{evidence_class}", 1)
            if record["source_main_dir"] != canonical["source_main_dir"]:
                counters.pipeline.update_counter(f"audit/drops/{record['variant']}/cross_source", 1)
            if record["raw_chars"] > canonical["raw_chars"]:
                counters.pipeline.update_counter(f"audit/drops/{record['variant']}/member_is_longer", 1)
            if record["raw_chars"] > TEXT_CAP_CHARS:
                counters.pipeline.update_counter(f"audit/drops/{record['variant']}/member_text_truncated", 1)
            if canonical["raw_chars"] > TEXT_CAP_CHARS:
                counters.pipeline.update_counter(f"audit/drops/{record['variant']}/canonical_text_truncated", 1)

        baseline_shared = len(set(canonical["baseline_buckets"]) & set(record["baseline_buckets"]))
        treatment_shared = len(set(canonical["treatment_buckets"]) & set(record["treatment_buckets"]))
        yield {
            "occurrence_key": record["occurrence_key"],
            "variant": record["variant"],
            "role": "canonical" if record["is_canonical"] else "drop",
            "evidence_class": evidence_class,
            "source_main_dir": record["source_main_dir"],
            "basename": record["basename"],
            "id": record["id"],
            "cluster_id": record["cluster_id"],
            "canonical_occurrence_key": canonical["occurrence_key"],
            "canonical_source_main_dir": canonical["source_main_dir"],
            "canonical_basename": canonical["basename"],
            "canonical_id": canonical["id"],
            "cross_source": record["source_main_dir"] != canonical["source_main_dir"],
            "raw_chars": record["raw_chars"],
            "canonical_raw_chars": canonical["raw_chars"],
            "raw_sha256": record["raw_sha256"],
            "canonical_raw_sha256": canonical["raw_sha256"],
            "member_is_longer": record["raw_chars"] > canonical["raw_chars"],
            "member_text_truncated_for_minhash": record["raw_chars"] > TEXT_CAP_CHARS,
            "canonical_text_truncated_for_minhash": canonical["raw_chars"] > TEXT_CAP_CHARS,
            "clean_chars": len(record["clean_text"]),
            "canonical_clean_chars": len(canonical["clean_text"]),
            "length_ratio": length_ratio,
            "exact_raw_text": exact_raw_text,
            "exact_clean_text": canonical["clean_text"] == record["clean_text"],
            "member_clean_text_contained": member_clean_text_contained,
            "char_5gram_jaccard": char_jaccard,
            "char_5gram_canonical_containment": canonical_char_containment,
            "char_5gram_member_containment": member_char_containment,
            "char_5gram_shorter_containment": max(canonical_char_containment, member_char_containment),
            "word_5gram_jaccard": word_jaccard,
            "word_5gram_canonical_containment": canonical_word_containment,
            "word_5gram_member_containment": member_word_containment,
            "word_5gram_shorter_containment": max(canonical_word_containment, member_word_containment),
            "baseline_shared_buckets": baseline_shared,
            "treatment_shared_buckets": treatment_shared,
            "text_prefix": record["text_prefix"],
            "text_suffix": record["text_suffix"],
            "canonical_text_prefix": canonical["text_prefix"],
            "canonical_text_suffix": canonical["text_suffix"],
        }


def _parquet_records(path: str) -> Iterator[dict[str, Any]]:
    with StoragePath(path).open("rb") as handle:
        for batch in pq.ParquetFile(handle).iter_batches():
            yield from batch.to_pylist()


def _validate_score_counts(
    score_counters: dict[str, int | float],
    baseline_dedup: dict[str, Any],
    treatment_dedup: dict[str, Any],
) -> None:
    """Require the score pass to cover every marker and drop in both artifacts."""
    for variant, dedup in (("baseline", baseline_dedup), ("treatment", treatment_dedup)):
        artifact_counters = dedup["counters"]
        expected_markers = int(artifact_counters.get("dedup/fuzzy/document/cluster_members", 0))
        canonicals = int(artifact_counters.get("dedup/fuzzy/document/canonicals", 0))
        expected_drops = expected_markers - canonicals
        actual_markers = int(score_counters.get(f"audit/markers/{variant}", 0))
        actual_drops = int(score_counters.get(f"audit/drops/{variant}", 0))
        if actual_markers != expected_markers or actual_drops != expected_drops:
            raise AssertionError(
                f"{variant} score coverage mismatch: markers={actual_markers}/{expected_markers}, "
                f"drops={actual_drops}/{expected_drops}"
            )


def _validate_comparison_counts(
    comparison_counters: dict[str, int | float],
    baseline_dedup: dict[str, Any],
    treatment_dedup: dict[str, Any],
) -> None:
    """Require every drop in each arm to appear in exactly one A/B category."""
    both_drop = int(comparison_counters.get("audit/comparison/both_drop", 0))
    baseline_only = int(comparison_counters.get("audit/comparison/baseline_drop_treatment_keep", 0))
    treatment_only = int(comparison_counters.get("audit/comparison/treatment_drop_baseline_keep", 0))

    expected: dict[str, int] = {}
    for variant, dedup in (("baseline", baseline_dedup), ("treatment", treatment_dedup)):
        artifact_counters = dedup["counters"]
        markers = int(artifact_counters.get("dedup/fuzzy/document/cluster_members", 0))
        canonicals = int(artifact_counters.get("dedup/fuzzy/document/canonicals", 0))
        expected[variant] = markers - canonicals

    actual_baseline = both_drop + baseline_only
    actual_treatment = both_drop + treatment_only
    if actual_baseline != expected["baseline"] or actual_treatment != expected["treatment"]:
        raise AssertionError(
            "A/B drop comparison mismatch: "
            f"baseline={actual_baseline}/{expected['baseline']}, "
            f"treatment={actual_treatment}/{expected['treatment']}"
        )


def _comparison_input_records(entry: dict[str, str]) -> Iterator[dict[str, Any]]:
    for record in _parquet_records(entry["path"]):
        if entry["kind"] == "graph_distance":
            yield {
                "occurrence_key": record["occurrence_key"],
                "kind": "graph_distance",
                "variant": "",
                "role": "",
                "evidence_class": "",
                "source_main_dir": "",
                "basename": "",
                "id": "",
                "cluster_id": "",
                "word_5gram_jaccard": -1.0,
                "baseline_shared_buckets": -1,
                "treatment_shared_buckets": -1,
                "graph_distance": record["graph_distance"],
            }
            continue
        yield {
            "occurrence_key": record["occurrence_key"],
            "kind": "score",
            "variant": record["variant"],
            "role": record["role"],
            "evidence_class": record["evidence_class"],
            "source_main_dir": record["source_main_dir"],
            "basename": record["basename"],
            "id": record["id"],
            "cluster_id": record["cluster_id"],
            "word_5gram_jaccard": record["word_5gram_jaccard"],
            "baseline_shared_buckets": record["baseline_shared_buckets"],
            "treatment_shared_buckets": record["treatment_shared_buckets"],
            "graph_distance": -1,
        }


def _comparison_category(baseline_role: str, treatment_role: str) -> str:
    if baseline_role == "drop" and treatment_role == "drop":
        return "both_drop"
    if baseline_role == "drop":
        return "baseline_drop_treatment_keep"
    if treatment_role == "drop":
        return "treatment_drop_baseline_keep"
    return "canonical_only"


def _baseline_only_attribution(
    baseline: dict[str, Any] | None,
    baseline_graph_distance: int | None,
    category: str,
) -> str:
    if category != "baseline_drop_treatment_keep" or baseline is None:
        return "not_applicable"
    assert baseline_graph_distance is not None
    direct_under_baseline = baseline_graph_distance == 1
    collides_under_treatment = baseline["treatment_shared_buckets"] > 0
    if not direct_under_baseline and not collides_under_treatment:
        return "transitive_closure_and_ngram"
    if not direct_under_baseline:
        return "transitive_closure"
    if not collides_under_treatment:
        return "word_ngram"
    return "canonical_or_graph_change"


def _compare_occurrence(occurrence_key: str, records: Iterator[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, dict[str, Any]] = {}
    baseline_graph_distance: int | None = None
    for record in records:
        if record["kind"] == "graph_distance":
            if baseline_graph_distance is not None:
                raise AssertionError(f"Duplicate baseline graph distance for {occurrence_key}")
            baseline_graph_distance = record["graph_distance"]
        else:
            variant = record["variant"]
            if variant in by_variant:
                raise AssertionError(f"Duplicate {variant} score for {occurrence_key}")
            by_variant[variant] = record
    baseline = by_variant.get("baseline")
    treatment = by_variant.get("treatment")
    if baseline is not None and baseline_graph_distance is None:
        raise AssertionError(f"Missing baseline graph distance for {occurrence_key}")
    if baseline is not None:
        expected_distance = 0 if baseline["role"] == "canonical" else None
        if expected_distance is not None and baseline_graph_distance != expected_distance:
            raise AssertionError(
                f"Baseline canonical {occurrence_key} has graph distance {baseline_graph_distance}, expected 0"
            )
        if baseline["role"] == "drop":
            direct_by_buckets = baseline["baseline_shared_buckets"] > 0
            if (baseline_graph_distance == 1) != direct_by_buckets:
                raise AssertionError(
                    f"Baseline direct-edge mismatch for {occurrence_key}: "
                    f"distance={baseline_graph_distance}, shared_buckets={baseline['baseline_shared_buckets']}"
                )
    baseline_role = baseline["role"] if baseline else "missing"
    treatment_role = treatment["role"] if treatment else "missing"
    category = _comparison_category(baseline_role, treatment_role)
    attribution = _baseline_only_attribution(baseline, baseline_graph_distance, category)
    counters.pipeline.update_counter(f"audit/comparison/{category}", 1)
    counters.pipeline.update_counter(f"audit/attribution/{attribution}", 1)
    reference = baseline or treatment
    assert reference is not None
    return {
        "occurrence_key": occurrence_key,
        "source_main_dir": reference["source_main_dir"],
        "basename": reference["basename"],
        "id": reference["id"],
        "baseline_role": baseline_role,
        "treatment_role": treatment_role,
        "category": category,
        "baseline_only_attribution": attribution,
        "baseline_evidence_class": baseline["evidence_class"] if baseline else "missing",
        "treatment_evidence_class": treatment["evidence_class"] if treatment else "missing",
        "baseline_cluster_id": baseline["cluster_id"] if baseline else "",
        "treatment_cluster_id": treatment["cluster_id"] if treatment else "",
        "baseline_graph_distance": baseline_graph_distance if baseline_graph_distance is not None else -1,
        "treatment_graph_distance": 0 if treatment_role == "canonical" else 1 if treatment_role == "drop" else -1,
        "baseline_word_5gram_jaccard": baseline["word_5gram_jaccard"] if baseline else -1.0,
        "treatment_word_5gram_jaccard": treatment["word_5gram_jaccard"] if treatment else -1.0,
        "baseline_shared_buckets": baseline["baseline_shared_buckets"] if baseline else -1,
        "treatment_shared_buckets": treatment["treatment_shared_buckets"] if treatment else -1,
    }


def audit(
    *,
    baseline_dedup_path: str,
    treatment_dedup_path: str,
    baseline_minhash_path: str,
    treatment_minhash_path: str,
    output_path: str,
    max_workers: int,
) -> DedupAuditData:
    """Run the exhaustive marker join, pair scoring, and A/B occurrence comparison."""
    baseline_dedup = _artifact_result(baseline_dedup_path)
    treatment_dedup = _artifact_result(treatment_dedup_path)
    baseline_minhash = _artifact_result(baseline_minhash_path)
    treatment_minhash = _artifact_result(treatment_minhash_path)
    _validate_arm(
        variant="baseline",
        dedup=baseline_dedup,
        minhash=baseline_minhash,
        expected_version=EXPECTED_BASELINE_VERSION,
        expected_ngram=EXPECTED_BASELINE_NGRAM,
    )
    _validate_arm(
        variant="treatment",
        dedup=treatment_dedup,
        minhash=treatment_minhash,
        expected_version=EXPECTED_TREATMENT_VERSION,
        expected_ngram=EXPECTED_TREATMENT_NGRAM,
    )
    entries = _source_shard_entries(
        baseline_dedup,
        treatment_dedup,
        baseline_minhash,
        treatment_minhash,
    )
    logger.info("Auditing %d co-partitioned source shards", len(entries))

    resources = ResourceConfig(cpu=2, ram="24g", disk="20g", preemptible=False)
    coordinator = ResourceConfig(cpu=4, ram="16g", disk="20g", preemptible=False)
    scores_dir = f"{output_path.rstrip('/')}/scores"
    score_context = ZephyrContext(
        name="dedup-ab-audit-scores",
        max_workers=max_workers,
        resources=resources,
        coordinator_resources=coordinator,
    )
    score_pipeline = (
        Dataset.from_list(entries)
        .flat_map(_joined_marker_records)
        .group_by(
            key=lambda record: record["pair_key"],
            sort_by=lambda record: f"{0 if record['is_canonical'] else 1}|{record['occurrence_key']}",
            reducer=_score_cluster,
            num_output_shards=max_workers,
        )
        .write_parquet(f"{scores_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    score_outcome = score_context.execute(score_pipeline, verbose=True)
    score_files = list(score_outcome.results)
    if not score_files:
        raise ValueError("A/B audit produced no score files")
    _validate_score_counts(dict(score_outcome.counters), baseline_dedup, treatment_dedup)

    graph_distances_dir = f"{output_path.rstrip('/')}/baseline-graph-distances"
    graph_context = ZephyrContext(
        name="dedup-ab-audit-graph-distances",
        max_workers=max_workers,
        resources=resources,
        coordinator_resources=coordinator,
    )
    graph_pipeline = (
        Dataset.from_list(_cc_distance_entries(baseline_dedup_path, set(baseline_dedup["sources"])))
        .flat_map(_graph_distance_records)
        .write_parquet(f"{graph_distances_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    graph_outcome = graph_context.execute(graph_pipeline, verbose=True)
    graph_files = list(graph_outcome.results)
    if not graph_files:
        raise ValueError("A/B audit produced no graph-distance files")
    baseline_markers = int(score_outcome.counters.get("audit/markers/baseline", 0))
    graph_markers = int(graph_outcome.counters.get("audit/graph_distance/markers", 0))
    if graph_markers != baseline_markers:
        raise AssertionError(f"Baseline graph distance covers {graph_markers} markers, expected {baseline_markers}")

    comparisons_dir = f"{output_path.rstrip('/')}/comparisons"
    comparison_context = ZephyrContext(
        name="dedup-ab-audit-comparisons",
        max_workers=max_workers,
        resources=resources,
        coordinator_resources=coordinator,
    )
    comparison_inputs = [
        *({"kind": "score", "path": path} for path in score_files),
        *({"kind": "graph_distance", "path": path} for path in graph_files),
    ]
    comparison_pipeline = (
        Dataset.from_list(comparison_inputs)
        .flat_map(_comparison_input_records)
        .group_by(
            key=lambda record: record["occurrence_key"],
            reducer=_compare_occurrence,
            num_output_shards=max_workers,
        )
        .write_parquet(f"{comparisons_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    comparison_outcome = comparison_context.execute(comparison_pipeline, verbose=True)
    _validate_comparison_counts(
        dict(comparison_outcome.counters),
        baseline_dedup,
        treatment_dedup,
    )
    merged_counters = {
        **{f"scores/{key}": value for key, value in score_outcome.counters.items()},
        **{f"graph_distances/{key}": value for key, value in graph_outcome.counters.items()},
        **{f"comparisons/{key}": value for key, value in comparison_outcome.counters.items()},
    }
    return DedupAuditData(
        baseline_dedup=baseline_dedup_path,
        treatment_dedup=treatment_dedup_path,
        baseline_minhash=baseline_minhash_path,
        treatment_minhash=treatment_minhash_path,
        scores_dir=scores_dir,
        graph_distances_dir=graph_distances_dir,
        comparisons_dir=comparisons_dir,
        counters=merged_counters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dedup", required=True)
    parser.add_argument("--treatment-dedup", required=True)
    parser.add_argument("--baseline-minhash", required=True)
    parser.add_argument("--treatment-minhash", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=128)
    args = parser.parse_args()
    configure_logging(logging.INFO)

    result = audit(
        baseline_dedup_path=args.baseline_dedup,
        treatment_dedup_path=args.treatment_dedup,
        baseline_minhash_path=args.baseline_minhash,
        treatment_minhash_path=args.treatment_minhash,
        output_path=args.output,
        max_workers=args.max_workers,
    )
    StoragePath(f"{args.output.rstrip('/')}/audit.json").write_text(result.model_dump_json(indent=2))
    logger.info("Wrote audit artifact to %s/audit.json", args.output.rstrip("/"))


if __name__ == "__main__":
    main()
