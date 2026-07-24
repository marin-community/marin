# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exhaustively compare capped and converged fuzzy-dedup marker artifacts."""

import argparse
import json
from collections.abc import Iterator, Mapping
from typing import Any

from fray.types import ResourceConfig
from pydantic import BaseModel
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.datakit.scripts.dedup_ab_audit import _artifact_result, _marker_map, _shards


class MarkerDiffData(BaseModel):
    """Paths and exact counters for one complete marker comparison."""

    version: str = "v1"
    capped_dedup: str
    converged_dedup: str
    differences_dir: str
    counters: dict[str, int | float]


def marker_differences(
    capped: Mapping[str, Mapping[str, Any]],
    converged: Mapping[str, Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield every marker whose presence or cluster attributes changed."""
    for doc_id in sorted(capped.keys() | converged.keys()):
        capped_marker = capped.get(doc_id)
        converged_marker = converged.get(doc_id)
        if capped_marker == converged_marker:
            continue
        if capped_marker is None:
            change_kind = "converged_only"
        elif converged_marker is None:
            change_kind = "capped_only"
        else:
            change_kind = "attributes_changed"
        yield {
            "id": doc_id,
            "change_kind": change_kind,
            "capped_cluster_id": capped_marker["dup_cluster_id"] if capped_marker is not None else None,
            "capped_is_canonical": bool(capped_marker["is_cluster_canonical"]) if capped_marker is not None else None,
            "converged_cluster_id": converged_marker["dup_cluster_id"] if converged_marker is not None else None,
            "converged_is_canonical": (
                bool(converged_marker["is_cluster_canonical"]) if converged_marker is not None else None
            ),
        }


def _diff_shard(entry: dict[str, str]) -> Iterator[dict[str, Any]]:
    capped = _marker_map(entry["capped_path"])
    converged = _marker_map(entry["converged_path"])
    counters.pipeline.update_counter("marker_diff/shards", 1)
    counters.pipeline.update_counter("marker_diff/capped_markers", len(capped))
    counters.pipeline.update_counter("marker_diff/converged_markers", len(converged))

    differences = list(marker_differences(capped, converged))
    changed_shared = sum(difference["change_kind"] == "attributes_changed" for difference in differences)
    counters.pipeline.update_counter(
        "marker_diff/unchanged",
        len(capped.keys() & converged.keys()) - changed_shared,
    )
    for difference in differences:
        counters.pipeline.update_counter("marker_diff/differences", 1)
        counters.pipeline.update_counter(f"marker_diff/{difference['change_kind']}", 1)
        yield {
            "source_main_dir": entry["source_main_dir"],
            "basename": entry["basename"],
            **difference,
        }


def _entries(capped: dict[str, Any], converged: dict[str, Any]) -> list[dict[str, str]]:
    capped_sources = set(capped["sources"])
    converged_sources = set(converged["sources"])
    if capped_sources != converged_sources:
        raise ValueError(
            f"Marker source mismatch: capped-only={sorted(capped_sources - converged_sources)}, "
            f"converged-only={sorted(converged_sources - capped_sources)}"
        )

    entries: list[dict[str, str]] = []
    for source_main_dir in sorted(capped_sources):
        capped_shards = _shards(capped["sources"][source_main_dir]["attr_dir"])
        converged_shards = _shards(converged["sources"][source_main_dir]["attr_dir"])
        if capped_shards.keys() != converged_shards.keys():
            raise ValueError(
                f"Marker shard mismatch for {source_main_dir}: "
                f"capped-only={sorted(capped_shards.keys() - converged_shards.keys())}, "
                f"converged-only={sorted(converged_shards.keys() - capped_shards.keys())}"
            )
        entries.extend(
            {
                "source_main_dir": source_main_dir,
                "basename": basename,
                "capped_path": capped_shards[basename],
                "converged_path": converged_shards[basename],
            }
            for basename in sorted(capped_shards)
        )
    return entries


def compare_markers(
    *,
    capped_dedup_path: str,
    converged_dedup_path: str,
    output_path: str,
    max_workers: int,
) -> MarkerDiffData:
    """Scan every marker shard and persist every capped/converged difference."""
    capped = _artifact_result(capped_dedup_path)
    converged = _artifact_result(converged_dedup_path)
    entries = _entries(capped, converged)
    resources = ResourceConfig(cpu=2, ram="8g", disk="20g", preemptible=False)
    coordinator = ResourceConfig(cpu=4, ram="16g", disk="20g", preemptible=False)
    differences_dir = f"{output_path.rstrip('/')}/differences"
    context = ZephyrContext(
        name="dedup-ab-marker-diff",
        max_workers=max_workers,
        resources=resources,
        coordinator_resources=coordinator,
    )
    pipeline = (
        Dataset.from_list(entries)
        .flat_map(_diff_shard)
        .write_parquet(f"{differences_dir}/part-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    outcome = context.execute(pipeline, verbose=True)
    result_counters = dict(outcome.counters)
    expected_capped = int(capped["counters"]["dedup/fuzzy/document/cluster_members"])
    expected_converged = int(converged["counters"]["dedup/fuzzy/document/cluster_members"])
    actual_capped = int(result_counters.get("marker_diff/capped_markers", 0))
    actual_converged = int(result_counters.get("marker_diff/converged_markers", 0))
    if (actual_capped, actual_converged) != (expected_capped, expected_converged):
        raise AssertionError(
            "Marker diff accounting mismatch: "
            f"capped={actual_capped}/{expected_capped}, converged={actual_converged}/{expected_converged}"
        )
    return MarkerDiffData(
        capped_dedup=capped_dedup_path,
        converged_dedup=converged_dedup_path,
        differences_dir=differences_dir,
        counters=result_counters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capped-dedup", required=True)
    parser.add_argument("--converged-dedup", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-workers", type=int, default=128)
    args = parser.parse_args()
    configure_logging()

    result = compare_markers(
        capped_dedup_path=args.capped_dedup,
        converged_dedup_path=args.converged_dedup,
        output_path=args.output,
        max_workers=args.max_workers,
    )
    StoragePath(f"{args.output.rstrip('/')}/marker-diff.json").write_text(
        json.dumps(result.model_dump(), indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
