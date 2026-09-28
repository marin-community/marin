# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from levanter.store.cache import CacheLedger, SerialCacheWriter, TreeCache, consolidate_shard_cache_ledgers

from experiments.domain_phase_mix.audit_delphi_tpp40_europe_runtime_caches import (
    DIGEST_ARTIFACT_PREFIXES,
    CachePair,
    ObjectMetadata,
    _is_non_payload_training_object,
    cache_layout,
    cache_pairs,
    compare_metadata,
    digest_artifact_path,
    digest_comparison_filename,
    logical_runtime_contract_sha256,
    shard_ledger_invariants,
    shard_ledger_runtime_layout,
    validate_digest_comparison,
    validate_stack_payload_stats,
)
from experiments.domain_phase_mix.compare_delphi_tpp40_runtime_digests import compare_digest_reports
from experiments.domain_phase_mix.delphi_tpp40_europe_runtime_caches import (
    EUROPE_HISTORICAL_STACK_MERGED_PATH,
    EUROPE_SOURCE_RUNTIME_CACHE_PATHS,
    EXPECTED_STACK_ELEMENTS,
)
from experiments.domain_phase_mix.digest_delphi_tpp40_runtime_cache import (
    ALGORITHM,
    RowRange,
    RuntimeObjectManifest,
    artifact_contract_sha256,
    digest_payload_sha256,
    digest_tree_cache,
    excluded_shard_ranges,
    runtime_object_manifest_binding,
    validate_digest_artifact,
    validate_runtime_evidence,
)
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (
    DOLMA3_AVAILABLE_TOKEN_COUNTS,
    TOP_LEVEL_DOMAIN_PARTITIONS,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION,
)


def test_europe_stack_runtime_binding_uses_historical_cache() -> None:
    assert (
        PREBUILT_MERGED_RUNTIME_CACHE_PATHS_BY_REGION["europe-west4"]["dolma3_stack_edu"]
        == EUROPE_HISTORICAL_STACK_MERGED_PATH
    )


def test_repaired_stem_digest_uses_isolated_europe_namespace() -> None:
    assert digest_artifact_path(region="europe", component="dolmino_stem_heavy_crawl") == (
        "gs://marin-eu-west4/experiments/domain_phase_mix/"
        "delphi_tpp40_multiregion_runtime_digests_v4_stem_metadata_repair/"
        "dolmino_stem_heavy_crawl.json"
    )
    assert digest_artifact_path(region="east5", component="dolmino_stem_heavy_crawl") == (
        "gs://marin-us-east5/experiments/domain_phase_mix/"
        "delphi_tpp40_multiregion_runtime_digests_v4/dolmino_stem_heavy_crawl.json"
    )


def test_runtime_cache_manifest_has_same_region_invariant_identity() -> None:
    pairs = cache_pairs(
        source_cache_resolver=lambda component, regions: (
            EUROPE_SOURCE_RUNTIME_CACHE_PATHS[component].replace(
                "gs://marin-eu-west4",
                "gs://marin-us-east5",
                1,
            )
            if regions == ("us-east5",)
            else None
        )
    )

    assert len(pairs) == 140
    assert len({pair.domain for pair in pairs}) == 39
    assert logical_runtime_contract_sha256(pairs, region="east5") == logical_runtime_contract_sha256(
        pairs,
        region="europe",
    )


def test_stack_identity_allows_reviewed_historical_namespace() -> None:
    pairs = cache_pairs(
        source_cache_resolver=lambda component, regions: (
            EUROPE_SOURCE_RUNTIME_CACHE_PATHS[component].replace(
                "gs://marin-eu-west4",
                "gs://marin-us-east5",
                1,
            )
            if regions == ("us-east5",)
            else None
        )
    )
    stack_index = next(index for index, pair in enumerate(pairs) if pair.component == "dolma3_stack_edu")
    pairs = (
        *pairs[:stack_index],
        replace(
            pairs[stack_index],
            europe_path=(
                "gs://marin-eu-west4/tokenized/merged/"
                "dolma3_dolmino_top_level_historical_full_document_v1/dolma3_stack_edu-newhash"
            ),
        ),
        *pairs[stack_index + 1 :],
    )

    assert logical_runtime_contract_sha256(pairs, region="east5") == logical_runtime_contract_sha256(
        pairs,
        region="europe",
    )


def test_digest_verified_identity_allows_reviewed_historical_namespace() -> None:
    pairs = (
        CachePair(
            domain="dolma3_finemath_3plus",
            component="finemath_3plus",
            east5_path="gs://marin-us-east5/tokenized/finemath_3_plus-a26b0f",
            europe_path="gs://marin-eu-west4/tokenized/finemath_3_plus_historical_full_document_v1-244ece",
        ),
    )

    assert logical_runtime_contract_sha256(pairs, region="east5") == logical_runtime_contract_sha256(
        pairs,
        region="europe",
    )


def test_compare_metadata_detects_missing_or_changed_training_objects() -> None:
    common = {
        ".stats.json": ObjectMetadata(size=12, crc32c="stats"),
        "input_ids/data/c/0": ObjectMetadata(size=100, crc32c="chunk"),
    }

    assert compare_metadata(common, dict(common)) == ()
    assert compare_metadata(common, {".stats.json": common[".stats.json"]}) == ("input_ids/data/c/0",)
    assert compare_metadata(
        common,
        {**common, "input_ids/data/c/0": ObjectMetadata(size=100, crc32c="changed")},
    ) == ("input_ids/data/c/0",)


@pytest.mark.parametrize(
    "relative_path",
    (
        "shard_ledger.json",
        "shard_ledger.json.bak",
        "___temp/input_ids/data/c/0",
        "part-00000.tmp/input_ids/data/c/0",
    ),
)
def test_non_payload_training_objects_exclude_only_metadata_and_scratch(relative_path: str) -> None:
    assert _is_non_payload_training_object(relative_path)


@pytest.mark.parametrize(
    "relative_path",
    (
        "part-00000/shard_ledger.json",
        "part-00000/input_ids/data/c/0",
        "input_ids/data/c/0",
    ),
)
def test_non_payload_training_objects_keep_reachable_runtime_payload(relative_path: str) -> None:
    assert not _is_non_payload_training_object(relative_path)


def test_digest_comparison_binds_acceptance_to_exact_cache_pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pair = CachePair(
        domain="domain",
        component="synth_math/verifiable_o4mini",
        east5_path="gs://marin-us-east5/cache",
        europe_path="gs://marin-eu-west4/cache",
    )
    east5_report = _digest_report(
        cache_path=pair.east5_path,
        selected_rows=173_680,
        selected_tokens=73_921_022,
    )
    europe_report = _digest_report(
        cache_path=pair.europe_path,
        selected_rows=173_680,
        selected_tokens=73_921_022,
    )
    artifact_filename = digest_comparison_filename(pair.component)
    east5_path = tmp_path / "east5" / artifact_filename
    europe_path = tmp_path / "europe" / artifact_filename
    east5_path.parent.mkdir()
    europe_path.parent.mkdir()
    east5_payload = (json.dumps(east5_report, indent=2, sort_keys=True) + "\n").encode()
    europe_payload = (json.dumps(europe_report, indent=2, sort_keys=True) + "\n").encode()
    east5_path.write_bytes(east5_payload)
    europe_path.write_bytes(europe_payload)
    comparison = compare_digest_reports(east5_report, europe_report, mode="acceptance")
    comparison["artifacts"] = {
        "east5": {"path": str(east5_path), "sha256": hashlib.sha256(east5_payload).hexdigest()},
        "europe": {"path": str(europe_path), "sha256": hashlib.sha256(europe_payload).hexdigest()},
    }
    current_manifests = {
        region: RuntimeObjectManifest(
            sha256="objects",
            objects=4,
            bytes=100,
            field_names=("input_ids",),
        )
        for region in ("east5", "europe")
    }
    monkeypatch.setitem(DIGEST_ARTIFACT_PREFIXES, "east5", str(east5_path.parent) + "/")
    monkeypatch.setitem(DIGEST_ARTIFACT_PREFIXES, "europe", str(europe_path.parent) + "/")

    validate_digest_comparison(pair, comparison, current_manifests=current_manifests)
    assert digest_comparison_filename(pair.component) == "synth_math_verifiable_o4mini.json"

    wrong_path = {**comparison, "cache_paths": {**comparison["cache_paths"], "europe": "gs://other/cache"}}
    with pytest.raises(ValueError, match="cache paths differ"):
        validate_digest_comparison(pair, wrong_path, current_manifests=current_manifests)

    diagnostic = {**comparison, "mode": "diagnostic"}
    with pytest.raises(ValueError, match="not an acceptance comparison"):
        validate_digest_comparison(pair, diagnostic, current_manifests=current_manifests)

    stale_hash = {
        **comparison,
        "artifacts": {
            **comparison["artifacts"],
            "europe": {**comparison["artifacts"]["europe"], "sha256": "0" * 64},
        },
    }
    with pytest.raises(ValueError, match="artifact SHA-256 differs"):
        validate_digest_comparison(pair, stale_hash, current_manifests=current_manifests)

    stale_manifest = {
        **current_manifests,
        "europe": RuntimeObjectManifest(
            sha256="new-objects",
            objects=4,
            bytes=100,
            field_names=("input_ids",),
        ),
    }
    with pytest.raises(ValueError, match=r"artifact for .* is stale"):
        validate_digest_comparison(pair, comparison, current_manifests=stale_manifest)


def test_shard_ledger_invariants_ignore_writer_metadata() -> None:
    east5 = {
        "total_num_rows": 10,
        "shard_rows": {"train": 10},
        "is_finished": True,
        "finished_shards": ["train"],
        "metadata": {"tokenizer": "legacy"},
    }
    europe = {**east5, "metadata": {"tokenizer": "current"}, "layout": "consolidated"}

    assert east5 != europe
    assert shard_ledger_invariants(east5) == shard_ledger_invariants(europe)


def test_consolidated_runtime_layout_ignores_finished_shard_bookkeeping() -> None:
    east5 = {
        "total_num_rows": 10,
        "shard_rows": {"train": 10},
        "is_finished": True,
        "finished_shards": [],
    }
    europe = {
        **east5,
        "layout": "consolidated",
        "finished_shards": ["train"],
    }

    assert shard_ledger_runtime_layout(east5) == shard_ledger_runtime_layout(europe)


def test_sharded_runtime_layout_requires_finished_shard_order() -> None:
    east5 = {
        "shard_rows": {"part-0": 4, "part-1": 6},
        "layout": "sharded",
        "finished_shards": ["part-0", "part-1"],
    }
    europe = {**east5, "finished_shards": ["part-1", "part-0"]}

    assert shard_ledger_runtime_layout(east5) != shard_ledger_runtime_layout(europe)


def test_cache_layout_defaults_to_consolidated_and_rejects_unknown_values() -> None:
    assert cache_layout({}) == "consolidated"
    assert cache_layout({"layout": "sharded"}) == "sharded"
    with pytest.raises(ValueError, match="Unknown cache layout"):
        cache_layout({"layout": "future-layout"})


def test_validate_stack_payload_stats_accepts_rechunked_exact_payload() -> None:
    inputs = {
        partition: {"total_tokens": DOLMA3_AVAILABLE_TOKEN_COUNTS[partition], "total_elements": index + 1}
        for index, partition in enumerate(TOP_LEVEL_DOMAIN_PARTITIONS["dolma3_stack_edu"])
    }
    first_partition = next(iter(inputs))
    inputs[first_partition]["total_elements"] += EXPECTED_STACK_ELEMENTS - sum(
        stats["total_elements"] for stats in inputs.values()
    )
    merged = {field: sum(stats[field] for stats in inputs.values()) for field in ("total_tokens", "total_elements")}

    validate_stack_payload_stats(
        east5_merged=merged,
        europe_merged=dict(merged),
        east5_inputs=inputs,
        europe_inputs={partition: dict(stats) for partition, stats in inputs.items()},
    )


def test_validate_stack_payload_stats_rejects_per_language_mismatch() -> None:
    inputs = {
        partition: {"total_tokens": DOLMA3_AVAILABLE_TOKEN_COUNTS[partition], "total_elements": index + 1}
        for index, partition in enumerate(TOP_LEVEL_DOMAIN_PARTITIONS["dolma3_stack_edu"])
    }
    first_partition = next(iter(inputs))
    inputs[first_partition]["total_elements"] += EXPECTED_STACK_ELEMENTS - sum(
        stats["total_elements"] for stats in inputs.values()
    )
    merged = {field: sum(stats[field] for stats in inputs.values()) for field in ("total_tokens", "total_elements")}
    europe_inputs = {partition: dict(stats) for partition, stats in inputs.items()}
    europe_inputs["stack_edu/Ruby"]["total_tokens"] += 1

    with pytest.raises(ValueError, match="per-language"):
        validate_stack_payload_stats(
            east5_merged=merged,
            europe_merged=dict(merged),
            east5_inputs=inputs,
            europe_inputs=europe_inputs,
        )


def _write_consolidated_cache(path: Path, rows: list[list[int]], dtype: str = "int32") -> TreeCache:
    exemplar = {"input_ids": np.zeros((0,), dtype=dtype)}
    records = [{"input_ids": np.asarray(row, dtype=dtype)} for row in rows]
    with SerialCacheWriter(str(path), exemplar) as writer:
        writer.write_batch(records)
    return TreeCache.load(str(path), exemplar)


def _write_sharded_cache(
    path: Path,
    rows: list[list[int]],
    shard_sizes: tuple[int, ...] | None = None,
    dtype: str = "int32",
) -> TreeCache:
    exemplar = {"input_ids": np.zeros((0,), dtype=dtype)}
    if shard_sizes is None:
        midpoint = len(rows) // 2
        shard_sizes = (midpoint, len(rows) - midpoint)
    assert sum(shard_sizes) == len(rows)
    shard_paths = []
    row_cursor = 0
    for shard_index, shard_size in enumerate(shard_sizes):
        shard_rows = rows[row_cursor : row_cursor + shard_size]
        row_cursor += shard_size
        shard_path = path / f"part-{shard_index}"
        records = [{"input_ids": np.asarray(row, dtype=dtype)} for row in shard_rows]
        with SerialCacheWriter(str(shard_path), exemplar) as writer:
            if records:
                writer.write_batch(records)
        shard_paths.append(str(shard_path))
    consolidate_shard_cache_ledgers(shard_paths, str(path), exemplar)
    return TreeCache.load(str(path), exemplar)


def test_logical_digest_is_independent_of_tree_cache_layout(tmp_path: Path) -> None:
    rows = [[1, 2], [3], [4, 5, 6], [7, 8], [9]]
    consolidated = _write_consolidated_cache(tmp_path / "consolidated", rows)
    sharded = _write_sharded_cache(tmp_path / "sharded", rows)

    expected_tokens = sum(map(len, rows))
    consolidated_digest = digest_tree_cache(
        consolidated,
        expected_rows=len(rows),
        expected_tokens=expected_tokens,
        block_rows=2,
    )
    sharded_digest = digest_tree_cache(
        sharded,
        expected_rows=len(rows),
        expected_tokens=expected_tokens,
        block_rows=2,
    )

    assert consolidated_digest["logical_payload_sha256"] == sharded_digest["logical_payload_sha256"]
    assert consolidated_digest["blocks"] == sharded_digest["blocks"]


def test_logical_digest_supports_int64_runtime_cache(tmp_path: Path) -> None:
    rows = [[1, 2], [3], [4, 5, 6]]
    consolidated = _write_consolidated_cache(tmp_path / "consolidated-int64", rows, dtype="int64")
    sharded = _write_sharded_cache(tmp_path / "sharded-int64", rows, dtype="int64")

    expected_tokens = sum(map(len, rows))
    consolidated_digest = digest_tree_cache(
        consolidated,
        expected_rows=len(rows),
        expected_tokens=expected_tokens,
        block_rows=2,
    )
    sharded_digest = digest_tree_cache(
        sharded,
        expected_rows=len(rows),
        expected_tokens=expected_tokens,
        block_rows=2,
    )

    assert consolidated_digest["dtype"] == "int64"
    assert sharded_digest["dtype"] == "int64"
    assert consolidated_digest["logical_payload_sha256"] == sharded_digest["logical_payload_sha256"]


def test_logical_digest_can_exclude_whole_inserted_row_ranges(tmp_path: Path) -> None:
    reference_rows = [[1, 2], [3], [4, 5, 6], [7]]
    expanded_rows = [reference_rows[0], [100], reference_rows[1], reference_rows[2], [200, 201], reference_rows[3]]
    reference = _write_consolidated_cache(tmp_path / "reference", reference_rows)
    expanded = _write_sharded_cache(tmp_path / "expanded", expanded_rows, shard_sizes=(1, 1, 2, 1, 1))
    runtime = validate_runtime_evidence(expanded)
    exclusions, evidence = excluded_shard_ranges(expanded, runtime, ("part-1", "part-3"))

    expected_tokens = sum(map(len, reference_rows))
    reference_digest = digest_tree_cache(
        reference,
        expected_rows=len(reference_rows),
        expected_tokens=expected_tokens,
        block_rows=3,
    )
    selected_digest = digest_tree_cache(
        expanded,
        expected_rows=len(reference_rows),
        expected_tokens=expected_tokens,
        block_rows=3,
        exclusions=exclusions,
        runtime=runtime,
    )

    assert [excluded.name for excluded in evidence] == ["part-1", "part-3"]
    assert selected_digest["source_rows"] == 6
    assert selected_digest["selected_rows"] == 4
    assert reference_digest["logical_payload_sha256"] == selected_digest["logical_payload_sha256"]


def test_logical_digest_detects_token_change_with_unchanged_shapes(tmp_path: Path) -> None:
    reference = _write_consolidated_cache(tmp_path / "reference", [[1, 2], [3], [4, 5]])
    changed = _write_sharded_cache(tmp_path / "changed", [[1, 2], [30], [4, 5]])

    reference_digest = digest_tree_cache(reference, expected_rows=3, expected_tokens=5, block_rows=2)
    changed_digest = digest_tree_cache(changed, expected_rows=3, expected_tokens=5, block_rows=2)

    assert reference_digest["logical_payload_sha256"] != changed_digest["logical_payload_sha256"]
    assert reference_digest["blocks"][0]["sha256"] != changed_digest["blocks"][0]["sha256"]


def test_logical_digest_matches_runtime_row_path_across_four_shards_and_empty_shard(tmp_path: Path) -> None:
    rows = [[1], [2, 3], [4], [5, 6, 7], [8]]
    sharded = _write_sharded_cache(tmp_path / "sharded", rows, shard_sizes=(1, 0, 2, 2))
    runtime_rows = [sharded[index]["input_ids"].tolist() for index in range(len(sharded))]
    runtime_materialized = _write_consolidated_cache(tmp_path / "runtime-materialized", runtime_rows)

    expected_tokens = sum(map(len, rows))
    sharded_digest = digest_tree_cache(
        sharded,
        expected_rows=len(rows),
        expected_tokens=expected_tokens,
        block_rows=2,
    )
    runtime_digest = digest_tree_cache(
        runtime_materialized,
        expected_rows=len(rows),
        expected_tokens=expected_tokens,
        block_rows=2,
    )

    assert sharded_digest["logical_payload_sha256"] == runtime_digest["logical_payload_sha256"]


def test_runtime_evidence_rejects_shard_ledger_token_skew(tmp_path: Path) -> None:
    cache_path = tmp_path / "sharded"
    _write_sharded_cache(cache_path, [[1, 2], [3], [4, 5], [6]], shard_sizes=(2, 2))
    ledger = CacheLedger.load(str(cache_path))
    first, second = ledger.finished_shards
    ledger.field_counts_by_shard[first]["input_ids"] += 1
    ledger.field_counts_by_shard[second]["input_ids"] -= 1
    ledger._serialize_and_commit(str(cache_path))
    skewed = TreeCache.load(str(cache_path), {"input_ids": np.zeros((0,), dtype=np.int32)})

    with pytest.raises(ValueError, match="token count differs from its ledger"):
        validate_runtime_evidence(skewed)


def test_runtime_evidence_rejects_unfinished_ledger(tmp_path: Path) -> None:
    cache_path = tmp_path / "unfinished"
    unfinished = _write_consolidated_cache(cache_path, [[1, 2], [3]])
    unfinished.ledger.is_finished = False

    with pytest.raises(ValueError, match="not finalized"):
        validate_runtime_evidence(unfinished)


def test_runtime_evidence_allows_missing_field_count_for_empty_shard(tmp_path: Path) -> None:
    cache_path = tmp_path / "sharded"
    _write_sharded_cache(cache_path, [[1], [2]], shard_sizes=(1, 0, 1))
    ledger = CacheLedger.load(str(cache_path))
    empty_shard = ledger.finished_shards[1]
    del ledger.field_counts_by_shard[empty_shard]["input_ids"]
    ledger._serialize_and_commit(str(cache_path))
    cache = TreeCache.load(str(cache_path), {"input_ids": np.zeros((0,), dtype=np.int32)})

    evidence = validate_runtime_evidence(cache)

    assert evidence.shard_stats[empty_shard].rows == 0
    assert evidence.shard_stats[empty_shard].tokens == 0


def test_runtime_object_manifest_binding_is_json_stable() -> None:
    manifest = RuntimeObjectManifest(
        sha256="digest",
        objects=4,
        bytes=100,
        field_names=("input_ids",),
    )

    binding = runtime_object_manifest_binding(manifest)

    assert json.loads(json.dumps(binding)) == binding


def test_logical_digest_rejects_empty_selection(tmp_path: Path) -> None:
    cache = _write_consolidated_cache(tmp_path / "cache", [[1], [2]])

    with pytest.raises(ValueError, match="cannot select zero rows"):
        digest_tree_cache(
            cache,
            expected_rows=0,
            expected_tokens=0,
            exclusions=(RowRange(0, 2),),
        )


def test_logical_digest_detects_wrong_same_length_exclusion(tmp_path: Path) -> None:
    reference = _write_consolidated_cache(tmp_path / "reference", [[1], [2], [3]])
    expanded = _write_sharded_cache(tmp_path / "expanded", [[1], [90], [2], [91], [3]], shard_sizes=(1, 1, 1, 1, 1))
    reference_digest = digest_tree_cache(reference, expected_rows=3, expected_tokens=3, block_rows=2)
    wrong_digest = digest_tree_cache(
        expanded,
        expected_rows=3,
        expected_tokens=3,
        block_rows=2,
        exclusions=(RowRange(1, 2), RowRange(4, 5)),
    )

    assert reference_digest["logical_payload_sha256"] != wrong_digest["logical_payload_sha256"]


def test_logical_digest_detects_row_boundary_change(tmp_path: Path) -> None:
    reference = _write_consolidated_cache(tmp_path / "reference", [[1, 2], [3]])
    changed = _write_sharded_cache(tmp_path / "changed", [[1], [2, 3]])

    reference_digest = digest_tree_cache(reference, expected_rows=2, expected_tokens=3)
    changed_digest = digest_tree_cache(changed, expected_rows=2, expected_tokens=3)

    assert reference_digest["logical_payload_sha256"] != changed_digest["logical_payload_sha256"]


def _digest_report(
    *,
    block_sha256: str = "a" * 64,
    selected_rows: int = 2,
    selected_tokens: int = 3,
    excluded_shards: list[dict[str, object]] | None = None,
    excluded_row_ranges: list[dict[str, object]] | None = None,
    metadata_sha256: str = "metadata",
    algorithm: str = ALGORITHM,
    cache_path: str = "gs://test/cache",
    source_rows: int | None = None,
) -> dict[str, object]:
    source_rows = selected_rows if source_rows is None else source_rows
    block_rows = max(4_096, selected_rows)
    report = {
        "status": "complete",
        "algorithm": algorithm,
        "block_rows": block_rows,
        "selected_rows": selected_rows,
        "source_rows": source_rows,
        "selected_tokens": selected_tokens,
        "source_tokens": selected_tokens,
        "dtype": "int32",
        "field_names": ["input_ids"],
        "blocks": [
            {
                "output_row_start": 0,
                "output_row_stop": selected_rows,
                "token_count": selected_tokens,
                "sha256": block_sha256,
            }
        ],
        "excluded_row_ranges": excluded_row_ranges or [],
        "binding": {
            "algorithm": algorithm,
            "cache_path": cache_path,
            "block_rows": block_rows,
            "expected_rows": selected_rows,
            "expected_tokens": selected_tokens,
            "ledger_sha256": "ledger",
            "preprocessor_metadata_sha256": metadata_sha256,
            "runtime_object_manifest": {
                "sha256": "objects",
                "objects": 4,
                "bytes": 100,
                "field_names": ["input_ids"],
            },
            "excluded_shards": excluded_shards or [],
        },
    }
    report["logical_payload_sha256"] = digest_payload_sha256(report)
    report["artifact_contract_sha256"] = artifact_contract_sha256(report)
    return report


def test_digest_comparator_accepts_exact_payload_without_exclusions() -> None:
    report = _digest_report()

    comparison = compare_digest_reports(report, dict(report), mode="acceptance")

    assert comparison["status"] == "equivalent"
    assert comparison["equivalent"] is True


def test_digest_comparator_rejects_exclusions_in_acceptance_mode() -> None:
    excluded = [{"name": "part-1", "source_row_start": 1, "source_row_stop": 2, "rows": 1, "tokens": 1}]
    report = _digest_report(
        excluded_shards=excluded,
        excluded_row_ranges=[{"start": 1, "stop": 2}],
        source_rows=3,
    )

    acceptance = compare_digest_reports(report, dict(report), mode="acceptance")
    diagnostic = compare_digest_reports(report, dict(report), mode="diagnostic")

    assert acceptance["status"] == "mismatch"
    assert acceptance["exclusion_gate_passes"] is False
    assert diagnostic["status"] == "equivalent"


def test_digest_comparator_rejects_derived_row_exclusion_in_acceptance_mode() -> None:
    report = _digest_report(
        excluded_row_ranges=[{"start": 2, "stop": 3}],
        source_rows=3,
    )

    acceptance = compare_digest_reports(report, dict(report), mode="acceptance")

    assert acceptance["status"] == "mismatch"
    assert acceptance["exclusion_gate_passes"] is False


def test_digest_comparator_reports_payload_and_provenance_mismatches() -> None:
    east5 = _digest_report()
    europe = _digest_report(block_sha256="b" * 64, selected_tokens=4, metadata_sha256="other")

    comparison = compare_digest_reports(east5, europe, mode="acceptance")

    assert comparison["status"] == "mismatch"
    assert comparison["payload_matches"] is False
    assert comparison["block_mismatch_count"] == 1
    assert comparison["provenance_mismatches"] == ["preprocessor_metadata_sha256"]


def test_digest_artifact_rejects_corrupted_root_hash() -> None:
    report = _digest_report()
    report["logical_payload_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="logical payload SHA-256 is invalid"):
        validate_digest_artifact(report)


def test_digest_artifact_rejects_zero_exclusion_source_token_mismatch() -> None:
    report = _digest_report()
    report["source_tokens"] = report["selected_tokens"] + 1
    report["artifact_contract_sha256"] = artifact_contract_sha256(report)

    with pytest.raises(ValueError, match="selected_tokens does not match source_tokens"):
        validate_digest_artifact(report)


def test_digest_comparator_classifies_algorithm_change_as_incomparable() -> None:
    east5 = _digest_report()
    europe = _digest_report(algorithm="future-algorithm")

    with pytest.raises(ValueError, match="algorithm"):
        compare_digest_reports(east5, europe, mode="acceptance")


def test_digest_artifact_rejects_relabelled_exclusion_story() -> None:
    excluded = [{"name": "part-1", "source_row_start": 1, "source_row_stop": 2, "rows": 1, "tokens": 1}]
    report = _digest_report(
        excluded_shards=excluded,
        excluded_row_ranges=[{"start": 1, "stop": 2}],
        source_rows=3,
    )
    report["source_rows"] = report["selected_rows"]
    report["excluded_row_ranges"] = []
    report["binding"]["excluded_shards"] = []

    with pytest.raises(ValueError, match="contract SHA-256"):
        validate_digest_artifact(report)
