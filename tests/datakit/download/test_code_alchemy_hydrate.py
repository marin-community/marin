# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

import marin.datakit.download.code_alchemy_hydrate as hydrate


def test_merge_deduplicates_identical_values_and_surfaces_conflicts():
    stack = pl.DataFrame(
        {
            "blob_id": ["aa01", "aa01", "aa02", None, "bb01"],
            "source": ["same", "same", "from stack", "null id", "wrong partition"],
        },
        schema=hydrate._SOURCE_SCHEMA,
    )
    downloaded = pl.DataFrame(
        {
            "blob_id": ["aa01", "aa02", "aa03", "aa04"],
            "source": ["same", "from swh", "download only", None],
        },
        schema=hydrate._SOURCE_SCHEMA,
    )
    fallback = pl.DataFrame(
        {"blob_id": ["aa05"], "source": ["mirror fallback"]},
        schema=hydrate._SOURCE_SCHEMA,
    )

    merged, conflicts, unresolved, metrics = hydrate.merge_source_frames(
        stack,
        downloaded,
        fallback,
        prefix="aa",
    )

    assert merged.to_dicts() == [
        {"blob_id": "aa01", "source": "same"},
        {"blob_id": "aa02", "source": "from stack"},
        {"blob_id": "aa03", "source": "download only"},
        {"blob_id": "aa05", "source": "mirror fallback"},
    ]
    assert conflicts.to_dicts() == [
        {"blob_id": "aa02", "source": "from stack", "origin": "stack_v3", "selected": True},
        {"blob_id": "aa02", "source": "from swh", "origin": "software_heritage", "selected": False},
    ]
    assert unresolved.select("blob_id", "reason").to_dicts() == [
        {"blob_id": None, "reason": "null_blob_id"},
        {"blob_id": "bb01", "reason": "prefix_mismatch"},
        {"blob_id": "aa04", "reason": "null_source"},
    ]
    assert metrics.output_rows == 4
    assert metrics.fallback_rows == 1
    assert metrics.duplicate_identical_rows == 2
    assert metrics.conflicting_blob_ids == 1
    assert metrics.conflicting_source_values == 2
    assert metrics.null_blob_id_rows == 1
    assert metrics.null_source_rows == 1
    assert metrics.prefix_mismatch_rows == 1


def test_merge_conflict_choice_is_independent_of_input_order():
    stack = pl.DataFrame({"blob_id": ["aa01", "aa01"], "source": ["z", "a"]})
    empty = pl.DataFrame(schema=hydrate._SOURCE_SCHEMA)

    forward = hydrate.merge_source_frames(stack, empty, prefix="aa")
    reverse = hydrate.merge_source_frames(stack.reverse(), empty, prefix="aa")

    assert forward[0].to_dicts() == [{"blob_id": "aa01", "source": "a"}]
    assert_frame_equal(forward[0], reverse[0])
    assert_frame_equal(forward[1], reverse[1])


def test_merge_partition_fails_on_conflicting_sources(tmp_path: Path):
    stack_path = tmp_path / "stack.parquet"
    pl.DataFrame(
        {"blob_id": ["aa01", "aa01"], "source": ["first", "second"]},
        schema=hydrate._SOURCE_SCHEMA,
    ).write_parquet(stack_path)
    output_path = tmp_path / "sources.parquet"
    conflicts_path = tmp_path / "conflicts.parquet"
    unresolved_path = tmp_path / "unresolved.parquet"
    task = hydrate.MergeSourceTask(
        prefix="aa",
        stack_paths=(str(stack_path),),
        downloaded_paths=(),
        fallback_paths=(),
        output_path=str(output_path),
        conflicts_path=str(conflicts_path),
        unresolved_path=str(unresolved_path),
    )

    with pytest.raises(RuntimeError, match="conflicting_blob_ids=1"):
        hydrate.merge_source_partition(task)

    assert conflicts_path.exists()
    assert unresolved_path.exists()
    assert not output_path.exists()


def test_hydrate_replaces_only_exact_marker_and_preserves_schema_rows_and_order():
    rows = pl.DataFrame(
        {
            "blob_id": ["aa01", "aa02", None, "aa03"],
            "text_with_placeholders": [
                f"before {hydrate.PLACEHOLDER} middle {hydrate.PLACEHOLDER} after",
                "unrelated {{REPLACE_WITH_BLOB_ID_SOURCE}} text",
                hydrate.PLACEHOLDER,
                "no placeholder",
            ],
            "score": [0.5, 1.25, 2.0, 3.5],
            "valid": [True, False, True, False],
        },
        schema={
            "blob_id": pl.String,
            "text_with_placeholders": pl.String,
            "score": pl.Float64,
            "valid": pl.Boolean,
        },
    )
    sources = pl.DataFrame({"blob_id": ["aa01"], "source": [" \nprint('$kept')\t"]})

    hydrated_rows, unresolved_rows, unresolved_ids, metrics = hydrate.hydrate_frame(
        rows,
        sources,
        subset="code-dev",
        prefix="aa",
    )

    assert hydrated_rows.columns == rows.columns
    assert hydrated_rows.schema == rows.schema
    assert hydrated_rows.get_column("blob_id").to_list() == rows.get_column("blob_id").to_list()
    assert hydrated_rows.get_column("text_with_placeholders").to_list() == [
        "before print('$kept') middle print('$kept') after",
        "unrelated {{REPLACE_WITH_BLOB_ID_SOURCE}} text",
        hydrate.PLACEHOLDER,
        "no placeholder",
    ]
    assert unresolved_rows.select("blob_id", "unresolved_reason").to_dicts() == [
        {"blob_id": None, "unresolved_reason": "null_blob_id"}
    ]
    assert unresolved_ids.to_dicts() == [
        {"blob_id": None, "unresolved_reason": "null_blob_id", "row_count": 1}
    ]
    assert metrics.input_rows == metrics.output_rows == 4
    assert metrics.rows_with_placeholder == 2
    assert metrics.rows_with_available_source == 1
    assert metrics.missing_source_rows == 3
    assert metrics.null_blob_id_rows == 1
    assert metrics.unresolved_placeholder_rows == 1


def test_hydrate_preserves_code_dialogue_integer_score_schema():
    rows = pl.DataFrame(
        {
            "blob_id": ["aa01"],
            "text_with_placeholders": [hydrate.PLACEHOLDER],
            "quality_score": [4],
            "has_reasoning": [True],
        },
        schema={
            "blob_id": pl.String,
            "text_with_placeholders": pl.String,
            "quality_score": pl.Int64,
            "has_reasoning": pl.Boolean,
        },
    )
    sources = pl.DataFrame({"blob_id": ["aa01"], "source": ["dialogue source"]})

    hydrated_rows, _, _, metrics = hydrate.hydrate_frame(
        rows,
        sources,
        subset="code-dialogue",
        prefix="aa",
    )

    assert hydrated_rows.schema == rows.schema
    assert hydrated_rows.to_dicts() == [
        {
            "blob_id": "aa01",
            "text_with_placeholders": "dialogue source",
            "quality_score": 4,
            "has_reasoning": True,
        }
    ]
    assert metrics.unresolved_placeholder_rows == 0


def test_source_that_contains_marker_remains_an_unresolved_diagnostic():
    rows = pl.DataFrame(
        {
            "blob_id": ["aa01"],
            "text_with_placeholders": [hydrate.PLACEHOLDER],
        }
    )
    sources = pl.DataFrame(
        {
            "blob_id": ["aa01"],
            "source": [f"source accidentally contains {hydrate.PLACEHOLDER}"],
        }
    )

    hydrated_rows, unresolved_rows, _, metrics = hydrate.hydrate_frame(
        rows,
        sources,
        subset="code-dialogue",
        prefix="aa",
    )

    assert hydrate.PLACEHOLDER in hydrated_rows.item(0, "text_with_placeholders")
    assert unresolved_rows.item(0, "unresolved_reason") == "replacement_source_contains_marker"
    assert metrics.unresolved_placeholder_rows == 1


def test_unresolved_placeholders_fail_unless_diagnostic_only():
    result = hydrate.HydrateResult(
        subset="code-dev",
        prefix="aa",
        input_rows=2,
        output_rows=2,
        rows_with_placeholder=1,
        rows_with_available_source=0,
        missing_source_rows=1,
        null_blob_id_rows=0,
        prefix_mismatch_rows=0,
        unresolved_placeholder_rows=1,
        unresolved_blob_ids=1,
    )

    with pytest.raises(RuntimeError, match="left 1 rows"):
        hydrate.ensure_no_unresolved_placeholders([result], diagnostic_only=False)
    hydrate.ensure_no_unresolved_placeholders([result], diagnostic_only=True)


def test_defaults_use_contracted_input_and_output_roots():
    cfg = hydrate.CodeAlchemyHydrateConfig()

    assert cfg.output_path == "s3://marin-us-east-02a/tmp/ttl=30d/code-alchemy-hydration"
    assert cfg.code_alchemy_path == "s3://marin-us-east-02a/tmp/ttl=30d/code-alchemy"
    assert cfg.stack_sources_path == f"{cfg.output_path}/stack-v3-matches"
    assert cfg.downloaded_sources_path == f"{cfg.output_path}/downloaded-sources"
    assert cfg.fallback_sources_path == f"{cfg.output_path}/fallback-sources"


def test_task_paths_are_prefix_aligned_and_keep_subset_partitions(monkeypatch, tmp_path: Path):
    input_root = tmp_path / "code-alchemy"
    for subset, prefix in (("code-dev", "aa"), ("code-dialogue", "null")):
        partition = input_root / "data" / f"subset={subset}" / f"blob_prefix={prefix}"
        partition.mkdir(parents=True)
        pl.DataFrame(
            {
                "blob_id": ["aa01" if prefix == "aa" else None],
                "text_with_placeholders": [hydrate.PLACEHOLDER],
            },
            schema={"blob_id": pl.String, "text_with_placeholders": pl.String},
        ).write_parquet(partition / "source.parquet")

    monkeypatch.setattr(
        hydrate,
        "_parquet_paths",
        lambda stage_path, prefix: (f"{stage_path}/data/blob_prefix={prefix}/part-00000.parquet",),
    )

    cfg = hydrate.CodeAlchemyHydrateConfig(
        output_path=str(tmp_path / "output"),
        code_alchemy_path=str(input_root),
        stack_sources_path=str(tmp_path / "stack-v3-matches"),
        downloaded_sources_path=str(tmp_path / "downloaded-sources"),
    )
    merge_tasks = hydrate.list_merge_tasks(cfg)
    hydration_tasks = hydrate.list_hydrate_tasks(cfg)

    assert len(merge_tasks) == 256
    assert merge_tasks[170].prefix == "aa"
    assert merge_tasks[170].output_path == str(
        tmp_path / "output" / "sources" / "data" / "blob_prefix=aa" / "part-00000.parquet"
    )
    assert [(task.subset, task.prefix) for task in hydration_tasks] == [
        ("code-dev", "aa"),
        ("code-dialogue", "null"),
    ]
    assert hydration_tasks[0].output_path == str(
        tmp_path / "output" / "hydrated" / "subset=code-dev" / "blob_prefix=aa" / "part-00000.parquet"
    )
    assert hydration_tasks[1].source_paths == ()
