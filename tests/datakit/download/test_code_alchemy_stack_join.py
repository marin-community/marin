# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import polars as pl
from marin.datakit.download.code_alchemy_stack_join import (
    BLOB_PREFIX_COUNT,
    CodeAlchemyStackJoinConfig,
    CodeAlchemyStackJoinTask,
    build_code_alchemy_stack_join_pipeline,
    join_code_alchemy_stack_prefix,
    list_code_alchemy_stack_join_tasks,
)
from zephyr.plan import compute_plan


def _task(tmp_path: Path, requested_paths: tuple[str, ...], stack_path: str, prefix: str = "00"):
    return CodeAlchemyStackJoinTask(
        prefix=prefix,
        requested_paths=requested_paths,
        stack_path=stack_path,
        matches_path=str(tmp_path / "matches" / "part.parquet"),
        missing_path=str(tmp_path / "missing" / "part.parquet"),
        requested_stage_path=str(tmp_path / "intermediate" / "requested.parquet"),
        stack_groups_stage_path=str(tmp_path / "intermediate" / "groups.parquet"),
    )


def test_prefix_join_deduplicates_requests_and_stack_sources(tmp_path: Path):
    dev = tmp_path / "dev.parquet"
    dialogue = tmp_path / "dialogue.parquet"
    stack = tmp_path / "stack.parquet"
    pl.DataFrame({"blob_id": ["00aa", "00bb", "00aa"], "dev_only": [1, 2, 3]}).write_parquet(dev)
    pl.DataFrame({"blob_id": ["00aa", "00cc"], "dialogue_only": [True, False]}).write_parquet(dialogue)
    pl.DataFrame(
        {
            "content_id": ["00AA", "00aa", "00bb", "00dd"],
            "content": ["alpha", "alpha", "beta", "irrelevant"],
        }
    ).write_parquet(stack)
    task = _task(tmp_path, (str(dev), str(dialogue)), str(stack))

    result = join_code_alchemy_stack_prefix(task)

    assert result.requested_unique_ids == 3
    assert result.matched_ids == 2
    assert result.missing_ids == 1
    assert result.stack_candidate_rows == 3
    assert result.stack_duplicate_rows_for_requested_ids == 1
    assert result.stack_conflicting_ids_for_requested_ids == 0
    assert pl.read_parquet(task.matches_path).to_dicts() == [
        {"blob_id": "00aa", "source": "alpha"},
        {"blob_id": "00bb", "source": "beta"},
    ]
    assert pl.read_parquet(task.missing_path).to_dicts() == [{"blob_id": "00cc"}]


def test_prefix_join_sends_conflicting_stack_sources_to_download(tmp_path: Path):
    requested = tmp_path / "requested.parquet"
    stack = tmp_path / "stack.parquet"
    pl.DataFrame({"blob_id": ["00aa"]}).write_parquet(requested)
    pl.DataFrame({"content_id": ["00aa", "00aa"], "content": ["alpha", "different"]}).write_parquet(stack)
    task = _task(tmp_path, (str(requested),), str(stack))

    result = join_code_alchemy_stack_prefix(task)

    assert result.matched_ids == 0
    assert result.missing_ids == 1
    assert result.stack_conflicting_ids_for_requested_ids == 1
    assert pl.read_parquet(task.matches_path).is_empty()
    assert pl.read_parquet(task.missing_path).to_dicts() == [{"blob_id": "00aa"}]


def test_task_listing_is_prefix_aligned_and_pipeline_has_256_shards(monkeypatch, tmp_path: Path):
    cfg = CodeAlchemyStackJoinConfig(
        output_path=str(tmp_path / "output"),
        code_alchemy_path=str(tmp_path / "code-alchemy"),
        stack_v3_path=str(tmp_path / "stack-v3"),
    )

    def fake_glob(path: str) -> tuple[str, ...]:
        return (f"{path}/shard.parquet",)

    monkeypatch.setattr("marin.datakit.download.code_alchemy_stack_join._glob_parquet", fake_glob)
    tasks = list_code_alchemy_stack_join_tasks(cfg)
    plan = compute_plan(build_code_alchemy_stack_join_pipeline(tasks, str(tmp_path / "metrics")))

    assert len(tasks) == BLOB_PREFIX_COUNT
    assert plan.num_shards == BLOB_PREFIX_COUNT
    assert [task.prefix for task in tasks] == [f"{index:02x}" for index in range(BLOB_PREFIX_COUNT)]
    for task in tasks:
        assert all(f"blob_prefix={task.prefix}" in path for path in task.requested_paths)
        assert f"content_prefix={task.prefix}" in task.stack_path
        assert f"blob_prefix={task.prefix}" in task.matches_path
        assert f"blob_prefix={task.prefix}" in task.missing_path
