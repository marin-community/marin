# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from rigging.filesystem.storage_path import StoragePath

from taskcompendium.importers.scienceagentbench import (
    KnowledgeCondition,
    VerifiedAnnotation,
    actor_context,
    read_verified_annotations,
)
from taskcompendium.pipeline.inputs import ConversionContext


@pytest.fixture
def annotation():
    return VerifiedAnnotation(
        instance_id=1,
        domain="synthetic-test-domain",
        subtask_categories="regression",
        github_name="private-source-repository",
        task_inst="Fit the observations and save predictions in pred_results/output.csv.",
        domain_knowledge="EXPERT_ONLY: the response is quadratic.",
        dataset_folder_tree="|-- observations/\n|---- train.csv\n|---- test.csv",
        dataset_preview="column_a,column_b\n" + "0,1\n" * 1000 + "LAST_PREVIEW_ROW",
        src_file_or_path="private-source-program.py",
        gold_program_name="private-reference.py",
        output_fname="pred_results/output.csv",
        eval_script_name="private-evaluator.py",
    )


def test_paired_inputs_change_only_expert_knowledge_and_preserve_full_preview(annotation):
    off = actor_context(annotation, KnowledgeCondition.OFF).model_dump_json()
    on = actor_context(annotation, KnowledgeCondition.ON).model_dump_json()
    assert annotation.domain_knowledge not in off
    assert annotation.domain_knowledge in on
    for private in (
        annotation.github_name,
        annotation.src_file_or_path,
        annotation.gold_program_name,
        annotation.eval_script_name,
    ):
        assert private not in off
        assert private not in on
    off_text = actor_context(annotation, KnowledgeCondition.OFF).events[0].content
    on_text = actor_context(annotation, KnowledgeCondition.ON).events[0].content
    assert on_text.replace(f"## Expert knowledge\n\n{annotation.domain_knowledge}\n\n", "") == off_text
    assert annotation.task_inst in off_text
    assert annotation.dataset_preview in off_text
    assert "benchmark/datasets/observations" in off_text


def test_null_optional_inputs_do_not_render_as_literal_none(annotation):
    without_optional = annotation.model_copy(update={"domain_knowledge": None, "dataset_preview": None})
    on = actor_context(without_optional, KnowledgeCondition.ON)
    off = actor_context(without_optional, KnowledgeCondition.OFF)
    assert on == off
    assert "None" not in on.events[0].content
    assert annotation.task_inst in on.events[0].content


def test_panel_reader_preserves_source_order_and_nulls(tmp_path, annotation):
    rows = [annotation.model_copy(update={"instance_id": i}).model_dump() for i in range(102, 0, -1)]
    rows[0]["domain_knowledge"] = None
    path = tmp_path / "verified.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    loaded = list(read_verified_annotations(StoragePath(str(path)), ConversionContext({}, None)))
    assert loaded == rows


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "unexpected"])
def test_incomplete_panel_yields_nothing(tmp_path, annotation, corruption):
    rows = [annotation.model_copy(update={"instance_id": i}).model_dump() for i in range(1, 103)]
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[-1]["instance_id"] = 103
    path = tmp_path / "verified.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path)
    reader = read_verified_annotations(StoragePath(str(path)), ConversionContext({}, None))
    with pytest.raises(ValueError, match="each instance ID"):
        next(reader)
