# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare NUPA sampling, manifest resolution, prompts, and native grades with frozen source evidence."""

import json
from pathlib import Path

import pytest
from fray.local_backend import LocalClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.convert.evaluation import source_evidence
from taskcompendium.grading import grade_answer
from taskcompendium.importers.evaluation_nupa import read_nupa_panel, read_nupa_sample
from taskcompendium.models import ConversationTrace, GradingAttempt, TextMessage
from taskcompendium.pipeline.evaluation import evaluation_cases, publish_evaluation_corpus
from taskcompendium.pipeline.inputs import ConversionContext
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.numeric import sources
from experiments.post_training.task_curation.pipeline import source_recipe

ORACLE = json.loads((Path(__file__).parent / "fixtures/evalchemy_nupa_oracle.json").read_text())


@pytest.fixture
def staged_source(tmp_path):
    primary = tmp_path / "source"
    primary.mkdir()
    (primary / "test.json").write_text(json.dumps(ORACLE["nested_source"]))
    panel = tmp_path / "panel"
    panel.mkdir()
    (panel / "nupa5k_manifest.jsonl").write_text(
        "".join(json.dumps(identity) + "\n" for identity in ORACLE["panel_manifest"])
    )
    return primary, panel


@pytest.mark.parametrize("reference", ORACLE["cases"], ids=lambda case: case["name"])
def test_nupa_source_rows_and_published_grades_match_source(reference, staged_source, tmp_path):
    primary, panel = staged_source
    dataset = next(source.config for source in sources() if source.config.protocol.benchmark == reference["name"])
    inputs = {"panel": str(panel)} if reference["name"] == "NUPA5K-Loose" else {}
    context = ConversionContext({name: StoragePath(path) for name, path in inputs.items()}, None)
    reader = read_nupa_panel if inputs else read_nupa_sample
    records = list(reader(StoragePath(str(primary / "test.json")), context))
    assert [
        {key: row[key] for key in expected} for row, expected in zip(records, reference["records"], strict=True)
    ] == reference["records"]
    protocol = dataset.protocol.model_copy(update={"expected_population": len(records)})
    recipe = source_recipe(dataset.recipe, inputs, None)
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as zephyr:
            manifest = publish_evaluation_corpus(
                recipe, protocol, dataset.item_builder, zephyr, str(primary), str(output)
            )
    finally:
        client.shutdown()
    cases = list(evaluation_cases(str(output)))
    assert manifest.rows == len(reference["specs"])
    for case, spec in zip(cases, reference["specs"], strict=True):
        assert [event.model_dump(include={"role", "content"}) for event in case.task.context.events] == spec["requests"][
            0
        ]["prompt"]
        assert case.item.groups["source_id"] == spec["source_id"]
        assert case.item.groups["task_bucket"] == spec["tags"]["task_bucket"]
        assert source_evidence(case.task)["answer"] == spec["target"]
    for grading in reference["grades"]:
        task = cases[grading["ordinal"]].task
        result = grade_answer(
            task,
            GradingAttempt(
                ConversationTrace(
                    events=(
                        *task.context.events,
                        TextMessage(role="assistant", content=grading["response"]),
                    )
                )
            ),
        )
        assert result.detail["metrics"] == grading["grade"]["metrics"]
        assert result.detail["extracted"] == grading["grade"]["filtered_resps"][0]


def test_nupa_panel_order_does_not_depend_on_source_task_order(staged_source):
    primary, panel = staged_source
    path = primary / "test.json"
    context = ConversionContext({"panel": StoragePath(str(panel))}, None)
    before = list(read_nupa_panel(StoragePath(str(path)), context))
    path.write_text(json.dumps(dict(reversed(list(ORACLE["nested_source"].items())))))
    after = list(read_nupa_panel(StoragePath(str(path)), context))
    assert after == before
