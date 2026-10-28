# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Pinned JEE source prompt, cohort, extraction, partial credit, and float-boundary evidence."""

import json
from pathlib import Path

from fray.local_backend import LocalClient
from taskcompendium.grading import grade_answer
from taskcompendium.models import ConversationTrace, GradingAttempt, TextMessage
from taskcompendium.pipeline.evaluation import evaluation_cases, publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.jee import sources
from experiments.post_training.task_curation.pipeline import source_recipe

ORACLE = json.loads((Path(__file__).parent / "fixtures/evalchemy_jee_oracle.json").read_text())


def test_jee_publication_preserves_source_prompts_cohort_and_grades(tmp_path):
    primary = tmp_path / "source"
    primary.mkdir()
    (primary / "test.json").write_text(json.dumps(ORACLE["rows"]))
    dataset = sources()[0].config
    protocol = dataset.protocol.model_copy(update={"expected_population": len(ORACLE["rows"])})
    recipe = source_recipe(dataset.recipe, {}, None)
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(primary), str(output))
    finally:
        client.shutdown()
    cases = list(evaluation_cases(str(output)))
    assert len(cases) * protocol.trials.count == len(ORACLE["specs"])
    assert [protocol.trials.seed_for(index) for index in range(protocol.trials.count)] == [None, None, None]
    for case, spec in zip(cases, ORACLE["specs"][::3], strict=True):
        assert [event.model_dump(include={"role", "content"}) for event in case.task.context.events] == spec["requests"][
            0
        ]["prompt"]
        assert case.item.groups["source_id"] == spec["source_id"]
        assert case.item.groups["type"] == spec["tags"]["type"]
    for grading in ORACLE["grades"]:
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
        assert result.reward == grading["grade"]["metrics"]["accuracy"]
        assert result.detail["extracted"] == grading["grade"]["filtered_resps"][0]
