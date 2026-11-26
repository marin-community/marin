# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MMLU-Redux publication renders ZeroEval's prompt and binds gold letters and subject groups."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.runtime.resources import resource_bytes
from zephyr.context import ZephyrContext

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation.pipeline import source_recipe


def test_publication_renders_zeroeval_prompt_with_subject_groups(tmp_path):
    [source] = evaluation_benchmarks()["mmlu-redux"].resolve()
    dataset = source.config
    rows = [
        {
            "id": "mmlu-redux-anatomy-#0",
            "question": "Which number is prime?",
            "choices": ["4", "6", "7", "9"],
            "correct_answer": "7",
            "source": "edinburgh-dawg/mmlu-redux",
            "config": "anatomy",
            "task_type": "multiple_choice",
        },
        {
            "id": "mmlu-redux-virology-#0",
            "question": "Which number is even?",
            "choices": ["8", "3", "5", "7"],
            "correct_answer": "8",
            "source": "edinburgh-dawg/mmlu-redux",
            "config": "virology",
            "task_type": "multiple_choice",
        },
    ]
    source_root = tmp_path / "source"
    primary = source_root / dataset.recipe.files[0]
    primary.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), primary)
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
    )
    recipe = source_recipe(dataset.recipe, {}, environment)
    protocol = dataset.protocol.model_copy(update={"expected_population": len(rows)})
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(source_root), str(output))
    finally:
        client.shutdown()
    cases = sorted(evaluation_cases(str(output)), key=lambda case: case.item.source_ordinal)
    assert [case.item.groups for case in cases] == [{"subject": "anatomy"}, {"subject": "virology"}]
    [event] = cases[0].task.context.events
    assert event.role == "user"
    assert event.content.startswith("\n## Question: \n\nWhich number is prime?\n\n## Choices:\n\n- (A) 4\n")
    configs = [
        json.loads(resource_bytes(next(r for r in case.task.resources.verifier if r.path == "config.json")))
        for case in cases
    ]
    assert configs == [{"expected": "C"}, {"expected": "A"}]
