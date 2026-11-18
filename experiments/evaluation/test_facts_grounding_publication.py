# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FACTS Grounding v2 publication sends the official full prompt and binds the judge's grounding inputs."""

import csv
import json

from fray.local_backend import LocalClient
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.runtime.resources import resource_bytes
from zephyr.context import ZephyrContext

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation.pipeline import source_recipe

# Longer than the standard library CSV reader's default 131,072-character field limit.
LONG_DOCUMENT = "The museum opens at nine. " * 6000


def test_publication_sends_full_prompt_and_binds_request_and_document(tmp_path):
    [source] = evaluation_benchmarks()["facts-grounding-v2"].resolve()
    dataset = source.config
    rows = [
        {
            "system_instruction": "Answer only from the context.",
            "user_request": "When does the museum open?",
            "context_document": LONG_DOCUMENT,
            "full_prompt": "Answer only from the context.\n\nWhen does the museum open?\n\n" + LONG_DOCUMENT,
            "domain": "Retail/Product",
            "type": "Fact Finding",
            "high_level_type": "Q&A",
        },
        {
            "system_instruction": "Summarize faithfully.",
            "user_request": 'Summarize the ruling, with "quotes".',
            "context_document": "Line one.\nLine two, with a comma.",
            "full_prompt": "Summarize faithfully.\nSummarize the ruling.\nLine one.\nLine two, with a comma.",
            "domain": "Legal",
            "type": "Summarization",
            "high_level_type": "Text Transformation",
        },
    ]
    source_root = tmp_path / "source"
    primary = source_root / dataset.recipe.files[0]
    primary.parent.mkdir(parents=True)
    with primary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
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
    assert [case.item.groups for case in cases] == [{"domain": "Retail/Product"}, {"domain": "Legal"}]
    assert {case.item.operation.response_selection for case in cases} == {"final_only"}
    for case, row in zip(cases, rows, strict=True):
        [event] = case.task.context.events
        assert (event.role, event.content) == ("user", row["full_prompt"])
        judge = case.task.grader.judge
        assert (judge.question, judge.references) == (row["user_request"], (row["context_document"],))
        config = json.loads(resource_bytes(next(r for r in case.task.resources.verifier if r.path == "config.json")))
        assert config == {"full_prompt": row["full_prompt"]}
