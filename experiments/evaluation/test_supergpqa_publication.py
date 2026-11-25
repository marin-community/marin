# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""SuperGPQA publication renders the authors' prompt and binds the reference MCQ grader and groups."""

import json

from fray.local_backend import LocalClient
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, GraderMachine, verifyit_spec
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from verifyit.spec import McqExtraction, McqSpec
from zephyr.context import ZephyrContext

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation.pipeline import source_recipe


def row(uuid, question, options, letter, discipline, difficulty):
    return {
        "uuid": uuid,
        "question": question,
        "options": options,
        "answer": options[ord(letter) - 65],
        "answer_letter": letter,
        "discipline": discipline,
        "field": "field",
        "subfield": "subfield",
        "difficulty": difficulty,
        "is_calculation": False,
    }


def test_publication_renders_reference_prompt_with_discipline_and_difficulty_groups(tmp_path):
    [source] = evaluation_benchmarks()["supergpqa"].resolve()
    dataset = source.config
    rows = [
        row("a", "Which number is prime?", ["4", "6", "7", "9"], "C", "Science", "easy"),
        row("b", "Which number is even?", ["8", "3", "5", "7", "9"], "A", "Engineering", "hard"),
    ]
    source_root = tmp_path / "source"
    primary = source_root / dataset.recipe.files[0]
    primary.parent.mkdir(parents=True)
    primary.write_text("".join(json.dumps(item) + "\n" for item in rows))
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
    )
    recipe = source_recipe(dataset.recipe, {}, environment)
    protocol = dataset.protocol
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(source_root), str(output))
    finally:
        client.shutdown()
    cases = sorted(evaluation_cases(str(output)), key=lambda case: case.item.source_ordinal)
    assert [case.item.groups for case in cases] == [
        {"discipline": "Science", "difficulty": "easy"},
        {"discipline": "Engineering", "difficulty": "hard"},
    ]
    [event] = cases[0].task.context.events
    assert event.role == "user"
    assert event.content.endswith("\n\nWhich number is prime?\nA) 4\nB) 6\nC) 7\nD) 9\n")
    reference = McqExtraction.SUPERGPQA_REFERENCE
    assert [verifyit_spec(case.task.grader) for case in cases] == [
        McqSpec("C", options=4, extraction=reference, option_texts=("4", "6", "7", "9")),
        McqSpec("A", options=5, extraction=reference, option_texts=("8", "3", "5", "7", "9")),
    ]
    assert all(case.task.grader.placement == GraderMachine(environment=environment) for case in cases)
