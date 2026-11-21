# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MMLU-Pro publication uses the staged category-specific fewshot split and sends its worked examples as turns."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.evaluation.corpus import EvaluationCase, evaluation_cases
from taskcompendium.evaluation.selection import prompt_variant
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks.reasoning import sources
from experiments.post_training.task_curation.pipeline import source_recipe
from experiments.post_training.task_curation.tests.conversion import converted_task

DOCUMENT = {
    "question": "Which integer is even?",
    "options": ["1", "2", "N/A"],
    "answer": "B",
    "category": "math",
    "cot_content": "A: Let's think step by step. The answer is (B).",
}
SUPPORT = [
    {**DOCUMENT, "question": "OTHER_CATEGORY_SENTINEL", "category": "physics"},
    *[{**DOCUMENT, "question": f"STAGED_FEWSHOT_{index}"} for index in range(6)],
]
# The source's one-message prompt, written out: the instruction, each worked example's question block and answer, then
# the question.
INSTRUCTION = (
    "The following are multiple choice questions (with answers) about math. Think step by step and then finish your "
    'answer with "the answer is (X)" where X is the correct letter choice.\n\n\n\n'
)
WORKED = tuple(
    (
        f"Question:\nSTAGED_FEWSHOT_{index}\nOptions:\nA. 1\nB. 2\n",
        "Answer: Let's think step by step. The answer is (B).\n\n",
    )
    for index in range(5)
)
QUESTION = "Question:\nWhich integer is even?\nOptions:\nA. 1\nB. 2\nAnswer: Let's think step by step."


class CharacterTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return [ord(character) for character in text]


def source_prompt(shots: int) -> str:
    return INSTRUCTION + "".join(question + answer for question, answer in WORKED[:shots]) + QUESTION


def mmlu_pro_dataset():
    return next(source.config for source in sources() if source.config.protocol.benchmark == "MMLUPro")


@pytest.mark.parametrize("as_url", [False, True])
def test_publication_compiles_mmlu_pro_variants_from_staged_fewshots(tmp_path, as_url):
    dataset = mmlu_pro_dataset()
    source_root = tmp_path / "source"
    fewshot_root = tmp_path / "fewshot"
    primary = source_root / dataset.recipe.files[0]
    primary.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([DOCUMENT]), primary)
    pool = fewshot_root / dataset.recipe.inputs["fewshot"].files[0]
    pool.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(SUPPORT), pool)
    inputs = {"fewshot": fewshot_root.as_uri() + "/" if as_url else str(fewshot_root)}
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
    )
    recipe = source_recipe(dataset.recipe, inputs, environment)
    protocol = dataset.protocol.model_copy(update={"expected_population": 1})
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            manifest = publish_evaluation_corpus(
                recipe, protocol, dataset.item_builder, context, str(source_root), str(output)
            )
    finally:
        client.shutdown()
    [case] = list(evaluation_cases(str(output)))
    assert manifest.rows == 1
    assert case.item.source_ordinal == 0
    assert case.item.groups == {"category": "math"}
    operation = case.item.operation
    assert operation.prompt_selection == "fit_context"
    assert operation.response_selection == "final_or_complete_reasoning"
    assert [variant.fewshot for variant in operation.variants] == list(range(6))
    for shots, variant in enumerate(operation.variants):
        events = variant.context.events
        assert [event.role for event in events] == ["user", "assistant"] * shots + ["user"]
        assert [event.content for event in events[1::2]] == [answer for _, answer in WORKED[:shots]]
        assert "".join(event.content for event in events) == source_prompt(shots)
    assert case.task.context == operation.variants[-1].context


@pytest.mark.parametrize("shots", range(6))
def test_mmlu_pro_context_fitting_keeps_the_most_shots_whose_source_prompt_fits(tmp_path, shots):
    dataset = mmlu_pro_dataset()
    pool = tmp_path / dataset.recipe.inputs["fewshot"].files[0]
    pool.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(SUPPORT), pool)
    task = converted_task(dataset.recipe, DOCUMENT, inputs={"fewshot": StoragePath(str(tmp_path))})
    case = EvaluationCase(task, dataset.item_builder(task))
    max_tokens = 100
    # `prompt_variant` admits prompts shorter than the window minus `max_tokens`, so the `shots`-example prompt fits
    # and the next does not.
    selected = prompt_variant(case, len(source_prompt(shots)) + 1 + max_tokens, max_tokens, CharacterTokenizer())
    assert selected.context == case.item.operation.variants[shots].context
