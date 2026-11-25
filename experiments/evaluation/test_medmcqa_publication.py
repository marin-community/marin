# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MedMCQA CoT publication renders lm-eval's prompt and binds the reference MCQ grader and subject groups."""

import pyarrow as pa
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, verifyit_spec
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from verifyit.spec import McqExtraction, McqSpec
from zephyr.context import ZephyrContext

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation.pipeline import source_recipe

# Rendered by lm-eval utils_medmcqa.doc_to_text_cot at d6de81643928d653435c431bae19945d41d32520.
SCURVY_PROMPT = (
    "Question: Which vitamin deficiency causes scurvy?\nChoices:\nA. Vitamin A\nB. Vitamin C\nC. Vitamin D\n"
    "D. Vitamin K\nFirst, think step-by-step. End your response with exactly this format:\nFinal Answer: X\n"
    "(where X is A, B, C, or D)."
)


def row(question: str, options: tuple[str, str, str, str], cop: int, subject: str) -> dict:
    return {
        "id": question,
        "question": question,
        "opa": options[0],
        "opb": options[1],
        "opc": options[2],
        "opd": options[3],
        "cop": cop,
        "choice_type": "single",
        "exp": None,
        "subject_name": subject,
        "topic_name": None,
    }


def test_publication_renders_lm_eval_cot_prompt_with_subject_groups(tmp_path):
    [source] = evaluation_benchmarks()["medmcqa-cot"].resolve()
    dataset = source.config
    rows = [
        row(
            "Which vitamin deficiency causes scurvy?",
            ("Vitamin A", "Vitamin C", "Vitamin D", "Vitamin K"),
            1,
            "Medicine",
        ),
        row("Which bone is in the forearm?", ("Radius", "Femur", "Tibia", "Fibula"), 0, "Anatomy"),
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
    assert [case.item.groups for case in cases] == [{"subject": "Medicine"}, {"subject": "Anatomy"}]
    assert {case.item.operation.response_selection for case in cases} == {"final_only"}
    [event] = cases[0].task.context.events
    assert (event.role, event.content) == ("user", SCURVY_PROMPT)
    reference = McqExtraction.MEDMCQA_COT_REFERENCE
    assert [verifyit_spec(case.task.grader) for case in cases] == [
        McqSpec("B", options=4, extraction=reference),
        McqSpec("A", options=4, extraction=reference),
    ]
