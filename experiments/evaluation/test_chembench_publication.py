# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ChemBench publication renders both question kinds and groups items by topic file."""

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

EXAMPLE = pa.struct([("input", pa.string()), ("target", pa.string()), ("target_scores", pa.string())])
SCHEMA = pa.schema(
    [
        ("canary", pa.string()),
        ("description", pa.string()),
        ("examples", pa.list_(EXAMPLE)),
        ("in_humansubset_w_tool", pa.bool_()),
        ("in_humansubset_wo_tool", pa.bool_()),
        ("keywords", pa.list_(pa.string())),
        ("metrics", pa.list_(pa.string())),
        ("name", pa.string()),
        ("preferred_score", pa.string()),
        ("uuid", pa.string()),
        ("subfield", pa.string()),
    ]
)


def chembench_row(name: str, example: dict, metrics: list[str]) -> dict:
    return {
        "canary": "canary",
        "description": "synthetic",
        "examples": [example],
        "in_humansubset_w_tool": False,
        "in_humansubset_wo_tool": False,
        "keywords": ["synthetic"],
        "metrics": metrics,
        "name": name,
        "preferred_score": metrics[0],
        "uuid": name,
        "subfield": "synthetic",
    }


def test_publication_renders_both_question_kinds_with_topic_groups(tmp_path):
    [source] = evaluation_benchmarks()["chembench"].resolve()
    dataset = source.config
    multiple_choice = chembench_row(
        "choice",
        {
            "input": "Which is a noble gas?",
            "target": None,
            "target_scores": json.dumps({"He": 1.0, "Na": 0.0, "Ne": 1.0}),
        },
        ["multiple_choice_grade"],
    )
    numeric = chembench_row(
        "number", {"input": "How many protons does \\ce{C} have?", "target": "6", "target_scores": None}, ["mae"]
    )
    source_root = tmp_path / "source"
    files = {"general_chemistry": multiple_choice, "physical_chemistry": numeric}
    for path in dataset.recipe.files:
        topic = path.split("/")[0]
        file = source_root / path
        file.parent.mkdir(parents=True)
        rows = [files[topic]] if topic in files else []
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), file)
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
    )
    recipe = source_recipe(dataset.recipe, {}, environment)
    protocol = dataset.protocol.model_copy(update={"expected_population": 2})
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(source_root), str(output))
    finally:
        client.shutdown()
    cases = sorted(evaluation_cases(str(output)), key=lambda case: case.item.groups["topic"])
    assert [case.item.groups for case in cases] == [{"topic": "general_chemistry"}, {"topic": "physical_chemistry"}]
    assert all(case.item.operation.response_selection == "final_only" for case in cases)
    choice_prompt, number_prompt = (case.task.context.events[0].content for case in cases)
    assert "Options:\nA. He\nB. Na\nC. Ne\n" in choice_prompt
    assert "Question: How many protons does C have?\n" in number_prompt
    configs = [
        json.loads(resource_bytes(next(r for r in case.task.resources.verifier if r.path == "config.json")))
        for case in cases
    ]
    assert configs == [
        {"kind": "multiple_choice", "letters": ["A", "B", "C"], "correct": ["A", "C"]},
        {"kind": "numeric", "target": 6.0},
    ]
