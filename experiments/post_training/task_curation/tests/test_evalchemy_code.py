# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare code preparation with frozen legacy programs without executing candidates."""

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from rigging.filesystem.storage_path import StoragePath
from shellbox.machine import PackageSpec
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import evaluation_cases, publish_evaluation_corpus
from taskcompendium.pipeline.inputs import ConversionContext
from taskcompendium.runtime.resources import resource_bytes
from taskcompendium.submission import conversation_messages
from verifyit.adapters.evalchemy_code import crux_answer, function_prompt, generation_code
from verifyit.adapters.evalchemy_code import test_program as python_test_program
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.code import read_crux, read_mbpp, sources
from experiments.post_training.task_curation.pipeline import source_recipe

ORACLE = json.loads((Path(__file__).parent / "fixtures/evalchemy_code_oracle.json").read_text())


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["suite"])
def test_python_source_preparation_preserves_source_program_bytes(case):
    problem = case["rows"][0]
    suite = case["suite"]
    for example in case["examples"]:
        response = example["response"]
        direction = example.get("direction", "")
        extracted = (
            crux_answer(response, direction) if suite == "cruxeval" else generation_code(problem, response, suite)
        )
        assert extracted == example["extracted"]
        program = python_test_program(problem, response, suite, direction)
        assert hashlib.sha256(program.encode()).hexdigest() == example["program_sha256"]


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["suite"])
def test_python_source_prompts_match_legacy_shots_and_direction_expansion(case, tmp_path):
    context = ConversionContext(inputs={}, grader_environment=None)
    if case["suite"] == "humaneval":
        prompts = [row["prompt"] for row in case["rows"]]
    elif case["suite"] == "humanevalplus":
        prompts = [function_prompt(row) for row in case["rows"]]
    elif case["suite"] == "mbppplus":
        path = tmp_path / "source.parquet"
        pq.write_table(pa.Table.from_pylist(case["rows"]), path)
        prompts = [row["evaluation_prompt"] for row in read_mbpp(StoragePath(str(path)), context)]
    else:
        path = tmp_path / "source.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in case["rows"]))
        prompts = [row["evaluation_prompt"] for row in read_crux(StoragePath(str(path)), context)]
    assert prompts == case["prompts"]


@pytest.mark.parametrize("case", ORACLE["cases"], ids=lambda case: case["suite"])
def test_python_publication_keeps_private_programs_and_exact_requests(case, tmp_path):
    source = next(source for source in sources() if source.info.id == "Evalchemy:" + case["suite"])
    dataset = source.config
    assert dataset is not None
    root = tmp_path / "source"
    path = root / dataset.files[0]
    path.parent.mkdir(parents=True)
    if path.suffix == ".parquet":
        pq.write_table(pa.Table.from_pylist(case["rows"]), path)
    else:
        path.write_text("".join(json.dumps(row) + "\n" for row in case["rows"]))
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        package_spec=PackageSpec(pypi_packages=("numpy==2.3.5",)),
    )
    protocol = dataset.protocol.model_copy(update={"expected_population": len(case["prompts"])})
    output = str(tmp_path / "published")
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(
                client=client,
                max_workers=1,
                chunk_storage_prefix=str(tmp_path / "chunks"),
            ) as context,
        ):
            publish_evaluation_corpus(
                source_recipe(dataset.recipe, {}, environment),
                protocol,
                dataset.item_builder,
                context,
                str(root),
                output,
            )
        published = list(evaluation_cases(output))
    finally:
        client.shutdown()
    assert [conversation_messages(item.task.context.events) for item in published] == [
        [{"role": "user", "content": prompt}] for prompt in case["prompts"]
    ]
    for item in published:
        assert not item.task.resources.worker
        config = next(resource for resource in item.task.resources.verifier if resource.path == "config.json")
        private = json.loads(resource_bytes(config))
        assert private["problem"]["code" if case["suite"] == "cruxeval" else "test"]
