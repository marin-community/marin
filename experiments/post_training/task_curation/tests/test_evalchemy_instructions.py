# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare complete local instruction release graphs with the frozen original scorers."""

import json
import os
import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from shellbox.machine import CommandSemantics, PackageSpec
from taskcompendium.evaluation import TextGeneration
from taskcompendium.models import EnvironmentRequirements, verifyit_spec
from taskcompendium.pipeline.evaluation import evaluation_cases, publish_evaluation_corpus
from taskcompendium.pipeline.inputs import SourceFormat
from taskcompendium.submission import conversation_messages
from verifyit.spec import spec_to_table
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.instructions import sources
from experiments.post_training.task_curation.pipeline import source_recipe

ORACLE = Path(__file__).parent / "fixtures/evalchemy_instructions_oracle.json"


@pytest.fixture(params=["ifeval", "IFBench"])
def instruction_corpus(request, tmp_path):
    reference = next(case for case in json.loads(ORACLE.read_text())["cases"] if case["name"] == request.param)
    source = next(source for source in sources() if source.config.protocol.benchmark == request.param)
    dataset = source.config
    root = tmp_path / "source"
    file = root / dataset.files[0]
    file.parent.mkdir(parents=True)
    if dataset.recipe.source.format is SourceFormat.PARQUET:
        pq.write_table(pa.Table.from_pylist(reference["rows"]), file)
    else:
        file.write_text("".join(json.dumps(row) + "\n" for row in reference["rows"]))
    source = replace(
        source,
        config=replace(
            dataset,
            protocol=dataset.protocol.model_copy(
                update={
                    "expected_population": len(reference["rows"]),
                }
            ),
        ),
    )
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        package_spec=PackageSpec(pypi_packages=dataset.recipe.grader.requirements),
    )
    output = str(tmp_path / "published")
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
        ):
            publish_evaluation_corpus(
                source_recipe(dataset.recipe, {}, environment),
                source.config.protocol,
                dataset.item_builder,
                context,
                str(root),
                output,
            )
        yield reference, list(evaluation_cases(output))
    finally:
        client.shutdown()


def test_source_instruction_prompts_metrics_and_sufficient_statistics(instruction_corpus, tmp_path):
    reference, cases = instruction_corpus
    assert len(cases) == len(reference["rows"])
    for case, original in zip(cases, reference["specs"], strict=True):
        if isinstance(case.item.operation, TextGeneration):
            assert case.item.operation.context == original["requests"][0]["prompt"]
        else:
            assert conversation_messages(case.task.context.events) == original["requests"][0]["prompt"]
    nltk_root = tmp_path / "nltk_data"
    with zipfile.ZipFile(ORACLE.parent / "nltk_english.zip") as archive:
        archive.extractall(nltk_root)
    jobs = [
        {"spec": spec_to_table(verifyit_spec(cases[recorded["ordinal"]].task.grader)), "response": recorded["response"]}
        for recorded in reference["grades"]
    ]
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import dataclasses,json,sys\n"
                "from verifyit.candidate import grade_candidate\n"
                "from verifyit.spec import spec_from_table\n"
                "jobs=json.load(sys.stdin)\n"
                "json.dump([dataclasses.asdict(grade_candidate(spec_from_table(job['spec']),job['response'],{})) "
                "for job in jobs],sys.stdout)\n"
            ),
        ],
        input=json.dumps(jobs),
        text=True,
        capture_output=True,
        check=True,
        env={**os.environ, "NLTK_DATA": str(nltk_root)},
    )
    for recorded, result in zip(reference["grades"], json.loads(process.stdout), strict=True):
        assert result["detail"]["metrics"] == recorded["grade"]["metrics"]
        if reference["name"] == "ifeval":
            assert result["detail"]["reducer_inputs"] == recorded["grade"]["reducer_inputs"]
        else:
            assert result["detail"]["metric_weights"] == recorded["grade"]["weights"]
            assert result["detail"]["instruction_pass"] == {
                mode: recorded["grade"]["audit"][mode + "_instruction_pass"] for mode in ("strict", "loose")
            }
