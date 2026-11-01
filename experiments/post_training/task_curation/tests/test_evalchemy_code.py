# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish code rows while keeping executable tests in verifier-only resources."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from shellbox.machine import PackageSpec
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import evaluation_cases, publish_evaluation_corpus
from taskcompendium.runtime.resources import resource_bytes
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.code import sources
from experiments.post_training.task_curation.pipeline import source_recipe

INPUTS = [
    {
        "suite": "humaneval",
        "rows": [
            {
                "task_id": "HumanEval/private",
                "prompt": "def identity(value):\n    ",
                "canonical_solution": "return value\n",
                "test": "def check(candidate):\n    assert candidate(7) == 7\n",
                "entry_point": "identity",
            }
        ],
    },
    {
        "suite": "cruxeval",
        "rows": [{"id": "identity", "code": "def f(x): return x", "input": "7", "output": "7"}],
    },
]


@pytest.mark.parametrize("case", INPUTS, ids=lambda case: case["suite"])
def test_python_publication_keeps_tests_private_and_expands_crux_directions(case, tmp_path):
    source = next(source for source in sources() if source.info.id == "Evalchemy:" + case["suite"])
    dataset = source.config
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
    protocol = dataset.protocol.model_copy(
        update={"expected_population": len(case["rows"]) * (2 if case["suite"] == "cruxeval" else 1)}
    )
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
    assert len(published) == protocol.expected_population
    if case["suite"] == "cruxeval":
        assert [item.item.groups["direction"] for item in published] == ["input", "output"]
        assert len({item.task.id for item in published}) == 2
    for item in published:
        assert not item.task.resources.worker
        config = next(resource for resource in item.task.resources.verifier if resource.path == "config.json")
        private = json.loads(resource_bytes(config))
        assert private["problem"]["code" if case["suite"] == "cruxeval" else "test"]
        if case["suite"] == "humaneval":
            assert private["problem"]["test"] not in item.task.context.events[0].content
