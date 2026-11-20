# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The catalog's hinted GSM8K publishes one grouped item per problem and level, with its reference."""

import json

import pyarrow as pa
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from taskcompendium.convert.answers import source_evidence
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.importers.evalchemy.hinted import HINT_LEVELS
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.runtime.resources import resource_bytes
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks.hinted import GSM8K_POOL_FILE
from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation.pipeline import source_recipe

TEST_ROWS = [
    {
        "question": "Tom reads 9 pages a day for 2 days. How many pages?",
        "answer": "Tom reads 9 * 2 = 18 pages.\n#### 18",
    },
    {"question": "Sue has 5 pens and loses 1. How many remain?", "answer": "Sue keeps 5 - 1 = 4 pens.\n#### 4"},
]
POOL_ROWS = [{"question": f"Pool question {index}?", "answer": f"Work {index}.\n#### {index}"} for index in range(7)]


def test_publication_groups_each_level_and_keeps_its_reference(tmp_path):
    [source] = evaluation_benchmarks()["gsm8k-hinted-truncation"].resolve()
    dataset = source.config
    assert dataset is not None
    source_root = tmp_path / "source"
    test_file = source_root / dataset.recipe.files[0]
    test_file.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(TEST_ROWS), test_file)
    pool_root = tmp_path / "fewshot"
    (pool_root / GSM8K_POOL_FILE).parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(POOL_ROWS), pool_root / GSM8K_POOL_FILE)
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
    )
    recipe = source_recipe(dataset.recipe, {"fewshot": str(pool_root)}, environment)
    protocol = dataset.protocol.model_copy(update={"expected_population": len(TEST_ROWS) * len(HINT_LEVELS)})
    output = tmp_path / "corpus"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(source_root), str(output))
    finally:
        client.shutdown()

    cases = sorted(evaluation_cases(str(output)), key=lambda case: case.item.source_ordinal)
    assert [case.item.groups for case in cases] == [
        {"hint_method": "truncation", "hint_level": f"{level:.1f}"} for _ in TEST_ROWS for level in HINT_LEVELS
    ]
    full = cases[len(HINT_LEVELS) - 1]
    evidence = source_evidence(full.task)
    assert evidence["reference_id"] == "openai/gsm8k@740312ad:test:0"
    assert evidence["reference"] == "Tom reads 9 * 2 = 18 pages."
    config = json.loads(resource_bytes(next(r for r in full.task.resources.verifier if r.path == "config.json")))
    assert config == {"expected": "18"}
