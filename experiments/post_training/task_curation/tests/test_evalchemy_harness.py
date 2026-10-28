# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise staged few-shot inputs through native publication and duplicate source identities."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from taskcompendium.evaluation import ChoiceLikelihood
from taskcompendium.pipeline.evaluation import evaluation_cases, publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.harness import sources
from experiments.post_training.task_curation.pipeline import source_recipe

INPUTS = json.loads((Path(__file__).parent / "fixtures/evalchemy_inputs.json").read_text())["harness"]


def declared_source(name):
    benchmark = "mmlu-abstract_algebra" if name == "mmlu" else name
    return next(source for source in sources() if source.config.protocol.benchmark == benchmark)


@pytest.mark.parametrize("name", ["mmlu", "drop"])
def test_publication_compiles_staged_fewshots_and_retains_duplicate_rows(name, tmp_path):
    reference = next(case for case in INPUTS if case["name"] == name)
    dataset = declared_source(name).config
    source_root = tmp_path / "source"
    fewshot_root = tmp_path / "fewshot"
    output = tmp_path / "output"
    primary = source_root / dataset.recipe.files[0]
    primary.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([reference["document"]] * 2), primary)
    pool = fewshot_root / dataset.recipe.inputs["fewshot"].files[0]
    pool.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [{**row, "question": "AUXILIARY_SENTINEL " + row["question"]} for row in reference["pool"]]
        ),
        pool,
    )
    recipe = source_recipe(dataset.recipe, {"fewshot": str(fewshot_root)}, None)
    protocol = dataset.protocol.model_copy(update={"expected_population": 2})
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            manifest = publish_evaluation_corpus(
                recipe, protocol, dataset.item_builder, context, str(source_root), str(output)
            )
    finally:
        client.shutdown()
    cases = list(evaluation_cases(str(output)))
    assert manifest.rows == 2
    assert [case.item.source_ordinal for case in cases] == [0, 1]
    assert cases[0].task.id != cases[1].task.id
    for case in cases:
        operation = case.item.operation
        context = operation.requests[0].context if isinstance(operation, ChoiceLikelihood) else operation.context
        assert "AUXILIARY_SENTINEL" in context
        assert reference["document"]["question"].strip() in context
