# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare compiled native requests with frozen requests from the pinned source harness."""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from taskcompendium.evaluation import ChoiceLikelihood, likelihood_metrics
from taskcompendium.importers.evaluation_harness import (
    HarnessTemplate,
    compile_choice_documents,
    compile_continuation_document,
)
from taskcompendium.pipeline.evaluation import evaluation_cases, publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.harness import sources
from experiments.post_training.task_curation.pipeline import source_recipe

ORACLE = json.loads((Path(__file__).parent / "fixtures/evalchemy_harness_oracle.json").read_text())
GENERATION_ORACLE = json.loads((Path(__file__).parent / "fixtures/evalchemy_harness_generation_oracle.json").read_text())


def declared_source(name):
    benchmark = "mmlu-abstract_algebra" if name == "mmlu" else name
    return next(source for source in sources() if source.config.protocol.benchmark == benchmark)


@pytest.mark.parametrize("reference", ORACLE["cases"], ids=lambda case: case["name"])
def test_compiled_requests_and_recorded_likelihoods_match_source(reference):
    name = reference["name"]
    if name == "lambada_openai":
        operation = compile_continuation_document(reference["document"])
        expected = reference["specs"][0]["requests"][0]
        assert operation.request.context == expected["prompt"]
        assert operation.request.continuation == expected["continuation"]
        protocol = declared_source(name).config.protocol
        for grade in reference["grades"]:
            assert (
                likelihood_metrics(
                    operation,
                    tuple(value[0] for value in grade["scores"]),
                    protocol,
                    is_greedy=tuple(value[1] for value in grade["scores"]),
                )
                == grade["reducer_inputs"]
            )
        return
    template_name = "mmlu" if name == "mmlu-smoke" else name
    template = HarnessTemplate(template_name, reference["shots"], "abstract_algebra" if template_name == "mmlu" else "")
    compiled = list(compile_choice_documents([reference["document"]] * 2, reference["pool"], template))
    operations = [ChoiceLikelihood.model_validate(row["operation"]) for row in compiled]
    for operation, spec in zip(operations, reference["specs"], strict=True):
        assert [request.context for request in operation.requests] == [request["prompt"] for request in spec["requests"]]
        assert [request.continuation for request in operation.requests] == [
            request["continuation"] for request in spec["requests"]
        ]
        if name == "truthfulqa_mc2":
            assert operation.gold == tuple(
                index for index, label in enumerate(reference["document"]["mc2_targets"]["labels"]) if label
            )
        elif name == "winogrande":
            assert operation.gold == (int(reference["document"]["answer"]) - 1,)
        else:
            assert operation.gold == (int(spec["target"]),)
    protocol = declared_source(name).config.protocol
    for grade in reference["grades"]:
        assert (
            likelihood_metrics(operations[grade["ordinal"]], tuple(value[0] for value in grade["scores"]), protocol)
            == grade["metrics"]
        )


@pytest.mark.parametrize("name", ["mmlu", "arc_challenge", "drop"])
def test_publication_compiles_staged_fewshots_and_retains_duplicate_rows(name, tmp_path):
    reference = next(case for case in [*ORACLE["cases"], *GENERATION_ORACLE["cases"]] if case["name"] == name)
    dataset = declared_source(name).config
    source_root = tmp_path / "source"
    fewshot_root = tmp_path / "fewshot"
    output = tmp_path / "output"
    primary = source_root / dataset.recipe.files[0]
    primary.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([reference["document"]] * 2), primary)
    pool = fewshot_root / dataset.recipe.inputs["fewshot"].files[0]
    pool.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(reference["pool"][:5] if name == "mmlu" else reference["pool"]), pool)
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
    for case, spec in zip(cases, reference["specs"], strict=True):
        operation = case.item.operation
        context = operation.requests[0].context if isinstance(operation, ChoiceLikelihood) else operation.context
        assert context == spec["requests"][0]["prompt"]
        assert "eval" in case.task.tags
        assert case.item.groups == ({"subject": "abstract_algebra"} if name == "mmlu" else {})
