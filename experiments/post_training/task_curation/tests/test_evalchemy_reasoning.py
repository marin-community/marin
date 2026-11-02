# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MMLU-Pro publication uses the staged category-specific fewshot split."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.local_backend import LocalClient
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.datasets.evalchemy.reasoning import sources
from experiments.post_training.task_curation.pipeline import source_recipe


@pytest.mark.parametrize("as_url", [False, True])
def test_publication_compiles_mmlu_pro_variants_from_staged_fewshots(tmp_path, as_url):
    dataset = next(source.config for source in sources() if source.config.protocol.benchmark == "MMLUPro")
    document = {
        "question": "Which integer is even?",
        "options": ["1", "2", "N/A"],
        "answer": "B",
        "category": "math",
        "cot_content": "A: Let's think step by step. The answer is (B).",
    }
    source_root = tmp_path / "source"
    fewshot_root = tmp_path / "fewshot"
    primary = source_root / dataset.recipe.files[0]
    primary.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([document]), primary)
    pool = fewshot_root / dataset.recipe.inputs["fewshot"].files[0]
    pool.parent.mkdir(parents=True)
    support = [
        {**document, "question": "OTHER_CATEGORY_SENTINEL", "category": "physics"},
        *[{**document, "question": f"STAGED_FEWSHOT_{index}"} for index in range(6)],
    ]
    pq.write_table(pa.Table.from_pylist(support), pool)
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
        [event] = variant.context.events
        prompt = event.content
        assert prompt.count("STAGED_FEWSHOT_") == shots
        for index in range(shots):
            assert f"STAGED_FEWSHOT_{index}" in prompt
        assert "OTHER_CATEGORY_SENTINEL" not in prompt
        assert "STAGED_FEWSHOT_5" not in prompt
        assert "N/A" not in prompt
        assert prompt.endswith(
            "Question:\nWhich integer is even?\nOptions:\nA. 1\nB. 2\nAnswer: Let's think step by step."
        )
    assert case.task.context == operation.variants[-1].context
