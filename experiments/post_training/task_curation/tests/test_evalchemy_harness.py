# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise staged few-shot inputs through native publication and duplicate source identities."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import CurationCatalog
from fray.local_backend import LocalClient
from marin.evaluation.model_config import ModelConfig
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.evaluation.curated import CuratedDefinition
from experiments.evaluation.evals import EVALS
from experiments.post_training.task_curation.campaign import CampaignStatus, SourceOutcome, write_campaign_report
from experiments.post_training.task_curation.catalog_snapshot import publish_catalog
from experiments.post_training.task_curation.pipeline import source_recipe
from experiments.post_training.task_curation.sources import evaluation_sources


@pytest.mark.parametrize("benchmark,shots", [("mmlu", 5), ("arc-easy", 0), ("arc-easy", 10)])
def test_publication_compiles_selected_fewshots_and_retains_duplicate_rows(tmp_path, benchmark, shots):
    base = EVALS[benchmark]
    assert isinstance(base, CuratedDefinition)
    selection = tmp_path / "eval.yaml"
    selection.write_text(f"name: selected\nsources: [{base.sources[0]}]\nnum_fewshot: {shots}\n")
    executor = CuratedDefinition.from_file(selection).executor_for(
        ModelConfig(name="test", location="local-model", tokenizer_revision="pinned"), None, None
    )
    source = evaluation_sources()[executor.datasets[0]]
    dataset = source.config
    assert dataset is not None
    document = {
        "question": "Which integer is even?",
        "subject": "abstract_algebra",
        "choices": (
            ["1", "2", "3", "5"]
            if benchmark == "mmlu"
            else {"text": ["1", "2", "3", "5"], "label": ["A", "B", "C", "D"]}
        ),
        "answer": 1,
        "answerKey": "B",
    }
    source_root = tmp_path / "source"
    fewshot_root = tmp_path / "fewshot"
    output = tmp_path / "output"
    primary = source_root / dataset.recipe.files[0]
    primary.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([document] * 2), primary)
    if shots:
        pool = fewshot_root / dataset.recipe.inputs["fewshot"].files[0]
        pool.parent.mkdir(parents=True)
        pq.write_table(
            pa.Table.from_pylist([{**document, "question": f"AUXILIARY_SENTINEL {i}"} for i in range(shots + 1)]),
            pool,
        )
    recipe = source_recipe(dataset.recipe, {"fewshot": str(fewshot_root)} if shots else {}, None)
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
        context = operation.requests[0].context
        assert context.count("AUXILIARY_SENTINEL") == shots
        assert document["question"].strip() in context
    catalog_root = str(tmp_path / "catalog")
    declared = replace(source, info=replace(source.info, count=2), config=replace(dataset, protocol=protocol))
    write_campaign_report(
        str(tmp_path / "campaign.json"),
        CampaignStatus.COMPLETED,
        mode="quick",
        outcomes=[SourceOutcome(declared.name, str(output), "completed", pipeline_key="fixture")],
        run_id="published-eval",
        started_at="2026-11-05T00:00:00+00:00",
        source_ids={declared.name: declared.info.id},
        execution={},
        catalog_root=catalog_root,
    )
    snapshot = publish_catalog(catalog_root, "published-eval", [declared])
    (published,) = CurationCatalog(catalog_root).sources(snapshot.snapshot_id)
    assert published.curation is not None
    assert published.curation.status == "completed"
    assert published.curation.processed_rows == 2
    assert published.curation.raw_population_count == 2
    assert published.declaration.type == "eval"
