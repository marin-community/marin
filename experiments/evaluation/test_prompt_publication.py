# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise staged few-shot inputs through native publication and duplicate source identities."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import CurationCatalog
from fray.local_backend import LocalClient
from marin.evaluation.records import PromptFormat
from shellbox.machine import PackageSpec
from taskcompendium.evaluation.corpus import evaluation_cases, evaluation_protocol
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.evaluation.curated import load_evaluations
from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation.campaign import CampaignStatus, SourceOutcome, write_campaign_report
from experiments.post_training.task_curation.catalog_snapshot import publish_catalog
from experiments.post_training.task_curation.pipeline import source_recipe


@pytest.mark.parametrize(
    "benchmark,shots,form",
    [
        ("mmlu", 5, None),
        ("arc-easy", 0, None),
        ("arc-easy", 10, None),
        *(("gsm8k", shots, form) for shots in (0, 5) for form in PromptFormat),
    ],
)
def test_publication_compiles_selected_fewshots_and_retains_duplicate_rows(tmp_path, benchmark, shots, form):
    selection = tmp_path / "eval.yaml"
    format_line = f"    prompt_format: {form.value}\n" if form is not None else ""
    selection.write_text(
        f"evaluations:\n  selected:\n    benchmark: {benchmark}\n    num_fewshot: {shots}\n{format_line}"
    )
    (selected,) = load_evaluations(selection)
    source = evaluation_benchmarks()[selected.benchmark].resolve(selected.protocol, selected.num_fewshot, form)[0]
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
    if benchmark == "gsm8k":
        document = {"question": "Which integer is even?", "answer": "Two is even.\n#### 2"}
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
    grader = (
        EnvironmentRequirements(
            command_semantics=CommandSemantics.LINUX_PROCESS,
            package_spec=PackageSpec(pypi_packages=dataset.recipe.grader.requirements),
        )
        if dataset.recipe.grader is not None
        else None
    )
    recipe = source_recipe(dataset.recipe, {"fewshot": str(fewshot_root)} if shots else {}, grader)
    protocol = dataset.protocol.model_copy(update={"expected_population": 2})
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            manifest = publish_evaluation_corpus(
                recipe,
                protocol,
                dataset.item_builder,
                context,
                str(source_root),
                str(output),
                protocol_name=dataset.protocol_name,
            )
    finally:
        client.shutdown()
    cases = list(evaluation_cases(str(output)))
    published_protocol, _ = evaluation_protocol(str(output))
    assert published_protocol.num_fewshot == shots
    assert manifest.rows == 2
    assert [case.item.source_ordinal for case in cases] == [0, 1]
    assert cases[0].task.id != cases[1].task.id
    for case in cases:
        operation = case.item.operation
        context = operation.requests[0].context if benchmark != "gsm8k" else case.task.context.events[0].content
        if benchmark == "gsm8k":
            assert operation.kind.value == ("chat_generation" if form is PromptFormat.CHAT else "text_generation")
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
