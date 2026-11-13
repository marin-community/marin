# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exercise staged few-shot inputs through native publication and duplicate source identities."""

import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.curation import CurationCatalog, WithinSourceDedupEvidence
from fray.local_backend import LocalClient
from marin.evaluation.records import PromptFormat
from shellbox.machine import PackageSpec
from taskcompendium.evaluation.corpus import evaluation_cases, evaluation_protocol
from taskcompendium.evaluation.selection import select_cases
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.pipeline.inputs import SourceFiles, SourceFormat
from zephyr.context import ZephyrContext

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.evaluation.curated import load_evaluations
from experiments.post_training.task_curation.campaign import CampaignStatus, SourceOutcome, write_campaign_report
from experiments.post_training.task_curation.catalog_snapshot import publish_catalog
from experiments.post_training.task_curation.pipeline import source_recipe


@pytest.mark.parametrize(
    "benchmark,shots,form,unique",
    [
        ("mmlu", 5, None, 1),
        ("arc-easy", 0, None, 1),
        ("arc-easy", 10, None, 2),
        *(("gsm8k", shots, form, 1 if shots == 0 else 2) for shots in (0, 5) for form in PromptFormat),
    ],
)
def test_publication_compiles_selected_fewshots_and_deduplicates_exact_contracts(
    tmp_path, benchmark, shots, form, unique
):
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
    conversion = json.loads((output / "manifest.json").read_text())
    measurement = conversion["within_source_dedup"]
    assert manifest.rows == len(cases) == unique
    assert measurement["raw_count"] == 2
    assert measurement["within_source_duplicates"] == 2 - unique
    assert cases[0].item.source_ordinal in {0, 1}
    for case in cases:
        operation = case.item.operation
        context = operation.requests[0].context if benchmark != "gsm8k" else case.task.context.events[0].content
        if benchmark == "gsm8k":
            assert operation.kind.value == ("chat_generation" if form is PromptFormat.CHAT else "text_generation")
        assert context.count("AUXILIARY_SENTINEL") == shots
        assert document["question"].strip() in context
    catalog_root = str(tmp_path / "catalog")
    declared = replace(
        source,
        info=replace(
            source.info,
            count=manifest.rows,
            within_source_dedup=WithinSourceDedupEvidence(
                raw_count=measurement["raw_count"],
                unique_count=measurement["unique_count"],
                within_source_duplicates=measurement["within_source_duplicates"],
                source_dataset_id=source.dataset.name,
                source_revision=source.dataset.revision,
                converter_version=source.version,
                converter_revision="fixture-converter",
                drop_manifest_uri=measurement["drop_manifest_uri"],
                evidence_url=str(output / "manifest.json"),
            ),
            daytona_snapshot_count=0,
            daytona_machine_slots=0,
            daytona_execution_failures=0,
            daytona_count_basis="Published fixture tasks contain no actor machine.",
        ),
        config=replace(dataset, protocol=protocol),
    )
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
    assert published.curation.processed_rows == unique
    assert published.curation.raw_population_count == 2
    assert published.declaration.type == "eval"


@pytest.mark.parametrize("benchmark", ["humanevalplus", "mbppplus", "ifbench"])
@pytest.mark.parametrize("run_cap", [4096, 768])
def test_generation_selection_preserves_context_derived_or_explicit_budget(tmp_path, benchmark, run_cap):
    (source,) = evaluation_benchmarks()[benchmark].resolve()
    dataset = source.config
    assert dataset is not None
    row = (
        {"prompt": "Write a short answer.", "instruction_id_list": [], "kwargs": []}
        if benchmark == "ifbench"
        else {"prompt": "def answer():\n    pass", "evaluation_prompt": "Write a function returning 42."}
    )
    inputs = tmp_path / "source"
    inputs.mkdir()
    (inputs / "row.jsonl").write_text(json.dumps(row) + "\n")
    recipe = replace(
        source_recipe(
            dataset.recipe,
            {},
            EnvironmentRequirements(
                command_semantics=CommandSemantics.LINUX_PROCESS,
                package_spec=PackageSpec(pypi_packages=dataset.recipe.grader.requirements),
            ),
        ),
        source=SourceFiles("budget-regression", "1", ("row.jsonl",), SourceFormat.JSONL),
    )
    protocol = dataset.protocol.model_copy(update={"expected_population": 1})
    root = str(tmp_path / "published")
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(inputs), root)
    finally:
        client.shutdown()
    (decision,) = select_cases(
        root,
        protocol,
        dataset_id="budget-regression",
        context_window=16384,
        max_tokens=run_cap,
        limit=None,
        tokenizer=None,
    )
    assert decision.case is not None
    assert decision.record.max_tokens == run_cap
