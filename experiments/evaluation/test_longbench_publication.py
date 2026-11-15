# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""LongBench v2 publication compiles middle-truncated prompt variants from the pinned rows."""

import json

from fray.local_backend import LocalClient
from taskcompendium.convert.answers import source_evidence
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.importers.evalchemy.long_context import official_tokenizer
from taskcompendium.importers.longbench_v2 import LONGBENCH_V2_TOKEN_BUDGETS
from taskcompendium.models import CommandSemantics, EnvironmentRequirements
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from zephyr.context import ZephyrContext

from experiments.evaluation.benchmarks.longbench import benchmarks
from experiments.post_training.task_curation.pipeline import source_recipe


def row(identifier: str, length: str, context: str) -> dict:
    return {
        "_id": identifier,
        "domain": "Single-Document QA",
        "sub_domain": "Financial",
        "difficulty": "hard",
        "length": length,
        "question": f"Question {identifier}?",
        "choice_A": "alpha",
        "choice_B": "beta",
        "choice_C": "gamma",
        "choice_D": "delta",
        "answer": "B",
        "context": context,
    }


def test_publication_compiles_middle_truncated_variants_smallest_first(tmp_path):
    [source] = benchmarks()[0].resolve("no-thinking")
    dataset = source.config
    long_context = "BEGIN " + "filler " * 20_000 + "END"
    rows = [row("long", "long", long_context), row("short", "short", "A short context.")]
    source_root = tmp_path / "source"
    source_root.mkdir()
    (source_root / dataset.recipe.files[0]).write_text(json.dumps(rows))
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
    )
    recipe = source_recipe(dataset.recipe, {}, environment)
    protocol = dataset.protocol.model_copy(update={"expected_population": 2})
    output = tmp_path / "output"
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(source_root), str(output))
    finally:
        client.shutdown()
    cases = {case.task.source.row.rsplit(":", 1)[1]: case for case in evaluation_cases(str(output))}
    long_case, short_case = cases["0"], cases["1"]

    tokenizer = official_tokenizer()
    [full] = long_case.task.context.events
    full_tokens = len(tokenizer.encode(full.content))
    budgets = [budget for budget in LONGBENCH_V2_TOKEN_BUDGETS if budget < full_tokens]
    variants = long_case.item.operation.variants
    assert [variant.fewshot for variant in variants] == list(range(len(budgets) + 1))
    assert variants[-1].context == long_case.task.context
    for budget, variant in zip(budgets, variants, strict=False):
        [event] = variant.context.events
        assert abs(len(tokenizer.encode(event.content)) - budget) <= 2
        assert event.content.startswith("Please read the following text")
        assert "BEGIN" in event.content and "END" in event.content
        assert event.content.endswith('Format your response as follows: "The correct answer is (insert answer here)".')
    assert long_case.item.groups == {"length": "long", "difficulty": "hard", "domain": "single_document_qa"}
    assert "context" not in source_evidence(long_case.task)

    [only] = short_case.item.operation.variants
    assert only.fewshot == 0 and only.context == short_case.task.context
    assert "<text>\nA short context.\n</text>" in short_case.task.context.events[0].content
