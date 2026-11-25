# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish a synthetic GenomeQA question through the evaluation catalog."""

import hashlib
import json
from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
from fray.local_backend import LocalClient
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.importers.genomeqa import FORMAT_BY_SHA256
from taskcompendium.models import CommandSemantics, EnvironmentRequirements, TextMessage
from taskcompendium.pipeline.evaluation import publish_evaluation_corpus
from taskcompendium.runtime.resources import resource_bytes
from zephyr.context import ZephyrContext

from experiments.evaluation.catalog import evaluation_benchmarks
from experiments.post_training.task_curation.pipeline import source_recipe


def test_genomeqa_publication(tmp_path, monkeypatch):
    [declaration] = evaluation_benchmarks()["genomeqa"].resolve()
    dataset = declaration.config
    source = tmp_path / "source"
    binary = dataset.recipe.files[0]
    parquet = source / binary
    parquet.parent.mkdir(parents=True)
    row = {"id": 1, "subtask": "6_motif", "question": "AcgT?", "options": ["alpha", "beta"], "answer_index": 1}
    pq.write_table(pa.Table.from_pylist([row]), parquet)
    (source / "run.py").write_text('raise RuntimeError("must not execute")\nSIMPLE_SYSTEM_PROMPT = "Fixture prompt"\n')
    environment = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS, docker_image="fixture@sha256:" + "0" * 64
    )
    recipe = source_recipe(dataset.recipe, {"prompt": str(source)}, environment)
    monkeypatch.setitem(FORMAT_BY_SHA256, hashlib.sha256(parquet.read_bytes()).hexdigest(), "binary")
    recipe = replace(recipe, source=replace(recipe.source, patterns=(binary,)))
    protocol = dataset.protocol
    output = str(tmp_path / "published")
    client = LocalClient()
    try:
        with ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context:
            publish_evaluation_corpus(recipe, protocol, dataset.item_builder, context, str(source), output)
    finally:
        client.shutdown()
    [case] = evaluation_cases(output)
    assert case.task.context.events == (
        TextMessage(role="system", content="Fixture prompt"),
        TextMessage(
            role="user", content="Question:\nAcgT?\n\nOptions:\nA. alpha\nB. beta\n\nPlease select the correct option."
        ),
    )
    assert case.item.groups == {"format": "binary", "subtask": "6_motif", "cell": "6_motif/binary"}
    assert not case.task.resources.worker
    config = next(resource for resource in case.task.resources.verifier if resource.path == "config.json")
    assert json.loads(resource_bytes(config)) == {"expected": "B", "options": "AB", "cell_weight": 5}
