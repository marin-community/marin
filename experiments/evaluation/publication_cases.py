# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish an evaluation source from local rows and return its corpus cases, for tests."""

from dataclasses import replace

from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.lazy import run
from taskcompendium.evaluation.corpus import evaluation_cases
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from zephyr.context import ZephyrContext

from experiments.evaluation.preparation import prepare_evaluation
from experiments.post_training.task_curation.campaign import CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions


def published_cases(source, write_rows, tmp_path, monkeypatch, population):
    dataset = source.config
    root = tmp_path / "source"
    file = root / dataset.files[0]
    file.parent.mkdir(parents=True)
    write_rows(file)
    source = replace(
        source,
        config=replace(dataset, protocol=dataset.protocol),
    )
    runtime = CampaignRuntime()
    options = PipelineOptions(SourceProcessingMode.QUICK, runtime, inputs=InputOverrides(root=str(root)))
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path / "artifacts"))
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=1, chunk_storage_prefix=str(tmp_path / "chunks")) as context,
            runtime.activate(context),
        ):
            artifact = run(prepare_evaluation(source, options), max_concurrent=1)[0]
        return list(evaluation_cases(artifact.path))
    finally:
        client.shutdown()
