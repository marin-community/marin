# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Entry point executed inside the pinned Atlas tree, never imported by main."""

import json
import sys
from dataclasses import replace
from pathlib import Path

from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from marin.execution.step_runner import StepRunner
from taskcompendium.pipeline.source_processing import SourceProcessingMode
from zephyr.context import ZephyrContext

from experiments.post_training.task_curation.campaign import CampaignArtifact, CampaignRuntime
from experiments.post_training.task_curation.config import InputOverrides, PipelineOptions
from experiments.post_training.task_curation.frozen_atlas_protocol import FrozenAtlasRequest
from experiments.post_training.task_curation.sources import all_sources


def main() -> None:
    request = FrozenAtlasRequest(**json.loads(Path(sys.argv[1]).read_text()))
    source = all_sources()[request.source_id]
    dataset = source.dataset
    assert dataset is not None and source.pipeline is not None
    expected = (request.name, request.version, request.repo, request.revision, request.files)
    actual = (source.name, source.version, dataset.name, dataset.revision, list(source.files))
    if actual != expected:
        raise ValueError(f"Frozen converter differs from the Atlas pin: {source.info.id}")
    source.require_license()
    runtime = CampaignRuntime()
    options = PipelineOptions(
        mode=SourceProcessingMode(request.mode),
        runtime=runtime,
        declared_task_format=source.info.task_format,
        inputs=InputOverrides(
            root=request.input_root,
            files={name: Path(path) for name, path in request.input_files.items()},
            auxiliary=request.auxiliary,
        ),
    )
    client = LocalClient()
    try:
        with (
            set_current_client(client),
            ZephyrContext(client=client, max_workers=2) as context,
            runtime.activate(context),
        ):
            step = source.pipeline(source, options)
            StepRunner().run([replace(step.lower(), override_output_path=request.output_path)], max_concurrent=1)
    finally:
        client.shutdown()
    artifact = CampaignArtifact.raw_load(request.output_path)
    Path(sys.argv[2]).write_text(artifact.model_dump_json())


if __name__ == "__main__":
    main()
