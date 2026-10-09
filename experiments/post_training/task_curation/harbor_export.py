# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind normalized curation artifacts to the packed Harbor consumer contract."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation import harbor, harbor_export_contract
from experiments.post_training.task_curation.environment import PINNED_IMAGE


class HarborExportArtifact(Artifact):
    verify_tool_ref: str
    exported_rows: int
    rejected_rows: int


@dataclass(frozen=True)
class HarborExportConfig:
    input_root: str
    output_root: str
    grader_image: str | None
    recipe: dict[str, str]


def run_harbor_export(config: HarborExportConfig) -> HarborExportArtifact:
    manifest = harbor.export_harbor(
        StoragePath(config.input_root), StoragePath(config.output_root), grader_image=config.grader_image
    )
    return HarborExportArtifact(
        path=config.output_root,
        verify_tool_ref=manifest["verify_tool_ref"],
        exported_rows=manifest["exported_rows"],
        rejected_rows=manifest["rejected_rows"],
    )


def harbor_export_step(
    normalized: ArtifactStep,
    *,
    name: str,
    version: str,
    grader_image: str | None,
) -> ArtifactStep[HarborExportArtifact]:
    """Export one normalized source; artifact identity includes the bundled verifier implementation."""
    if grader_image is not None and PINNED_IMAGE.fullmatch(grader_image) is None:
        raise ValueError("The verifier image must be explicitly pinned by digest")
    recipe = {__name__: hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for module in (harbor, harbor_export_contract):
        source_file = module.__file__
        assert source_file is not None
        recipe[module.__name__] = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    recipe.update({name: hashlib.sha256(content).hexdigest() for name, content in harbor.verifier_runtime().items()})

    def build_config(ctx: StepContext) -> HarborExportConfig:
        return HarborExportConfig(ctx.artifact_path(normalized), ctx.output_path, grader_image, recipe)

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=HarborExportArtifact,
        run=run_harbor_export,
        build_config=build_config,
        deps=(normalized,),
    )
