# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind normalized curation artifacts to the packed Harbor consumer contract."""

import json
from dataclasses import dataclass
from pathlib import Path

import click
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.datasets.environments import VERIFYIT_PACKAGE
from experiments.post_training.task_curation.environment import PINNED_IMAGE
from experiments.post_training.task_curation.images.build import BASE_IMAGE
from experiments.post_training.task_curation.source import RlDataSource
from experiments.post_training.task_curation.sources import all_sources
from experiments.post_training.task_curation.tasktrove import harbor_export as harbor
from experiments.post_training.task_curation.tasktrove.harbor_export import MANIFEST_FILENAME, HarborSourceMetadata


class HarborExportArtifact(Artifact):
    verify_tool_ref: str
    exported_rows: int
    rejected_rows: int


@dataclass(frozen=True)
class HarborExportConfig:
    input_root: str
    output_root: str
    grader_image: str | None
    fallback_actor_image: str
    verifyit_package_root: Path
    source: HarborSourceMetadata


def run_harbor_export(config: HarborExportConfig) -> HarborExportArtifact:
    manifest = harbor.export_harbor(
        StoragePath(config.input_root),
        StoragePath(config.output_root),
        grader_image=config.grader_image,
        source=config.source,
        fallback_actor_image=config.fallback_actor_image,
        verifyit_package_root=config.verifyit_package_root,
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
    source: RlDataSource,
    name: str,
    version: str,
    grader_image: str | None,
    verifyit_package_root: Path = VERIFYIT_PACKAGE,
) -> ArtifactStep[HarborExportArtifact]:
    """Export one normalized source; bump the explicit version when lowering or bundled code changes."""
    if grader_image is not None and PINNED_IMAGE.fullmatch(grader_image) is None:
        raise ValueError("The verifier image must be explicitly pinned by digest")
    metadata = HarborSourceMetadata(source.name, source.info.id, source.info.family)

    def build_config(ctx: StepContext) -> HarborExportConfig:
        return HarborExportConfig(
            ctx.artifact_path(normalized),
            ctx.output_path,
            grader_image,
            BASE_IMAGE,
            verifyit_package_root,
            metadata,
        )

    return ArtifactStep(
        name=name,
        version=version,
        artifact_type=HarborExportArtifact,
        run=run_harbor_export,
        build_config=build_config,
        deps=(normalized,),
    )


@click.command(help=__doc__)
@click.option("--input-root", type=click.Path(exists=True, file_okay=False, path_type=Path), required=True)
@click.option("--output-root", type=click.Path(file_okay=False, path_type=Path), required=True)
@click.option(
    "--verifyit-package-root",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=VERIFYIT_PACKAGE,
    show_default=True,
)
@click.option(
    "--grader-image",
    help="Pinned verifier base for tasks without their own build recipe; Harbor builds private tests on top.",
)
def main(input_root: Path, output_root: Path, grader_image: str | None, verifyit_package_root: Path) -> None:
    manifest = json.loads((input_root / MANIFEST_FILENAME).read_text())
    sources = {source.name: source for source in all_sources().values() if source.pipeline is not None}
    source = sources[manifest["source"]]
    result = harbor.export_harbor(
        StoragePath(str(input_root)),
        StoragePath(str(output_root)),
        grader_image=grader_image,
        source=HarborSourceMetadata(source.name, source.info.id, source.info.family),
        fallback_actor_image=BASE_IMAGE,
        verifyit_package_root=verifyit_package_root,
    )
    click.echo(json.dumps({key: value for key, value in result.items() if key != "rejections"}))


if __name__ == "__main__":
    main()
