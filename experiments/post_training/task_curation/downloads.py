# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build download artifacts from recipe-owned pinned input declarations."""

import hashlib
from dataclasses import dataclass

import requests
from fray.types import ResourceConfig
from marin.datakit.download.huggingface import DownloadConfig, download_hf
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.data import hf_download, raw_download
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.inputs import HubDownload, UrlDownload
from taskcompendium.pipeline.models import DatasetRecipe

DOWNLOAD_VERSION = "2026.10.02.1"


@dataclass(frozen=True)
class DownloadInputs:
    downloads: tuple[HubDownload | UrlDownload, ...]
    output_path: str


def download_inputs(config: DownloadInputs) -> None:
    for declaration in config.downloads:
        if isinstance(declaration, HubDownload):
            destination = StoragePath(config.output_path)
            if declaration.subdirectory:
                destination = destination / declaration.subdirectory
            download_hf(
                DownloadConfig(
                    hf_dataset_id=declaration.dataset,
                    revision=declaration.revision,
                    hf_urls_glob=list(declaration.patterns),
                    gcs_output_path=str(destination),
                    wait_for_completion=True,
                )
            )
        else:
            with requests.get(declaration.url, stream=True, timeout=60) as response:
                response.raise_for_status()
                destination = StoragePath(config.output_path) / declaration.filename
                with destination.open("wb", auto_mkdir=True) as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)


def source_download(recipe: DatasetRecipe, resources: ResourceConfig) -> ArtifactStep[Artifact]:
    """Download complete pinned inputs independently of the audit row limit."""
    declarations = recipe.inputs.downloads
    if not declarations:
        raise ValueError(f"Recipe {recipe.name} requires externally staged inputs")
    selection = hashlib.sha256(repr(declarations).encode()).hexdigest()[:16]
    name = f"task-curation/download/{recipe.source.dataset}/{selection}"
    if len(declarations) == 1 and isinstance(declarations[0], HubDownload) and not declarations[0].subdirectory:
        declaration = declarations[0]
        return hf_download(
            name,
            hf_id=declaration.dataset,
            revision=declaration.revision,
            version=DOWNLOAD_VERSION,
            urls_glob=declaration.patterns,
            resources=resources,
        )

    def config(ctx: StepContext) -> DownloadInputs:
        return DownloadInputs(declarations, ctx.output_path)

    return raw_download(name, fn=download_inputs, build_config=config, version=DOWNLOAD_VERSION, resources=resources)
