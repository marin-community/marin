# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build download artifacts from recipe-owned pinned input declarations."""

import hashlib

from fray.types import ResourceConfig
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.execution.remote import remote
from marin.experiment.data import hf_download, raw_download
from taskcompendium.pipeline.inputs import HubDownload
from taskcompendium.pipeline.models import DatasetRecipe

from experiments.post_training.task_curation.staging import DownloadInputs, download_inputs

DOWNLOAD_VERSION = "2026.10.02.1"


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

    return raw_download(
        name,
        fn=remote(download_inputs, resources=resources, pip_packages=["./lib/taskcompendium[pipeline]"]),
        build_config=config,
        version=DOWNLOAD_VERSION,
    )
