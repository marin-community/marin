# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run Atlas-pinned evaluation converters in their original dependency tree."""

import io
import json
import os
import subprocess
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from finestore.rl.pinned_file import PinnedHfFile, read_pinned_hf_file
from marin.execution.lazy import ArtifactStep
from taskcompendium.pipeline.models import IntendedUse

from experiments.post_training.task_curation.campaign import CampaignArtifact
from experiments.post_training.task_curation.config import PipelineOptions
from experiments.post_training.task_curation.frozen_atlas_protocol import FrozenAtlasRequest
from experiments.post_training.task_curation.source import RlDataSource, SourceReference

FROZEN_IMPORT_VERSION = "2026.11.25.1"


@dataclass(frozen=True)
class FrozenAtlasConversion:
    """The exact conversion identity and the source tree that implements it."""

    source_id: str
    name: str
    version: str
    dataset: SourceReference
    files: tuple[str, ...]
    archive: PinnedHfFile
    intended_use: IntendedUse


@dataclass(frozen=True)
class FrozenAtlasRun:
    conversion: FrozenAtlasConversion
    output_path: str
    mode: str
    input_root: str | None
    input_files: dict[str, str]
    auxiliary: dict[str, str]


def run_frozen_atlas(invocation: FrozenAtlasRun) -> CampaignArtifact:
    """Execute the original converter and reject any difference in its declared pin."""
    config = invocation.conversion
    request = FrozenAtlasRequest(
        config.source_id,
        config.name,
        config.version,
        config.dataset.name,
        config.dataset.revision,
        list(config.files),
        invocation.output_path,
        invocation.mode,
        invocation.input_root,
        invocation.input_files,
        invocation.auxiliary,
    )
    with TemporaryDirectory(prefix="atlas-converter-") as temporary:
        root = Path(temporary)
        checkout = root / "checkout"
        checkout.mkdir()
        with tarfile.open(fileobj=io.BytesIO(read_pinned_hf_file(config.archive)), mode="r:gz") as archive:
            archive.extractall(checkout, filter="data")
        request_path, result_path = root / "request.json", root / "result.json"
        request_path.write_text(json.dumps(asdict(request)))
        protocol = Path(__file__).with_name("frozen_atlas_protocol.py")
        (checkout / "experiments/post_training/task_curation" / protocol.name).write_bytes(protocol.read_bytes())
        worker = Path(__file__).with_name("frozen_atlas_worker.py").read_text()
        environment = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "VIRTUAL_ENV"}}
        environment["MARIN_PREFIX"] = str(root / "artifacts")
        subprocess.run(
            ["uv", "run", "--isolated", "--frozen", "python", "-c", worker, str(request_path), str(result_path)],
            cwd=checkout,
            env=environment,
            check=True,
        )
        return CampaignArtifact.model_validate_json(result_path.read_text())


def import_frozen_atlas(
    source: RlDataSource[FrozenAtlasConversion], options: PipelineOptions
) -> ArtifactStep[CampaignArtifact]:
    """Keep the fingerprinted converter and its dependency models isolated from main."""
    source.require_license()
    config = source.config
    assert config is not None

    def build_config(context) -> FrozenAtlasRun:
        return FrozenAtlasRun(
            config,
            context.output_path,
            options.mode.value,
            options.inputs.root,
            {name: str(path) for name, path in options.inputs.files.items()},
            dict(options.inputs.auxiliary),
        )

    return ArtifactStep(
        name=f"atlas-frozen/{config.name}/{config.archive.revision}",
        version=FROZEN_IMPORT_VERSION,
        artifact_type=CampaignArtifact,
        run=run_frozen_atlas,
        build_config=build_config,
        deps=(),
    )
