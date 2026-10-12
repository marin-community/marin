# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Adopt retained full curation outputs without repeating conversion or controls."""

from dataclasses import dataclass, replace

from marin.execution.artifact import Artifact
from marin.execution.lazy import OUT, ArtifactStep, apply
from rigging.filesystem.storage_path import StoragePath
from taskcompendium.pipeline.assessment import read_json, write_json
from taskcompendium.pipeline.models import SourceStatus

from experiments.post_training.task_curation.campaign import CampaignArtifact, PipelineResult
from experiments.post_training.task_curation.source import RlDataSource

REASSESSMENT_VERSION = "2026.10.12.1"


@dataclass(frozen=True)
class CuratedSource:
    source: str
    dataset: str | None
    dataset_revision: str | None
    recipe_revision: str
    dataset_url: str | None = None


def retained_sources(inventory_path: str) -> dict[str, str]:
    inventory = read_json(inventory_path)
    entries = inventory["sources"]
    roots = {entry["source"]: str(StoragePath(entry["manifest"]).parent) for entry in entries}
    if len(roots) != len(entries):
        raise ValueError("Retained inventory contains duplicate source names")
    return roots


def _validated_curated(*, manifest_path: str, expected: CuratedSource, output_root: str) -> CampaignArtifact:
    manifest = read_json(manifest_path)
    actual = CuratedSource(
        manifest["source"],
        manifest["source_dataset"],
        manifest["source_revision"],
        manifest["recipe_revision"],
    )
    # HF manifests name a repository; URL inputs name their download URL.
    if (
        actual.source != expected.source
        or actual.dataset not in (expected.dataset, expected.dataset_url)
        or actual.dataset_revision != expected.dataset_revision
        or actual.recipe_revision != expected.recipe_revision
    ):
        raise ValueError(f"Retained curation pin differs from declaration: {actual} != {expected}")
    if manifest["mode"] != "full":
        raise ValueError("Reassessment requires a full curation output")
    if manifest["processed_rows"] + manifest["unprocessed_rows"] != manifest["raw_population_count"]:
        raise ValueError("Retained curation population accounting is inconsistent")
    views = manifest["datasets"]
    # An inconclusive control result remains evidence; reassessment changes no admission.
    result = PipelineResult(
        SourceStatus(manifest["status"]),
        {name: views[name] for name in ("normalize", "final")},
        {**{name: views[name] for name in ("download", "review", "verify")}, "manifest": manifest_path},
        ("adopt_curated",),
    )
    write_json(str(StoragePath(output_root) / "manifest.json"), manifest)
    return CampaignArtifact(path=output_root, status=result.status, result=result)


def reassessment_input(source: RlDataSource, root: str) -> ArtifactStep[CampaignArtifact]:
    """Keep the original producer's name/version as the adopted upstream key."""
    path = StoragePath(root)
    original = ArtifactStep.adopt(
        f"retained-curation/{path.parent.name}",
        path.name,
        str(path / "manifest.json"),
        kind=Artifact,
        config={"producer": f"data/rl/{path.parent.name}@{path.name}"},
    )
    dataset = source.dataset
    expected = CuratedSource(
        source.name,
        dataset.name if dataset else None,
        dataset.revision if dataset else None,
        source.version,
        dataset.url if dataset else None,
    )
    step = apply(
        f"rl-assessment/{source.name}/curated",
        _validated_curated,
        version=REASSESSMENT_VERSION,
        artifact_type=CampaignArtifact,
        manifest_path=original,
        expected=expected,
        output_root=OUT,
    )
    return replace(step, name=f"{step.name}-{step.fingerprint()[:16]}")
