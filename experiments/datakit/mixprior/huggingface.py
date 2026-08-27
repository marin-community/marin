# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize hash-verified campaign records from Hugging Face."""

from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import cast

from huggingface_hub import HfFileSystem
from rigging.filesystem.storage_path import prefix_join

from experiments.datakit.mixprior.campaign import (
    CAMPAIGN_MANIFEST,
    CampaignManifest,
    ContentBasisManifest,
    SwarmManifest,
    SwarmRegistry,
)
from experiments.datakit.mixprior.data import ArtifactReference, read_record, sha256

HF_COMMIT_URI = re.compile(r"^hf://datasets/[^/@]+/[^/@]+@[0-9a-f]{40}/.+$")


def download_campaign(campaign_uri: str, campaign_sha256: str, destination: Path) -> Path:
    """Materialize one commit-pinned campaign from Hugging Face."""
    filesystem = HfFileSystem(token=False)
    if not HF_COMMIT_URI.fullmatch(campaign_uri):
        raise ValueError("Campaign URI must pin a 40-character Hugging Face commit")
    campaign_root_uri = campaign_uri.rsplit("/", 1)[0]
    destination.mkdir(parents=True, exist_ok=False)
    manifest_path = destination / CAMPAIGN_MANIFEST
    _download(filesystem, campaign_uri, manifest_path)
    if sha256(manifest_path) != campaign_sha256:
        raise ValueError("Campaign manifest hash mismatch")

    manifest = cast(CampaignManifest, read_record(manifest_path))
    registry_reference = manifest["registry"]
    registry_path = destination / _safe_relative_path(registry_reference["path"])
    _download_reference(filesystem, campaign_root_uri, destination, registry_reference, registry_path)
    registry = cast(SwarmRegistry, read_record(registry_path))
    swarms_by_id = {reference["swarm_id"]: reference for reference in registry["swarms"]}

    basis_ids = set()
    for swarm_id in [manifest["target_swarm"], *manifest["source_swarms"]]:
        swarm_reference = swarms_by_id[swarm_id]
        swarm_path = destination / _safe_relative_path(swarm_reference["path"])
        _download_reference(filesystem, campaign_root_uri, destination, swarm_reference, swarm_path)
        swarm = cast(SwarmManifest, read_record(swarm_path))
        for reference in (swarm["observations"], swarm["buckets"], swarm["content"]):
            artifact_path = swarm_path.parent / _safe_relative_path(reference["path"])
            _download_reference(filesystem, campaign_root_uri, destination, reference, artifact_path)
        content_path = swarm_path.parent / _safe_relative_path(swarm["content"]["path"])
        basis_ids.add(read_record(content_path)["basis_id"])

    bases_by_id = {reference["basis_id"]: reference for reference in registry["content_bases"]}
    for basis_id in basis_ids:
        basis_reference = bases_by_id[basis_id]
        basis_path = destination / _safe_relative_path(basis_reference["path"])
        _download_reference(filesystem, campaign_root_uri, destination, basis_reference, basis_path)
        basis = cast(ContentBasisManifest, read_record(basis_path))
        lookup_reference = basis["lookup"]
        lookup_path = basis_path.parent / _safe_relative_path(lookup_reference["path"])
        _download_reference(filesystem, campaign_root_uri, destination, lookup_reference, lookup_path)
    return manifest_path


def _download_reference(
    filesystem: HfFileSystem,
    campaign_root_uri: str,
    campaign_directory: Path,
    reference: ArtifactReference,
    destination: Path,
) -> None:
    relative = destination.relative_to(campaign_directory).as_posix()
    _download(filesystem, prefix_join(campaign_root_uri, relative), destination)
    if sha256(destination) != reference["sha256"]:
        raise ValueError(f"Downloaded artifact hash mismatch: {destination}")


def _download(filesystem: HfFileSystem, source_uri: str, destination: Path) -> None:
    if not HF_COMMIT_URI.fullmatch(source_uri):
        raise ValueError("Campaign artifacts must use commit-pinned Hugging Face URIs")
    destination.parent.mkdir(parents=True, exist_ok=True)
    filesystem.get_file(source_uri, destination)


def _safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Campaign artifact path escapes its root: {value}")
    return path
