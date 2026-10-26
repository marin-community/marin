# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Load portable curator packets into fully hydrated open-ended task bundles."""

import json
from pathlib import Path
from typing import Any

from taskcompendium.models import Source
from taskcompendium.runtime.resources import inline_resource
from taskforge.spec.draft import requirements

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedTaskBrief, OpenEndedTaskBundle
from experiments.post_training.capability_driven_envs.open_ended_packet_schema import (
    MANIFEST_PATH,
    PacketFileDeclaration,
    PacketManifest,
    PacketVisibility,
    inventory_files,
)


def load_open_ended_packet(root: Path, *, expected_bundle_digest: str | None = None) -> OpenEndedTaskBundle:
    """Verify ``root`` and return its fully hydrated bundle.

    The manifest is the sole mapping from packet-local paths to TaskResource
    paths. Loading performs no source lookup and rejects missing, undeclared,
    modified, or symbolic-link files.

    Args:
        root: Curator packet directory containing ``manifest.json``.
        expected_bundle_digest: Optional canonical bundle digest recorded by a
            campaign manifest. A mismatch rejects the packet.

    Raises:
        ValueError: the packet or expected digest is inconsistent.
    """
    if not root.is_dir():
        raise ValueError(f"packet root is not a directory: {root}")
    manifest_path = root / MANIFEST_PATH
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"packet needs a regular {MANIFEST_PATH}")
    manifest = PacketManifest.model_validate(_read_json(manifest_path))
    declarations = tuple(
        PacketFileDeclaration(
            packet_path=item.packet_path,
            resource_path=item.resource_path,
            visibility=item.visibility,
        )
        for item in manifest.files
    )
    measured = inventory_files(root, declarations)
    expected = {item.packet_path: item for item in manifest.files}
    observed = {item.packet_path: item for item in measured}
    if observed != expected:
        changed = sorted(path for path in expected if expected.get(path) != observed.get(path))
        raise ValueError(f"packet file size or digest differs from manifest: {changed}")

    brief = OpenEndedTaskBrief.model_validate(_read_json(root / manifest.brief_path))
    solver_resources = []
    author_evidence = []
    for item in manifest.files:
        if item.visibility is PacketVisibility.METADATA:
            continue
        assert item.resource_path is not None
        resource = inline_resource(item.resource_path, (root / item.packet_path).read_bytes())
        if item.visibility is PacketVisibility.SOLVER:
            solver_resources.append(resource)
        else:
            author_evidence.append(resource)
    bundle = OpenEndedTaskBundle(
        bundle_id=manifest.bundle_id,
        split_group=manifest.split_group,
        brief=brief,
        solver_resources=tuple(solver_resources),
        author_evidence=tuple(author_evidence),
        source=Source.model_validate(manifest.source.model_dump()),
        environment_requirements=requirements(
            image=None,
            packages=manifest.environment.packages,
            workdir=manifest.environment.workdir,
        ),
        release_status=manifest.release_status,
        tags=manifest.tags,
    )
    if expected_bundle_digest is not None and bundle.content_digest != expected_bundle_digest:
        raise ValueError(
            f"bundle digest differs from expected value: expected {expected_bundle_digest}, got {bundle.content_digest}"
        )
    return bundle


def _read_json(path: Path) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"JSON object in {path} repeats key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(path.read_text(), object_pairs_hook=unique_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read JSON file {path}: {error}") from error
