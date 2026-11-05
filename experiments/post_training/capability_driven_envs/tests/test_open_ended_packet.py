# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from taskcompendium.runtime.resources import resource_bytes

from experiments.post_training.capability_driven_envs.open_ended_packet import load_open_ended_packet
from experiments.post_training.capability_driven_envs.open_ended_packet_schema import (
    OpenEndedReleaseStatus,
    PacketEnvironment,
    PacketFileDeclaration,
    PacketManifest,
    PacketSource,
    PacketVisibility,
    inventory_files,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def packet(root: Path) -> None:
    write_json(
        root / "brief.json",
        {
            "goal": "Assess whether receptor usage supports a cell annotation.",
            "expected_deliverable": "A donor-aware comparison with a calibrated interpretation.",
            "question_family": "annotation_validation",
            "avoid": ["Reveal the expected answer."],
        },
    )
    (root / "solver").mkdir()
    (root / "solver/counts.csv").write_text("group,success,total\nA,8,10\nB,2,10\n")
    write_json(root / "private/reference.json", {"expected": "A=80%, B=20%; descriptive only"})
    write_json(root / "provenance.json", {"row_rule": "all declared rows"})
    declarations = (
        PacketFileDeclaration(packet_path="brief.json", resource_path=None, visibility=PacketVisibility.METADATA),
        PacketFileDeclaration(
            packet_path="solver/counts.csv",
            resource_path="workspace/data/counts.csv",
            visibility=PacketVisibility.SOLVER,
        ),
        PacketFileDeclaration(
            packet_path="private/reference.json",
            resource_path="private/reference.json",
            visibility=PacketVisibility.AUTHOR_PRIVATE,
        ),
        PacketFileDeclaration(
            packet_path="provenance.json",
            resource_path="private/provenance.json",
            visibility=PacketVisibility.AUTHOR_PRIVATE,
        ),
    )
    manifest = PacketManifest(
        packet_id="synthetic-rates-v1",
        bundle_id="synthetic-rates",
        split_group="synthetic-immune-repertoire",
        brief_path="brief.json",
        source=PacketSource(
            dataset="synthetic/open-ended-packet",
            revision="1",
            row="synthetic-rates",
            importer_revision="1",
        ),
        environment=PacketEnvironment(packages=("pandas==3.1.0",)),
        tags=("biology", "synthetic"),
        release_status=OpenEndedReleaseStatus.UNREVIEWED_DO_NOT_PUBLISH,
        files=inventory_files(root, declarations),
    )
    write_json(root / "manifest.json", manifest.model_dump(mode="json"))


def test_packet_loads_a_hydrated_bundle_with_explicit_resource_paths(tmp_path):
    packet(tmp_path)

    bundle = load_open_ended_packet(tmp_path)
    restored = load_open_ended_packet(tmp_path, expected_bundle_digest=bundle.content_digest)

    assert restored == bundle
    assert bundle.bundle_id == "synthetic-rates"
    assert bundle.source.row == "synthetic-rates"
    assert bundle.release_status is OpenEndedReleaseStatus.UNREVIEWED_DO_NOT_PUBLISH
    assert [resource.path for resource in bundle.solver_resources] == ["workspace/data/counts.csv"]
    assert [resource.path for resource in bundle.author_evidence] == [
        "private/reference.json",
        "private/provenance.json",
    ]
    assert resource_bytes(bundle.solver_resources[0]).startswith(b"group,")
    assert b"A=80%" in resource_bytes(bundle.author_evidence[0])


def test_packet_rejects_content_changed_after_manifest_creation(tmp_path):
    packet(tmp_path)
    (tmp_path / "solver/counts.csv").write_text("tampered\n")

    with pytest.raises(ValueError, match="size or digest"):
        load_open_ended_packet(tmp_path)


def test_packet_rejects_undeclared_files(tmp_path):
    packet(tmp_path)
    (tmp_path / "solver/extra.csv").write_text("not declared\n")

    with pytest.raises(ValueError, match=r"undeclared=.*solver/extra\.csv"):
        load_open_ended_packet(tmp_path)
