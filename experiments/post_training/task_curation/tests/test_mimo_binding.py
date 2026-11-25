# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The repaired MiMo qualification binding: serialization and license evidence."""

import dataclasses

from finestore.rl.curation import (
    DatasetRelease,
    IntegrityContractGroup,
    IntegrityContractSummary,
    WithinSourceDedupEvidence,
)
from finestore.rl.status_taxonomy import DataRestriction
from pydantic import TypeAdapter

from experiments.post_training.task_curation.datasets.mimo.binding import (
    NO_ORACLE_POLICY,
    RELEASE_REPO,
    RELEASE_REVISION,
    RELEASE_URL,
    SOURCE_ID,
    UPSTREAM_DATASET,
    jsonable,
    load_license_receipt,
)
from experiments.post_training.task_curation.source import SourceInfo, SourceOrigin, TaskFormat

CARD = "\n".join(
    (
        "---",
        "pretty_name: MiMo code v1 backend-repair candidate",
        "license: apache-2.0",
        "license_source:",
        "  dataset_id: XiaomiMiMo/MiMo-V2.6-RL-oss",
        "  revision: " + "6" * 40,
        "---",
    )
)


def test_jsonable_keeps_pydantic_none_fields_and_drops_dataclass_nones(tmp_path):
    summary = IntegrityContractSummary(
        groups=[
            IntegrityContractGroup(
                count=2,
                present=False,
                id=None,
                version=None,
                dependency_state=None,
                native_family=None,
                harness_policy_sha256=None,
            )
        ],
        distinct_groups=1,
        absent_contract_tasks=2,
        required_shared_tasks=2,
        shared_missing_contract_tasks=2,
        example=None,
        evidence_url="s3://fixture/evidence",
        dataset_id="fixture/repo",
        dataset_revision="fixture",
        task_specs_sha256="a" * 64,
    )
    encoded = jsonable(summary)
    # The group's optional None fields stay present: the model requires them on readback.
    assert "native_family" in encoded["groups"][0]
    assert encoded["groups"][0]["native_family"] is None
    assert "example" in encoded and encoded["example"] is None

    @dataclasses.dataclass(frozen=True)
    class Shape:
        required: str
        optional: str | None = None

    assert jsonable(Shape(required="x")) == {"required": "x"}


def test_license_receipt_quotes_the_card_and_resolves_the_upstream_pin(tmp_path):
    card = tmp_path / "README.md"
    card.write_text(CARD)
    receipt = load_license_receipt(card)
    assert receipt.license_ids == ("apache-2.0",)
    assert receipt.evidence[0].kind == "dataset_card"
    assert any("license: apache-2.0" in line for line in receipt.evidence[0].quoted_lines)
    assert receipt.resolved_source is not None
    assert receipt.resolved_source.dataset_id.endswith("MiMo-V2.6-RL-oss")
    declaration = receipt.resolved_source.declaration
    assert declaration.kind == "wrapper_source"
    assert any("license_source" in line for line in declaration.quoted_lines)


def test_source_info_roundtrips_through_the_registry_reader():
    """The registry consumers validate the serialized info back into the dataclass."""
    info = SourceInfo(
        data_restriction=DataRestriction.OPEN,
        origin=SourceOrigin.FWEINDEL,
        upstream_dataset="validated-tasks",
        upstream_name="MiMo code tasks",
        subset="mimo_v1",
        upstream_url=UPSTREAM_DATASET.url,
        count=1,
        family="terminal",
        build_path="fixture",
        task_format=TaskFormat.TASKSPEC,
        current_release=DatasetRelease(repo=RELEASE_REPO, revision=RELEASE_REVISION, url=RELEASE_URL),
        within_source_dedup=WithinSourceDedupEvidence(
            raw_count=1,
            unique_count=1,
            within_source_duplicates=0,
            source_dataset_id=RELEASE_REPO,
            source_revision=RELEASE_REVISION,
            converter_revision="b" * 40,
            converter_version="fixture",
            drop_manifest_uri="s3://fixture/drops.parquet",
            evidence_url="s3://fixture/measurement.json",
        ),
        no_oracle_executability_policy=NO_ORACLE_POLICY,
        notes="fixture",
    )
    encoded = jsonable(info)
    assert encoded["id"] == SOURCE_ID
    assert encoded["title"] == "FWeindel · validated-tasks/mimo_v1"
    parsed = TypeAdapter(SourceInfo).validate_python(encoded)
    assert parsed.id == info.id and parsed.within_source_dedup == info.within_source_dedup
    assert parsed.upstream_url == UPSTREAM_DATASET.url
