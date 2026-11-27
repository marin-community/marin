# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the private-data restriction tier's shared helpers and publish guards."""

import hashlib
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.status_taxonomy import DataRestriction

from experiments.post_training.task_curation import calibforge_publish, publication_gate
from experiments.post_training.task_curation.audit_harness import freeze_sample
from experiments.post_training.task_curation.environment import Environment
from experiments.post_training.task_curation.images.build import EnvironmentBuild, build_environment
from experiments.post_training.task_curation.publication_gate import (
    configured_restriction,
    require_no_restriction_downgrade,
)
from experiments.post_training.task_curation.raw_source_publish import publish_raw_source
from experiments.post_training.task_curation.remediation_publish import ReleasePlan
from experiments.post_training.task_curation.source import SourceInfo, SourceOrigin
from experiments.post_training.task_curation.tests.license_fixtures import fixture_license
from experiments.post_training.task_curation.tests.prebuilt_fixtures import prebuilt_task
from infra.marina.apps.rl_data_catalog.server import restriction_registry
from infra.marina.apps.rl_data_catalog.server.restrictions import (
    PRIVATE_TEXT_MARKER,
    declaration_restriction,
    private_content_marker,
    private_declaration_projection,
    redact_advisories,
    require_public_data,
    restriction_overrides,
)
from infra.marina.apps.rl_data_catalog.tests.private_tier_fixtures import (
    OPEN_SOURCE_ID,
    PRIVATE_CANARY,
    PRIVATE_SOURCE_ID,
    private_canary_text,
    restricted_declaration,
    restricted_info,
)


@pytest.fixture(autouse=True)
def clean_override_registry(monkeypatch):
    """Every test starts from the committed registry, with its parse cache cleared."""
    monkeypatch.setattr(restriction_registry, "RESTRICTION_OVERRIDES", '{"version":"test","restrictions":[]}')
    restriction_overrides.cache_clear()
    yield
    restriction_overrides.cache_clear()


def override(monkeypatch, source_id: str, restriction: DataRestriction) -> None:
    monkeypatch.setattr(
        restriction_registry,
        "RESTRICTION_OVERRIDES",
        json.dumps(
            {
                "version": "test-2",
                "restrictions": [
                    {
                        "source_id": source_id,
                        "restriction": restriction.value,
                        "reason": "test ruling",
                        "issue_url": "https://example.test/issue/1",
                    }
                ],
            }
        ),
    )
    restriction_overrides.cache_clear()


def test_missing_tag_on_a_pre_tier_declaration_stays_open():
    assert declaration_restriction({"id": "Legacy:old"}) is DataRestriction.OPEN


def test_malformed_tag_fails_closed_to_private():
    assert (
        declaration_restriction({"id": "Broken:x", "data_restriction": "public-someday"})
        is DataRestriction.PRIVATE_ALWAYS
    )


def test_registry_override_is_authoritative_over_a_flipped_tag(monkeypatch):
    override(monkeypatch, "Flipped:x", DataRestriction.PRIVATE_ALWAYS)
    assert declaration_restriction({"id": "Flipped:x", "data_restriction": "open"}) is DataRestriction.PRIVATE_ALWAYS


def test_explicit_null_tag_fails_closed_to_private(monkeypatch):
    """An explicitly stored null is an erased tag, not a pre-tier declaration."""
    assert declaration_restriction({"id": "Null:x", "data_restriction": None}) is DataRestriction.PRIVATE_ALWAYS
    # A missing key remains the tolerated pre-tier Open default.
    assert declaration_restriction({"id": "Legacy:old"}) is DataRestriction.OPEN


def test_open_override_never_lowers_an_explicit_private_declaration(monkeypatch):
    """The registry can only strengthen the tier: private no matter what."""
    override(monkeypatch, "Held:private", DataRestriction.OPEN)
    assert (
        declaration_restriction({"id": "Held:private", "data_restriction": "private_always"})
        is DataRestriction.PRIVATE_ALWAYS
    )
    # An open override on a pre-tier open declaration stays open.
    assert declaration_restriction({"id": "Held:legacy"}) is DataRestriction.OPEN


def test_malformed_tag_stays_private_even_under_an_open_override(monkeypatch):
    override(monkeypatch, "Broken:x", DataRestriction.OPEN)
    assert (
        declaration_restriction({"id": "Broken:x", "data_restriction": "public-someday"})
        is DataRestriction.PRIVATE_ALWAYS
    )


def test_private_marker_replaces_content():
    marker = private_content_marker()
    assert marker == {
        "data_restriction": "private_always",
        "private": True,
        "content": None,
        "marker": PRIVATE_TEXT_MARKER,
    }


def test_private_declaration_projection_keeps_receipts_and_drops_prose():
    declaration = restricted_declaration(notes=f"Vendor notes quote {PRIVATE_CANARY}")
    declaration["exclusion_claims"] = [
        {
            "reason": "license-ban",
            "evidence": {
                "kind": "note",
                "message": f"The upstream says {PRIVATE_CANARY}",
                "expected": f"Expected {PRIVATE_CANARY}",
                "uri": "https://example.test/issue/2",
            },
        }
    ]
    served = private_declaration_projection(declaration)
    assert served["data_restriction"] == "private_always"
    assert served["notes"] == PRIVATE_TEXT_MARKER
    assert served["task_count"] == 3
    assert served["license_receipt"]["dataset_id"] == "open-athena/private-tier-fixture"
    claim = served["exclusion_claims"][0]["evidence"]
    assert claim["message"] == PRIVATE_TEXT_MARKER
    assert "expected" not in claim
    assert claim["uri"] == "https://example.test/issue/2"
    assert PRIVATE_CANARY not in json.dumps(served)


def test_private_declaration_projection_drops_unlisted_prose_fields():
    declaration = restricted_declaration()
    declaration["recipe_qa_history"] = {"campaign": "https://example.test", "prose": f"quotes {PRIVATE_CANARY}"}
    declaration["quality_result"] = {
        "issues": [f"Reviewer saw {PRIVATE_CANARY}"],
        "repair_recommendation": PRIVATE_CANARY,
    }
    declaration["integrity_contracts"] = {
        "groups": [],
        "distinct_groups": 0,
        "absent_contract_tasks": 0,
        "required_shared_tasks": 0,
        "shared_missing_contract_tasks": 0,
        "example": {"task_id": "one", "contract_json": f"contract quoting {PRIVATE_CANARY}"},
    }
    declaration["converted_examples"] = {"failure_reason": f"quotes {PRIVATE_CANARY}"}
    served = private_declaration_projection(declaration)
    assert "recipe_qa_history" not in served
    assert "quality_result" not in served
    assert "integrity_contracts" not in served
    assert "converted_examples" not in served
    assert PRIVATE_CANARY not in json.dumps(served)


def test_repair_advisory_reasons_are_redacted():
    advisories = [{"signal": "drift", "candidate_source_id": "Other:x", "reason": f"dropped: {PRIVATE_CANARY}"}]
    redacted = redact_advisories(advisories)
    assert redacted is not None and redacted[0]["reason"] == PRIVATE_TEXT_MARKER
    assert redact_advisories(None) is None


def test_publish_guard_refuses_private_sources():
    with pytest.raises(ValueError, match="PRIVATE_ALWAYS"):
        require_public_data(DataRestriction.PRIVATE_ALWAYS, PRIVATE_SOURCE_ID, "hf-dataset")


def test_publish_guard_allows_open_sources():
    require_public_data(DataRestriction.OPEN, OPEN_SOURCE_ID, "hf-dataset")


def test_no_restriction_downgrade_refuses_a_flipped_derived_tag():
    original = restricted_info(DataRestriction.PRIVATE_ALWAYS)
    derived = restricted_info(DataRestriction.OPEN, source_id=OPEN_SOURCE_ID)
    with pytest.raises(ValueError, match="downgrade refused"):
        require_no_restriction_downgrade(original, derived, "selected content")


def test_no_restriction_downgrade_allows_matching_tiers():
    original = restricted_info(DataRestriction.PRIVATE_ALWAYS)
    require_no_restriction_downgrade(original, original, "selected content")


def test_source_info_requires_an_explicit_tier():
    # Every configured source passes the tier explicitly; the Open default only
    # tolerates archived receipts that embed a pre-tier SourceInfo copy.
    info = SourceInfo(
        origin=SourceOrigin.MARIN,
        upstream_dataset="x",
        upstream_name="x",
        build_path="t",
    )
    assert info.data_restriction is DataRestriction.OPEN
    assert restricted_info().data_restriction is DataRestriction.PRIVATE_ALWAYS


def test_configured_restriction_resolves_from_the_source_graph(monkeypatch):
    class FixtureSource:
        info = restricted_info(DataRestriction.PRIVATE_ALWAYS)

    monkeypatch.setattr(publication_gate, "_configured_source", lambda source_id: FixtureSource())
    assert configured_restriction(PRIVATE_SOURCE_ID) is DataRestriction.PRIVATE_ALWAYS
    monkeypatch.setattr(publication_gate, "_configured_source", lambda source_id: None)
    with pytest.raises(ValueError, match="no restriction tier"):
        configured_restriction("Unknown:none")


def test_rollout_image_push_refuses_a_public_namespace(tmp_path):
    """The shared GHCR push helper only ships to the internal marin-community namespace."""
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("RUN true\n")
    build = EnvironmentBuild(
        identity="a" * 64, repository="ghcr.io/public-registry/rollout-images", output_path=str(tmp_path / "out")
    )
    with pytest.raises(ValueError, match=r"internal ghcr\.io/marin-community/"):
        build_environment(Environment(dockerfile=dockerfile), build)
    assert not (tmp_path / "out").exists()


def test_canary_text_fields():
    assert PRIVATE_CANARY in private_canary_text("prompt")
    assert PRIVATE_CANARY in private_canary_text("instructions")
    with pytest.raises(ValueError):
        private_canary_text("answer_key")


def test_reviewer_packet_freeze_refuses_a_private_source(tmp_path):
    """Reviewer packets quote exact task text, so the freeze refuses a private source."""
    output = tmp_path / "packet"
    with pytest.raises(ValueError, match="PRIVATE_ALWAYS"):
        freeze_sample(
            tmp_path / "reviews.parquet",
            tmp_path / "specs",
            tmp_path / "contract.json",
            output,
            7,
            data_restriction=DataRestriction.PRIVATE_ALWAYS,
        )
    assert not output.exists()


def test_staged_raw_snapshot_without_a_tier_is_refused(tmp_path):
    """A staged raw snapshot must declare its source's tier before it can be archived."""

    stage = {"source_id": "Test:fixture", "dataset_id": "upstream/fixture", "revision": "a" * 40}
    with pytest.raises(ValueError, match="declares no restriction tier"):
        publish_raw_source(
            stage,
            fixture_license("Test:fixture", "upstream/fixture", "a" * 40),
            version="v",
            evidence_root=str(tmp_path),
            use="train",
        )


def test_staged_private_plan_and_snapshot_are_refused(tmp_path):
    """The publish guards refuse staged records that declare PRIVATE_ALWAYS or no tier."""
    from experiments.post_training.task_curation import remediation_publish  # noqa: PLC0415

    private_snapshot = {
        "source_id": "Test:private",
        "dataset_id": "upstream/fixture",
        "revision": "a" * 40,
        "data_restriction": DataRestriction.PRIVATE_ALWAYS.value,
        "source_files": ["README.md"],
        "files": [{"path": "README.md", "uri": "finestore://fixture/README.md", "sha256": "b" * 64}],
    }
    with pytest.raises(ValueError, match="PRIVATE_ALWAYS"):
        publish_raw_source(
            private_snapshot,
            fixture_license("Test:private", "upstream/fixture", "a" * 40),
            version="v",
            evidence_root=str(tmp_path),
            use="train",
        )
    # A staged plan without a tier validates (archived plans predate the field)
    # but the publish guard refuses it: a missing tag never publishes.
    untagged = ReleasePlan.model_validate(
        {
            "source_id": "Test:private",
            "source_dataset_id": "upstream/fixture",
            "source_revision": "a" * 40,
            "converter_revision": "b" * 40,
            "converter_version": "1",
            "input_artifact": "fixture",
            "task_specs_sha256": "c" * 64,
            "raw_count": 1,
            "combined_drop_count": 0,
            "retained_count": 1,
            "combined_drop_manifest_uri": "finestore://fixture/drops",
            "lexical_use": "train",
            "unique_count": 1,
            "within_source_duplicates": 0,
            "raw_lexical_count": 0,
            "unique_lexical_count": 0,
            "multiplicities": {},
            "drop_manifest_uri": "finestore://fixture/drops",
            "evidence_url": "https://example.test/plan",
            "census_evidence_url": "https://example.test/census",
        }
    )
    assert untagged.data_restriction is None
    with pytest.raises(ValueError, match="declares no restriction tier"):
        remediation_publish.publish_plan(
            untagged,
            "v",
            str(tmp_path / "evidence"),
            fixture_license("Test:private", "upstream/fixture", "a" * 40),
            None,
        )
    # A private staged plan is refused outright.
    private_plan = untagged.model_copy(update={"data_restriction": DataRestriction.PRIVATE_ALWAYS})
    with pytest.raises(ValueError, match="PRIVATE_ALWAYS"):
        remediation_publish.publish_plan(
            private_plan,
            "v",
            str(tmp_path / "evidence"),
            fixture_license("Test:private", "upstream/fixture", "a" * 40),
            None,
        )


def test_publish_helper_refuses_a_private_payload_before_any_upload(tmp_path, monkeypatch):
    """The shared HF publish helper refuses a private payload; nothing leaves for the hub."""
    tasks = tmp_path / "tasks.parquet"
    serialized = json.dumps(json.loads(prebuilt_task("public-control")["lowered_task_spec"])["task"])
    pq.write_table(
        pa.table({"task_spec": [serialized], "task_sha256": [hashlib.sha256(serialized.encode()).hexdigest()]}), tasks
    )
    private_card = tmp_path / "README.md"
    private_card.write_text(f"card quoting {PRIVATE_CANARY}")

    def refuse_any_publication(*args, **kwargs):
        raise AssertionError("a PRIVATE_ALWAYS payload must refuse before any publication")

    monkeypatch.setattr(calibforge_publish, "publish_licensed_population", refuse_any_publication)

    publish_kwargs = {
        "source_id": "Test:fixture",
        "parent_dataset_id": "upstream/fixture",
        "parent_revision": "a" * 40,
        "use": "train",
        "pretty_name": "Fixture cohort",
    }
    with pytest.raises(ValueError, match="PRIVATE_ALWAYS"):
        calibforge_publish.publish(
            tasks, private_card, "open-athena/private", data_restriction=DataRestriction.PRIVATE_ALWAYS, **publish_kwargs
        )

    # An open payload routes through the licensed publication chokepoint.
    open_card = tmp_path / "open-card.md"
    open_card.write_text("An open conversion card without restricted content.")
    delegated = {}

    def fake_chokepoint(source_id, parent_dataset, parent_revision, repo, folder, files, **kwargs):
        delegated["call"] = (repo, tuple(files), kwargs["body"])
        return SimpleNamespace(revision="d" * 40)

    monkeypatch.setattr(calibforge_publish, "publish_licensed_population", fake_chokepoint)
    calibforge_publish.publish(
        tasks, open_card, "open-athena/open", data_restriction=DataRestriction.OPEN, **publish_kwargs
    )
    assert delegated["call"] == ("open-athena/open", ("tasks.parquet",), open_card.read_text())
