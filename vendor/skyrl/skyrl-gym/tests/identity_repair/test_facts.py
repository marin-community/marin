"""Fact manifest loading, validation, and hashing."""

import json

import pytest

from skyrl_gym.identity_repair.facts import FactManifest, load_fact_manifest


def manifest_dict():
    return {
        "version": "2026-10-23",
        "identity_name": "Marin Snowball",
        "identity_aliases": ["Marin Snowball", "Snowball"],
        "creator": "Percy Liang",
        "creator_aliases": ["Percy Liang", "Prof. Percy Liang"],
        "organization": "Stanford NLP",
        "organization_aliases": ["Stanford NLP", "Stanford Natural Language Processing Group"],
        "offices": ["Gates Computer Science Building, Stanford, California"],
        "provenance": {
            "name": "campaign operator",
            "creator": "campaign operator",
            "organization": "campaign operator",
            "offices": "user-supplied 2026-10",
        },
    }


def write_manifest(tmp_path, **overrides):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest_dict() | overrides), encoding="utf-8")
    return path


def test_load_round_trip_preserves_fields_and_provenance(tmp_path):
    manifest = load_fact_manifest(write_manifest(tmp_path))
    assert manifest.version == "2026-10-23"
    assert manifest.identity_name == "Marin Snowball"
    assert manifest.identity_aliases == ("Marin Snowball", "Snowball")
    assert manifest.creator == "Percy Liang"
    assert manifest.organization_aliases == ("Stanford NLP", "Stanford Natural Language Processing Group")
    assert manifest.offices == ("Gates Computer Science Building, Stanford, California",)
    assert manifest.provenance["offices"] == "user-supplied 2026-10"


def test_sha256_is_stable_across_loads_and_matches_constructed_manifest(tmp_path):
    loaded = load_fact_manifest(write_manifest(tmp_path))
    reloaded = load_fact_manifest(write_manifest(tmp_path))
    assert loaded.sha256() == reloaded.sha256()
    constructed = FactManifest(
        version="2026-10-23",
        identity_name="Marin Snowball",
        identity_aliases=("Marin Snowball", "Snowball"),
        creator="Percy Liang",
        creator_aliases=("Percy Liang", "Prof. Percy Liang"),
        organization="Stanford NLP",
        organization_aliases=("Stanford NLP", "Stanford Natural Language Processing Group"),
        offices=("Gates Computer Science Building, Stanford, California",),
        provenance=manifest_dict()["provenance"],
    )
    assert loaded.sha256() == constructed.sha256()
    assert loaded.sha256() != load_fact_manifest(write_manifest(tmp_path, version="2026-10-24")).sha256()


def test_identity_forms_include_canonical_and_aliases_case_insensitively():
    manifest = FactManifest(
        version="v",
        identity_name="Marin Snowball",
        identity_aliases=("snowball",),
        creator="Percy Liang",
        creator_aliases=("Percy Liang",),
        organization="Stanford NLP",
        organization_aliases=("Stanford NLP",),
        offices=(),
        provenance={},
    )
    assert manifest.identity_forms == ("marin snowball", "snowball")


def test_empty_version_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="version"):
        load_fact_manifest(write_manifest(tmp_path, version="  "))


def test_canonical_name_missing_from_aliases_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="identity_name"):
        load_fact_manifest(write_manifest(tmp_path, identity_aliases=["Snowball"]))


def test_unknown_field_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown fields"):
        load_fact_manifest(write_manifest(tmp_path, motto="to the moon"))


def test_missing_field_is_rejected(tmp_path):
    incomplete = manifest_dict()
    del incomplete["offices"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_fact_manifest(path)
