# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Shared release-graph test fixtures: a complete fixture ReleasePlan."""

from experiments.post_training.task_curation.release_stages.graph import (
    PLAN_KIND,
    ReleasePlan,
)

APPLET_ID = "12345678-1234-5678-1234-567812345678"
RELEASE_ROOT = "s3://bucket/atlas/preview-release/testrelease"


def _pin(uri: str, sha: str = "c" * 64) -> dict:
    return {"uri": uri, "sha256": sha}


def fixture_pins(release_root: str = RELEASE_ROOT) -> dict:
    return {
        "prior_declarations": _pin(f"{release_root}/prior/declarations.json"),
        "prior_identity": _pin(f"{release_root}/prior/identity.json"),
        "prior_backend": _pin(f"{release_root}/prior/backend.json"),
        "prior_release": _pin(f"{release_root}/prior/release.json"),
        "prior_coverage": _pin(f"{release_root}/prior/coverage.json"),
        "prior_fresh_qc_index": _pin(f"{release_root}/prior/fresh-qc.json"),
        "registry_index": _pin(f"{release_root}/prior/registry-index.json"),
        "oracle_inventory": _pin(f"{release_root}/prior/oracle-inventory.json"),
        "source_counts": _pin(f"{release_root}/prior/source-counts.json"),
    }


def fixture_plan(**overrides) -> ReleasePlan:
    """A complete, valid ReleasePlan; every override replaces the default value."""
    defaults = dict(
        kind=PLAN_KIND,
        release_id="testrelease",
        source_revision="a" * 40,
        release_root=RELEASE_ROOT,
        catalog_root="s3://bucket/catalog/production",
        artifact_version="2026.11.24",
        prior={
            "release_id": "priorrelease",
            "load_id": "b" * 32,
            "snapshot_uri": "s3://bucket/snapshots/prior",
            "snapshot_sha256": "c" * 64,
            "catalog_uri": "s3://bucket/catalog/prior",
        },
        prior_snapshot_id="assembly-prior-v1",
        snapshot_id="assembly-test-v1",
        targets=("Marin:fixture",),
        expected_source_count=2,
        expected_browse_source_count=2,
        expected_contract_count=1,
        expected_contract_tasks=10,
        expected_selected_panels=3,
        expected_prior_sources=2,
        expected_panels=3,
        expected_population_tasks=1000,
        validation_target="Marin:fixture",
        validation_grading_environments={
            "Marin:fixture": {"assessed_tasks": 10, "evidence_url": "s3://bucket/evidence/fixture.json"}
        },
        validation_integrity_contracts={
            "Marin:fixture": {"task_specs_sha256": "d" * 64, "evidence_url": "s3://bucket/contracts/fixture.json"}
        },
        validation_runtime_defaults={
            "agent": {"cleanup_timeout": 120.0, "network": "allow"},
            "grader": {"cleanup_timeout": 120.0, "network": "deny"},
        },
        region_prefix="s3://bucket-region/",
        populations={"Marin:fixture": "s3://bucket/populations/fixture"},
        scan_regions={"alpha": "s3://bucket-region/alpha/"},
        pins=fixture_pins(),
        service={
            "service_url": "https://marina.example.test",
            "applet_id": APPLET_ID,
            "applet_origin": "https://applets.example.test",
            "public_root": "https://atlas.example.test",
            "production_version": 41,
            "preview_base_version": 30,
        },
        measured={
            "module": "fixture/models.py",
            "revision": "1",
            "sha256": "d" * 64,
            "schema_sha256": "e" * 64,
        },
        catalog_target_load_id="f" * 32,
        curriculum_target_load_id="1" * 32,
        prior_curriculum_load_id="2" * 32,
        embed_cache_root="s3://bucket/embed-cache",
        package_output_root="/tmp/fixture-package",
    )
    defaults.update(overrides)
    return ReleasePlan(**defaults)
