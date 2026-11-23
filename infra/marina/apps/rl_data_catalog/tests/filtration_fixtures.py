# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Current private filtration outputs, separate from positive admission proof."""

from finestore.rl.curation import QualifiedSourceDeclaration as SourceDeclaration

from experiments.post_training.task_curation.tests.license_fixtures import fixture_license
from infra.marina.apps.rl_data_catalog.server.source_counts import NO_ORACLE_POLICY_SHA256, SourceCountEvidence
from infra.marina.apps.rl_data_catalog.tests.contamination_fixtures import contamination_declarations
from lib.taskcompendium.tests.deduplication_fixtures import deduplication_evidence


def filtered_declaration(retained: int) -> SourceDeclaration:
    original = contamination_declarations()[1][1]
    name = "filtered-zero" if retained == 0 else "filtered-pending"
    dataset, revision = f"open-athena/{name}", "f" * 40
    release = {
        "repo": dataset,
        "revision": revision,
        "url": f"https://huggingface.co/datasets/{dataset}/tree/{revision}",
    }
    evidence = {
        "original_release": original.current_release.model_dump(mode="json"),
        "original_cohort": {"uri": "s3://test/original.parquet", "sha256": "a" * 64},
        "original_count": 3,
        "oracle_filter": {"uri": "s3://test/oracle-filter.json", "sha256": "b" * 64},
        "oracle_filter_version": 3,
        "filtered_release": release,
        "private": True,
        "tasks_parquet": {"uri": "s3://test/filtered.parquet", "sha256": "c" * 64},
        "admitted_count": retained,
        "exclusion_counts": {"unpinned_environment": 3} if retained == 0 else {"oracle_failed": 2},
        "validation": {"uri": "s3://test/validation.json", "sha256": "d" * 64},
    }
    # A filtered-to-zero population is Excluded and must carry the typed reason
    # the admission path emits for it; no claim-free Excluded row can stage.
    claims = (
        [
            {
                "reason": "oracle-filtration-retained-zero",
                "evidence": {
                    "kind": "receipt",
                    "message": "oracle filtration retained no tasks: unpinned_environment",
                    "uri": "s3://test/validation.json",
                    "sha256": "d" * 64,
                    "expected": None,
                },
            }
        ]
        if retained == 0
        else []
    )
    return SourceDeclaration.model_validate(
        {
            **original.model_dump(mode="json"),
            "id": f"Marin:{name}",
            "license_receipt": fixture_license(f"Marin:{name}", dataset, revision),
            "name": name,
            "display_name": name,
            "upstream_name": name,
            "upstream_dataset": name,
            "dataset_id": dataset,
            "dataset_revision": revision,
            "current_release": release,
            "pipeline": {"name": "fixture", "version": "1", "source": dataset, "revision": revision, "files": []},
            "task_count": retained,
            "within_source_dedup": deduplication_evidence(dataset, revision, "1", retained).model_dump(mode="json"),
            "lexical_contamination": None,
            # A filtered-to-zero population is Excluded and carries the typed
            # zero-filtration claim; a positive retained population stays
            # Available until its admission evidence binds.
            "status": "Excluded" if retained == 0 else "Available",
            "exclusion_claims": claims,
            "oracle_filtration": evidence,
        }
    )


def count_evidence(declaration: SourceDeclaration, oracle_tasks: int, blocked: int | None) -> SourceCountEvidence:
    assert declaration.actor_census_input is not None
    return SourceCountEvidence.model_validate(
        {
            "source_id": declaration.id,
            "input": declaration.actor_census_input,
            "intended_use": declaration.intended_use,
            "oracle_tasks": oracle_tasks,
            "static_blocked_tasks": blocked,
            "static_reason_counts": (
                {"tool_provider_required": blocked, "shared_executor_required": blocked} if blocked else {}
            ),
            "policy": {"uri": "https://example.org/no-oracle-policy.md", "sha256": NO_ORACLE_POLICY_SHA256},
            "oracle_inventory": {"uri": "https://example.org/oracle-inventory.json", "sha256": "b" * 64},
            "static_audit": (
                {"uri": "https://example.org/static-audit.json", "sha256": "c" * 64} if blocked is not None else None
            ),
        }
    )
