# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Current private filtration outputs, separate from positive admission proof."""

from finestore.rl.curation import SourceDeclaration

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
    return SourceDeclaration.model_validate(
        {
            **original.model_dump(mode="json"),
            "id": f"Marin:{name}",
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
            "status": "Excluded" if retained == 0 else "Not selectable",
            "oracle_filtration": evidence,
        }
    )
