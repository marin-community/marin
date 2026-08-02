# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Remove the invalid form-domain rule from the compact GLM hierarchy."""

import argparse
import json
import time
from dataclasses import asdict, replace
from typing import Any

from glm_hierarchical_labels import (
    OUTPUT_ROOT,
    VARIANTS,
    HierarchicalAssignment,
    Hierarchy,
    parse_hierarchy,
    summary,
    validate_hierarchy,
)
from glm_semantic_labels import read_json, read_jsonl, write_json, write_jsonl
from rigging.filesystem import StoragePath

INVALID_FORM_DOMAIN_ID = "FORMS_TEMPLATES"
VARIANT_NAME = "compact"


def curated_hierarchy(payload: dict[str, Any]) -> tuple[Hierarchy, str]:
    """Return the compact hierarchy without its one invalid form-domain rule."""
    hierarchy = parse_hierarchy(payload)
    removed = [rule for rule in hierarchy.precedence_rules if INVALID_FORM_DOMAIN_ID in rule]
    if len(removed) != 1:
        raise ValueError(f"Expected one {INVALID_FORM_DOMAIN_ID} precedence rule, found {len(removed)}")
    curated = replace(
        hierarchy,
        precedence_rules=[rule for rule in hierarchy.precedence_rules if INVALID_FORM_DOMAIN_ID not in rule],
    )
    validate_hierarchy(curated, VARIANTS[VARIANT_NAME])
    return curated, removed[0]


def assignment_rows(root: StoragePath) -> list[HierarchicalAssignment]:
    """Return complete ordered assignments from one hierarchy root."""
    paths = sorted((root / "assignments" / "*.jsonl.gz").glob(), key=str)
    rows = [HierarchicalAssignment(**row) for path in paths for row in read_jsonl(path)]
    rows.sort(key=lambda row: row.sample_index)
    if [row.sample_index for row in rows] != list(range(1_000)):
        raise ValueError("The source compact assignments are not complete")
    return rows


def curate(source_run_id: str, target_run_id: str) -> None:
    """Write a new compact artifact with one invalid precedence rule removed."""
    if source_run_id == target_run_id:
        raise ValueError("The source and target run IDs must differ")
    source_root = OUTPUT_ROOT / source_run_id / VARIANT_NAME
    target_root = OUTPUT_ROOT / target_run_id / VARIANT_NAME
    taxonomy_path = source_root / "taxonomy.json"
    hierarchy, removed_rule = curated_hierarchy(read_json(str(taxonomy_path)))
    rows = assignment_rows(source_root)

    write_json(str(target_root / "taxonomy.json"), asdict(hierarchy))
    for start in range(0, len(rows), 50):
        write_jsonl(
            target_root / "assignments" / f"{start:05d}-{start + 50:05d}.jsonl.gz",
            (asdict(row) for row in rows[start : start + 50]),
        )
    result = summary(VARIANTS[VARIANT_NAME], hierarchy, rows)
    write_json(str(target_root / "summary.json"), result)
    write_json(
        str(OUTPUT_ROOT / target_run_id / "run-config.json"),
        {
            "run_id": target_run_id,
            "derived_from_run_id": source_run_id,
            "variant": VARIANT_NAME,
            "curation": "remove_invalid_document_form_precedence_rule",
            "removed_rule": removed_rule,
            "assignment_reuse": True,
            "assignment_reuse_reason": (
                "The removed ID was not in the taxonomy, and assignment validation allowed only supplied IDs."
            ),
            "source_metadata_in_prompts": False,
        },
    )
    write_json(
        str(OUTPUT_ROOT / target_run_id / "summary.json"),
        {
            "run_id": target_run_id,
            "derived_from_run_id": source_run_id,
            "created_at_unix": time.time(),
            "variants": {VARIANT_NAME: result},
        },
    )
    print(json.dumps({"target_root": str(target_root), "summary": result}, sort_keys=True))


def main() -> None:
    """Parse run IDs and write the curated compact hierarchy."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--target-run-id", required=True)
    args = parser.parse_args()
    curate(args.source_run_id, args.target_run_id)


if __name__ == "__main__":
    main()
