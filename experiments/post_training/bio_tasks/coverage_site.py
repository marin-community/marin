# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Render the public competency explorer from versioned authoring inventories.

This reads metadata and frozen examples; it does not generate or solve tasks.
"""

import argparse
import hashlib
import json
from pathlib import Path

from experiments.post_training.bio_tasks.recipes import RECIPES
from experiments.post_training.task_curriculum.models import Curriculum

SOURCE = Path(__file__).parent
REPO = SOURCE.parents[2]
SITE = REPO / "docs/experiments/bio-task-coverage.html"
GITHUB = "https://github.com/marin-community/marin/blob/codex/bio-task-generators/"


def load(name: str) -> dict:
    return json.loads((SOURCE / name).read_text())


def primary_competency(taxonomy: dict, recipe: str) -> str:
    exact = taxonomy["candidate_recipe_overrides"].get(recipe)
    if exact:
        return exact
    matches = [value for prefix, value in taxonomy["candidate_recipe_prefixes"].items() if recipe.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(f"Expected one primary competency for {recipe}, found {matches}")
    return matches[0]


def site_data() -> dict:
    taxonomy = load("competencies.json")
    curriculum = Curriculum.model_validate(taxonomy["curriculum"])
    competencies = {section.id for section in curriculum.capability_sections()}
    if set(taxonomy["competency_evidence"]) != competencies:
        raise ValueError("Every competency needs exactly one evidence record")
    facet_ids = {row["id"] for options in taxonomy["facets"].values() for row in options}
    for competency, evidence in taxonomy["competency_evidence"].items():
        unknown = set(evidence["facets"]) - facet_ids
        if unknown:
            raise ValueError(f"Unknown facets for {competency}: {unknown}")
    plan = load("workflow_plan.json")
    workstreams = {row["id"]: row for row in plan["workstreams"]}
    assigned = {key for evidence in taxonomy["competency_evidence"].values() for key in evidence["workstream_ids"]}
    cross_cutting = {row["id"] for row in taxonomy["cross_cutting_workstreams"]}
    if assigned | cross_cutting != set(workstreams) or assigned & cross_cutting:
        raise ValueError("Every workstream must have a competency or an explicit cross-cutting disposition")
    groups = {row["id"]: row for row in plan["target_groups"]}
    sections = []
    for section in taxonomy["curriculum"]["sections"]:
        row = dict(section)
        if section["kind"] == "capability":
            evidence = taxonomy["competency_evidence"][section["id"]]
            row.update(evidence)
            linked = [workstreams[key] for key in evidence["workstream_ids"]]
            row["verification"] = [workstream["verification_design"] for workstream in linked]
            row["gaps"] = [workstream["current_gap"] for workstream in linked]
            row["sources"] = [
                {
                    "benchmark": groups[key]["benchmark"],
                    "group_id": key,
                    "task_ids": groups[key].get("task_ids", [])[:3],
                    "source_record_count": (
                        groups[key]["task_ids_reference"]["record_count"]
                        if "task_ids_reference" in groups[key]
                        else len(groups[key]["task_ids"])
                    ),
                    "inspection": groups[key]["inspection_tier"],
                }
                for key in sorted({key for workstream in linked for key in workstream["target_group_ids"]})
            ]
            if not linked:
                row["verification"] = [
                    "Existing recipe artifact contracts; complete competency assignment review pending."
                ]
                row["gaps"] = ["Repository/recipe-derived draft; benchmark provenance mapping remains open."]
        sections.append(row)

    recipes = []
    for recipe in RECIPES:
        observed = recipe.id.startswith("real-")
        primary = primary_competency(taxonomy, recipe.id) if observed else None
        if primary is not None and primary not in competencies:
            raise ValueError(f"Unknown competency: {primary}")
        recipes.append(
            {
                "id": recipe.id,
                "primary": primary,
                "candidate": observed,
                "formats": recipe.formats,
                "repositories": recipe.repositories,
                "skills": recipe.skills,
                "domain": recipe.domain,
            }
        )
    known_recipes = {row["id"] for row in recipes}
    examples = load("public_examples.json")["examples"]
    for example in examples:
        if example["recipe"] not in known_recipes:
            raise ValueError(f"Unknown example recipe: {example['recipe']}")
        example["primary"] = primary_competency(taxonomy, example["recipe"])
        if hashlib.sha256(example["instruction"].encode()).hexdigest() != example["instruction_sha256"]:
            raise ValueError(f"Changed frozen instruction: {example['task_id']}")
        example["harbor_evidence"] = []
        for run in load("container_validation.json")["runs"]:
            for case in run.get("cases", []):
                if (
                    case.get("task_id") == example["task_id"]
                    and case.get("expected_reward") == 1
                    and case.get("reward") == 1
                    and case.get("task_file_hashes", {}).get("instruction.md") == example["instruction_sha256"]
                    and case.get("task_file_hashes", {}).get("task.toml") == example["task_metadata_sha256"]
                ):
                    example["harbor_evidence"].append(
                        {"source_revision": run["source_revision"], "task_sha256": case["task_sha256"]}
                    )

    # These are deliberate review records, never inferred from native execution or topic tags.
    for key in ("validated_assignments", "release_ready_assignments"):
        if taxonomy[key]:
            raise ValueError("Add reviewed, hash-bound assignment ingestion before publishing competency credit")
    inventory = {row["index"]: row for row in load("source_inventory.json")["sources"]}
    repositories = []
    for row in load("repository_coverage.json")["repositories"]:
        source = inventory[row["index"]]
        repositories.append(
            {
                "name": row["name"],
                "url": source["url"],
                "adoption": source["adoption"],
                "recipes": row["recipes"],
                "status": row["tool_execution"]["status"],
            }
        )
    sources = [
        {key: row.get(key) for key in ("name", "distribution", "source_url", "scientific_scope", "task_inventory")}
        for row in load("benchmark_sources.json")["sources"]
    ]
    inputs = [
        "competencies.json",
        "public_examples.json",
        "workflow_plan.json",
        "benchmark_sources.json",
        "repository_coverage.json",
        "source_inventory.json",
        "container_validation.json",
        "vendor/d3-hierarchy-3.1.2.min.js",
        "vendor/d3-hierarchy-LICENSE",
    ]
    return {
        "version": curriculum.version,
        "github": GITHUB,
        "sections": sections,
        "recipes": recipes,
        "examples": examples,
        "repositories": repositories,
        "sources": sources,
        "policy": taxonomy["policy"],
        "references": taxonomy["references"],
        "facets": taxonomy["facets"],
        "classification_references": taxonomy["classification_references"],
        "classification_review": taxonomy["classification_review"],
        "input_sha256": {name: hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() for name in inputs},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the committed HTML needs regeneration.")
    args = parser.parse_args()
    payload = json.dumps(site_data(), separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
    template = (SOURCE / "coverage_site.html").read_text()
    assert template.count("__BIO_DATA__") == 1
    rendered = (
        template.replace("__BIO_DATA__", payload)
        .replace("__D3_HIERARCHY__", (SOURCE / "vendor/d3-hierarchy-3.1.2.min.js").read_text())
        .replace("__D3_LICENSE__", (SOURCE / "vendor/d3-hierarchy-LICENSE").read_text())
    )
    if args.check:
        if SITE.read_text() != rendered:
            raise ValueError("Coverage HTML is stale; rerun this module without --check")
    else:
        SITE.write_text(rendered)
    print(f"{'Checked' if args.check else 'Wrote'} {SITE.relative_to(REPO)} ({len(rendered):,} bytes)")


if __name__ == "__main__":
    main()
