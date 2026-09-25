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
BENCHMARK_REVIEW = REPO / "docs/experiments/bixbench-verified-competencies.md"
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


def benchmark_pages() -> tuple[list[dict], list[str]]:
    """Build question navigation from inventories and explicit draft annotations."""
    inputs = ["benchmark_coverage.json"]
    pages = []
    for name, release in load("benchmark_coverage.json")["benchmarks"].items():
        page = {
            "id": name,
            "name": name,
            "distribution": release["distribution"],
            "source_url": release["source_url"],
            "inventory": release["tasks_file"],
            "inspection": release["task_inventory_status"],
            "questions": [],
            "review": None,
        }
        # OOD source metadata is visible, but its question content is not ingested.
        if release["distribution"] != "ID" or not release["training_mapping_allowed"]:
            pages.append(page)
            continue
        inputs.append(release["tasks_file"])
        inventory = load(release["tasks_file"])
        review_file = "benchmark_competencies/" + Path(release["tasks_file"]).name
        annotations = {}
        if (SOURCE / review_file).exists():
            inputs.append(review_file)
            review = load(review_file)
            if review["benchmark"] != name or review["source_revision"] != inventory["source_revision"]:
                raise ValueError(f"Competency review source mismatch: {name}")
            annotations = {row["task_id"]: row for row in review["tasks"]}
            if len(annotations) != len(review["tasks"]) or set(annotations) != {
                row["task_id"] for row in inventory["tasks"]
            }:
                raise ValueError(f"Competency review must cover each question exactly once: {name}")
            competency_ids = {row["id"] for row in review["competencies"]}
            if len(competency_ids) != len(review["competencies"]):
                raise ValueError(f"Duplicate competency definitions: {name}")
            for row in inventory["tasks"]:
                annotation = annotations[row["task_id"]]
                labels = annotation["competencies"]
                if len(labels) != len(set(labels)) or not set(labels) <= competency_ids:
                    raise ValueError(f"Invalid competency labels: {name}/{row['task_id']}")
                eligible = annotation["verification_status"] == "numeric-contract"
                if bool(labels) != eligible:
                    raise ValueError(f"Only verifiable questions may have competency labels: {name}/{row['task_id']}")
                if annotation["source_question_sha256"] != row["source_question_sha256"]:
                    raise ValueError(f"Question changed since review: {name}/{row['task_id']}")
            page["review"] = {key: value for key, value in review.items() if key != "tasks"}
            page["review_file"] = review_file
        for task in inventory["tasks"]:
            family = task.get("workflow_family", "")
            expanded = {
                **inventory.get("task_defaults", {}),
                **inventory.get("patterns", {}).get(family, {}),
                **task,
            }
            page["questions"].append(
                {
                    "id": task["task_id"],
                    "family": family,
                    "summary": task.get("workflow_pattern", family.replace("-", " ")),
                    "source_url": task.get("source_metadata_url", release["source_url"]),
                    "source_capsule": task.get("source_capsule_id"),
                    "formats": expanded.get("required_formats", []),
                    "stages": expanded.get("required_stages", []),
                    "status": expanded.get("status", "unmapped"),
                    "recipes": expanded.get("recipes", []),
                    "annotation": annotations.get(task["task_id"]),
                }
            )
        pages.append(page)
    inventoried = {page["inventory"] for page in pages}
    for source in load("benchmark_sources.json")["sources"]:
        if source.get("task_inventory") in inventoried:
            continue
        pages.append(
            {
                "id": source["name"],
                "name": source["name"],
                "distribution": source["distribution"],
                "source_url": source["source_url"],
                "inventory": None,
                "inspection": source["inspection"],
                "questions": [],
                "review": None,
            }
        )
    return pages, inputs


def benchmark_markdown(review: dict) -> str:
    """Render the flat question review for editing alongside the explorer."""
    names = {row["id"]: row["name"] for row in review["competencies"]}
    included = [row for row in review["tasks"] if row["competencies"]]
    excluded = [row for row in review["tasks"] if not row["competencies"]]
    counts = sorted(
        (
            (competency, sum(competency["id"] in row["competencies"] for row in included))
            for competency in review["competencies"]
        ),
        key=lambda row: (-row[1], row[0]["name"]),
    )
    lines = [
        "# BixBench-Verified: flat competency review",
        "",
        f"Draft for review: {len(included)} of {len(review['tasks'])} source questions have "
        f"proposed executable checks and {len(review['competencies'])} provisional competencies. "
        "Task generation remains paused. No LLM judge is in scope.",
        "",
        "An **analysis competency** is a reusable scientific analysis with a checkable outcome. "
        "A **workflow recipe** connects competencies to answer a scientific question on observed data. "
        "Filters, covariates, model options and denominators belong in the question-specific contract "
        "unless they change the analysis being assessed. These boundaries are open for review.",
        "",
        "Use the same competency when two tasks require the same scientific analysis and a comparable "
        "output contract, even if species, tool or threshold changes. Split it when the scientific "
        "decision or required artifacts change substantially. A task can carry several competencies "
        "when its verifier checks the connected intermediate results.",
        "",
        "Each question may require several competencies; it is counted once per assigned label. "
        "The counts overlap and do not measure unique studies, independent workflows or validated generated tasks.",
        "",
        "[Versioned annotations, original questions and verification notes]"
        "(../../experiments/post_training/bio_tasks/benchmark_competencies/bixbench-verified-50.json) · "
        "[Source inventory](../../experiments/post_training/bio_tasks/benchmark_tasks/bixbench-verified-50.json) · "
        "[Licensed source dataset](https://huggingface.co/datasets/phylobio/BixBench-Verified-50)",
        "",
        "The naming follows [EDAM's separation of operations, topics, data and formats]"
        "(https://edamontology.org/). These are local draft labels, not official EDAM terms.",
        "",
        "## Ranked competencies",
        "",
        "| Competency | Questions | Checkable outcome |",
        "| --- | ---: | --- |",
    ]
    for competency, count in counts:
        lines.append(f"| {competency['name']} | {count} | {competency['outcome']} |")
    lines.extend(
        [
            "",
            "## Questions with proposed executable checks",
            "",
            "These are prospective verifier designs. No new Harbor validation is claimed.",
            "",
            "| Question | Short description | Competencies |",
            "| --- | --- | --- |",
        ]
    )
    for row in included:
        labels = "; ".join(names[key] for key in row["competencies"])
        lines.append(f"| {row['task_id']} | {row['question_summary']} | {labels} |")
    lines.extend(
        [
            "",
            "## Set aside for now",
            "",
            "These questions have no competency assignment or count while their complete endpoint lacks "
            "an executable contract.",
            "",
            "| Question | Short description | Reason |",
            "| --- | --- | --- |",
        ]
    )
    for row in excluded:
        lines.append(f"| {row['task_id']} | {row['question_summary']} | {row['decisions']} |")
    lines.extend(
        [
            "",
            "## Boundaries to review",
            "",
            "- Differential expression analysis is one competency here; shrinkage, design formula and "
            "filtering specify the task instance.",
            "- Phylogenetic tree metrics currently share one competency; we could split it if the metrics "
            "require distinct assessment contracts.",
            "- Spearman correlation and Mann-Whitney tests are counted as competencies. They might instead "
            "be cross-cutting statistical tags.",
            "- Eligibility and denominator choices are required verifier checks, not separate competencies "
            "in this draft.",
            "- A question requiring both differential expression and pathway enrichment counts toward "
            "both competencies. A generated task must validate the connected workflow.",
            "",
            "Rebuild this Markdown and the HTML with "
            "`uv run python -m experiments.post_training.bio_tasks.coverage_site`.",
        ]
    )
    return "\n".join(lines) + "\n"


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
    benchmarks, benchmark_inputs = benchmark_pages()
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
        "benchmark_pages.js",
        *benchmark_inputs,
    ]
    return {
        "version": curriculum.version,
        "github": GITHUB,
        "sections": sections,
        "recipes": recipes,
        "examples": examples,
        "repositories": repositories,
        "sources": sources,
        "benchmarks": benchmarks,
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
        .replace("__BENCHMARK_PAGES__", (SOURCE / "benchmark_pages.js").read_text())
    )
    markdown = benchmark_markdown(load("benchmark_competencies/bixbench-verified-50.json"))
    if args.check:
        if SITE.read_text() != rendered:
            raise ValueError("Coverage HTML is stale; rerun this module without --check")
        if BENCHMARK_REVIEW.read_text() != markdown:
            raise ValueError("Benchmark review Markdown is stale; rerun this module without --check")
    else:
        SITE.write_text(rendered)
        BENCHMARK_REVIEW.write_text(markdown)
    print(f"{'Checked' if args.check else 'Wrote'} {SITE.relative_to(REPO)} ({len(rendered):,} bytes)")


if __name__ == "__main__":
    main()
