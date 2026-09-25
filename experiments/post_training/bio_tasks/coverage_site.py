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
BENCHMARK_SITE = REPO / "docs/experiments/bio-benchmarks.html"
BENCHMARK_REVIEW = REPO / "docs/experiments/bixbench-verified-competencies.md"
CATEGORY_REVIEW = REPO / "docs/experiments/bio-benchmark-categories.md"
GITHUB = "https://github.com/marin-community/marin/blob/codex/bio-task-generators/"


def load(name: str) -> dict:
    return json.loads((SOURCE / name).read_text())


def load_benchmark_review(name: str) -> dict:
    """Load a review manifest and its bounded, editable question files."""
    review = load(name)
    review["tasks"] = [row for path in review["task_files"] for row in load(path)["tasks"]]
    if len(review["tasks"]) != review["task_count"]:
        raise ValueError(f"Review record count mismatch: {name}")
    return review


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
            review = load_benchmark_review(review_file)
            inputs.extend(review["task_files"])
            if review["schema_version"] != 2 or review["benchmark"] != name:
                raise ValueError(f"Competency review schema/source mismatch: {name}")
            if review["inventory_sha256"] != hashlib.sha256((SOURCE / release["tasks_file"]).read_bytes()).hexdigest():
                raise ValueError(f"Inventory changed since review: {name}")
            annotations = {row["task_id"]: row for row in review["tasks"]}
            if len(annotations) != len(review["tasks"]) or set(annotations) != {
                row["task_id"] for row in inventory["tasks"]
            }:
                raise ValueError(f"Competency review must cover each question exactly once: {name}")
            definitions = load("benchmark_competencies/taxonomy.json")
            for annotation in annotations.values():
                identity = f"{name}/{annotation['task_id']}"
                eligible = annotation["disposition"] == "reframe"
                if annotation["disposition"] not in {"reframe", "excluded", "unavailable"}:
                    raise ValueError(f"Unknown question disposition: {identity}")
                for facet in ("skills", "applications"):
                    labels = annotation[facet]
                    allowed = {row["id"] for row in definitions[facet]}
                    if (
                        len(labels) != len(set(labels))
                        or not set(labels) <= allowed
                        or (labels and not eligible)
                        or (eligible and facet == "skills" and not labels)
                    ):
                        raise ValueError(f"Invalid {facet} assignment: {identity}")
                if eligible and not all(
                    annotation[key] for key in ("source_question", "reframed_question", "outputs", "verification")
                ):
                    raise ValueError(f"Missing executable framing: {identity}")
                question = annotation["source_question"]
                if question and hashlib.sha256(question.encode()).hexdigest() != annotation["source_question_sha256"]:
                    raise ValueError(f"Question text/hash mismatch: {identity}")
                if not eligible and not annotation["reason"]:
                    raise ValueError(f"Missing exclusion/access reason: {identity}")
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
    """Render an editable view of the BixBench review and its independent facets."""
    github = GITHUB.replace("blob/codex/", "blob/codex%2F")
    taxonomy = load("benchmark_competencies/taxonomy.json")
    names = {row["id"]: row["name"] for row in taxonomy["skills"] + taxonomy["applications"]}
    included = [row for row in review["tasks"] if row["disposition"] == "reframe"]
    counts = sorted(
        ((skill, sum(skill["id"] in row["skills"] for row in included)) for skill in taxonomy["skills"]),
        key=lambda row: (-row[1], row[0]["name"]),
    )
    lines = [
        "# BixBench-Verified: skills and applications",
        "",
        f"Draft: {len(included)} of {len(review['tasks'])} questions have a proposed executable framing. "
        "This includes explicit adaptations of underspecified or interpretive source questions. "
        "These are authoring proposals, not newly validated Harbor tasks. No LLM judge is in scope.",
        "",
        "A **skill** is a reusable analysis operation with a checkable result. An **application** "
        "describes its biological setting. A **workflow** connects operations, scientific decisions "
        "and artifacts to answer a question. Keep these as independent fields: application is not "
        "a second level below skill, and not every possible skill/application pair is meaningful.",
        "",
        "Use workflow proposals to select the next tasks. Prefer missing combinations of required "
        "operations and decisions on observed data, with a credible executable check. Vary studies, "
        "contrasts, input stages and methods while retaining a scientific purpose. Do not allocate "
        "tasks merely in proportion to label counts or count incidental input terminology as work.",
        "",
        "Labels describe requirements, not demonstrated competence. An endpoint-only reward does "
        "not prove that each intermediate skill was exercised. Group related questions and overlapping "
        "benchmark releases before using frequencies as planning weights.",
        "",
        "[Annotations and exact original questions]"
        f"({github}experiments/post_training/bio_tasks/benchmark_competencies/bixbench-verified-50.json) · "
        "[Shared flat vocabulary]"
        f"({github}experiments/post_training/bio_tasks/benchmark_competencies/taxonomy.json)",
        "",
        "The facets borrow [EDAM's distinction between operations and topics]"
        "(https://edamontologydocs.readthedocs.io/en/latest/editors_guide.html). These draft labels "
        "are not official EDAM terms. Differential expression remains a recognizable specialized "
        "skill; normalization, transformation and dimensionality reduction can be separate skills "
        "when the solver actually performs them.",
        "",
        "## Skills ranked by eligible questions",
        "",
        "Counts overlap. Each source question counts at most once per skill.",
        "",
        "| Skill | Questions | Checkable outcome |",
        "| --- | ---: | --- |",
    ]
    for skill, count in counts:
        if count:
            lines.append(f"| {skill['name']} | {count} | {skill['outcome']} |")
    lines += ["", "## Question assignments", "", "| Question | Skills | Applications |", "| --- | --- | --- |"]
    for row in included:
        skills = "; ".join(names[key] for key in row["skills"])
        applications = "; ".join(names[key] for key in row["applications"])
        lines.append(f"| {row['task_id']} | {skills} | {applications} |")
    lines += ["", "## Original questions and proposed checks", ""]
    for row in review["tasks"]:
        lines += [f"### {row['task_id']}", "", row["source_question"], ""]
        if row["disposition"] != "reframe":
            lines += [row["reason"], ""]
            continue
        lines += ["**Proposed framing:** " + row["reframed_question"], "", "**Check:** " + row["verification"], ""]
        lines += ["- " + item for item in row["decisions"] + row["scope_changes"]]
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def benchmark_statistics(benchmarks: list[dict]) -> dict:
    """Count category requirements across eligible ID records and source releases."""
    id_benchmarks = [benchmark for benchmark in benchmarks if benchmark["distribution"] == "ID"]
    reviewed = [benchmark for benchmark in id_benchmarks if benchmark["review"]]
    eligible = {
        benchmark["id"]: [q["annotation"] for q in benchmark["questions"] if q["annotation"]["disposition"] == "reframe"]
        for benchmark in reviewed
    }
    taxonomy = load("benchmark_competencies/taxonomy.json")
    facets = {}
    for facet in ("skills", "applications"):
        counts = []
        for category in taxonomy[facet]:
            sources = []
            for name, questions in eligible.items():
                count = sum(category["id"] in question[facet] for question in questions)
                if count:
                    sources.append({"benchmark": name, "count": count, "eligible": len(questions)})
            counts.append(
                {
                    "id": category["id"],
                    "name": category["name"],
                    "count": sum(source["count"] for source in sources),
                    "benchmarks": len(sources),
                    "sources": sorted(sources, key=lambda row: (-row["count"], row["benchmark"])),
                }
            )
        facets[facet] = sorted(counts, key=lambda row: (-row["count"], row["name"]))
    return {
        "inventoried_releases": len(reviewed),
        "source_records": sum(len(benchmark["questions"]) for benchmark in reviewed),
        "eligible_records": sum(len(questions) for questions in eligible.values()),
        "unresolved_applications": sum(not q["applications"] for rows in eligible.values() for q in rows),
        "operation_reviewed_records": sum(bool(q.get("category_review")) for rows in eligible.values() for q in rows),
        "missing_inventories": [benchmark["id"] for benchmark in id_benchmarks if not benchmark["review"]],
        "facets": facets,
    }


def category_markdown(statistics: dict) -> str:
    """Render the global rankings with explicit denominators and access gaps."""
    total = statistics["eligible_records"]
    lines = [
        "# Categories across ID benchmarks",
        "",
        f"{total:,} candidate verifiable source records across {statistics['inventoried_releases']} "
        f"inventoried ID releases, out of {statistics['source_records']:,} inventoried records. "
        "Excluded questions and unavailable instructions contribute no category counts.",
        "",
        "Assignments are provisional requirements, not validated task coverage. Each question counts "
        "once per assigned category. Categories overlap; percentages need not sum to 100%. Releases "
        "are not deduplicated: overlapping benchmarks and shared protocols can inflate raw frequency. "
        "Release counts show breadth of representation, not the number of independent studies.",
        "",
        "Use these counts to find candidate workflows for review, alongside scientific decisions, "
        "available observed inputs and an executable reward. They are not generation quotas.",
        "",
        f"{statistics['operation_reviewed_records']:,} records have a source-protocol or output-contract "
        "review of operation boundaries. This is not an individual verifier audit. "
        f"{statistics['unresolved_applications']:,} records have no resolved biological application; "
        "they remain in the eligible denominator and skill counts.",
    ]
    for facet, title in (("skills", "Analytical skills"), ("applications", "Biological applications")):
        lines += [
            "",
            f"## {title}",
            "",
            "| Category | Question records | % of eligible records | ID releases |",
            "| --- | ---: | ---: | ---: |",
        ]
        for row in statistics["facets"][facet]:
            percentage = 100 * row["count"] / total if total else 0
            lines.append(f"| {row['name']} | {row['count']} | {percentage:.1f}% | {row['benchmarks']} |")
    lines += [
        "",
        "## Inventory gaps",
        "",
        "These ID sources have benchmark pages but no task manifest in this review. "
        "Their unknown task counts are not treated as zero skill demand.",
        "",
    ]
    lines += ["- " + name for name in statistics["missing_inventories"]]
    lines += ["", "Rebuild with `uv run python -m experiments.post_training.bio_tasks.coverage_site`.", ""]
    return "\n".join(lines)


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
        "benchmark_site.html",
        "benchmark_competencies/taxonomy.json",
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
        "review_taxonomy": load("benchmark_competencies/taxonomy.json"),
        "category_statistics": benchmark_statistics(benchmarks),
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
    data = site_data()
    payload = json.dumps(data, separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
    template = (SOURCE / "coverage_site.html").read_text()
    assert template.count("__BIO_DATA__") == 1
    rendered = (
        template.replace("__BIO_DATA__", payload)
        .replace("__D3_HIERARCHY__", (SOURCE / "vendor/d3-hierarchy-3.1.2.min.js").read_text())
        .replace("__D3_LICENSE__", (SOURCE / "vendor/d3-hierarchy-LICENSE").read_text())
        .replace("__BENCHMARK_PAGES__", (SOURCE / "benchmark_pages.js").read_text())
    )
    benchmark_data = {key: data[key] for key in ("github", "benchmarks", "review_taxonomy", "category_statistics")}
    benchmark_payload = json.dumps(benchmark_data, separators=(",", ":"), ensure_ascii=True).replace("<", "\\u003c")
    benchmark_rendered = (
        (SOURCE / "benchmark_site.html")
        .read_text()
        .replace("__BIO_DATA__", benchmark_payload)
        .replace("__BENCHMARK_PAGES__", (SOURCE / "benchmark_pages.js").read_text())
    )
    markdown = benchmark_markdown(load_benchmark_review("benchmark_competencies/bixbench-verified-50.json"))
    categories = category_markdown(data["category_statistics"])
    if args.check:
        if BENCHMARK_SITE.read_text() != benchmark_rendered:
            raise ValueError("Benchmark site is stale; rerun this module without --check")
        if CATEGORY_REVIEW.read_text() != categories:
            raise ValueError("Global category Markdown is stale; rerun this module without --check")
        if SITE.read_text() != rendered:
            raise ValueError("Coverage HTML is stale; rerun this module without --check")
        if BENCHMARK_REVIEW.read_text() != markdown:
            raise ValueError("Benchmark review Markdown is stale; rerun this module without --check")
    else:
        SITE.write_text(rendered)
        BENCHMARK_SITE.write_text(benchmark_rendered)
        BENCHMARK_REVIEW.write_text(markdown)
        CATEGORY_REVIEW.write_text(categories)
    for path, content in ((SITE, rendered), (BENCHMARK_SITE, benchmark_rendered)):
        print(f"{'Checked' if args.check else 'Wrote'} {path.relative_to(REPO)} ({len(content):,} bytes)")


if __name__ == "__main__":
    main()
