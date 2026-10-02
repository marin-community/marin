# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Expand heterogeneous selections using published component metadata."""

import math
import re
from dataclasses import dataclass
from typing import Any

from .nemotron_counts import NEMOTRON_COUNTS
from .nemotron_records import record_source

NEMOTRON = "nvidia/Nemotron-RL-Ultra-Training-Blends"
NEMOTRON_ENV = "nemotron_ultra"
HH_RLHF = "Anthropic/hh-rlhf"
KTO_MIX = "trl-lib/kto-mix-14k"


def canonical_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore registry selections before refreshing their component inventory."""
    parents = {}
    for saved in rows:
        row = dict(saved)
        parent_id = row.get("canonical_id", row["id"])
        if parent_id in parents:
            continue
        if row.get("component_name"):
            row.update(
                id=parent_id,
                name=row["registry_name"],
                display_name=row["canonical_source"],
                url=row["canonical_url"],
                task_count=row["canonical_task_count"],
                turns="Mixed",
            )
            for key in (
                "component_name",
                "component_ratio",
                "canonical_id",
                "registry_name",
                "canonical_task_count",
                "component_selector",
                "component_file_sha256",
            ):
                row.pop(key, None)
        parents[parent_id] = row
    return list(parents.values())


@dataclass(frozen=True)
class ComponentClassification:
    task_type: str
    interaction: str
    family: str


def component_classification(dataset_id: str) -> ComponentClassification:
    """Return type, interaction capability, and family for audited blend components."""
    name = dataset_id.lower()
    if "swe" in name:
        return ComponentClassification("Agentic", "Multi-turn", "swe-repo")
    if "agentic" in name:
        return ComponentClassification("Agentic", "Multi-turn", "tool-use")
    if "rlhf" in name or "safety" in name:
        return ComponentClassification("Alignment", "Single-turn", "preference" if "rlhf" in name else "safety")
    if "multiturnchat" in name:
        return ComponentClassification("RLVR", "Multi-turn", "instruction-following")
    if "instruction" in name or "litmus" in name:
        return ComponentClassification("RLVR", "Single-turn", "instruction-following")
    if "math" in name:
        return ComponentClassification("RLVR", "Single-turn", "math-proof" if "proof" in name else "math-answer")
    if "coding" in name:
        return ComponentClassification("RLVR", "Single-turn", "competitive-programming")
    if "arc-agi" in name:
        return ComponentClassification("RLVR", "Single-turn", "arc-agi")
    if "reasoninggym" in name:
        return ComponentClassification("RLVR", "Single-turn", "reasoning-gym")
    if "science" in name or "mcqa" in name:
        return ComponentClassification("RLVR", "Single-turn", "qa-multiple-choice")
    if "abstention" in name:
        return ComponentClassification("RLVR", "Single-turn", "qa-abstention")
    raise ValueError(f"Unreviewed Nemotron component: {dataset_id}")


def child_row(parent: dict[str, Any], name: str, count: int | None) -> dict[str, Any]:
    row = dict(parent)
    row.update(
        id=f"{parent['id']}/{name}",
        name=f"{parent['name']}/{name}",
        registry_name=parent["name"],
        canonical_id=parent["id"],
        canonical_source=parent["display_name"],
        canonical_url=parent["url"],
        canonical_task_count=parent["task_count"],
        component_name=name,
        task_count=count,
        difficulty=None,
        quality=None,
        traces=None,
    )
    return row


def nemotron_components(parent: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    blend = parent["name"].removeprefix("nemotron_ultra_")
    audit = NEMOTRON_COUNTS["blends"].get(blend)
    if audit and (
        parent.get("dataset_revision") == NEMOTRON_COUNTS["revision"]
        or info.get("file_sha256", {}).get(f"{blend}.jsonl") == audit["sha256"]
    ):
        return counted_nemotron_components(parent, audit)
    if info.get("metadata_error"):
        return [parent]
    section = re.search(rf"^### {re.escape(blend)}\s*\n(.*?)(?=^##|\Z)", info["card_text"], re.MULTILINE | re.DOTALL)
    if section is None:
        raise ValueError(f"Nemotron card has no composition for {blend}")
    groups = []
    for line in section[1].splitlines():
        ratio = re.search(r"\|\s*([\d.]+)%\s*\|", line)
        datasets = re.findall(r"https://huggingface.co/datasets/([^\s)]+)", line)
        if ratio and datasets:
            groups.append((datasets, float(ratio[1])))
    total_ratio = sum(ratio for _, ratio in groups)
    if not groups or abs(total_ratio - 100) > 0.1:
        raise ValueError(f"Nemotron {blend} composition does not cover the complete blend: {total_ratio}%")
    total = parent["task_count"]
    if total is None:
        raise ValueError(f"Nemotron {blend} total count is unavailable")
    # Rounded percentages are estimates. Allocate rounding residue so a complete
    # breakdown preserves the selected population's total, never its raw ratio sum.
    expected = [total * ratio / total_ratio for _, ratio in groups]
    counts = [math.floor(value) for value in expected]
    order = sorted(range(len(groups)), key=lambda index: expected[index] - counts[index], reverse=True)
    for index in order[: total - sum(counts)]:
        counts[index] += 1
    rows = []
    for (datasets, ratio), count in zip(groups, counts, strict=True):
        for dataset in datasets:
            row = child_row(parent, dataset, count if len(datasets) == 1 else None)
            classification = component_classification(dataset)
            row.update(
                display_name=f"{dataset} · {blend}",
                url=f"https://huggingface.co/datasets/{dataset}",
                type=classification.task_type,
                turns=classification.interaction,
                family=classification.family,
                component_ratio=f"{ratio:g}%" + (" combined SWE share" if len(datasets) > 1 else ""),
                count_precision="estimated" if len(datasets) == 1 else "unknown",
                count_basis=(
                    f"Estimated blend contribution from rounded card share {ratio:g}%; normalized rounding"
                    if len(datasets) == 1
                    else f"Card combines {' + '.join(datasets)}: {ratio:g}% (approximately {count:,} rows total); "
                    "individual contributions are not published"
                ),
                count_url=info["card_url"],
                family_basis="Component dataset and Nemotron composition card audited 2026-09-28",
                family_url=info["card_url"],
                classification_basis="Component reward/task contract; Agentic components use Multi-turn",
                notes="Count describes this component's contribution to the selected blend.",
            )
            rows.append(row)
    return rows


def component_rows(parent: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    """Expand mixed selections once, retaining their canonical source and provenance."""
    if parent["dataset_id"] == NEMOTRON:
        return nemotron_components(parent, info)
    if parent["dataset_id"] not in {HH_RLHF, KTO_MIX}:
        return [parent]
    counts = info["components"]
    if sum(counts.values()) != parent["task_count"]:
        raise ValueError(f"Component counts disagree with {parent['dataset_id']} selected population")
    rows = []
    for name, count in counts.items():
        row = child_row(parent, name, count)
        hh = parent["dataset_id"] == HH_RLHF
        row.update(
            display_name=f"{parent['dataset_id']} · {name}" if hh else f"{name} · KTO",
            url=(
                f"{parent['url']}/tree/{parent['dataset_revision']}/{name}"
                if hh
                else f"https://huggingface.co/datasets/{name}"
            ),
            turns="Multi-turn" if hh or "capybara" in name else "Single-turn",
            type="Alignment",
            family="preference",
            count_precision="exact",
            count_basis=(
                "Named HH collection train rows, corroborated by tasksource mirror metadata"
                if hh
                else "Argilla DPO mix train source frequencies x 2: one positive and one negative KTO record per pair"
            ),
            count_url=info["composition_url"],
            classification_basis=(
                "HH/Capybara collections support Multi-turn; " "Orca/UltraFeedback use Single-turn prompts"
            ),
            notes=(
                "Interaction describes supported conversation structure. "
                "Conversational collections can include one-turn examples."
            ),
        )
        rows.append(row)
    return rows


def counted_nemotron_components(parent: dict[str, Any], audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Return Nemotron component rows with audited task counts."""
    if audit["total"] != parent["task_count"]:
        raise ValueError("Nemotron record audit disagrees with selected blend total")
    blend = parent["name"].removeprefix("nemotron_ultra_")
    rows = []
    for group in audit["groups"]:
        source = record_source(group["dataset"], group["swe_source"])
        key = group["dataset"] + ("/" + group["swe_source"] if group["swe_source"] else "")
        row = child_row(parent, key, group["count"])
        selection = (" · " + source.selection) if source.selection else ""
        row.update(
            display_name=f"{source.repository}{selection} · {blend}",
            url=f"https://huggingface.co/datasets/{source.repository}",
            type=source.task_type,
            turns=source.interaction,
            family=source.family,
            component_selector=group["dataset"],
            component_ratio=f"{100 * group['count'] / audit['total']:.4f}%",
            component_file_sha256=audit["sha256"],
            count_precision="exact",
            count_basis="Every record in the pinned original blend JSONL was counted once by dataset selection. "
            + (NEMOTRON_COUNTS["swe_attribution"] if group["swe_source"] else ""),
            count_url=f"{parent['url']}/blob/{parent['dataset_revision']}/{blend}.jsonl",
            family_basis="Actual record dataset/agent contract and component card, audited 2026-09-28",
            family_url=parent["family_url"],
            classification_basis="Actual selected record population; Agentic tasks use Multi-turn",
            notes="Complete-file count of this component selection; not the original component repository size.",
        )
        rows.append(row)
    if sum(row["task_count"] for row in rows) != audit["total"]:
        raise ValueError("Nemotron record audit component counts disagree with blend total")
    return rows
