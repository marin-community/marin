# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# ///

"""Export the Zotero Data Mixture collection and its two-phase evidence ledger."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_ZOTERO_DATABASE = Path.home() / "Zotero" / "zotero.sqlite"
DEFAULT_ZOTERO_STORAGE = Path.home() / "Zotero" / "storage"
DEFAULT_THEORY = (
    Path.home()
    / "Library/CloudStorage/GoogleDrive-pinlinxu@stanford.edu/My Drive/Research/Marin/data_mixing_paper/theory.md"
)
DEFAULT_OUTPUT = Path(__file__).parent / "reference_outputs" / "two_phase_theory_literature_synthesis_20260726"


@dataclass(frozen=True)
class ZoteroItem:
    item_id: int
    item_key: str
    item_type: str
    title: str
    creators: str
    date: str
    doi: str
    url: str
    abstract: str
    attachment_paths: str


@dataclass(frozen=True)
class EvidenceAssessment:
    title: str
    review_basis: str
    mechanism: str
    exposure_control: str
    relationship_to_marin_estimand: str
    principal_result: str
    limiting_assumption_or_confounder: str
    implication_for_theory: str
    evidence_class: str


ASSESSMENTS = (
    EvidenceAssessment(
        title="Algorithmic Stability and Uniform Generalization",
        review_basis="Full PDF",
        mechanism="Mutual stability between the inferred hypothesis and a random training observation.",
        exposure_control="Not a schedule intervention.",
        relationship_to_marin_estimand="Boundary result: it studies generalization sensitivity, not fixed-aggregate phase order.",
        principal_result="Its stability notion is necessary and sufficient for uniform generalization over bounded parametric losses.",
        limiting_assumption_or_confounder=(
            "The observations are i.i.d. and the learning algorithm is explicitly assumed permutation-invariant, "
            "so training-example order is irrelevant by construction."
        ),
        implication_for_theory=(
            "Use stability to model finite-support sensitivity, endpoint variance, and effective observation size. "
            "Do not use it to infer the sign of a phase-order effect without a time-labelled extension."
        ),
        evidence_class="theoretical boundary",
    ),
    EvidenceAssessment(
        title="Maximize Your Data's Potential: Enhancing LLM Accuracy with Two-Phase Pretraining",
        review_basis="Full PDF",
        mechanism="Quality- and epoch-aware broad-first, high-quality-late two-phase schedule.",
        exposure_control="BASE-RO uses the same aggregate dataset weights in random order.",
        relationship_to_marin_estimand=(
            "Suggestive same-aggregate evidence that one selected phased policy can beat a random-order baseline."
        ),
        principal_result=(
            "The selected two-phase recipe reports a 3.4% relative average-accuracy improvement "
            "(1.74 percentage points) over BASE-RO at 8B/1T."
        ),
        limiting_assumption_or_confounder=(
            "The phase blends and boundary were selected through a multi-stage search; uncertainty and the globally optimal "
            "tied policy are not reported, and data timing remains tied to optimizer time."
        ),
        implication_for_theory=(
            "The response should permit a nonzero signed order channel, but the unreplicated result does not establish "
            "its magnitude or lower-bound the global two-phase versus optimized-tied class gap in Marin."
        ),
        evidence_class="direct selected-policy evidence",
    ),
    EvidenceAssessment(
        title="Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics",
        review_basis="Full PDF",
        mechanism="Linguistically ordered samples alter gradient noise and time spent in shared latent training phases.",
        exposure_control="Uses the same fixed Pile samples and compute while changing their order.",
        relationship_to_marin_estimand="Direct order intervention, but not an aggregate-preserving domain-mixture contrast.",
        principal_result=(
            "Direction matters in a reverse-order control; stability effects are strongest at 14M-160M and smaller by 410M-1B."
        ),
        limiting_assumption_or_confounder=(
            "The curriculum is a fine-grained sample ordering, whereas Marin controls 39 bucket weights in two coarse phases."
        ),
        implication_for_theory=(
            "Model phase value through optimizer-state- and scale-dependent dynamics rather than a scale-invariant late-data scalar."
        ),
        evidence_class="direct order mechanism",
    ),
    EvidenceAssessment(
        title="Midtraining Bridges Pretraining and Posttraining Distributions",
        review_basis="Full PDF",
        mechanism="Gradient alignment, posttraining gradient energy, and a time-dependent plasticity window.",
        exposure_control="Start time and mixture weight jointly change specialized-token exposure.",
        relationship_to_marin_estimand="Mechanistically relevant but not fixed-aggregate and evaluated after a future target phase.",
        principal_result="Timing and mixture weight interact; early introduction can outperform stronger late mixing.",
        limiting_assumption_or_confounder=(
            "The endpoint includes posttraining, and the specialized data is selected to bridge to that future target distribution."
        ),
        implication_for_theory=(
            "A phase-order state should depend on transported gradients and future objectives; effective exposure alone is insufficient."
        ),
        evidence_class="adjacent mechanistic evidence",
    ),
    EvidenceAssessment(
        title="On The Power of Curriculum Learning in Training Deep Networks",
        review_basis="Full PDF",
        mechanism="Curriculum changes optimization-landscape steepness while preserving an ideal curriculum's global optimum.",
        exposure_control="Presentation order and pacing change; exposure is not generally fixed.",
        relationship_to_marin_estimand="Clarifies finite-time trajectory value versus a distinct policy-class endpoint optimum.",
        principal_result="An ideal curriculum can accelerate optimization without changing the original objective's global optimum.",
        limiting_assumption_or_confounder="Image classification and idealized curriculum assumptions differ from LLM pretraining.",
        implication_for_theory=(
            "A finite-budget phase gain need not imply a different asymptotic optimum; retain explicit dependence on budget B."
        ),
        evidence_class="theoretical mechanism",
    ),
    EvidenceAssessment(
        title="Replaying pre-training data improves fine-tuning",
        review_basis="Full PDF and source audit",
        mechanism="Generic-data replay and earlier target exposure soften a pretraining-to-target distribution shift.",
        exposure_control="Figure 7 fixes target exposure across sampled schedules, but does not include the corresponding tied point.",
        relationship_to_marin_estimand="Closest small-model analogue, but it does not compare the best schedule to a tied policy.",
        principal_result="Some fixed-target-exposure schedules outperform other schedules for code, math, and instruction targets.",
        limiting_assumption_or_confounder=(
            "The endpoint is target adaptation; the documented schedule grid stops before the constant-mixture point."
        ),
        implication_for_theory=(
            "Separate aggregate exposure from shift cost and replay. A tied mixture already removes the abrupt stage discontinuity."
        ),
        evidence_class="near-direct incomplete grid",
    ),
    EvidenceAssessment(
        title="The Finetuner's Fallacy: When to Pretrain with Your Finetuning Data",
        review_basis="Full PDF",
        mechanism="Early specialized exposure stretches scarce-data utility and reduces later overfitting and forgetting.",
        exposure_control="Specialized pretraining adds exposure before a common finetuning stage.",
        relationship_to_marin_estimand="Supports early exposure, not a fixed-aggregate two-phase advantage.",
        principal_result="Specialized pretraining improves domain performance and preserves general capabilities after finetuning.",
        limiting_assumption_or_confounder="Aggregate specialized-token exposure and the post-finetuning endpoint both differ.",
        implication_for_theory=(
            "Repetition harm and retention are aggregate-dependent; 'high quality late' is not a universal signed rule."
        ),
        evidence_class="adjacent exposure evidence",
    ),
    EvidenceAssessment(
        title="Skill-it! A Data-Driven Skills Framework for Understanding and Training Language Models",
        review_basis="Metadata and paper abstract",
        mechanism="Directed prerequisite relations between skills determine a useful data order.",
        exposure_control="Total compute is fixed, but cumulative exposure by skill changes under the adaptive sampler.",
        relationship_to_marin_estimand="Supports structured order effects, not random fixed-aggregate phase contrast.",
        principal_result="Prerequisite ordering and adaptive skill mixtures improve data efficiency on synthetic and real tasks.",
        limiting_assumption_or_confounder="Requires identifiable skill dependencies and target-specific online adaptation.",
        implication_for_theory=(
            "Directed semantic structure is worth testing. The paper does not imply that Marin's 39-bucket order "
            "gradient is low rank; that remains a separate falsifiable hypothesis."
        ),
        evidence_class="adjacent structured-order evidence",
    ),
    EvidenceAssessment(
        title="Does your data spark joy? Performance gains from domain upsampling at the end of training",
        review_basis="Metadata and prior full-paper audit",
        mechanism="Late domain upsampling during an annealing segment.",
        exposure_control="Replacing the final segment changes aggregate source exposure.",
        relationship_to_marin_estimand="Production recipe evidence, not a fixed-aggregate order effect.",
        principal_result="Late upsampling improves selected downstream metrics for a 7B model trained on 1T tokens.",
        limiting_assumption_or_confounder="Data, terminal learning-rate position, and aggregate exposure move together.",
        implication_for_theory="Treat the learning-rate schedule and phase boundary as part of the condition, not as token exposure.",
        evidence_class="adjacent recipe evidence",
    ),
    EvidenceAssessment(
        title="Data Mixing Can Induce Phase Transitions in Knowledge Acquisition",
        review_basis="Metadata and paper abstract",
        mechanism="Mixture-dependent capability acquisition exhibits threshold-like transitions.",
        exposure_control="Static mixture weights and training scale vary.",
        relationship_to_marin_estimand="Supports nonlinear aggregate response and budget dependence, not chronology.",
        principal_result="Changing data composition can move the onset of knowledge acquisition across training.",
        limiting_assumption_or_confounder="The intervention is aggregate mixture, not a fixed-aggregate schedule.",
        implication_for_theory="Permit budget-dependent aggregate curvature and thresholds before attributing residuals to phase order.",
        evidence_class="aggregate mechanism",
    ),
    EvidenceAssessment(
        title="Scaling Data-Constrained Language Models",
        review_basis="Metadata and prior full-paper audit",
        mechanism="Repeated data has diminishing marginal value and eventually produces overfitting.",
        exposure_control="Repetition and data uniqueness vary at fixed-compute comparison points.",
        relationship_to_marin_estimand="Mechanism for exposure saturation and phase-dependent retention, not order identification.",
        principal_result="A few repeats can be nearly harmless before repeated-token value decays.",
        limiting_assumption_or_confounder="Does not hold phase order fixed while varying repetition.",
        implication_for_theory="Represent repetition through dimensionless epochs and keep its mean and variance effects distinct.",
        evidence_class="aggregate repetition mechanism",
    ),
    EvidenceAssessment(
        title="Scaling Laws for Mixture Pretraining Under Data Constraints",
        review_basis="Metadata and paper abstract",
        mechanism="Diminishing repeated-target utility plus regularization from generic data.",
        exposure_control="Target fraction, dataset size, compute, and model scale vary.",
        relationship_to_marin_estimand="Strong aggregate exposure law; no phase-order isolation.",
        principal_result="Optimal target repetition depends on target size, compute, and model scale.",
        limiting_assumption_or_confounder="Static two-source mixtures do not identify retention or chronology.",
        implication_for_theory="Use a fitted aggregate backbone with explicit data shortage and repetition before fitting an order residual.",
        evidence_class="aggregate repetition mechanism",
    ),
    EvidenceAssessment(
        title="DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining",
        review_basis="Metadata and prior full-paper audit",
        mechanism="Online domain reweighting minimizes worst-group excess loss relative to a reference model.",
        exposure_control="The cumulative aggregate mixture changes.",
        relationship_to_marin_estimand="Demonstrates large aggregate-mixture value, not phase-order value.",
        principal_result="Learned domain weights improve pretraining efficiency over standard static baselines.",
        limiting_assumption_or_confounder="No fixed aggregate and no explicit two-phase restriction.",
        implication_for_theory="Estimate aggregate value separately because it dominates the observed 39-bucket signal.",
        evidence_class="aggregate policy evidence",
    ),
    EvidenceAssessment(
        title="Aioli: A Unified Optimization Framework for Language Model Data Mixing",
        review_basis="Metadata and paper abstract",
        mechanism="Online estimation of a shared mixing law with dynamic domain allocation.",
        exposure_control="The cumulative aggregate mixture changes online.",
        relationship_to_marin_estimand="Useful warning about misspecified mixing laws, not direct phase-order evidence.",
        principal_result="Existing methods are inconsistent when their mixing-law parameters are inaccurate.",
        limiting_assumption_or_confounder="Dynamic allocations and target losses differ from Marin's fixed two-phase endpoint model.",
        implication_for_theory="Treat surrogate-law fidelity as falsifiable and do not infer mechanism from optimizer success alone.",
        evidence_class="methodological evidence",
    ),
    EvidenceAssessment(
        title="Bayesian Experimental Design: A Review",
        review_basis="Metadata and paper abstract",
        mechanism="Decision-theoretic selection of experiments under a prior and explicit utility.",
        exposure_control="Not a training-schedule study.",
        relationship_to_marin_estimand="Design framework for allocating a fixed swarm budget.",
        principal_result="Experimental design criteria can be unified through decision-theoretic utility.",
        limiting_assumption_or_confounder="Does not supply a data-mixing response law.",
        implication_for_theory=(
            "Allocate tied and contrast samples against explicit uncertainty in aggregate value, signed order, and curvature."
        ),
        evidence_class="experimental-design background",
    ),
    EvidenceAssessment(
        title="Conservative Q-Learning for Offline Reinforcement Learning",
        review_basis="Metadata and paper abstract",
        mechanism="Conservative value estimation under action-distribution shift.",
        exposure_control="Not a data-mixture schedule study.",
        relationship_to_marin_estimand="Analogy for optimization outside fit support, not an admissible mean surrogate.",
        principal_result="A conservative objective reduces offline-policy overestimation.",
        limiting_assumption_or_confounder="Bellman structure and logged action coverage have no direct analogue in endpoint-only mixture fits.",
        implication_for_theory=(
            "Keep deployment conservatism separate from evidence that the mechanistic response surface is correct."
        ),
        evidence_class="deployment analogy",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zotero-database", type=Path, default=DEFAULT_ZOTERO_DATABASE)
    parser.add_argument("--zotero-storage", type=Path, default=DEFAULT_ZOTERO_STORAGE)
    parser.add_argument("--theory", type=Path, default=DEFAULT_THEORY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def database_connection(database: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{database}?immutable=1", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def collection_descendants(connection: sqlite3.Connection, collection_id: int) -> list[int]:
    rows = connection.execute(
        """
        WITH RECURSIVE descendants(collectionID) AS (
            SELECT ?
            UNION ALL
            SELECT collections.collectionID
            FROM collections
            JOIN descendants ON collections.parentCollectionID = descendants.collectionID
        )
        SELECT collectionID FROM descendants
        """,
        (collection_id,),
    ).fetchall()
    return [int(row["collectionID"]) for row in rows]


def populated_collection(connection: sqlite3.Connection, collection_name: str) -> tuple[int, list[int]]:
    candidates = connection.execute(
        "SELECT collectionID FROM collections WHERE collectionName = ?",
        (collection_name,),
    ).fetchall()
    if not candidates:
        raise ValueError(f"Zotero collection not found: {collection_name}")

    scored: list[tuple[int, int, list[int]]] = []
    for candidate in candidates:
        collection_id = int(candidate["collectionID"])
        descendants = collection_descendants(connection, collection_id)
        placeholders = ",".join("?" for _ in descendants)
        count = int(
            connection.execute(
                f"SELECT COUNT(DISTINCT itemID) AS n FROM collectionItems WHERE collectionID IN ({placeholders})",
                descendants,
            ).fetchone()["n"]
        )
        scored.append((count, collection_id, descendants))
    count, collection_id, descendants = max(scored)
    if count == 0:
        raise ValueError(f"All Zotero collections named {collection_name!r} are empty")
    return collection_id, descendants


def item_fields(connection: sqlite3.Connection, item_ids: list[int]) -> dict[int, dict[str, str]]:
    placeholders = ",".join("?" for _ in item_ids)
    rows = connection.execute(
        f"""
        SELECT itemData.itemID, fieldsCombined.fieldName, itemDataValues.value
        FROM itemData
        JOIN fieldsCombined ON fieldsCombined.fieldID = itemData.fieldID
        JOIN itemDataValues ON itemDataValues.valueID = itemData.valueID
        WHERE itemData.itemID IN ({placeholders})
        """,
        item_ids,
    ).fetchall()
    fields_by_item: dict[int, dict[str, str]] = {item_id: {} for item_id in item_ids}
    for row in rows:
        fields_by_item[int(row["itemID"])][str(row["fieldName"])] = str(row["value"])
    return fields_by_item


def item_creators(connection: sqlite3.Connection, item_ids: list[int]) -> dict[int, str]:
    placeholders = ",".join("?" for _ in item_ids)
    rows = connection.execute(
        f"""
        SELECT itemCreators.itemID, creators.firstName, creators.lastName
        FROM itemCreators
        JOIN creators ON creators.creatorID = itemCreators.creatorID
        WHERE itemCreators.itemID IN ({placeholders})
        ORDER BY itemCreators.itemID, itemCreators.orderIndex
        """,
        item_ids,
    ).fetchall()
    names_by_item: dict[int, list[str]] = {item_id: [] for item_id in item_ids}
    for row in rows:
        name = " ".join(part for part in (row["firstName"], row["lastName"]) if part)
        names_by_item[int(row["itemID"])].append(name)
    return {item_id: "; ".join(names) for item_id, names in names_by_item.items()}


def item_attachments(
    connection: sqlite3.Connection,
    item_ids: list[int],
    storage_root: Path,
) -> dict[int, str]:
    placeholders = ",".join("?" for _ in item_ids)
    rows = connection.execute(
        f"""
        SELECT itemAttachments.parentItemID, itemAttachments.path, items.key
        FROM itemAttachments
        JOIN items ON items.itemID = itemAttachments.itemID
        WHERE itemAttachments.parentItemID IN ({placeholders})
        ORDER BY itemAttachments.parentItemID, itemAttachments.itemID
        """,
        item_ids,
    ).fetchall()
    paths_by_item: dict[int, list[str]] = {item_id: [] for item_id in item_ids}
    for row in rows:
        stored_path = str(row["path"] or "")
        if stored_path.startswith("storage:"):
            resolved = storage_root / str(row["key"]) / stored_path.removeprefix("storage:")
        else:
            resolved = Path(stored_path)
        paths_by_item[int(row["parentItemID"])].append(str(resolved))
    return {item_id: "; ".join(paths) for item_id, paths in paths_by_item.items()}


def load_items(
    connection: sqlite3.Connection,
    collection_ids: list[int],
    storage_root: Path,
) -> list[ZoteroItem]:
    placeholders = ",".join("?" for _ in collection_ids)
    rows = connection.execute(
        f"""
        SELECT DISTINCT items.itemID, items.key, itemTypes.typeName
        FROM collectionItems
        JOIN items ON items.itemID = collectionItems.itemID
        JOIN itemTypes ON itemTypes.itemTypeID = items.itemTypeID
        WHERE collectionItems.collectionID IN ({placeholders})
        """,
        collection_ids,
    ).fetchall()
    item_ids = [int(row["itemID"]) for row in rows]
    metadata = item_fields(connection, item_ids)
    creators = item_creators(connection, item_ids)
    attachments = item_attachments(connection, item_ids, storage_root)
    types = {int(row["itemID"]): str(row["typeName"]) for row in rows}
    keys = {int(row["itemID"]): str(row["key"]) for row in rows}

    items = [
        ZoteroItem(
            item_id=item_id,
            item_key=keys[item_id],
            item_type=types[item_id],
            title=metadata[item_id].get("title", ""),
            creators=creators[item_id],
            date=metadata[item_id].get("date", ""),
            doi=metadata[item_id].get("DOI", ""),
            url=metadata[item_id].get("url", ""),
            abstract=metadata[item_id].get("abstractNote", ""),
            attachment_paths=attachments[item_id],
        )
        for item_id in item_ids
    ]
    return sorted(items, key=lambda item: (item.title.casefold(), item.item_id))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with database_connection(args.zotero_database) as connection:
        collection_id, collection_ids = populated_collection(connection, "Data Mixture")
        items = load_items(connection, collection_ids, args.zotero_storage)

    by_title: dict[str, ZoteroItem] = {}
    for item in items:
        by_title.setdefault(item.title, item)
    missing = [assessment.title for assessment in ASSESSMENTS if assessment.title not in by_title]
    if missing:
        raise ValueError(f"Curated assessments missing from Zotero collection: {missing}")

    write_csv(args.output_dir / "zotero_data_mixture_inventory.csv", [asdict(item) for item in items])
    ledger_rows = []
    for assessment in ASSESSMENTS:
        item = by_title[assessment.title]
        ledger_rows.append(
            {
                **asdict(assessment),
                "zotero_item_id": item.item_id,
                "zotero_item_key": item.item_key,
                "date": item.date,
                "doi": item.doi,
                "url": item.url,
                "attachment_paths": item.attachment_paths,
            }
        )
    write_csv(args.output_dir / "two_phase_evidence_ledger.csv", ledger_rows)
    shutil.copyfile(args.theory, args.output_dir / "theory_snapshot.md")

    summary = {
        "selected_collection_id": collection_id,
        "collection_ids": collection_ids,
        "inventory_rows": len(items),
        "unique_titles": len(by_title),
        "curated_evidence_rows": len(ledger_rows),
        "items_with_local_attachments": sum(bool(item.attachment_paths) for item in items),
        "theory_source": str(args.theory),
        "zotero_database": str(args.zotero_database),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
