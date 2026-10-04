# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the frozen champion and parent on the separate coding panel."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, dataclass

from finestore.eval import sample_from_archive_row
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext, artifact_identity
from rigging.filesystem.storage_path import StoragePath, prefix_join

from experiments.evaluation.pipeline import EvaluationResult
from experiments.post_training.mismatch_probe.metrics import prompt_cluster_bootstrap
from experiments.post_training.russell_rsi.bootstrap_loop import pinned_file
from experiments.post_training.russell_rsi.coding_eval_feedback import (
    CODING_SUITES,
    CodingEvidenceConfig,
    CodingEvidenceRow,
    CodingPanel,
    PanelItem,
    coding_evidence_rows,
    digest,
    load_coding_archives,
    protocol_digest,
)

BOOTSTRAP_SEED = 9528
BOOTSTRAP_DRAWS = 10_000
PANEL_ITEMS_PER_SUITE = 32
FINAL_ITEMS_PER_SUITE = 2 * PANEL_ITEMS_PER_SUITE


@dataclass(frozen=True)
class HeldoutComparisonConfig:
    parent: CodingEvidenceConfig
    champion: CodingEvidenceConfig
    manifest_uri: str
    manifest_sha256: str
    terminal_state: dict
    output_path: str


def heldout_panel_ids(working_panel: CodingPanel, manifest: dict) -> set[tuple[str, str]]:
    """Validate the complete panel before the controller allocates training GPUs."""
    suites = set(CODING_SUITES)
    working = {(item.suite, item.benchmark_id) for item in working_panel.items}
    if (
        set(working_panel.protocols) != suites
        or len(working) != len(working_panel.items)
        or any(sum(key[0] == suite for key in working) != PANEL_ITEMS_PER_SUITE for suite in suites)
        or {key[0] for key in working} != suites
        or set(manifest["suites"]) != suites
    ):
        raise ValueError("Freeze exactly 32 unique working items per coding suite")
    heldout = set()
    for suite, values in manifest["suites"].items():
        identifiers = [str(value) for value in values["sample_ids"]]
        if len(identifiers) != PANEL_ITEMS_PER_SUITE or len(set(identifiers)) != PANEL_ITEMS_PER_SUITE:
            raise ValueError("Freeze exactly 32 unique held-out items per coding suite")
        if set(identifiers) & {str(value) for value in values.get("demonstration_ids", [])}:
            raise ValueError("Held-out tasks cannot be prompt demonstrations")
        heldout.update((suite, identifier) for identifier in identifiers)
    if working & heldout:
        raise ValueError("The held-out and working panels must be disjoint")
    return heldout


def heldout_rows(
    records: tuple[dict, ...],
    archives: tuple[list[dict], ...],
    working_panel: CodingPanel,
    manifest: dict,
) -> tuple[tuple[list[dict], ...], CodingPanel]:
    """Validate the full final archives before selecting the held-out rows."""
    working = {(item.suite, item.benchmark_id): item.prompt_sha256 for item in working_panel.items}
    heldout = heldout_panel_ids(working_panel, manifest)
    expected = set(working) | heldout
    selected, items, protocols = [], [], {}
    seen: set[tuple[str, str]] = set()
    for record, rows in zip(records, archives, strict=True):
        suite = record["eval"]["name"]
        normalized = deepcopy(record)
        evaluation = normalized["eval"]
        if evaluation["evalchemy"]["max_eval_instances"] != FINAL_ITEMS_PER_SUITE:
            raise ValueError("The final evaluation must run the frozen 64-item union per suite")
        evaluation["evalchemy"]["max_eval_instances"] = PANEL_ITEMS_PER_SUITE
        for task in evaluation["tasks"]:
            if task["benchmark"]["n_attempted"] != FINAL_ITEMS_PER_SUITE:
                raise ValueError("The final evaluation did not attempt the complete union")
            task["benchmark"]["n_attempted"] = PANEL_ITEMS_PER_SUITE
        if protocol_digest(normalized) != working_panel.protocols[suite]:
            raise ValueError("Final evaluation changed more than the declared sample limit")
        if not record["provenance"]["eval_runtime"].endswith("@" + manifest["evalchemy_commit"]):
            raise ValueError("Final evaluator differs from the pinned dataset source revision")
        protocols[suite] = protocol_digest(record)
        subset = []
        for row in rows:
            sample = sample_from_archive_row(row)
            key = (suite, str(json.loads(sample.doc)["task_id"]))
            if key not in expected or key in seen:
                raise ValueError("Final archives contain an unknown or duplicate task")
            if sample.task != suite or row.get("filter") != "none" or row.get("trial_id") != "":
                raise ValueError("Final archives changed the task, filter, or trial")
            if sample.prompt_text is None or sample.output is None or sample.metrics.get("pass_rate") not in (0.0, 1.0):
                raise ValueError("Final archives contain an incomplete measurement")
            if sample.grading is not None and sample.grading.score is None:
                raise ValueError("Final archives contain an infrastructure grading failure")
            prompt_hash = hashlib.sha256(sample.prompt_text.encode()).hexdigest()
            if key in working and prompt_hash != working[key]:
                raise ValueError("Final evaluation changed a frozen working prompt")
            seen.add(key)
            if key in heldout:
                items.append(PanelItem(suite, key[1], prompt_hash))
                subset.append(row)
        selected.append(subset)
    if seen != expected:
        raise ValueError("Final archives do not contain the complete frozen task union")
    return tuple(selected), CodingPanel(tuple(items), protocols)


def paired_scores(parent: tuple[CodingEvidenceRow, ...], champion: tuple[CodingEvidenceRow, ...]) -> dict[str, dict]:
    """Report paired binary outcomes without treating missing results as failures."""
    left = {(row.suite, row.benchmark_id): row.pass_rate for row in parent}
    right = {(row.suite, row.benchmark_id): row.pass_rate for row in champion}
    if left.keys() != right.keys():
        raise ValueError("The compared checkpoints have different measured tasks")
    results = {}
    for suite in CODING_SUITES:
        keys = sorted(key for key in left if key[0] == suite)
        differences = [right[key] - left[key] for key in keys]

        def calculate(indices: list[int], *, values: tuple[float, ...] = tuple(differences)) -> dict[str, float | int]:
            return {"difference_pp": 100 * sum(values[index] for index in indices) / len(indices)}

        bootstrap = prompt_cluster_bootstrap(
            [identifier for _, identifier in keys], calculate, seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS
        )
        results[suite] = {
            "count": len(keys),
            "parent_correct": sum(left[key] for key in keys),
            "champion_correct": sum(right[key] for key in keys),
            "difference_pp": bootstrap.point["difference_pp"],
            "ci95_pp": bootstrap.intervals["difference_pp"],
            "gains": sum(left[key] == 0 and right[key] == 1 for key in keys),
            "losses": sum(left[key] == 1 and right[key] == 0 for key in keys),
            "both_correct": sum(left[key] == right[key] == 1 for key in keys),
            "both_incorrect": sum(left[key] == right[key] == 0 for key in keys),
            "outcomes": [{"benchmark_id": key[1], "parent": left[key], "champion": right[key]} for key in keys],
        }
    return results


def compare_heldout_evaluations(config: HeldoutComparisonConfig) -> None:
    """Write the final comparison after the controller freezes its champion."""
    manifest = json.loads(pinned_file(config.manifest_uri, config.manifest_sha256))
    parent_records, parent_archives = load_coding_archives(config.parent)
    parent_selected, panel = heldout_rows(parent_records, parent_archives, config.parent.panel, manifest)
    parent = coding_evidence_rows(parent_records, parent_selected, config.parent.model_identity, panel)
    same_checkpoint = config.parent.model_identity == config.champion.model_identity
    if same_checkpoint:
        if config.parent != config.champion:
            raise ValueError("An unchanged champion must reuse the parent evaluation result")
        champion_records, champion = parent_records, parent
    else:
        champion_records, champion_archives = load_coding_archives(config.champion)
        champion_selected, _ = heldout_rows(champion_records, champion_archives, config.parent.panel, manifest)
        champion = coding_evidence_rows(champion_records, champion_selected, config.champion.model_identity, panel)
    report = {
        "status": "completed",
        "kind": "held_out_id_coding_comparison",
        "no_checkpoint_promoted": same_checkpoint,
        "manifest": manifest,
        "manifest_sha256": config.manifest_sha256,
        "terminal_state": config.terminal_state,
        "terminal_state_sha256": digest(config.terminal_state),
        "parent_identity": config.parent.model_identity,
        "champion_identity": config.champion.model_identity,
        "inputs": {"parent": asdict(config.parent), "champion": asdict(config.champion)},
        "records_sha256": {
            "parent": [digest(record) for record in parent_records],
            "champion": [digest(record) for record in champion_records],
        },
        "measured_rows_sha256": {
            "parent": [row.source_sha256 for row in parent],
            "champion": [row.source_sha256 for row in champion],
        },
        "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "unit": "paired_benchmark_id"},
        "scores": paired_scores(parent, champion),
        "limits": (
            "Intervals describe task variation in this small fixed panel. They do not measure training-seed variation. "
            "Dataset byte hashes come from the frozen source audit. "
            "Final run records identify the pinned evaluator revision."
        ),
    }
    StoragePath(prefix_join(config.output_path, "comparison.json")).write_text(json.dumps(report) + "\n")


def heldout_comparison_step(
    parent_evaluation: ArtifactStep[EvaluationResult],
    champion_evaluation: ArtifactStep[EvaluationResult],
    *,
    parent_identity: str,
    champion_identity: str,
    working_panel: CodingPanel,
    manifest_uri: str,
    manifest_sha256: str,
    terminal_state: dict,
    version: str,
) -> ArtifactStep[Artifact]:
    """Bind the final comparison to completed evaluator records and terminal state."""
    identities = (artifact_identity(parent_evaluation), artifact_identity(champion_evaluation))

    def build_config(ctx: StepContext) -> HeldoutComparisonConfig | dict:
        if ctx.is_fingerprint:
            return {
                "evaluations": identities,
                "models": (parent_identity, champion_identity),
                "working_panel": asdict(working_panel),
                "manifest_sha256": manifest_sha256,
                "terminal_state": terminal_state,
            }

        def evidence(step: ArtifactStep[EvaluationResult], identity: str) -> CodingEvidenceConfig:
            result = ctx.resolved(step)
            return CodingEvidenceConfig(
                result.records_prefix, result.run_ids, result.results_paths, identity, working_panel, ctx.output_path
            )

        return HeldoutComparisonConfig(
            evidence(parent_evaluation, parent_identity),
            evidence(champion_evaluation, champion_identity),
            manifest_uri,
            manifest_sha256,
            terminal_state,
            ctx.output_path,
        )

    return ArtifactStep(
        name="documents/russell-rsi-final-heldout-comparison",
        version=version,
        artifact_type=Artifact,
        deps=(parent_evaluation,) if identities[0] == identities[1] else (parent_evaluation, champion_evaluation),
        build_config=build_config,
        run=compare_heldout_evaluations,
    )
