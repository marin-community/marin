# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "tabulate>=0.9"]
# ///
"""Audit the immutable 710-row heldout archive and adversarial provenance."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
REFERENCE_ROOT = TWO_PHASE_ROOT / "reference_outputs"
OUTPUT_ROOT = REFERENCE_ROOT / "mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round58_heldout_provenance"
HELDOUT = REFERENCE_ROOT / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
PREDICTIONS = REFERENCE_ROOT / "delphi_3e18_adversarial_generalization_20260718" / "heldout_predictions.csv"
ADVERSARIAL_MANIFEST = REFERENCE_ROOT / "delphi_3e18_adversarial_stress_panel_20260716" / "candidate_manifest.csv"

NEW_ONE_PHASE_SERIES = "delphi_one_phase_augmented_swarm_3e18_20260715"
ADVERSARIAL_SERIES = "delphi_3e18_adversarial_stress_panel_20260716"


def tag_dict(raw: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for tag in json.loads(raw):
        if "=" not in tag:
            continue
        key, value = tag.split("=", maxsplit=1)
        result[key] = value
    return result


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = pd.read_csv(HELDOUT)
    predictions = pd.read_csv(PREDICTIONS)
    adversarial_manifest = pd.read_csv(ADVERSARIAL_MANIFEST)

    if all_rows["heldout_id"].duplicated().any():
        raise ValueError("Heldout IDs must be unique")
    overlap_counts = all_rows["fit_panel_overlap"].value_counts().to_dict()
    if overlap_counts != {"coordinate_disjoint": 710, "exact_coordinate": 12}:
        raise ValueError(f"Unexpected fit-panel overlap counts: {overlap_counts}")
    aliases = all_rows.loc[all_rows["fit_panel_overlap"].eq("exact_coordinate")].copy()
    heldout = all_rows.loc[all_rows["fit_panel_overlap"].eq("coordinate_disjoint")].copy()
    if aliases["fit_panel_run_name"].isna().any() or aliases["fit_panel_run_name"].astype(str).str.strip().eq("").any():
        raise ValueError("Every excluded coordinate alias must identify its fit-panel run")
    if not heldout["training_state"].eq("finished").all():
        raise ValueError("Heldout archive contains unfinished training")
    if not heldout["checkpoint_declared_complete"].eq(1).all():
        raise ValueError("Heldout archive contains an incomplete checkpoint")
    for target in ("uncheatable_bpb", "table9_macro_bpb"):
        values = pd.to_numeric(heldout[target], errors="coerce")
        if values.isna().any():
            raise ValueError(f"Heldout archive has missing {target}")

    coordinate_counts = heldout.groupby("mixture_sha256").size()
    repeats = heldout.loc[heldout["mixture_sha256"].isin(coordinate_counts[coordinate_counts.gt(1)].index)].copy()
    repeat_groups = (
        repeats.groupby("mixture_sha256")
        .agg(
            row_count=("heldout_id", "size"),
            distinct_data_seeds=("data_seed", "nunique"),
            run_names=("wandb_run_name", lambda values: "|".join(sorted(values))),
        )
        .reset_index()
        .sort_values(["row_count", "mixture_sha256"], ascending=[False, True])
    )
    repeat_groups["repeat_kind"] = "independent_seed_repeat"
    repeat_groups.loc[repeat_groups["distinct_data_seeds"].lt(repeat_groups["row_count"]), "repeat_kind"] = (
        "mixed_seed_and_launcher_duplicate"
    )
    repeat_groups.to_csv(ROUND_DIR / "coordinate_repeat_groups.csv", index=False)

    split = pd.Series("historical", index=heldout.index)
    split.loc[heldout["training_series"].eq(NEW_ONE_PHASE_SERIES)] = "new_one_phase"
    split.loc[heldout["training_series"].eq(ADVERSARIAL_SERIES)] = "adversarial"
    heldout["archive_split"] = split
    split_summary = (
        heldout.groupby(["archive_split", "policy_class"])
        .size()
        .rename("row_count")
        .reset_index()
        .sort_values(["archive_split", "policy_class"])
    )
    if heldout["archive_split"].value_counts().to_dict() != {
        "historical": 352,
        "new_one_phase": 238,
        "adversarial": 120,
    }:
        raise ValueError("Archive split counts have drifted")
    split_summary.to_csv(ROUND_DIR / "archive_split_summary.csv", index=False)
    heldout[
        [
            "heldout_id",
            "wandb_run_name",
            "training_series",
            "objective",
            "policy_class",
            "data_seed",
            "trainer_seed",
            "mixture_sha256",
            "archive_split",
            "uncheatable_bpb",
            "table9_macro_bpb",
        ]
    ].to_csv(ROUND_DIR / "heldout_provenance_index.csv", index=False)
    aliases[
        ["heldout_id", "wandb_run_name", "mixture_sha256", "fit_panel_run_name", "fit_panel_max_abs_distance"]
    ].to_csv(ROUND_DIR / "excluded_coordinate_aliases.csv", index=False)

    expected_predictions = len(heldout) * 11 * 2
    if len(predictions) != expected_predictions:
        raise ValueError(f"Expected {expected_predictions} predictions, found {len(predictions)}")
    if predictions[["row_id", "model", "target"]].duplicated().any():
        raise ValueError("Prediction archive contains duplicate row/model/target keys")
    if predictions["model"].nunique() != 11 or set(predictions["target"]) != {"uncheatable", "table9"}:
        raise ValueError("Prediction archive model/target coverage has drifted")
    if not predictions.groupby("row_id").size().eq(22).all():
        raise ValueError("Every heldout row must have 11 model predictions for both targets")

    adversarial = heldout.loc[heldout["training_series"].eq(ADVERSARIAL_SERIES)].copy()
    tags = adversarial["tags_json"].map(tag_dict)
    adversarial["candidate_id"] = tags.map(lambda values: values.get("source_run", ""))
    adversarial["candidate_target_from_tags"] = tags.map(lambda values: values.get("target", ""))
    adversarial["selection_stratum_from_tags"] = tags.map(lambda values: values.get("selection", ""))
    if adversarial["candidate_id"].str.strip().eq("").any():
        raise ValueError("Adversarial run is missing source_run provenance")
    joined = adversarial.merge(
        adversarial_manifest,
        on="candidate_id",
        suffixes=("_run", "_manifest"),
        validate="one_to_one",
    )
    if len(joined) != 120:
        raise ValueError("Not every adversarial checkpoint joins to its frozen manifest")
    if not joined["objective"].eq(joined["target"]).all():
        raise ValueError("Adversarial objective does not match manifest candidate target")
    if not joined["candidate_target_from_tags"].eq(joined["target"]).all():
        raise ValueError("Adversarial target tag does not match the frozen manifest")
    if not joined["selection_stratum_from_tags"].eq(joined["selection_stratum"]).all():
        raise ValueError("Adversarial selection tag does not match the frozen manifest")
    if not joined["policy_class_run"].eq(joined["policy_class_manifest"]).all():
        raise ValueError("Adversarial policy class does not match the frozen manifest")

    joined[
        [
            "candidate_id",
            "heldout_id",
            "wandb_run_name",
            "target",
            "policy_class_manifest",
            "selection_stratum",
            "origin",
            "proposal_models",
            "mixture_sha256",
            "uncheatable_bpb",
            "table9_macro_bpb",
        ]
    ].rename(columns={"policy_class_manifest": "policy_class", "target": "candidate_target"}).to_csv(
        ROUND_DIR / "adversarial_provenance.csv", index=False
    )

    report = "\n".join(
        [
            "# Round 58: heldout archive provenance audit",
            "",
            "This audit reads provenance and the already-exposed development archive. It does not read any sealed confirmation outcome or fit a model.",
            "",
            "## Archive invariants",
            "",
            f"- Source registry rows: {len(all_rows)}.",
            f"- Coordinate-disjoint development rows: {len(heldout)}.",
            f"- Exact fit-panel coordinate aliases excluded: {len(aliases)}.",
            f"- Unique policy hashes: {heldout['mixture_sha256'].nunique()}.",
            f"- Repeated-coordinate groups retained as stochastic evidence: {len(repeat_groups)} groups, {len(repeats)} rows, {len(repeats) - len(repeat_groups)} extra repeats.",
            "- All 710 rows are finished, checkpoint-declared complete, and have both Uncheatable and Table-9 BPB.",
            f"- Fully distinct-seed repeat groups: {repeat_groups['repeat_kind'].eq('independent_seed_repeat').sum()}; mixed seed/launcher duplicate groups: {repeat_groups['repeat_kind'].eq('mixed_seed_and_launcher_duplicate').sum()}.",
            "- All rows remain in the append-only provenance index, but coordinate-balanced sensitivity is required before treating 710 as an effective sample size.",
            "",
            "## Split and policy coverage",
            "",
            split_summary.to_markdown(index=False),
            "",
            "## Adversarial provenance",
            "",
            "All 120 exposed adversarial checkpoints join one-to-one to the frozen candidate manifest. Candidate target, policy class, and selection stratum agree between the run tags and manifest. Origin and proposal-model metadata are retained row by row. Both observed targets remain attached to every checkpoint so target-matched use can be separated from cross-target transfer use.",
            "",
            "## Prediction coverage",
            "",
            f"The frozen prediction archive contains {len(predictions)} rows: 710 policies x 11 models x 2 targets, with no duplicate row/model/target key and no coverage gap.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
