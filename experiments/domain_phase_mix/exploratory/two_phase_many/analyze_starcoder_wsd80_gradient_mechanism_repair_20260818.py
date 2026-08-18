# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.5.1",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "tabulate>=0.9",
# ]
# ///
"""Analyze the post-outcome H1/H2/H3/H5 gradient-mechanism repair."""

import argparse
import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd

from experiments.domain_phase_mix import starcoder_wsd80_gradient_mechanism_repair as runtime
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_mechanism_repair_20260818 as freeze,
)

DEFAULT_OUTPUT_DIR = freeze.OUTPUT_DIR.parent / "starcoder_wsd80_gradient_mechanism_repair_results_v8_20260818"
ARTIFACT_VERSION = runtime.ARTIFACT_VERSION
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 2_026_081_801
MAX_WORKERS = 64

PL_TARGET = "paloma_programming_languages"
C4_TARGET = "paloma_c4_en"
GITHUB_TARGET = "uncheatable_github_python"
WIKIPEDIA_TARGET = "uncheatable_wikipedia_english"
TARGETS = (PL_TARGET, C4_TARGET, GITHUB_TARGET, WIKIPEDIA_TARGET)
PRIMARY_CONTRAST = f"{freeze.GLOBAL_STARCODER}__minus__{freeze.NEMOTRON}"
SUPPORT_CONTRAST = f"{freeze.SUPPORT_STARCODER}__minus__{freeze.GLOBAL_STARCODER}"


def _read_manifest() -> list[dict[str, str]]:
    with freeze.FULL_MANIFEST_PATH.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _read_document(fs: gcsfs.GCSFileSystem, row: Mapping[str, str], release_sha256: str) -> dict[str, Any]:
    uri = f"{freeze.RESULT_ROOT}/full/{row['group_id']}/{ARTIFACT_VERSION}/rows/{row['row_id']}.json"
    with fs.open(uri.removeprefix("gs://"), "rb") as handle:
        document = json.load(handle)
    if document.get("schema_version") != runtime.SCHEMA_VERSION or document.get("kind") != "gradient_mechanism_repair":
        raise RuntimeError(f"Repair output schema/kind mismatch: {uri}")
    if document.get("payload_sha256") != freeze.canonical_sha256({**document, "payload_sha256": ""}):
        raise RuntimeError(f"Repair output payload hash mismatch: {uri}")
    if (
        document["release_sha256"] != release_sha256
        or document["row"] != row
        or document.get("identity_sha256") != runtime._row_identity(row, release_sha256)
    ):
        raise RuntimeError(f"Repair output identity mismatch: {uri}")
    if document.get("endpoint_metrics_read") is not False:
        raise RuntimeError(f"Endpoint leakage marker is not false: {uri}")
    if document.get("scientific_status") != freeze.SCIENTIFIC_STATUS:
        raise RuntimeError(f"Repair scientific status drifted: {uri}")
    document["source_uri"] = uri
    return document


def load_documents(fs: gcsfs.GCSFileSystem, release_sha256: str) -> list[dict[str, Any]]:
    rows = _read_manifest()
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(lambda row: _read_document(fs, row, release_sha256), rows))


def flatten_alignment(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in documents:
        metadata = document["row"]
        for target, contrasts in document["target_source_choice_contrasts"].items():
            for contrast_name, contrast in contrasts.items():
                for geometry in ("raw", "projected"):
                    for component, statistics in contrast["statistic"][geometry].items():
                        if statistics.get("cosine_defined") is not True or statistics.get("cosine") is None:
                            raise RuntimeError(
                                "Undefined alignment cosine for "
                                f"{metadata['row_id']}/{target}/{contrast_name}/{geometry}/{component}"
                            )
                        rows.append(
                            {
                                "row_id": metadata["row_id"],
                                "trajectory_id": metadata["trajectory_id"],
                                "training_seed": int(metadata["training_seed"]),
                                "support_id": metadata["support_id"],
                                "policy_role": metadata["policy_role"],
                                "analysis_role": metadata["analysis_role"],
                                "checkpoint_label": metadata["checkpoint_label"],
                                "target": target,
                                "contrast": contrast_name,
                                "geometry": geometry,
                                "component": component,
                                "X_y": float(statistics["dot"]),
                                "A_y": float(statistics["cosine"]),
                                "target_gradient_norm": float(statistics["left_norm"]),
                                "source_update_contrast_norm": float(statistics["right_norm"]),
                                "source_uri": document["source_uri"],
                            }
                        )
    return pd.DataFrame(rows)


def flatten_utilities(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in documents:
        metadata = document["row"]
        for target, sources in document["target_source_utility_statistics"].items():
            for source, bundle in sources.items():
                for geometry in ("raw", "projected"):
                    for component, statistics in bundle[geometry].items():
                        if statistics.get("cosine_defined") is not True or statistics.get("cosine") is None:
                            identity = f"{metadata['row_id']}/{target}/{source}/{geometry}/{component}"
                            raise RuntimeError(f"Undefined utility cosine for {identity}")
                        rows.append(
                            {
                                "row_id": metadata["row_id"],
                                "trajectory_id": metadata["trajectory_id"],
                                "training_seed": int(metadata["training_seed"]),
                                "support_id": metadata["support_id"],
                                "policy_role": metadata["policy_role"],
                                "analysis_role": metadata["analysis_role"],
                                "checkpoint_label": metadata["checkpoint_label"],
                                "target": target,
                                "source": source,
                                "geometry": geometry,
                                "component": component,
                                "U_y": float(statistics["dot"]),
                                "cosine": float(statistics["cosine"]),
                                "target_gradient_norm": float(statistics["left_norm"]),
                                "source_update_norm": float(statistics["right_norm"]),
                                "source_uri": document["source_uri"],
                            }
                        )
    return pd.DataFrame(rows)


def flatten_h1(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in documents:
        metadata = document["row"]
        for pair in document["source_pair_statistics"].values():
            for statistic_name in ("gradient", "optimizer_update"):
                for geometry in ("raw", "projected"):
                    for component, statistics in pair[statistic_name][geometry].items():
                        if statistics.get("cosine_defined") is not True or statistics.get("cosine") is None:
                            raise RuntimeError(
                                f"Undefined H1 cosine for {metadata['row_id']}/{statistic_name}/{geometry}/{component}"
                            )
                        rows.append(
                            {
                                "row_id": metadata["row_id"],
                                "trajectory_id": metadata["trajectory_id"],
                                "training_seed": int(metadata["training_seed"]),
                                "support_id": metadata["support_id"],
                                "policy_role": metadata["policy_role"],
                                "analysis_role": metadata["analysis_role"],
                                "checkpoint_label": metadata["checkpoint_label"],
                                "statistic": statistic_name,
                                "geometry": geometry,
                                "component": component,
                                "cosine": float(statistics["cosine"]),
                                "dot": float(statistics["dot"]),
                                "left_norm": float(statistics["left_norm"]),
                                "right_norm": float(statistics["right_norm"]),
                                "source_uri": document["source_uri"],
                            }
                        )
    return pd.DataFrame(rows)


def select_h1_documents(documents: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    contract = freeze.ANALYSIS_CONTRACT["estimands"]["h1"]
    expected_manifest = [
        row
        for row in _read_manifest()
        if row["analysis_role"] == "h1_trajectory_extension" and row["checkpoint_label"] in set(contract["states"])
    ]
    expected_ids = {row["row_id"] for row in expected_manifest}
    selected = [document for document in documents if document["row"]["row_id"] in expected_ids]
    observed_ids = [document["row"]["row_id"] for document in selected]
    if len(expected_manifest) != int(contract["row_count"]) or len(expected_ids) != len(expected_manifest):
        raise RuntimeError("Frozen H1 manifest inventory is inconsistent with its contract")
    if len(observed_ids) != len(set(observed_ids)) or set(observed_ids) != expected_ids:
        raise RuntimeError("H1 document inventory drifted from the frozen restricted subset")
    if any(
        document["row"]["analysis_role"] != "h1_trajectory_extension"
        or document["row"]["checkpoint_label"] not in set(contract["states"])
        for document in selected
    ):
        raise RuntimeError("H1 document selection admitted a row outside the frozen restricted subset")
    return selected


def _bootstrap_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise RuntimeError("Bootstrap input is empty or non-finite")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    means = array[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def _exact_sign_flip_p(values: Sequence[float], *, alternative: str) -> float:
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.isfinite(array).all():
        raise RuntimeError("Sign-flip input is empty or non-finite")
    observed = float(array.mean())
    total = 1 << len(array)
    extreme = 0
    tolerance = 1e-15
    chunk = 1 << 18
    bit_positions = np.arange(len(array), dtype=np.uint64)
    for start in range(0, total, chunk):
        codes = np.arange(start, min(start + chunk, total), dtype=np.uint64)[:, None]
        signs = 2.0 * ((codes >> bit_positions) & 1).astype(float) - 1.0
        null = (signs @ array) / len(array)
        if alternative == "greater":
            extreme += int(np.sum(null >= observed - tolerance))
        elif alternative == "less":
            extreme += int(np.sum(null <= observed + tolerance))
        elif alternative == "two_sided":
            extreme += int(np.sum(np.abs(null) >= abs(observed) - tolerance))
        else:
            raise ValueError(f"Unknown sign-flip alternative: {alternative}")
    return extreme / total


def _summary(values: Sequence[float], *, name: str, alternative: str, role: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    low, high = _bootstrap_interval(array)
    return {
        "contrast": name,
        "scientific_status": freeze.SCIENTIFIC_STATUS,
        "evidence_role": role,
        "n_paired_seeds": len(array),
        "mean": float(array.mean()),
        "seed_sd": float(array.std(ddof=1)),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "exact_sign_flip_p_unadjusted": _exact_sign_flip_p(array, alternative=alternative),
        "alternative": alternative,
        "positive_pairs": int(np.sum(array > 0)),
    }


def _assert_exact_manifest_inventory(
    frame: pd.DataFrame,
    *,
    roles: set[str],
    labels: set[str],
    include_sources: bool,
    name: str,
) -> None:
    manifest = [row for row in _read_manifest() if row["analysis_role"] in roles and row["checkpoint_label"] in labels]
    expected: set[tuple[str, ...]] = set()
    for row in manifest:
        targets = json.loads(row["target_distribution_ids_json"])
        sources = json.loads(row["source_distribution_ids_json"]) if include_sources else (None,)
        for target in targets:
            for source in sources:
                key = (row["row_id"], target)
                if include_sources:
                    key = (*key, source)
                expected.add(key)
    columns = ["row_id", "target", *(("source",) if include_sources else ())]
    observed_rows = [tuple(str(row[column]) for column in columns) for _, row in frame.iterrows()]
    observed = set(observed_rows)
    if len(observed_rows) != len(observed):
        raise RuntimeError(f"{name} contains duplicate row/target inventories")
    if observed != expected:
        missing = sorted(expected - observed)[:10]
        unexpected = sorted(observed - expected)[:10]
        raise RuntimeError(f"{name} inventory drifted: missing={missing}, unexpected={unexpected}")


def h2_h3_analysis(alignment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roles = {"h2_primary", "h3_full_support_pair", "h3_second_pool_sensitivity"}
    labels = {label for states in freeze.ANALYSIS_CONTRACT["estimands"]["h2"]["states"].values() for label in states}
    primary = alignment[
        alignment["geometry"].eq("projected")
        & alignment["component"].eq("trunk")
        & alignment["contrast"].eq(PRIMARY_CONTRAST)
        & alignment["analysis_role"].isin(roles)
        & alignment["checkpoint_label"].isin(labels)
    ].copy()
    _assert_exact_manifest_inventory(
        primary,
        roles=roles,
        labels=labels,
        include_sources=False,
        name="H2/H3 alignment",
    )
    period_by_state = {
        state: period
        for period, states in freeze.ANALYSIS_CONTRACT["estimands"]["h2"]["states"].items()
        for state in states
    }
    primary["period"] = primary["checkpoint_label"].map(period_by_state)
    period = primary.groupby(["training_seed", "support_id", "target", "period"], as_index=False)["A_y"].mean()
    wide = period.pivot(index=["training_seed", "support_id", "target"], columns="period", values="A_y").reset_index()
    required_periods = ["mid", "late_pre_decay", "late_post_decay"]
    if wide[required_periods].isna().any().any():
        raise RuntimeError("H2 period inventory is incomplete")
    seed_rows = wide[["training_seed", "support_id"]].drop_duplicates().reset_index(drop=True)
    for horizon in ("late_pre_decay", "late_post_decay"):
        wide[f"{horizon}_minus_mid"] = wide[horizon] - wide["mid"]
        target_wide = wide.pivot(
            index=["training_seed", "support_id"], columns="target", values=f"{horizon}_minus_mid"
        ).reset_index()
        target_wide = target_wide.rename(columns={target: f"{horizon}_{target}" for target in TARGETS})
        seed_rows = seed_rows.merge(target_wide, on=["training_seed", "support_id"], validate="one_to_one")
    expected_columns = [f"{horizon}_{target}" for horizon in ("late_pre_decay", "late_post_decay") for target in TARGETS]
    if seed_rows[expected_columns].isna().any().any():
        raise RuntimeError("H2 target inventory is incomplete")
    expected_support_seeds = {"m100a": 24, "full": 24, "m100b": 8}
    observed_support_seeds = seed_rows.groupby("support_id")["training_seed"].nunique().to_dict()
    if observed_support_seeds != expected_support_seeds:
        raise RuntimeError(
            f"H2/H3 seed inventory drifted: observed={observed_support_seeds}, expected={expected_support_seeds}"
        )
    for prefix in ("late_pre_decay", "late_post_decay"):
        seed_rows[f"{prefix}_pl_minus_c4"] = seed_rows[f"{prefix}_{PL_TARGET}"] - seed_rows[f"{prefix}_{C4_TARGET}"]
        seed_rows[f"{prefix}_github_minus_c4"] = (
            seed_rows[f"{prefix}_{GITHUB_TARGET}"] - seed_rows[f"{prefix}_{C4_TARGET}"]
        )
        seed_rows[f"{prefix}_wikipedia_minus_c4"] = (
            seed_rows[f"{prefix}_{WIKIPEDIA_TARGET}"] - seed_rows[f"{prefix}_{C4_TARGET}"]
        )

    summaries: list[dict[str, Any]] = []
    for support, group in seed_rows.groupby("support_id", sort=True):
        role = "sensitivity_only" if support == "m100b" else "development_repair_of_frozen_estimand"
        for horizon, horizon_role in (
            ("late_pre_decay", role),
            ("late_post_decay", "secondary_post_decay_development_evidence" if support != "m100b" else role),
        ):
            summaries.append(
                {
                    **_summary(
                        group[f"{horizon}_pl_minus_c4"],
                        name=f"H2 PL-minus-C4 {horizon}-minus-mid revaluation ({support})",
                        alternative="greater",
                        role=horizon_role,
                    ),
                    "github_same_sign": bool(
                        np.sign(group[f"{horizon}_github_minus_c4"].mean())
                        == np.sign(group[f"{horizon}_pl_minus_c4"].mean())
                    ),
                    "wikipedia_negative_control_mean": float(group[f"{horizon}_wikipedia_minus_c4"].mean()),
                }
            )
    h3 = seed_rows.pivot(index="training_seed", columns="support_id", values="late_pre_decay_pl_minus_c4")
    if len(h3) != 24 or h3[["m100a", "full"]].isna().any().any():
        raise RuntimeError("H3 does not contain all 24 frozen m100a/full seed pairs")
    summaries.append(
        _summary(
            h3["m100a"] - h3["full"],
            name="H3 m100a-minus-full H2 interaction",
            alternative="two_sided",
            role="development_repair_of_frozen_estimand",
        )
    )
    return seed_rows, pd.DataFrame(summaries)


def support_separation_analysis(alignment: pd.DataFrame) -> pd.DataFrame:
    return alignment[
        alignment["geometry"].eq("projected")
        & alignment["component"].eq("trunk")
        & alignment["contrast"].eq(SUPPORT_CONTRAST)
        & alignment["analysis_role"].isin(("h2_primary", "h3_full_support_pair", "h3_second_pool_sensitivity"))
    ].copy()


def h3_repetition_mechanism_analysis(utilities: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roles = {"h2_primary", "h3_full_support_pair", "h3_second_pool_sensitivity"}
    states = freeze.ANALYSIS_CONTRACT["estimands"]["h2"]["states"]
    labels = set(states["mid"]) | set(states["late_pre_decay"])
    selected = utilities[
        utilities["geometry"].eq("projected")
        & utilities["component"].eq("trunk")
        & utilities["analysis_role"].isin(roles)
        & utilities["checkpoint_label"].isin(labels)
    ].copy()
    _assert_exact_manifest_inventory(
        selected,
        roles=roles,
        labels=labels,
        include_sources=True,
        name="H3 source utility",
    )
    period_by_state = {state: period for period in ("mid", "late_pre_decay") for state in states[period]}
    selected["period"] = selected["checkpoint_label"].map(period_by_state)
    periods = selected.groupby(["training_seed", "support_id", "target", "source", "period"], as_index=False)[
        "U_y"
    ].mean()
    by_source = periods.pivot(
        index=["training_seed", "support_id", "target", "period"], columns="source", values="U_y"
    ).reset_index()
    required_sources = [freeze.GLOBAL_STARCODER, freeze.SUPPORT_STARCODER]
    if by_source[required_sources].isna().any().any():
        raise RuntimeError("H3 source utility inventory is incomplete")
    by_source["support_minus_global"] = by_source[freeze.SUPPORT_STARCODER] - by_source[freeze.GLOBAL_STARCODER]
    mid = (
        by_source[by_source["period"].eq("mid")]
        .drop(columns="period")
        .rename(
            columns={
                freeze.GLOBAL_STARCODER: "global_mid",
                "support_minus_global": "support_minus_global_mid",
            }
        )
    )
    late = (
        by_source[by_source["period"].eq("late_pre_decay")]
        .drop(columns="period")
        .rename(
            columns={
                freeze.GLOBAL_STARCODER: "global_late_pre_decay",
                "support_minus_global": "support_minus_global_late_pre_decay",
            }
        )
    )
    keys = ["training_seed", "support_id", "target"]
    seed_rows = mid[[*keys, "global_mid", "support_minus_global_mid"]].merge(
        late[[*keys, "global_late_pre_decay", "support_minus_global_late_pre_decay"]],
        on=keys,
        validate="one_to_one",
    )
    seed_rows["unseen_global_utility_decline"] = seed_rows["global_late_pre_decay"] - seed_rows["global_mid"]
    seed_rows["support_separation_growth"] = (
        seed_rows["support_minus_global_late_pre_decay"] - seed_rows["support_minus_global_mid"]
    )
    expected_support_seeds = {"m100a": 24, "full": 24, "m100b": 8}
    for target in TARGETS:
        observed = seed_rows[seed_rows["target"].eq(target)].groupby("support_id")["training_seed"].nunique().to_dict()
        if observed != expected_support_seeds:
            raise RuntimeError(f"H3 utility seed inventory drifted for {target}: {observed}")

    summaries: list[dict[str, Any]] = []
    for (support, target), group in seed_rows.groupby(["support_id", "target"], sort=True):
        role = "sensitivity_only" if support == "m100b" else "development_repair_of_frozen_estimand"
        summaries.append(
            _summary(
                group["unseen_global_utility_decline"],
                name=f"H3 unseen global-source utility late-minus-mid ({support}; {target})",
                alternative="less",
                role=role,
            )
        )
        summaries.append(
            _summary(
                group["support_separation_growth"],
                name=f"H3 included-support separation growth ({support}; {target})",
                alternative="greater",
                role=role,
            )
        )
    paired = seed_rows[seed_rows["support_id"].isin(("m100a", "full"))].pivot(
        index=["training_seed", "target"],
        columns="support_id",
        values=["unseen_global_utility_decline", "support_separation_growth"],
    )
    if len(paired) != 24 * len(TARGETS) or paired.isna().any().any():
        raise RuntimeError("H3 utility interaction omits a frozen m100a/full seed pair")
    for target in TARGETS:
        target_rows = paired.xs(target, level="target")
        summaries.append(
            _summary(
                target_rows[("unseen_global_utility_decline", "m100a")]
                - target_rows[("unseen_global_utility_decline", "full")],
                name=f"H3 m100a-minus-full unseen utility decline ({target})",
                alternative="less",
                role="development_repair_of_frozen_estimand",
            )
        )
        summaries.append(
            _summary(
                target_rows[("support_separation_growth", "m100a")] - target_rows[("support_separation_growth", "full")],
                name=f"H3 m100a-minus-full support-separation growth ({target})",
                alternative="greater",
                role="development_repair_of_frozen_estimand",
            )
        )
    return seed_rows, pd.DataFrame(summaries)


def h5_profile_analysis(alignment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    roles = {"h5_preregistered_profile"}
    labels = {
        label for states in freeze.ANALYSIS_CONTRACT["estimands"]["h5_profile"]["periods"].values() for label in states
    }
    profile = alignment[
        alignment["geometry"].eq("projected")
        & alignment["component"].eq("trunk")
        & alignment["contrast"].eq(PRIMARY_CONTRAST)
        & alignment["analysis_role"].eq("h5_preregistered_profile")
        & alignment["checkpoint_label"].isin(labels)
    ].copy()
    _assert_exact_manifest_inventory(
        profile,
        roles=roles,
        labels=labels,
        include_sources=False,
        name="H5 profile alignment",
    )
    period_by_state = {
        state: period
        for period, states in freeze.ANALYSIS_CONTRACT["estimands"]["h5_profile"]["periods"].items()
        for state in states
    }
    profile["period"] = profile["checkpoint_label"].map(period_by_state)
    periods = profile.groupby(["training_seed", "policy_role", "target", "period"], as_index=False)["A_y"].mean()
    wide = periods.pivot(index=["training_seed", "target", "period"], columns="policy_role", values="A_y").reset_index()
    if wide[["boundary_beta_0p60", "boundary_beta_0p85"]].isna().any().any():
        raise RuntimeError("H5 policy-pair inventory is incomplete")
    wide["D_beta0p60_minus_beta0p85"] = wide["boundary_beta_0p60"] - wide["boundary_beta_0p85"]
    period_wide = wide.pivot(
        index=["training_seed", "target"], columns="period", values="D_beta0p60_minus_beta0p85"
    ).reset_index()
    if period_wide[["mid", "pre", "post"]].isna().any().any():
        raise RuntimeError("H5 period inventory is incomplete")
    if period_wide.groupby("target")["training_seed"].nunique().to_dict() != {target: 16 for target in TARGETS}:
        raise RuntimeError("H5 does not contain all 16 frozen paired seeds for every target")
    period_wide["D_pre_minus_D_mid"] = period_wide["pre"] - period_wide["mid"]
    period_wide["D_post_minus_D_pre"] = period_wide["post"] - period_wide["pre"]
    summaries: list[dict[str, Any]] = []
    for target, group in period_wide.groupby("target", sort=True):
        summaries.append(
            _summary(
                group["D_pre_minus_D_mid"],
                name=f"H5 profile D_pre-minus-D_mid ({target})",
                alternative="two_sided",
                role="secondary_development_repair_of_frozen_profile",
            )
        )
        summaries.append(
            _summary(
                group["D_post_minus_D_pre"],
                name=f"H5 profile D_post-minus-D_pre ({target})",
                alternative="two_sided",
                role="secondary_development_repair_of_frozen_profile",
            )
        )
    return period_wide, pd.DataFrame(summaries)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False, floatfmt=".6g")


def write_report(
    output_dir: Path,
    h2_h3_summary: pd.DataFrame,
    h3_repetition_summary: pd.DataFrame,
    h5_summary: pd.DataFrame,
    h1: pd.DataFrame,
    support_separation_summary: pd.DataFrame,
) -> None:
    h1_primary = h1[h1["geometry"].eq("projected") & h1["component"].eq("trunk")]
    h1_summary = (
        h1_primary.groupby(
            ["analysis_role", "policy_role", "support_id", "checkpoint_label", "statistic"], as_index=False
        )["cosine"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    report = f"""# StarCoder WSD80 gradient-mechanism repair

This is a **post-outcome development repair**, not untouched confirmation. It recomputes cross-statistics omitted by
the sealed v6 schema from the same permanent checkpoints. It does not repair H4's missing preregistered mapping.

## H2 and H3

{_markdown_table(h2_h3_summary)}

## H3 repetition-mechanism signatures

{_markdown_table(h3_repetition_summary)}

## H5 mechanism profile

{_markdown_table(h5_summary)}

## H3 support-reference separation

{_markdown_table(support_separation_summary)}

## H1 source geometry

H1 is restricted to the 56 trajectories already selected for H2, H3, or H5, at the five states named in the frozen
contract. It is descriptive subset coverage, not complete H1 coverage of the original trajectory panel.

{_markdown_table(h1_summary)}

All p-values are unadjusted exact seed-level sign-flip values. No confirmatory familywise claim is made.
"""
    (output_dir / "report.md").write_text(report)


def _directory_hashes(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _publish_create_only(staging: Path, output_dir: Path) -> None:
    if output_dir.exists():
        if _directory_hashes(output_dir) != _directory_hashes(staging):
            raise RuntimeError(f"Existing analysis output differs from the replay: {output_dir}")
        shutil.rmtree(staging)
        return
    staging.replace(output_dir)


def analyze(output_dir: Path) -> None:
    release = runtime._load_release(json.loads(freeze.RELEASE_PATH.read_text())["release_sha256"])
    runtime_audit = runtime.audit_outputs("full", release)
    fs = gcsfs.GCSFileSystem()
    documents = load_documents(fs, release["release_sha256"])
    alignment = flatten_alignment(documents)
    utilities = flatten_utilities(documents)
    h1 = flatten_h1(select_h1_documents(documents))
    h2_h3_rows, h2_h3_summary = h2_h3_analysis(alignment)
    h3_repetition_rows, h3_repetition_summary = h3_repetition_mechanism_analysis(utilities)
    support_separation = support_separation_analysis(alignment)
    support_separation_summary = (
        support_separation.groupby(["support_id", "target", "checkpoint_label"], as_index=False)[["X_y", "A_y"]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    h5_rows, h5_summary = h5_profile_analysis(alignment)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    alignment.to_csv(staging / "target_source_choice_alignment.csv", index=False)
    utilities.to_csv(staging / "target_source_utilities.csv", index=False)
    h1.to_csv(staging / "source_source_geometry.csv", index=False)
    h2_h3_rows.to_csv(staging / "h2_h3_seed_statistics.csv", index=False)
    h2_h3_summary.to_csv(staging / "h2_h3_summary.csv", index=False)
    h3_repetition_rows.to_csv(staging / "h3_repetition_mechanism_seed_statistics.csv", index=False)
    h3_repetition_summary.to_csv(staging / "h3_repetition_mechanism_summary.csv", index=False)
    support_separation.to_csv(staging / "h3_support_separation.csv", index=False)
    support_separation_summary.to_csv(staging / "h3_support_separation_summary.csv", index=False)
    h5_rows.to_csv(staging / "h5_profile_seed_statistics.csv", index=False)
    h5_summary.to_csv(staging / "h5_profile_summary.csv", index=False)
    write_report(staging, h2_h3_summary, h3_repetition_summary, h5_summary, h1, support_separation_summary)
    audit = {
        "release_sha256": release["release_sha256"],
        "scientific_status": freeze.SCIENTIFIC_STATUS,
        "manifest_rows": len(_read_manifest()),
        "documents": len(documents),
        "alignment_rows": len(alignment),
        "utility_rows": len(utilities),
        "h1_rows": len(h1),
        "endpoint_metrics_read": runtime_audit["endpoint_metrics_read"],
        "h4_included": False,
    }
    (staging / "analysis_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    _publish_create_only(staging, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    analyze(args.output_dir)


if __name__ == "__main__":
    main()
