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
"""Post-review sensitivity audit for the WSD80 gradient-mechanism repair."""

import asyncio
import csv
import hashlib
import json
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd

from experiments.domain_phase_mix import starcoder_wsd80_gradient_mechanism_repair as runtime
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    freeze_starcoder_wsd80_gradient_mechanism_repair_20260818 as freeze,
)

INPUT_DIR = (
    Path(__file__).parent / "reference_outputs" / "starcoder_wsd80_gradient_mechanism_repair_results_v10_20260821"
)
OUTPUT_DIR = (
    Path(__file__).parent / "reference_outputs" / "starcoder_wsd80_gradient_mechanism_repair_review_sensitivity_20260821"
)
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 2_026_082_101
GCS_READ_ATTEMPTS = 4
GCS_READ_CONCURRENCY = 32
GCS_READ_TIMEOUT = 45.0
TARGETS = (
    "paloma_c4_en",
    "paloma_programming_languages",
    "uncheatable_github_python",
    "uncheatable_wikipedia_english",
)
GLOBAL_STARCODER = freeze.GLOBAL_STARCODER
SUPPORT_STARCODER = freeze.SUPPORT_STARCODER
STAGE_AUDIT_HASHES = {
    "stage_1": "ed9090fb5a480813ed9a63686bae66d5a681179745b327a6ea412a2f666213a8",
    "stage_2": "7ccab32821012bceea777c098a341b3cf299441547e556c1509455fc2936aeb4",
    "stage_3": "ce7cfc6370cbc9594d37475bb32f445fa171f8c80d7cd07bb29f8a80ea9daa83",
}
H5_TOTAL_STEPS = 28_160
H5_DECAY_STEP = 22_528
H5_PHASE_WEIGHTS = {
    "boundary_beta_0p60": (0.60, 40 / 2_048, 860 / 2_048),
    "boundary_beta_0p85": (0.85, 245 / 2_048, 1_065 / 2_048),
}


def _bootstrap_interval(values: Sequence[float], name: str) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    name_seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big")
    rng = np.random.default_rng(BOOTSTRAP_SEED ^ name_seed)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    means = array[indices].mean(axis=1)
    low, high = np.quantile(means, (0.025, 0.975))
    return float(low), float(high)


def _exact_two_sided_sign_flip_p(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=float)
    observed = abs(float(array.mean()))
    total = 1 << len(array)
    extreme = 0
    chunk = 1 << 18
    bit_positions = np.arange(len(array), dtype=np.uint64)
    scale = max(float(np.abs(array).sum() / len(array)), observed, np.finfo(float).tiny)
    tolerance = 64.0 * np.finfo(float).eps * scale
    for start in range(0, total, chunk):
        codes = np.arange(start, min(start + chunk, total), dtype=np.uint64)[:, None]
        signs = 2.0 * ((codes >> bit_positions) & 1).astype(float) - 1.0
        null = np.abs((signs @ array) / len(array))
        extreme += int(np.sum(null >= observed - tolerance))
    if extreme < 2:
        raise RuntimeError("A two-sided exact sign-flip test excluded the observed sign assignments")
    return extreme / total


def _summary(values: Sequence[float], *, name: str, family: str) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    low, high = _bootstrap_interval(array, name)
    return {
        "family": family,
        "contrast": name,
        "n_paired_seeds": len(array),
        "mean": float(array.mean()),
        "seed_sd": float(array.std(ddof=1)),
        "bootstrap_ci95_low": low,
        "bootstrap_ci95_high": high,
        "exact_two_sided_sign_flip_p": _exact_two_sided_sign_flip_p(array),
    }


def _holm_adjust(p_values: pd.Series) -> pd.Series:
    order = np.argsort(p_values.to_numpy(), kind="stable")
    ordered = p_values.to_numpy()[order]
    adjusted_ordered = np.maximum.accumulate((len(ordered) - np.arange(len(ordered))) * ordered)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = np.minimum(adjusted_ordered, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def _recompute_frozen_tests() -> pd.DataFrame:
    summaries: list[dict[str, Any]] = []
    h2 = pd.read_csv(INPUT_DIR / "h2_h3_seed_statistics.csv")
    for support, group in h2.groupby("support_id", sort=True):
        for horizon in ("late_pre_decay", "late_post_decay"):
            column = f"{horizon}_pl_minus_c4"
            summaries.append(
                _summary(
                    group[column],
                    name=f"H2 PL-minus-C4 {horizon}-minus-mid revaluation ({support})",
                    family="h2_h3_frozen",
                )
            )
    h3 = h2.pivot(index="training_seed", columns="support_id", values="late_pre_decay_pl_minus_c4")
    summaries.append(
        _summary(
            h3["m100a"] - h3["full"],
            name="H3 m100a-minus-full H2 interaction",
            family="h2_h3_frozen",
        )
    )

    repetition = pd.read_csv(INPUT_DIR / "h3_repetition_mechanism_seed_statistics.csv")
    for (support, target), group in repetition.groupby(["support_id", "target"], sort=True):
        summaries.append(
            _summary(
                group["unseen_global_utility_decline"],
                name=f"H3 unseen global-source utility late-minus-mid ({support}; {target})",
                family="h3_repetition_frozen",
            )
        )
        summaries.append(
            _summary(
                group["support_separation_growth"],
                name=f"H3 included-support separation growth ({support}; {target})",
                family="h3_repetition_frozen",
            )
        )
    paired = repetition[repetition["support_id"].isin(("m100a", "full"))].pivot(
        index=["training_seed", "target"],
        columns="support_id",
        values=["unseen_global_utility_decline", "support_separation_growth"],
    )
    for target in TARGETS:
        target_rows = paired.xs(target, level="target")
        summaries.append(
            _summary(
                target_rows[("unseen_global_utility_decline", "m100a")]
                - target_rows[("unseen_global_utility_decline", "full")],
                name=f"H3 m100a-minus-full unseen utility decline ({target})",
                family="h3_repetition_frozen",
            )
        )
        summaries.append(
            _summary(
                target_rows[("support_separation_growth", "m100a")] - target_rows[("support_separation_growth", "full")],
                name=f"H3 m100a-minus-full support-separation growth ({target})",
                family="h3_repetition_frozen",
            )
        )

    h5 = pd.read_csv(INPUT_DIR / "h5_profile_seed_statistics.csv")
    for target, group in h5.groupby("target", sort=True):
        summaries.append(
            _summary(
                group["D_pre_minus_D_mid"],
                name=f"H5 profile D_pre-minus-D_mid ({target})",
                family="h5_profile_frozen",
            )
        )
        summaries.append(
            _summary(
                group["D_post_minus_D_pre"],
                name=f"H5 profile D_post-minus-D_pre ({target})",
                family="h5_profile_frozen",
            )
        )

    frame = pd.DataFrame(summaries)
    if len(frame) != 47:
        raise RuntimeError(f"Expected 47 frozen-analysis tests, found {len(frame)}")
    frame["holm_p_within_family"] = frame.groupby("family", group_keys=False)["exact_two_sided_sign_flip_p"].apply(
        _holm_adjust
    )
    frame["holm_p_across_47"] = _holm_adjust(frame["exact_two_sided_sign_flip_p"])
    return frame


def _normalized_h3_tests() -> pd.DataFrame:
    utilities = pd.read_csv(INPUT_DIR / "target_source_utilities.csv")
    roles = {"h2_primary", "h3_full_support_pair", "h3_second_pool_sensitivity"}
    period_by_label = {
        "fraction_0p40": "mid",
        "fraction_0p55": "mid",
        "decay_minus_256": "late_pre_decay",
        "decay_minus_64": "late_pre_decay",
    }
    selected = utilities[
        utilities["geometry"].eq("projected")
        & utilities["component"].eq("trunk")
        & utilities["analysis_role"].isin(roles)
        & utilities["checkpoint_label"].isin(period_by_label)
    ].copy()
    selected["period"] = selected["checkpoint_label"].map(period_by_label)
    periods = selected.groupby(["training_seed", "support_id", "target", "source", "period"], as_index=False)[
        "cosine"
    ].mean()
    wide = periods.pivot(
        index=["training_seed", "support_id", "target"],
        columns=["source", "period"],
        values="cosine",
    )
    seed_rows = pd.DataFrame(
        {
            "global_change": wide[(GLOBAL_STARCODER, "late_pre_decay")] - wide[(GLOBAL_STARCODER, "mid")],
            "separation_change": (
                wide[(SUPPORT_STARCODER, "late_pre_decay")]
                - wide[(GLOBAL_STARCODER, "late_pre_decay")]
                - wide[(SUPPORT_STARCODER, "mid")]
                + wide[(GLOBAL_STARCODER, "mid")]
            ),
        }
    ).reset_index()
    summaries: list[dict[str, Any]] = []
    for (support, target), group in seed_rows.groupby(["support_id", "target"], sort=True):
        summaries.append(
            _summary(
                group["global_change"],
                name=f"Normalized H3 global-source cosine late-minus-mid ({support}; {target})",
                family="h3_normalized_sensitivity",
            )
        )
        summaries.append(
            _summary(
                group["separation_change"],
                name=f"Normalized H3 support-separation cosine growth ({support}; {target})",
                family="h3_normalized_sensitivity",
            )
        )
    paired = seed_rows[seed_rows["support_id"].isin(("m100a", "full"))].pivot(
        index=["training_seed", "target"], columns="support_id", values=["global_change", "separation_change"]
    )
    for target in TARGETS:
        target_rows = paired.xs(target, level="target")
        summaries.append(
            _summary(
                target_rows[("global_change", "m100a")] - target_rows[("global_change", "full")],
                name=f"Normalized H3 m100a-minus-full global-source change ({target})",
                family="h3_normalized_sensitivity",
            )
        )
        summaries.append(
            _summary(
                target_rows[("separation_change", "m100a")] - target_rows[("separation_change", "full")],
                name=f"Normalized H3 m100a-minus-full support-separation change ({target})",
                family="h3_normalized_sensitivity",
            )
        )
    frame = pd.DataFrame(summaries)
    if len(frame) != 32:
        raise RuntimeError(f"Expected 32 normalized H3 sensitivity tests, found {len(frame)}")
    frame["holm_p_across_32"] = _holm_adjust(frame["exact_two_sided_sign_flip_p"])
    return frame


def _h5_event_time_tests() -> pd.DataFrame:
    alignment = pd.read_csv(INPUT_DIR / "target_source_choice_alignment.csv")
    labels = ("data_switch_minus_64", "data_switch", "data_switch_plus_64")
    selected = alignment[
        alignment["analysis_role"].eq("h5_preregistered_profile")
        & alignment["geometry"].eq("projected")
        & alignment["component"].eq("trunk")
        & alignment["contrast"].eq(f"{GLOBAL_STARCODER}__minus__{freeze.NEMOTRON}")
        & alignment["checkpoint_label"].isin(labels)
    ]
    wide = selected.pivot(
        index=["training_seed", "policy_role", "target"], columns="checkpoint_label", values="A_y"
    ).reset_index()
    if wide[list(labels)].isna().any().any() or len(wide) != 2 * 16 * len(TARGETS):
        raise RuntimeError("H5 event-time inventory is incomplete")
    wide["plus64_minus_minus64"] = wide["data_switch_plus_64"] - wide["data_switch_minus_64"]
    summaries: list[dict[str, Any]] = []
    for (policy, target), group in wide.groupby(["policy_role", "target"], sort=True):
        summaries.append(
            _summary(
                group["plus64_minus_minus64"],
                name=f"H5 switch-relative plus64-minus-minus64 ({policy}; {target})",
                family="h5_event_time_post_review",
            )
        )
    paired = wide.pivot(index=["training_seed", "target"], columns="policy_role", values="plus64_minus_minus64")
    for target in TARGETS:
        target_rows = paired.xs(target, level="target")
        summaries.append(
            _summary(
                target_rows["boundary_beta_0p60"] - target_rows["boundary_beta_0p85"],
                name=f"H5 switch-response beta0p60-minus-beta0p85 ({target})",
                family="h5_event_time_post_review",
            )
        )
    frame = pd.DataFrame(summaries)
    frame["holm_p_across_12"] = _holm_adjust(frame["exact_two_sided_sign_flip_p"])
    return frame


def _cumulative_exposure(policy: str, step: int) -> float:
    beta, phase_0, phase_1 = H5_PHASE_WEIGHTS[policy]
    switch_step = int(beta * H5_TOTAL_STEPS)
    if step <= switch_step:
        return phase_0
    return (switch_step * phase_0 + (step - switch_step) * phase_1) / step


def _h5_exposure_gap_table() -> pd.DataFrame:
    periods = {
        "mid": (int(0.40 * H5_TOTAL_STEPS), int(0.55 * H5_TOTAL_STEPS)),
        "pre": (H5_DECAY_STEP - 256, H5_DECAY_STEP - 64),
        "post": (H5_DECAY_STEP, H5_DECAY_STEP + 64),
    }
    rows = []
    for period, steps in periods.items():
        beta_0p60 = np.mean([_cumulative_exposure("boundary_beta_0p60", step) for step in steps])
        beta_0p85 = np.mean([_cumulative_exposure("boundary_beta_0p85", step) for step in steps])
        rows.append(
            {
                "period": period,
                "checkpoint_steps": json.dumps(steps),
                "beta_0p60_cumulative_starcoder": beta_0p60,
                "beta_0p85_cumulative_starcoder": beta_0p85,
                "beta_0p60_minus_beta_0p85_exposure_gap": beta_0p60 - beta_0p85,
            }
        )
    return pd.DataFrame(rows)


def _analysis_path_sensitivity() -> pd.DataFrame:
    alignment = pd.read_csv(INPUT_DIR / "target_source_choice_alignment.csv")
    primary_contrast = f"{GLOBAL_STARCODER}__minus__{freeze.NEMOTRON}"
    h2_roles = {"h2_primary", "h3_full_support_pair", "h3_second_pool_sensitivity"}
    h2_period = {
        "fraction_0p40": "mid",
        "fraction_0p55": "mid",
        "decay_minus_256": "late_pre_decay",
        "decay_minus_64": "late_pre_decay",
    }
    h5_period = {
        "fraction_0p40": "mid",
        "fraction_0p55": "mid",
        "optimizer_decay_minus_256": "pre",
        "optimizer_decay_minus_64": "pre",
    }
    rows: list[dict[str, Any]] = []
    for (geometry, component), path_frame in alignment.groupby(["geometry", "component"], sort=True):
        h2 = path_frame[
            path_frame["contrast"].eq(primary_contrast)
            & path_frame["analysis_role"].isin(h2_roles)
            & path_frame["checkpoint_label"].isin(h2_period)
        ].copy()
        h2["period"] = h2["checkpoint_label"].map(h2_period)
        h2_periods = h2.groupby(["training_seed", "support_id", "target", "period"], as_index=False)["A_y"].mean()
        h2_wide = h2_periods.pivot(
            index=["training_seed", "support_id", "target"], columns="period", values="A_y"
        ).reset_index()
        h2_wide["late_minus_mid"] = h2_wide["late_pre_decay"] - h2_wide["mid"]
        h2_targets = h2_wide.pivot(
            index=["training_seed", "support_id"], columns="target", values="late_minus_mid"
        ).reset_index()
        h2_targets["pl_minus_c4"] = h2_targets["paloma_programming_languages"] - h2_targets["paloma_c4_en"]
        h2_supports = h2_targets.pivot(index="training_seed", columns="support_id", values="pl_minus_c4")

        h5 = path_frame[
            path_frame["contrast"].eq(primary_contrast)
            & path_frame["analysis_role"].eq("h5_preregistered_profile")
            & path_frame["checkpoint_label"].isin(h5_period)
        ].copy()
        h5["period"] = h5["checkpoint_label"].map(h5_period)
        h5_periods = h5.groupby(["training_seed", "policy_role", "target", "period"], as_index=False)["A_y"].mean()
        h5_wide = h5_periods.pivot(
            index=["training_seed", "target", "period"], columns="policy_role", values="A_y"
        ).reset_index()
        h5_wide["D"] = h5_wide["boundary_beta_0p60"] - h5_wide["boundary_beta_0p85"]
        h5_seed = h5_wide.pivot(index=["training_seed", "target"], columns="period", values="D").reset_index()
        h5_seed["pre_minus_mid"] = h5_seed["pre"] - h5_seed["mid"]

        path_summary: dict[str, Any] = {
            "geometry": geometry,
            "component": component,
            "h2_m100a_pl_minus_c4_mean": float(h2_targets[h2_targets["support_id"].eq("m100a")]["pl_minus_c4"].mean()),
            "h3_m100a_minus_full_mean": float((h2_supports["m100a"] - h2_supports["full"]).mean()),
        }
        for target in TARGETS:
            path_summary[f"h5_pre_minus_mid_{target}_mean"] = float(
                h5_seed[h5_seed["target"].eq(target)]["pre_minus_mid"].mean()
            )
        rows.append(path_summary)
    frame = pd.DataFrame(rows)
    if len(frame) != 22:
        raise RuntimeError(f"Expected 22 geometry/component analysis paths, found {len(frame)}")
    return frame


def _target_probe_block_counts() -> dict[str, int]:
    with freeze.FULL_MANIFEST_PATH.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row["analysis_role"] == "h5_preregistered_profile"]
    inventories = {row["distribution_block_counts_json"] for row in rows}
    if len(inventories) != 1:
        raise RuntimeError("H5 target probe block counts are not constant")
    counts = json.loads(inventories.pop())
    return {target: int(counts[target]) for target in TARGETS}


async def _read_gcs_payloads(paths: Sequence[str]) -> dict[str, bytes]:
    fs = gcsfs.GCSFileSystem(asynchronous=True)
    semaphore = asyncio.Semaphore(GCS_READ_CONCURRENCY)

    async def read_one(path: str) -> tuple[str, bytes]:
        for attempt in range(GCS_READ_ATTEMPTS):
            try:
                async with semaphore:
                    payload = await asyncio.wait_for(fs._cat_file(path), timeout=GCS_READ_TIMEOUT)
                return path, payload
            except Exception as error:
                if attempt + 1 == GCS_READ_ATTEMPTS:
                    raise RuntimeError(f"Failed to read gs://{path} after {GCS_READ_ATTEMPTS} attempts") from error
                await asyncio.sleep(2**attempt)
        raise AssertionError("GCS read retry loop exhausted without returning or raising")

    try:
        return dict(await asyncio.gather(*(read_one(path) for path in paths)))
    finally:
        if fs._session is not None:
            await fs._session.close()


def _payload_content_audit() -> dict[str, Any]:
    with freeze.FULL_MANIFEST_PATH.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    uris = sorted(
        f"{freeze.RESULT_ROOT}/full/{row['group_id']}/{runtime.ARTIFACT_VERSION}/rows/{row['row_id']}.json"
        for row in manifest
    )
    paths = [uri.removeprefix("gs://") for uri in uris]
    payloads = asyncio.run(_read_gcs_payloads(paths))
    identities: list[dict[str, str]] = []
    endpoint_markers = set()
    for path in paths:
        document = json.loads(payloads[path])
        expected_hash = freeze.canonical_sha256({**document, "payload_sha256": ""})
        if document["payload_sha256"] != expected_hash:
            raise RuntimeError(f"Payload hash mismatch: gs://{path}")
        identities.append(
            {
                "row_id": str(document["row"]["row_id"]),
                "payload_sha256": str(document["payload_sha256"]),
            }
        )
        endpoint_markers.add(document.get("endpoint_metrics_read"))
    identities.sort(key=lambda item: item["row_id"])
    if len(identities) != 960 or len({item["row_id"] for item in identities}) != 960:
        raise RuntimeError("Payload identity inventory is not exactly 960 unique rows")
    return {
        "release_sha256": json.loads(freeze.RELEASE_PATH.read_text())["release_sha256"],
        "stage_audit_sha256": STAGE_AUDIT_HASHES,
        "row_payload_count": len(identities),
        "row_payload_identity_sha256": freeze.canonical_sha256(identities),
        "endpoint_metrics_read_markers": sorted(endpoint_markers),
        "endpoint_marker_scope": (
            "Emitted row invariant only; endpoint blindness is supported by source inspection, not by this flag alone."
        ),
    }


def _h1_inventory_audit() -> dict[str, Any]:
    with freeze.FULL_MANIFEST_PATH.open(newline="") as handle:
        manifest = list(csv.DictReader(handle))
    h1 = [row for row in manifest if row["analysis_role"] == "h1_trajectory_extension"]
    h2_h3 = [
        row
        for row in manifest
        if row["analysis_role"] in {"h2_primary", "h3_full_support_pair", "h3_second_pool_sensitivity"}
    ]
    h5 = [row for row in manifest if row["analysis_role"] == "h5_preregistered_profile"]
    h1_trajectories = {row["trajectory_id"] for row in h1}
    return {
        "h1_rows": len(h1),
        "h1_trajectories": len(h1_trajectories),
        "h1_policy_roles": sorted({row["policy_role"] for row in h1}),
        "h1_overlap_with_h2_h3_trajectories": len(h1_trajectories & {row["trajectory_id"] for row in h2_h3}),
        "h1_overlap_with_h5_trajectories": len(h1_trajectories & {row["trajectory_id"] for row in h5}),
        "corrected_interpretation": (
            "H1 contains the 56 common-tied H2/H3 trajectories only; it contains no H5 trajectory."
        ),
    }


def _directory_hashes(path: Path) -> dict[str, str]:
    return {
        str(file.relative_to(path)): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"))
        if file.is_file()
    }


def _publish_create_only(staging: Path) -> None:
    if OUTPUT_DIR.exists():
        if _directory_hashes(OUTPUT_DIR) != _directory_hashes(staging):
            raise RuntimeError(f"Existing sensitivity output differs from replay: {OUTPUT_DIR}")
        shutil.rmtree(staging)
        return
    staging.replace(OUTPUT_DIR)


def main() -> None:
    frozen_tests = _recompute_frozen_tests()
    normalized_h3 = _normalized_h3_tests()
    h5_event_time = _h5_event_time_tests()
    h5_exposure = _h5_exposure_gap_table()
    path_sensitivity = _analysis_path_sensitivity()
    target_probe_blocks = _target_probe_block_counts()
    payload_audit = _payload_content_audit()
    h1_audit = _h1_inventory_audit()

    h5_pre = frozen_tests[frozen_tests["contrast"].str.contains("H5 profile D_pre-minus-D_mid")]
    significant_global = frozen_tests[frozen_tests["holm_p_across_47"] < 0.05]
    normalized_significant = normalized_h3[normalized_h3["holm_p_across_32"] < 0.05]
    event_significant = h5_event_time[h5_event_time["holm_p_across_12"] < 0.05]
    h5_columns = [column for column in path_sensitivity if column.startswith("h5_pre_minus_mid_")]
    h5_path_sign_counts = {
        column.removeprefix("h5_pre_minus_mid_").removesuffix("_mean"): int((path_sensitivity[column] < 0).sum())
        for column in h5_columns
    }
    report = f"""# Gradient-mechanism repair: post-review sensitivity audit

This audit does not replace or modify the frozen development analysis. It addresses reviewer-identified scale,
multiplicity, provenance, and inventory ambiguities. Every inferential test below is two-sided.

## Verdict

- The frozen analysis contains **47 tests**: 7 H2/H3, 32 H3 repetition signatures, and 8 H5 profiles.
- **{len(significant_global)} tests survive Holm adjustment across all 47.** They are the four H5
  `D_pre-minus-D_mid` contrasts, one for every target. Their mean shifts range from
  {h5_pre['mean'].min():+.6f} to {h5_pre['mean'].max():+.6f} in normalized alignment units.
- The scale-free H3 sensitivity has **{len(normalized_significant)}/32** tests surviving its Holm adjustment.
  It therefore does not rescue the preregistered repetition mechanism from changing target-gradient or update norms.
- H2 temporal PL-versus-C4 revaluation and the primary m100a-versus-full H3 interaction remain null.

## Interpretation boundary

The robust H5 result is not a clean boundary or decay effect. Holding aggregate and contrast fixed while changing
`beta` forces the phase mixtures to change: the beta=0.60 and beta=0.85 arms use phase-0 StarCoder weights 0.0195 and
0.1196. Their cumulative-exposure gap is -0.1001 in `mid`, -0.0022 in `pre`, and +0.0004 in `post`; 62/64 `D_mid`
values are positive. Much of `D_pre-D_mid` therefore tracks closure of a designed-in exposure gap rather than an
isolated switch-time intervention.

The alignment difference remains negative when cumulative exposures nearly match, so cumulative exposure alone does
not fully collapse the state. The already-collected switch-relative sensitivity finds {len(event_significant)}/12
Holm-adjusted changes, but it lacks a same-time tied/no-switch control and remains post-outcome. The H5 shift has the
same sign for PL, C4, GitHub Python, and Wikipedia, so it is not specific evidence that the PL target becomes relatively
more valuable late. It does not identify gradient conflict, finite-support repetition, decay alignment, or causal
mediation of endpoint BPB.

H1's source-gradient cosine is descriptive. Its optimizer-update cosine does not become more conflicting before the
final zero-learning-rate state; the raw-gradient cosine becomes negative only at `final`. H1 therefore does not show
an actionable late-training conflict trend before the phase decision.

## Audit corrections

- H1 contains {h1_audit['h1_trajectories']} common-tied H2/H3 trajectories and zero H5 trajectories; the frozen
  report's “H2, H3, or H5” wording is incorrect.
- The repair already contains 96 `data_switch_*` rows, now reported as a post-review event-time sensitivity. The
  purpose-built `boundary_tied_018` no-switch control has zero repair rows and remains the missing decisive comparator.
- H2/H3/H5 did not freeze geometry/component. The reported projected-trunk path is one of 22 persisted paths. H5's
  pre-minus-mid sign is negative on {h5_path_sign_counts}; this is descriptive path robustness, not a repaired freeze.
- H5 target probe blocks are unequal: {target_probe_blocks}. Cross-target effect-size comparisons are therefore
  confounded by target-gradient precision; the four-block Wikipedia series is not a strong negative control.
- `endpoint_metrics_read=false` is an emitted invariant, not an independent leakage detector. Source inspection found
  no endpoint read path, but the flag alone cannot prove that fact.
- The true content digest covers all {payload_audit['row_payload_count']} row payload hashes:
  `{payload_audit['row_payload_identity_sha256']}`.
- All stage audits passed: `{STAGE_AUDIT_HASHES}`.

## Consequence for modeling

The first state variable justified by H5 is **cumulative per-source exposure**. The residual after exposure matching is
hypothesis-generating evidence that richer trajectory state may matter, not proof of a particular temporal mechanism.
A defensible surrogate should start with cumulative exposure and test whether gradient descriptors add held-out branch
prediction beyond it. It should not add a decay-alignment term or a specific finite-support repetition law from this
panel. The cheapest decisive repair is the 64-row `boundary_tied_018` no-switch gradient control, followed by a new
design that breaks the beta/phase-0-mixture collinearity.
"""

    OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{OUTPUT_DIR.name}.", dir=OUTPUT_DIR.parent))
    frozen_tests.to_csv(staging / "recomputed_frozen_tests_two_sided.csv", index=False)
    normalized_h3.to_csv(staging / "h3_normalized_sensitivity.csv", index=False)
    h5_event_time.to_csv(staging / "h5_event_time_sensitivity.csv", index=False)
    h5_exposure.to_csv(staging / "h5_cumulative_exposure_gap.csv", index=False)
    path_sensitivity.to_csv(staging / "analysis_path_sensitivity.csv", index=False)
    (staging / "target_probe_block_counts.json").write_text(
        json.dumps(target_probe_blocks, indent=2, sort_keys=True) + "\n"
    )
    (staging / "payload_content_audit.json").write_text(json.dumps(payload_audit, indent=2, sort_keys=True) + "\n")
    (staging / "h1_inventory_audit.json").write_text(json.dumps(h1_audit, indent=2, sort_keys=True) + "\n")
    (staging / "report.md").write_text(report)
    _publish_create_only(staging)


if __name__ == "__main__":
    main()
