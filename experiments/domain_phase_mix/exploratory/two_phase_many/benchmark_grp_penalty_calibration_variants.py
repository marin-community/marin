# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Benchmark GRP penalty/CES follow-up variants on the many-domain packet."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    PENALTY_CALIBRATION_VARIANT_NAMES,
    build_penalty_calibration_surrogate,
    compute_penalty_calibration_metrics,
    deploy_penalty_calibration_gaincapped_topkactual,
    penalty_calibration_variant_parameter_counts,
    tune_penalty_calibration_params,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

DEFAULT_VARIANTS = (
    "power_family",
    "power_boxcox_family",
    "power_family_penalty",
    "power_shared_penalty",
    "power_boxcox_family_penalty",
    "power_family_penalty_global_ftotal",
    "power_shared_penalty_global_ftotal",
    "power_boxcox_family_penalty_global_ftotal",
    "power_family_penalty_global_ftotal_pairces",
    "power_shared_penalty_global_ftotal_pairces",
    "power_boxcox_family_penalty_global_ftotal_pairces",
)
FAMILY_CURVATURE_BEST_CSV = Path(__file__).resolve().parent / "grp_family_curvature_benchmark_best.csv"
REFERENCE_STATE_PATH = (
    Path(__file__).resolve().parent / "chatgpt_pro_grp_recovery_packet" / "data" / "current_reference_state.json"
)
DEFAULT_OUTPUT_STEM = "grp_penalty_calibration_variants"

POWER_FAMILY_PENALTY_SEED = {
    "eta": 5.9558,
    "lam": 0.02468,
    "reg": 1e-3,
    "beta": 0.2613,
    "a_broad_text": 0.7271,
    "a_tech_code": 0.0200,
    "a_reasoning": 1.5122,
    "tau_broad_text": 2.869,
    "tau_tech_code": 4.013,
    "tau_reasoning": 5.352,
}
POWER_FAMILY_GLOBAL_FTOTAL_SEED = {
    "eta": 6.064786562155571,
    "lam": 0.0454985094487219,
    "reg": 0.0080164462678229,
    "beta": 0.2923458959006473,
    "a_broad_text": 0.5379544556598576,
    "a_tech_code": 0.0200,
    "a_reasoning": 0.1656222972637272,
    "tau": 4.1476331239284905,
    "tau_broad_text": 2.8690667025602514,
    "tau_tech_code": 4.050688977814653,
    "tau_reasoning": 5.338284119101838,
}
FAMILY_TAU_TEMPLATES = (
    (2.6, 3.8, 4.8),
    (2.9, 4.1, 5.3),
    (3.2, 4.5, 5.8),
)
GLOBAL_TAU_TEMPLATES = (2.8, 3.5, 4.2)
PAIR_RHO_TEMPLATES = (0.6, 0.8, 1.0)


def _output_paths(output_stem: str) -> dict[str, Path]:
    base_dir = Path(__file__).resolve().parent
    return {
        "coarse_csv": base_dir / f"{output_stem}_coarse.csv",
        "refine_csv": base_dir / f"{output_stem}_refine.csv",
        "best_csv": base_dir / f"{output_stem}_best.csv",
        "deploy_csv": base_dir / f"{output_stem}_deployments.csv",
        "summary_json": base_dir / f"{output_stem}_summary.json",
    }


def _validated_anchor_arrays() -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(REFERENCE_STATE_PATH.read_text())
    weights = np.stack(
        [
            np.asarray(payload["validated_global"]["phase_weights"], dtype=float),
            np.asarray(payload["validated_pair"]["phase_weights"], dtype=float),
        ],
        axis=0,
    )
    targets = np.asarray([payload["validated_global_bpb"], payload["validated_pair_bpb"]], dtype=float)
    return weights, targets


def _existing_best_params(variant_name: str) -> dict[str, float]:
    with FAMILY_CURVATURE_BEST_CSV.open() as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        if row["variant"] != variant_name:
            continue
        params: dict[str, float] = {
            "eta": float(row["eta"]),
            "lam": float(row["lam"]),
            "reg": float(row["reg"]),
            "beta": float(row["beta"]),
        }
        if row.get("alpha"):
            params["alpha"] = float(row["alpha"])
        for family_name in GENERIC_FAMILY_NAMES:
            params[f"a_{family_name}"] = float(row[f"a_{family_name}"])
        if row.get("tau"):
            params["tau"] = float(row["tau"])
        return params
    raise ValueError(f"Variant {variant_name!r} missing from {FAMILY_CURVATURE_BEST_CSV}")


def _dedupe_start_bank(start_bank: list[dict[str, float]]) -> tuple[dict[str, float], ...]:
    seen: set[tuple[tuple[str, float], ...]] = set()
    unique: list[dict[str, float]] = []
    for row in start_bank:
        key = tuple(sorted((key, round(float(value), 8)) for key, value in row.items()))
        if key in seen:
            continue
        seen.add(key)
        unique.append({key: float(value) for key, value in row.items()})
    return tuple(unique)


def _with_family_taus(
    base: dict[str, float],
    tau_broad_text: float,
    tau_tech_code: float,
    tau_reasoning: float,
) -> dict[str, float]:
    row = dict(base)
    row["tau_broad_text"] = float(tau_broad_text)
    row["tau_tech_code"] = float(tau_tech_code)
    row["tau_reasoning"] = float(tau_reasoning)
    return row


def _geometric_mean(values: tuple[float, ...]) -> float:
    return float(np.exp(np.mean(np.log(np.asarray(values, dtype=float)))))


def _collapse_family_curvature(params: dict[str, float]) -> dict[str, float]:
    row = dict(params)
    row["a"] = _geometric_mean(
        (
            float(row.pop("a_broad_text")),
            float(row.pop("a_tech_code")),
            float(row.pop("a_reasoning")),
        )
    )
    return row


def _baseline_start_bank(variant_name: str) -> tuple[dict[str, float], ...]:
    base = _existing_best_params(variant_name)
    starts = [
        dict(base),
        {
            **base,
            "eta": base["eta"] * 0.8,
            "reg": max(base["reg"] * 0.5, 1e-6),
            "beta": min(base["beta"] + 0.08, 0.95),
            "tau": base["tau"] + 0.6,
        },
        {
            **base,
            "eta": base["eta"] * 1.2,
            "reg": min(max(base["reg"] * 2.0, 1e-6), 1.0),
            "beta": max(base["beta"] - 0.08, 0.05),
            "tau": base["tau"] - 0.6,
        },
    ]
    return _dedupe_start_bank(starts)


def _variant_start_bank(variant_name: str) -> tuple[dict[str, float], ...]:
    if variant_name in {"power_family", "power_boxcox_family"}:
        return _baseline_start_bank(variant_name)
    if variant_name.startswith("power_shared_"):
        starts: list[dict[str, float]] = []
        if "pairces" in variant_name or "global_ftotal" in variant_name:
            seed = _collapse_family_curvature(POWER_FAMILY_GLOBAL_FTOTAL_SEED)
        else:
            seed = _collapse_family_curvature(POWER_FAMILY_PENALTY_SEED)
        if "pairces" in variant_name:
            seed["pair_rho"] = 0.8
        starts.append(dict(seed))

        shared_base = _collapse_family_curvature(_existing_best_params("power_family"))
        for tau_broad, tau_tech, tau_reason in FAMILY_TAU_TEMPLATES:
            row = _with_family_taus(shared_base, tau_broad, tau_tech, tau_reason)
            if "global_ftotal" in variant_name:
                row["tau"] = float(seed["tau"])
            if "pairces" in variant_name:
                row["pair_rho"] = 0.8
            starts.append(row)

        for a_scale in (0.7, 1.0, 1.3):
            row = dict(seed)
            row["a"] = float(np.clip(seed["a"] * a_scale, 0.02, 2.0))
            starts.append(row)

        if "global_ftotal" in variant_name:
            for tau in GLOBAL_TAU_TEMPLATES:
                row = dict(seed)
                row["tau"] = float(tau)
                starts.append(row)

        if "pairces" in variant_name:
            pair_base = dict(seed)
            for pair_rho in PAIR_RHO_TEMPLATES:
                starts.append({**pair_base, "pair_rho": float(pair_rho)})

        return _dedupe_start_bank(starts)

    is_mixed = variant_name.startswith("power_boxcox")
    base_variant = "power_boxcox_family" if is_mixed else "power_family"
    base = _existing_best_params(base_variant)
    starts: list[dict[str, float]] = []

    if "pairces" in variant_name or "global_ftotal" in variant_name:
        seed = dict(POWER_FAMILY_GLOBAL_FTOTAL_SEED)
    else:
        seed = dict(POWER_FAMILY_PENALTY_SEED)

    if is_mixed:
        seed["alpha"] = float(base["alpha"])
        for family_name in GENERIC_FAMILY_NAMES:
            seed[f"a_{family_name}"] = float(base[f"a_{family_name}"])
    if "pairces" in variant_name:
        seed["pair_rho"] = 0.8

    starts.append(dict(seed))

    for tau_broad, tau_tech, tau_reason in FAMILY_TAU_TEMPLATES:
        row = _with_family_taus(base, tau_broad, tau_tech, tau_reason)
        if "global_ftotal" in variant_name:
            row["tau"] = float(seed["tau"])
        if "pairces" in variant_name:
            row["pair_rho"] = 0.8
        starts.append(row)

    if "global_ftotal" in variant_name:
        for tau in GLOBAL_TAU_TEMPLATES:
            row = _with_family_taus(base, seed["tau_broad_text"], seed["tau_tech_code"], seed["tau_reasoning"])
            row["tau"] = float(tau)
            if "pairces" in variant_name:
                row["pair_rho"] = 0.8
            starts.append(row)

    if "pairces" in variant_name:
        pair_base = _with_family_taus(base, seed["tau_broad_text"], seed["tau_tech_code"], seed["tau_reasoning"])
        pair_base["tau"] = float(seed["tau"])
        for pair_rho in PAIR_RHO_TEMPLATES:
            starts.append({**pair_base, "pair_rho": float(pair_rho)})

    if "penalty" in variant_name and "global_ftotal" not in variant_name:
        starts.append(_with_family_taus(base, 2.8, 4.0, 5.3))

    return _dedupe_start_bank(starts)


def _deployment_row(packet, best_row: dict[str, Any]) -> dict[str, Any]:
    variant = str(best_row["variant"])
    model = build_penalty_calibration_surrogate(
        packet,
        params=_best_params_dict(best_row),
        variant_name=variant,
    ).fit(packet.base.w, packet.base.y)
    deployment = deploy_penalty_calibration_gaincapped_topkactual(packet, model, best_row)
    deploy_weights = np.asarray(deployment["weights"], dtype=float)
    distances = average_phase_tv_distance(packet.base.w, deploy_weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    return {
        "variant": variant,
        "predicted_optimum_value": float(deployment["predicted_optimum_value"]),
        "raw_predicted_optimum_value": float(deployment["raw_predicted_optimum_value"]),
        "hull_predicted_optimum_value": float(deployment["hull_predicted_optimum_value"]),
        "gain_budget": float(deployment["gain_budget"]),
        "delta": float(deployment["delta"]),
        "phase0_lt_1e4": int(np.sum(deploy_weights[0] < 1e-4)),
        "phase1_lt_1e4": int(np.sum(deploy_weights[1] < 1e-4)),
        "phase0_max_weight": float(np.max(deploy_weights[0])),
        "phase1_max_weight": float(np.max(deploy_weights[1])),
        "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_value": float(packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances[nearest_idx]),
    }


def _best_params_dict(row: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in row.items()
        if key
        in {
            "eta",
            "lam",
            "reg",
            "beta",
            "alpha",
            "tau",
            "tau_broad_text",
            "tau_tech_code",
            "tau_reasoning",
            "a_broad_text",
            "a_tech_code",
            "a_reasoning",
            "pair_rho",
        }
    }


def _write_outputs(
    coarse_frames: list[pd.DataFrame],
    refine_frames: list[pd.DataFrame],
    best_rows: list[dict[str, Any]],
    deploy_rows: list[dict[str, Any]],
    *,
    variants: list[str],
    output_stem: str,
) -> None:
    paths = _output_paths(output_stem)
    coarse_frame = pd.concat(coarse_frames, ignore_index=True) if coarse_frames else pd.DataFrame()
    refine_frame = pd.concat(refine_frames, ignore_index=True) if refine_frames else pd.DataFrame()
    best_frame = pd.DataFrame(best_rows)
    if not best_frame.empty:
        best_frame = best_frame.sort_values(
            ["objective", "cv_rmse", "cv_depopt_best8", "cv_rawopt_nearest_tv"],
            ascending=[True, True, True, True],
        )
    deploy_frame = pd.DataFrame(deploy_rows)
    if not deploy_frame.empty:
        deploy_frame = deploy_frame.sort_values(
            ["predicted_optimum_value", "delta"],
            ascending=[True, True],
        )

    coarse_frame.to_csv(paths["coarse_csv"], index=False)
    refine_frame.to_csv(paths["refine_csv"], index=False)
    best_frame.to_csv(paths["best_csv"], index=False)
    deploy_frame.to_csv(paths["deploy_csv"], index=False)
    paths["summary_json"].write_text(
        json.dumps(
            {
                "variants": variants,
                "best_rows": best_frame.to_dict(orient="records"),
                "deployment_rows": deploy_frame.to_dict(orient="records"),
                "coarse_csv": str(paths["coarse_csv"]),
                "refine_csv": str(paths["refine_csv"]),
                "best_csv": str(paths["best_csv"]),
                "deployment_csv": str(paths["deploy_csv"]),
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="*", default=list(DEFAULT_VARIANTS))
    parser.add_argument("--method", default="coarse")
    parser.add_argument("--coarse-top-k", type=int, default=3)
    parser.add_argument("--max-start-bank", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-stem", default=DEFAULT_OUTPUT_STEM)
    args = parser.parse_args()

    for variant in args.variants:
        if variant not in PENALTY_CALIBRATION_VARIANT_NAMES:
            raise ValueError(f"Unsupported variant {variant!r}")

    packet = load_generic_family_packet()
    valid_weights, valid_y = _validated_anchor_arrays()
    counts_by_variant = {
        variant: penalty_calibration_variant_parameter_counts(packet, variant) for variant in args.variants
    }

    coarse_frames: list[pd.DataFrame] = []
    refine_frames: list[pd.DataFrame] = []
    best_rows: list[dict[str, Any]] = []
    deploy_rows: list[dict[str, Any]] = []

    for variant in args.variants:
        start_bank = _variant_start_bank(variant)
        if args.max_start_bank is not None:
            start_bank = start_bank[: max(int(args.max_start_bank), 1)]
        print(f"[{variant}] tuning from {len(start_bank)} starts", flush=True)
        coarse, refine, tuned, _ = tune_penalty_calibration_params(
            packet,
            variant_name=variant,
            start_bank=start_bank,
            method=args.method,
            coarse_top_k=args.coarse_top_k,
            seed=args.seed,
        )
        coarse_frames.append(coarse.assign(**counts_by_variant[variant]))
        refine_frames.append(refine.assign(**counts_by_variant[variant]))

        model = build_penalty_calibration_surrogate(
            packet,
            params=_best_params_dict(tuned),
            variant_name=variant,
        ).fit(packet.base.w, packet.base.y)
        metrics = compute_penalty_calibration_metrics(
            packet,
            model,
            seed=args.seed,
            valid_weights=valid_weights,
            valid_y=valid_y,
        )
        best_row = {**tuned, **metrics, **counts_by_variant[variant]}
        best_rows.append(best_row)
        deploy_rows.append({**_deployment_row(packet, best_row), **counts_by_variant[variant]})
        print(
            f"[{variant}] objective={best_row['objective']:.6f} deploy={deploy_rows[-1]['predicted_optimum_value']:.6f}",
            flush=True,
        )
        _write_outputs(
            coarse_frames,
            refine_frames,
            best_rows,
            deploy_rows,
            variants=args.variants,
            output_stem=args.output_stem,
        )


if __name__ == "__main__":
    main()
