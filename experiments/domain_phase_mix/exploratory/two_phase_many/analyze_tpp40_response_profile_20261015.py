# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]
# ///
"""Author replay and paired descriptive readout of completed TP40-036 heads.

No fitting, feature integration, policy optimization, filtering unresolved heads,
or scientific gate changes. Requires the complete approved 240-head artifact.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.domain_phase_mix.exploratory.two_phase_many import run_tpp40_response_profile_20261015 as runner

OUTPUT = runner.OUTPUT / "descriptive"
ROLES = ("asymmetric", "matched_tied", "actual_tied")
PAIRS = ((runner.ARMS[1], runner.ARMS[0]), (runner.ARMS[2], runner.ARMS[0]), (runner.ARMS[2], runner.ARMS[1]))


def replay(matrix: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    """Reconstruct saved predictions explicitly, without calling the fitter."""
    selected = fit.get("selected")
    if selected is None:
        head = fit["profile"]["head"]
        eta = head["intercept"] + matrix @ np.asarray(head["coefficients"])
        return head["floor"] + np.exp(np.clip(eta, -30, 30))
    if selected["scale"] == 0:
        return np.full(len(matrix), selected["floor"])
    eta = matrix @ np.asarray(selected["amplitudes"]) - selected["normalization"]
    return selected["floor"] + np.exp(math.log(selected["scale"]) + eta)


def trial_coverage(fit: dict[str, Any], starts: list[dict[str, Any]]) -> None:
    """Check retained search history and training-only incumbent selection."""
    if "selected" not in fit:
        return
    trials, solves = fit["trials"], fit["solves"]
    if fit["starts"] != starts or len(solves) != 5:
        raise ValueError("The shared five initialization records differ")
    if [trial["trial_id"] for trial in trials] != list(range(len(trials))):
        raise ValueError("A trial was dropped or renumbered")
    finite = [item for item in trials if item["status"] == "finite"]
    winner = min(finite, key=lambda item: (item["scaled_objective"], item["trial_id"]))
    if fit["selected_trial"] != winner["trial_id"] or fit["selected"] != winner:
        raise ValueError("Saved incumbent is not the prescribed finite training-objective minimum")
    for solve in solves:
        items = trials[solve["first_trial"] : solve["last_trial"] + 1]
        if len(items) != solve["recorded_objective_calls"] or len(items) > 5000:
            raise ValueError("The recorded evaluation budget differs")
        if any(item["start_index"] != solve["start_index"] for item in items):
            raise ValueError("A solve contains another start's evaluations")


def role_records(frame: pd.DataFrame, spec: dict[str, Any], arm: str, unresolved: bool) -> list[dict[str, Any]]:
    """Retain empty strata and unresolved heads in the descriptive table."""
    records = []
    for role in ROLES:
        values = frame.loc[frame.role == role]
        error = values.predicted.to_numpy() - values.truth.to_numpy()
        records.append(
            {
                "world": spec["world"]["name"],
                "noise": spec["noise"],
                "noise_group": "clean" if spec["noise"] == "clean" else "noisy",
                "outer_fold": spec["outer_fold"],
                "arm": arm,
                "role": role,
                "count": len(error),
                "mse": float(np.mean(error**2)) if len(error) else None,
                "bias": float(np.mean(error)) if len(error) else None,
                "max_abs_error": float(np.max(abs(error))) if len(error) else None,
                "unresolved": unresolved,
            }
        )
    return records


def paired_roles(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute same-context differences; positive means the first arm is worse."""
    keys = ["world", "noise", "noise_group", "outer_fold", "role"]
    records = []
    for first, second in PAIRS:
        a = frame.loc[frame.arm == first]
        b = frame.loc[frame.arm == second]
        joined = a.merge(b, on=keys, suffixes=("_first", "_second"), validate="one_to_one")
        if len(joined) != len(a) or len(joined) != len(b):
            raise ValueError("A paired context is missing")
        for item in joined.to_dict("records"):
            if item["count_first"] != item["count_second"]:
                raise ValueError("Paired query strata differ")
            records.append(
                {key: item[key] for key in keys}
                | {
                    "comparison": f"{first}_minus_{second}",
                    "count": item["count_first"],
                    "mse_difference": item["mse_first"] - item["mse_second"],
                    "bias_difference": item["bias_first"] - item["bias_second"],
                    "max_abs_error_difference": item["max_abs_error_first"] - item["max_abs_error_second"],
                    "either_unresolved": item["unresolved_first"] or item["unresolved_second"],
                }
            )
    return pd.DataFrame(records)


def main() -> None:
    frozen = runner.execution_identity()
    if not runner.receipt(runner.OUTPUT, {"stage": "summary"}, frozen):
        raise ValueError("The full diagnostic must complete first")
    heads, roles, starts, dependencies = [], [], [], []
    maximum_replay_error = 0.0
    for spec in runner.contexts():
        world, outer, noise = spec["world"], spec["outer_fold"], spec["noise"]
        folder = runner.OUTPUT / "cells" / f"outer{outer}" / noise / world["name"]
        initialization = runner.read(folder / "initialization/starts.json")
        matrix = np.asarray(initialization["matrix"])
        data = runner.geometry(outer)
        with closing(sqlite3.connect(f"file:{runner.CACHE_PATH}?mode=ro&immutable=1", uri=True)) as connection:
            cache = runner.FrozenFeatures(connection, data, frozen)
            query_matrix = cache.matrix(data["query0"], data["query1"], world)
        for arm in runner.ARMS:
            target = folder / arm
            if not runner.receipt(target, spec | {"arm": arm}, frozen):
                raise ValueError("Do not omit incomplete or unresolved heads")
            fit, scores = runner.read(target / "fit.json"), runner.read(target / "scores.json")
            query = pd.read_csv(target / "predictions.csv")
            training = pd.read_csv(target / "training.csv")
            trial_coverage(fit, initialization["starts"])
            for design, frame in ((matrix, training), (query_matrix, query)):
                predicted = replay(design, fit)
                difference = float(np.max(abs(predicted - frame.predicted.to_numpy())))
                maximum_replay_error = max(maximum_replay_error, difference)
                np.testing.assert_allclose(predicted, frame.predicted, atol=2e-12, rtol=2e-13)
            np.testing.assert_allclose(training.observed, initialization["observed"], atol=2e-15, rtol=2e-15)
            if not query[["role", "source_index", "row_id"]].equals(data["query"]):
                raise ValueError("The query ordering differs from the frozen covariates")
            calculated = runner.score(query.predicted.to_numpy(), query.truth.to_numpy(), query)
            for name, expected in calculated.items():
                if expected is None:
                    if scores[name] is not None:
                        raise ValueError("An empty query stratum was silently filled")
                else:
                    np.testing.assert_allclose(scores[name], expected, atol=2e-12, rtol=2e-13)
            selected = fit.get("selected")
            selected_start = None if selected is None else selected["start_index"]
            head = fit["profile"]["head"] if selected is None else selected
            heads.append(
                {
                    "world": world["name"],
                    "noise": noise,
                    "noise_group": "clean" if noise == "clean" else "noisy",
                    "outer_fold": outer,
                    "arm": arm,
                    **scores,
                    "floor": head["floor"],
                    "floor_error": head["floor"] - 0.6,
                    "scale": None if selected is None else selected["scale"],
                    "selected_face": None if selected is None else selected["face"],
                    "projected_gradient_inf": None if selected is None else selected["projected_gradient_inf"],
                    "selected_profile_singular_values": (
                        None if selected is None else selected["profile_singular_values"]
                    ),
                    "amplitudes": head["coefficients"] if selected is None else selected["amplitudes"],
                }
            )
            roles.extend(role_records(query, spec, arm, scores["unresolved"]))
            for solve in fit.get("solves", []):
                starts.append(
                    {
                        "world": world["name"],
                        "noise": noise,
                        "outer_fold": outer,
                        "arm": arm,
                        "selected_start": solve["start_index"] == selected_start,
                        "best_start_objective": (
                            None
                            if solve["best_trial"] is None
                            else fit["trials"][solve["best_trial"]]["scaled_objective"]
                        ),
                        **{
                            name: solve.get(name)
                            for name in (
                                "start_index",
                                "success",
                                "status",
                                "message",
                                "termination",
                                "nit",
                                "nfev",
                                "njev",
                                "recorded_objective_calls",
                            )
                        },
                    }
                )
            dependencies.append(target / "complete.json")
    head_frame, role_frame = pd.DataFrame(heads), pd.DataFrame(roles)
    pairs = paired_roles(role_frame)
    pooled = (
        role_frame.groupby(["world", "noise_group", "arm", "role"], sort=True)
        .agg(
            contexts=("count", "size"),
            nonempty_contexts=("mse", "count"),
            equal_context_mean_mse=("mse", "mean"),
            equal_context_mean_bias=("bias", "mean"),
            unresolved_contexts=("unresolved", "sum"),
        )
        .reset_index()
    )
    paired_pooled = (
        pairs.groupby(["world", "noise_group", "comparison", "role"], sort=True)
        .agg(
            contexts=("count", "size"),
            nonempty_contexts=("mse_difference", "count"),
            equal_context_mean_mse_difference=("mse_difference", "mean"),
            either_unresolved_contexts=("either_unresolved", "sum"),
        )
        .reset_index()
    )
    gain_pairs = []
    for first, second in PAIRS:
        keys = ["world", "noise", "noise_group", "outer_fold"]
        joined = head_frame.loc[head_frame.arm == first].merge(
            head_frame.loc[head_frame.arm == second], on=keys, suffixes=("_first", "_second"), validate="one_to_one"
        )
        for item in joined.to_dict("records"):
            gain_pairs.append(
                {key: item[key] for key in keys}
                | {
                    "comparison": f"{first}_minus_{second}",
                    "gain_mse_difference": item["gain_rmse_first"] ** 2 - item["gain_rmse_second"] ** 2,
                    "floor_difference": item["floor_first"] - item["floor_second"],
                    "either_unresolved": item["unresolved_first"] or item["unresolved_second"],
                }
            )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    tables = {
        "heads.csv": head_frame,
        "roles.csv": role_frame,
        "starts.csv": pd.DataFrame(starts),
        "paired_roles.csv": pairs,
        "pooled_roles.csv": pooled,
        "paired_pooled_roles.csv": paired_pooled,
        "paired_gain.csv": pd.DataFrame(gain_pairs),
    }
    for name, table in tables.items():
        table.to_csv(OUTPUT / name, index=False)
    runner.write(
        OUTPUT / "readout.json",
        {
            "status": "author_saved_prediction_replay_and_descriptive_readout",
            "heads": len(heads),
            "starts": len(starts),
            "maximum_prediction_replay_error": maximum_replay_error,
            "unresolved_heads_retained": int(head_frame.unresolved.sum()),
            "failed_starts_retained": int(head_frame.failed_start_count.sum()),
            "failed_nonwinning_starts_retained": int(head_frame.nonwinning_failed_start_count.sum()),
            "clean_reference_applicable": int(head_frame.clean_correct_model_reference_applicable.sum()),
            "clean_reference_misses": int(
                (
                    head_frame.clean_query_recovery_reference_met.eq(False)
                    & head_frame.clean_correct_model_reference_applicable
                ).sum()
            ),
            "scientific_promotion": False,
            "limitations": [
                "This author replay reuses the frozen cache reader and scorer; "
                "it is not an independent end-to-end numerical audit.",
                "No conditional or nonlinear solver is rerun; "
                "unselected profiles receive structural coverage checks only.",
                "Paired differences use all contexts, including unresolved heads. Equal-context MSE is reported "
                "separately by world and role; empty actual-tied strata remain visible with explicit nonempty counts.",
                "Overlapping outer folds and three noise vectors do not justify calibrated confidence intervals "
                "or tail guarantees.",
                "No frontier/regret or raw optimum claims are added; inherited learned-floor failures remain failed.",
            ],
        },
    )
    runner.write(
        OUTPUT / "readout_receipt.json",
        {
            "source_sha256": runner.digest(Path(__file__)),
            "execution_identity": frozen,
            "dependencies": {str(path): runner.digest(path) for path in dependencies},
            "files": {name: runner.digest(OUTPUT / name) for name in (*tables, "readout.json")},
        },
    )
    print(json.dumps({"heads": len(heads), "starts": len(starts), "output": str(OUTPUT)}))


if __name__ == "__main__":
    main()
