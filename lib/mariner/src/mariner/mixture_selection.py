# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Epoch-aware pretraining-mixture selection from a proxy swarm, in one file.

This is the authoritative implementation of the procedure of the one-phase data-mixing paper: fit one
epoch-aware surrogate per evaluation task on a swarm of proxy runs, aggregate the tasks with the evaluation's own
weights, and minimize the aggregate over the target-feasible mixtures. It was extracted from the Marin research
code path and reproduces that code's fits on the calibration-pinned protocol to 1e-8 and its materialized policies
exactly (``tests/test_parity.py``).

Procedure (paper Section 4):

1. Materialized epochs: E_i(w) = w_i * (D_tgt / P_i), the number of passes over bucket i at weight w_i.
2. Per task t, a floored log-deficit surrogate on the shared response functions
   f_t(w) = phi_t + exp(c_t - sum_i alpha_ti b_t(E_i) + sum_i beta_ti h_t(E_i)), alpha, beta >= 0,
   b_t(E) = 1 - exp(-(rho_t E)^kappa_t), h_t(E) = softplus(log(1 + E) - tau_t)^2.
3. The floor phi_t = anchor_t - max(nu_t (anchor_t - min_r y_rt), 3 sigma_t), with the proportional mixture's
   mean and repeat SD as anchor and sigma. The head is fitted by nonnegative least squares on the log-deficit
   log(y - phi_t) with centered, unscaled columns and ridge weight gamma_t; the shape (rho_t, kappa_t, tau_t) and
   gamma_t come from a fixed grid by mixture-blocked inner cross-validation of the predictions in BPB at a
   provisional nu of 2.5; nu_t by a bounded scalar search on log nu in [1, 6] of the same inner-CV error, with a
   boundary rule: when the search ends within 10% of the log upper bound, nu_t is set to 1.5 (a conservative
   default chosen on development data, not a curvature test).
4. Aggregate sum_t a_t f_t(w), minimized by SLSQP from several feasible starts over the simplex (an epoch cap is
   optional and not part of the procedure: the reported optima are unconstrained), polished, then rounded to the
   sampler grid of 2,048 blocks with an exchange search. SLSQP finds a local minimizer; the multistart is the only
   guard against local optima.

Also included: the Olmix baseline as an independent implementation: the positive log-linear law per task fitted
with a Huber loss by scipy multistart (not upstream's torch fitter), and the exact KL-regularized proposer, which
solves the same convex program as allenai/olmix's exact proposer.

Inputs are CSV files (see ``data/`` and the README): swarm weights, swarm outcomes, objectives (component
weights), anchors, optional fold assignments, bucket pool sizes, and held-out banks for evaluation.

usage:
  python mixture_selection.py fit --objective table9 --out fit_table9.json
  python mixture_selection.py predict --fit fit_table9.json --mixtures mixtures.csv --out predictions.csv
  python mixture_selection.py optimize --fit fit_table9.json --cap 6 --out policy.csv
  python mixture_selection.py evaluate --fit fit_table9.json --bank data/heldout_bank_table9.csv
  python mixture_selection.py olmix --objective table9 --kl 0.005 --cap 4 --out olmix_table9.csv
  python mixture_selection.py self-test
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from mariner.fit import final_inner_folds, fit_objective
from mariner.model import Anchor, Objective, ObjectiveFit, Swarm
from mariner.olmix import FIT_SEED, FIT_STARTS, HUBER_DELTA, exact_proposal, fit_log_linear
from mariner.optimize import MIXTURE_BLOCK_SIZE, PANEL_STARTS, START_SEED, continuous_optimum, runtime_policy

# ---------------------------------------------------------------------------------------------------------------
# Constants of the frozen procedure (every value is the one the paper's runs used)
# ---------------------------------------------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

EXPOSURE_REFERENCE_TOKENS = 6_325_183_647_689  # D_tgt: the production budget whose repetition the proxies reproduce
CALIBRATION_MARKER = "baseline_proportional"  # swarm run(s) that anchor the floors; never scored out of fold


OUTER_FOLDS = 5

ACTIVE_TOLERANCE = 1e-10


PARITY_TOLERANCE = 1e-8


# ---------------------------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------------------------


def read_swarm(weights_path: Path, outcomes_path: Path, buckets_path: Path) -> Swarm:
    buckets_table = pd.read_csv(buckets_path)
    buckets = tuple(str(b) for b in buckets_table["bucket"])
    if "epochs_per_unit_weight" in buckets_table:
        epochs_per_unit_weight = buckets_table["epochs_per_unit_weight"].to_numpy(float)
    else:
        epochs_per_unit_weight = EXPOSURE_REFERENCE_TOKENS / buckets_table["pool_tokens"].to_numpy(float)
    weights_table = pd.read_csv(weights_path)
    outcomes_table = pd.read_csv(outcomes_path)
    if list(weights_table["run"]) != list(outcomes_table["run"]):
        raise ValueError("swarm weights and outcomes must list the same runs in the same order")
    weights = weights_table[list(buckets)].to_numpy(float)
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
        raise ValueError("swarm weights must sum to one per run")
    runs = tuple(str(r) for r in weights_table["run"])
    if "calibration" in weights_table:
        calibration = weights_table["calibration"].astype(bool).to_numpy()
    else:
        calibration = np.asarray([CALIBRATION_MARKER in r for r in runs])
    outcomes = outcomes_table.drop(columns=["run"]).astype(float)
    return Swarm(runs, buckets, weights, epochs_per_unit_weight, outcomes, calibration)


def read_objectives(path: Path) -> dict[str, Objective]:
    table = pd.read_csv(path)
    result = {}
    for name, group in table.groupby("objective", sort=False):
        result[str(name)] = Objective(str(name), tuple(group["component"].astype(str)), group["weight"].to_numpy(float))
    return result


def read_anchors(path: Path, objective: str) -> dict[str, Anchor]:
    table = pd.read_csv(path)
    table = table[table["objective"].eq(objective)]
    return {str(r.component): Anchor(float(r.proportional_bpb), float(r.repeat_sd)) for r in table.itertuples()}


# ---------------------------------------------------------------------------------------------------------------
# Evaluation on a held-out bank
# ---------------------------------------------------------------------------------------------------------------


def bank_metrics(predicted: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    pick = int(np.argmin(predicted))
    return {
        "rows": float(len(measured)),
        "regret_at_1": float(measured[pick] - measured.min()),
        "selected_rank": float(np.sum(measured <= measured[pick])),
        "optimism_at_pick": float(measured[pick] - predicted[pick]),
        "rmse": float(np.sqrt(np.mean((predicted - measured) ** 2))),
        "spearman": float(spearmanr(predicted, measured).correlation),
    }


# ---------------------------------------------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------------------------------------------


def _default_inputs(args: argparse.Namespace) -> tuple[Swarm, dict[str, Objective], Path, pd.DataFrame | None, Path]:
    swarm = read_swarm(args.swarm_weights, args.swarm_outcomes, args.buckets)
    objectives = read_objectives(args.objectives)
    splits = pd.read_csv(args.splits) if args.splits is not None and Path(args.splits).exists() else None
    return swarm, objectives, args.anchors, splits, args.buckets


def _add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--swarm-weights", type=Path, default=DATA / "swarm_weights.csv")
    parser.add_argument("--swarm-outcomes", type=Path, default=DATA / "swarm_outcomes.csv")
    parser.add_argument("--buckets", type=Path, default=DATA / "buckets.csv")
    parser.add_argument("--objectives", type=Path, default=DATA / "objectives.csv")
    parser.add_argument("--anchors", type=Path, default=DATA / "anchors.csv")
    parser.add_argument(
        "--splits", type=Path, default=DATA / "splits.csv", help="fold table; recomputed with k-means if absent"
    )


def command_fit(args: argparse.Namespace) -> None:
    swarm, objectives, anchors_path, splits, _ = _default_inputs(args)
    objective = objectives[args.objective]
    fit = fit_objective(
        swarm, objective, read_anchors(anchors_path, args.objective), final_inner_folds(swarm, splits), print
    )
    Path(args.out).write_text(json.dumps(fit.to_json(), indent=1))
    print(f"wrote {args.out}: {len(fit.tasks)} tasks")


def command_predict(args: argparse.Namespace) -> None:
    fit = ObjectiveFit.from_json(json.loads(Path(args.fit).read_text()))
    table = pd.read_csv(args.mixtures)
    mixtures = table[list(fit.buckets)].to_numpy(float)
    out = table[[c for c in table.columns if c not in fit.buckets]].copy()
    out[f"predicted_{fit.objective}"] = fit.predict(mixtures)
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


def command_optimize(args: argparse.Namespace) -> None:
    swarm = read_swarm(args.swarm_weights, args.swarm_outcomes, args.buckets)
    fit = ObjectiveFit.from_json(json.loads(Path(args.fit).read_text()))
    weights, records = continuous_optimum(fit, swarm.weights, args.cap, args.kl)
    counts, summary = runtime_policy(fit, weights, args.cap, args.kl)
    table = pd.DataFrame(
        {
            "bucket": fit.buckets,
            "continuous_weight": weights,
            "runtime_count": counts,
            "runtime_weight": counts / MIXTURE_BLOCK_SIZE,
            "materialized_epochs": counts / MIXTURE_BLOCK_SIZE * fit.epochs_per_unit_weight,
        }
    )
    table.to_csv(args.out, index=False)
    diagnostics = {
        "fit": str(args.fit),
        "cap": args.cap,
        "kl": args.kl,
        "block_size": MIXTURE_BLOCK_SIZE,
        "start_seed": START_SEED,
        "panel_starts": PANEL_STARTS,
        **summary,
        "starts": records,
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(diagnostics, indent=1, default=float))
    print(json.dumps(diagnostics, indent=1, default=float))


def command_evaluate(args: argparse.Namespace) -> None:
    fit = ObjectiveFit.from_json(json.loads(Path(args.fit).read_text()))
    bank = pd.read_csv(args.bank)
    predicted = fit.predict(bank[list(fit.buckets)].to_numpy(float))
    print(json.dumps(bank_metrics(predicted, bank["measured_mean_bpb"].to_numpy(float)), indent=1))


def command_olmix(args: argparse.Namespace) -> None:
    swarm, objectives, _, _, _ = _default_inputs(args)
    objective = objectives[args.objective]
    laws = [fit_log_linear(swarm.weights, swarm.outcomes[c].to_numpy(float), args.delta) for c in objective.tasks]
    natural = 1.0 / swarm.epochs_per_unit_weight
    natural = natural / natural.sum()
    caps = np.minimum(1.0, args.cap / swarm.epochs_per_unit_weight)
    solution, predicted, status = exact_proposal(laws, objective.weights, natural, caps, args.kl)
    pd.DataFrame(
        {"bucket": swarm.buckets, "weight": solution, "materialized_epochs": solution * swarm.epochs_per_unit_weight}
    ).to_csv(args.out, index=False)
    laws_json = [
        {"task": c, "log_c": law.log_c, "coefficients": law.coefficients.tolist(), "huber_loss": law.huber_loss}
        for c, law in zip(objective.tasks, laws, strict=True)
    ]
    record = {
        "objective": args.objective,
        "kl_reg": args.kl,
        "cap": args.cap,
        "huber_delta": args.delta,
        "starts": FIT_STARTS,
        "seed": FIT_SEED,
        "solver": status,
        "predicted": predicted,
        "max_epochs": float((solution * swarm.epochs_per_unit_weight).max()),
        "laws": laws_json,
    }
    Path(args.out).with_suffix(".json").write_text(json.dumps(record, indent=1))
    print(json.dumps({k: v for k, v in record.items() if k != "laws"}, indent=1))


def self_test() -> None:
    """Refit both objectives on the bundled swarm and check parity with the bundled reference fits and policies."""
    swarm = read_swarm(DATA / "swarm_weights.csv", DATA / "swarm_outcomes.csv", DATA / "buckets.csv")
    objectives = read_objectives(DATA / "objectives.csv")
    splits = pd.read_csv(DATA / "splits.csv")
    folds = final_inner_folds(swarm, splits)
    policies = pd.read_csv(DATA / "reference_policies.csv")
    # The unconstrained optimum is the procedure's proposal; the capped cases are the sensitivity policies.
    for name, cases in (
        ("uncheatable", ((None, "lwspu_u_snc_cap06"), (6, "lwspu_u_snc_cap06"))),
        ("table9", ((None, "lwspu_t9_snc_cap08"), (6, "lwspu_t9_snc_cap06"), (8, "lwspu_t9_snc_cap08"))),
    ):
        fit = fit_objective(swarm, objectives[name], read_anchors(DATA / "anchors.csv", name), folds)
        reference = pd.read_csv(DATA / f"reference_swarm_predictions_{name}.csv")
        error = float(
            np.max(np.abs(fit.predict_tasks(swarm.weights) - reference[list(objectives[name].tasks)].to_numpy(float)))
        )
        bank = pd.read_csv(DATA / f"heldout_bank_{name}.csv")
        reference_bank = pd.read_csv(DATA / f"reference_bank_predictions_{name}.csv")
        bank_error = float(
            np.max(
                np.abs(
                    fit.predict_tasks(bank[list(swarm.buckets)].to_numpy(float))
                    - reference_bank[list(objectives[name].tasks)].to_numpy(float)
                )
            )
        )
        print(f"{name}: swarm prediction parity {error:.1e}, bank {bank_error:.1e}")
        if max(error, bank_error) > PARITY_TOLERANCE:
            raise AssertionError(f"{name}: prediction parity failed")
        for cap, candidate in cases:
            weights, _ = continuous_optimum(fit, swarm.weights, cap)
            counts, summary = runtime_policy(fit, weights, cap)
            expected = (
                policies[policies.candidate_id.eq(candidate)]
                .set_index("domain")
                .runtime_count.reindex(swarm.buckets)
                .fillna(0)
                .to_numpy(int)
            )
            same = bool(np.array_equal(counts, expected))
            label = "unconstrained" if cap is None else f"cap {cap}"
            print(
                f"{name} {label}: predicted {summary['runtime_prediction']:.4f}, "
                f"realized policy identical to reference: {same}"
            )
            if not same:
                raise AssertionError(f"{name} {label}: realized policy differs from the reference")
    print("self-test passed")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit", help="fit one objective's surrogate on a swarm")
    _add_data_arguments(fit)
    fit.add_argument("--objective", required=True)
    fit.add_argument("--out", required=True)
    fit.set_defaults(func=command_fit)
    predict = commands.add_parser(
        "predict", help="predict the objective for mixtures in a CSV with one column per bucket"
    )
    predict.add_argument("--fit", required=True)
    predict.add_argument("--mixtures", required=True)
    predict.add_argument("--out", required=True)
    predict.set_defaults(func=command_predict)
    optimize = commands.add_parser(
        "optimize", help="minimize the fitted aggregate over the simplex and round to the sampler grid"
    )
    _add_data_arguments(optimize)
    optimize.add_argument("--fit", required=True)
    optimize.add_argument(
        "--cap",
        type=float,
        default=None,
        help="optional per-bucket epoch cap (not part of the procedure; the reported optima are unconstrained)",
    )
    optimize.add_argument(
        "--kl",
        type=float,
        default=0.0,
        help="optional KL penalty toward the proportional mixture (not part of the procedure)",
    )
    optimize.add_argument("--out", required=True)
    optimize.set_defaults(func=command_optimize)
    evaluate = commands.add_parser("evaluate", help="selection metrics on a held-out bank")
    evaluate.add_argument("--fit", required=True)
    evaluate.add_argument("--bank", required=True)
    evaluate.set_defaults(func=command_evaluate)
    olmix = commands.add_parser("olmix", help="the Olmix baseline: log-linear laws and the exact KL proposer")
    _add_data_arguments(olmix)
    olmix.add_argument("--objective", required=True)
    olmix.add_argument("--kl", type=float, required=True)
    olmix.add_argument("--cap", type=float, default=4.0)
    olmix.add_argument("--delta", type=float, default=HUBER_DELTA)
    olmix.add_argument("--out", required=True)
    olmix.set_defaults(func=command_olmix)
    test = commands.add_parser("self-test", help="parity with the bundled reference fits and policies")
    test.set_defaults(func=lambda args: self_test())
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
