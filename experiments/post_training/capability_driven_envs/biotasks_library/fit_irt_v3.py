"""Fit exploratory Bayesian 2PL on the immutable BioTasks v3 response snapshot.

Requires CmdStan and cmdstanpy. Keep the private input and per-item outputs
outside git. This fits observed scored responses; it does not model missingness.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SNAPSHOT_SHA256 = "acf9b5aff6a47230e2e0d0c751554e03969d4af9fbbb22099154db1aed039cda"
MODELS = 12
ITEMS = 230


def load_responses(
    path: Path,
    *,
    include_submission_failures: bool,
    additional_path: Path | None = None,
    additional_sha256: str | None = None,
) -> tuple[dict[str, Any], list[str], list[str], dict[str, Any]]:
    """Verify frozen source hashes and join complete matrices without imputation."""
    if (additional_path is None) != (additional_sha256 is None):
        raise ValueError("The additional snapshot and its SHA-256 must be supplied together")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != SNAPSHOT_SHA256:
        raise ValueError("Snapshot SHA-256 does not match frozen v3 response artifact")
    rows = [json.loads(line) for line in content.splitlines() if line]
    build_data(rows, include_submission_failures=include_submission_failures)
    hashes = [SNAPSHOT_SHA256]
    if additional_path is not None and additional_sha256 is not None:
        new_content = additional_path.read_bytes()
        if hashlib.sha256(new_content).hexdigest() != additional_sha256:
            raise ValueError("Additional response snapshot SHA-256 mismatch")
        additional = [json.loads(line) for line in new_content.splitlines() if line]
        build_data(additional, include_submission_failures=include_submission_failures, expected_models=20)
        if {row["task_id"] for row in rows} != {row["task_id"] for row in additional}:
            raise ValueError("Additional snapshot uses different tasks")
        rows.extend(additional)
        hashes.append(additional_sha256)
    data, models, items = build_data(
        rows, include_submission_failures=include_submission_failures, expected_models=MODELS if len(hashes) == 1 else 32
    )
    metadata = {
        "input_snapshot_sha256": hashes,
        "n_submission_failure_zeros_included": (
            sum(row["grade_status"] == "submission_failure" for row in rows) if include_submission_failures else 0
        ),
    }
    return data, models, items, metadata


def build_data(
    rows: list[dict[str, Any]], *, include_submission_failures: bool, expected_models: int = MODELS
) -> tuple[dict[str, Any], list[str], list[str]]:
    """Validate population and map observed scores into Stan's indexing."""
    models = sorted({r["model"] for r in rows})
    items = sorted({r["task_id"] for r in rows})
    if len(models) != expected_models or len(items) != ITEMS or len(rows) != expected_models * ITEMS:
        raise ValueError("Unexpected model/item matrix dimensions")
    if len({(r["model"], r["task_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate respondent/item pair")
    if any(
        r["grade_status"] not in {"graded", "submission_failure", "ungraded", "missing_infrastructure"} for r in rows
    ):
        raise ValueError("Unknown grade status")
    if any(
        (r["score"] in (0, 1)) != (r["grade_status"] in {"graded", "submission_failure"})
        or (r["grade_status"] == "submission_failure" and r["score"] != 0)
        for r in rows
    ):
        raise ValueError("Inconsistent frozen grade/score")
    selected = [
        r for r in rows if r["score"] is not None and (include_submission_failures or r["grade_status"] == "graded")
    ]
    model_index = {m: i + 1 for i, m in enumerate(models)}
    item_index = {item: i + 1 for i, item in enumerate(items)}
    data = {
        "M": len(models),
        "I": len(items),
        "N": len(selected),
        "respondent": [model_index[r["model"]] for r in selected],
        "item": [item_index[r["task_id"]] for r in selected],
        "y": [r["score"] for r in selected],
    }
    return data, models, items


def interval(draws: np.ndarray) -> dict[str, float]:
    """Return a posterior median and equal-tailed 95% credible interval."""
    low, median, high = np.quantile(draws, (0.025, 0.5, 0.975))
    return {"median": float(median), "low": float(low), "high": float(high)}


def summarize(
    fit: Any, data: dict[str, Any], models: list[str], items: list[str], mode: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize identification, item slopes and model-rank uncertainty."""
    theta = fit.stan_variable("theta")
    log_a = fit.stan_variable("log_discrimination")
    difficulty = fit.stan_variable("difficulty")
    a = np.exp(log_a)
    rank = np.argsort(np.argsort(-theta, axis=1), axis=1) + 1
    model_rows = [
        {
            "model": model,
            "ability": interval(theta[:, index]),
            "rank": interval(rank[:, index]),
            "prob_best": float(np.mean(rank[:, index] == 1)),
            "n_scored": int(np.sum(np.asarray(data["respondent"]) == index + 1)),
        }
        for index, model in enumerate(models)
    ]
    item_rows = [
        {
            "item": item,
            "difficulty": interval(difficulty[:, index]),
            "discrimination": interval(a[:, index]),
            "log_discrimination_sd": float(np.std(log_a[:, index], ddof=1)),
            "prob_discrimination_gt_1": float(np.mean(a[:, index] > 1)),
            "n_scored": int(np.sum(np.asarray(data["item"]) == index + 1)),
        }
        for index, item in enumerate(items)
    ]
    diagnostics = fit.summary()
    variables = fit.method_variables()
    divergences = int(np.sum(variables["divergent__"]))
    parameter_rows = diagnostics.loc[diagnostics.index.str.startswith(("theta[", "difficulty[", "log_discrimination["))]
    assert len(parameter_rows) == len(models) + 2 * len(items)
    report = {
        "status": "EXPLORATORY_NOT_VALIDATED",
        "snapshot_sha256": SNAPSHOT_SHA256,
        "mode": mode,
        "n_scored": data["N"],
        "n_correct": int(sum(data["y"])),
        "n_submission_failure_zeros_included": 396 if mode == "all_scored" else 0,
        "n_items_with_no_scored_responses": int(sum(r["n_scored"] == 0 for r in item_rows)),
        "divergences": divergences,
        "parameter_rhat_max": float(parameter_rows["R_hat"].max()),
        "parameter_ess_bulk_min": float(parameter_rows["ESS_bulk"].min()),
        "median_item_log_discrimination_sd": float(np.median([r["log_discrimination_sd"] for r in item_rows])),
        "items_with_95pct_discrimination_gt_1": sum(r["prob_discrimination_gt_1"] > 0.95 for r in item_rows),
        "items_with_95pct_discrimination_lt_1": sum(r["prob_discrimination_gt_1"] < 0.05 for r in item_rows),
        "model_posteriors": model_rows,
    }
    return report, item_rows


def main() -> None:
    from cmdstanpy import CmdStanModel, from_csv  # noqa: PLC0415 - optional fitting dependency

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Private frozen JSONL; hash verified before fitting")
    parser.add_argument("--additional-input", type=Path, help="Private frozen 20-model JSONL")
    parser.add_argument("--additional-sha256", help="SHA-256 from the effective20 frozen manifest")
    parser.add_argument(
        "--output", type=Path, required=True, help="Private local output directory; never commit raw draws"
    )
    parser.add_argument("--mode", choices=("all_scored", "graded_only"), required=True)
    parser.add_argument("--warmup", type=int, default=600)
    parser.add_argument("--samples", type=int, default=600)
    parser.add_argument("--seed", type=int, default=324)
    parser.add_argument("--reuse-csv", action="store_true", help="Analyze already sampled CSV chains in output")
    args = parser.parse_args()
    if args.warmup < 100 or args.samples < 100:
        parser.error("At least 100 warmup and sampling iterations required")
    args.output.mkdir(parents=True, exist_ok=True)
    data, models, items, input_metadata = load_responses(
        args.input,
        include_submission_failures=args.mode == "all_scored",
        additional_path=args.additional_input,
        additional_sha256=args.additional_sha256,
    )
    if args.reuse_csv:
        chains = sorted(args.output.glob("irt_v3-*.csv"))
        if len(chains) != 4:
            raise ValueError(f"Expected exactly four saved chains, found {len(chains)}")
        fit = from_csv([str(chain) for chain in chains], method="sample")
        if fit is None:
            raise ValueError("Could not load saved Stan chains")
    else:
        model = CmdStanModel(stan_file=str(Path(__file__).with_name("irt_v3.stan")))
        fit = model.sample(
            data=data,
            chains=4,
            parallel_chains=4,
            iter_warmup=args.warmup,
            iter_sampling=args.samples,
            seed=args.seed,
            adapt_delta=0.95,
            output_dir=str(args.output),
            show_progress=False,
        )
    report, item_rows = summarize(fit, data, models, items, args.mode)
    report.update(input_metadata)
    report["n_models"] = len(models)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "private_item_posteriors.json").write_text(json.dumps(item_rows, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
