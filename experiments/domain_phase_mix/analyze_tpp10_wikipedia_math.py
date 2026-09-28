# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare verified Wikipedia and FineMath proxy math likelihoods without launching jobs."""

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from experiments.domain_phase_mix import analyze_tpp10_finemath_math as previous
from experiments.domain_phase_mix import evaluate_tpp10_finemath_math as evaluation
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

DIRECTORY = evaluation.DIRECTORY.parent / "wikipedia_math_eval_20260912"
WRAPPER = Path(__file__).with_name("evaluate_tpp10_wikipedia_math.py")
DOMAINS = {"wikipedia": "Wikipedia", "finemath_3plus": "FineMath-3+"}
COLORS = {"wikipedia": "#0072B2", "finemath_3plus": "#D55E00"}
SHARED_FIELDS = (
    "sources",
    "tokenizer_pins",
    "primary",
    "secondary",
    "metric",
    "format",
    "context",
    "stride",
    "eos_policy",
    "boundary_policy",
    "aggregation",
    "paloma_path",
    "batch_size",
    "code_sha256",
    "population_counts",
)


def validate_comparison(spec: dict) -> dict:
    """Require identical scoring, data, tokenizer, and the same zero-fraction checkpoint."""
    evaluation.validate_spec(spec)
    reference = spec["reference_spec"]
    evaluation.validate_spec(reference)
    if file_sha256(WRAPPER) != spec["wrapper_code_sha256"]:
        raise ValueError("Wikipedia evaluation wrapper changed after the specification was frozen")
    for field in SHARED_FIELDS:
        if spec[field] != reference[field]:
            raise ValueError(f"Wikipedia and FineMath scoring specifications differ: {field}")
    if spec["endpoints"][0] != reference["endpoints"][0]:
        raise ValueError("The two curves do not share the identical zero-fraction endpoint")
    for domain, current in (("wikipedia", spec), ("finemath_3plus", reference)):
        for endpoint in current["endpoints"][1:]:
            request = endpoint["request"]
            if request["domain"] != domain or request["arm"] != "matched":
                raise ValueError(f"Expected only matched {domain} proxies: {request['run_name']}")
    return reference


def archived_points(spec: dict, domain: str, archive: Path) -> dict[int, dict]:
    """Reproduce each endpoint's materialized epochs from the frozen block allocation."""
    preflight = previous.read_json(archive / "preflight.json")
    selected = [
        curve
        for curve in previous.read_json(archive / "curves.json")["curves"]
        if curve["domain"] == domain and curve["arm"] == "matched"
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected exactly one archived matched curve for {domain}")
    points = {point["percent"]: point for point in selected[0]["points"]}
    if len(points) != len(selected[0]["points"]) or tuple(sorted(points)) != evaluation.GRID:
        raise ValueError(f"Archived {domain} curve does not contain the unique complete grid")
    coordinates = {point["percent"]: point for point in preflight["allocation"]["coordinates"]}
    pool_tokens = preflight["data"]["caches"][f"{domain}/matched"]["tokens"]
    for endpoint in spec["endpoints"]:
        request = endpoint["request"]
        percent = request["percent"]
        point = points[percent]
        expected_name = "shared p0" if percent == 0 else request["run_name"]
        if point["run_name"] != expected_name:
            raise ValueError(f"Archived run identity differs: {domain}, p={percent}")
        if not math.isclose(point["epochs"], coordinates[percent]["matched_epochs"], abs_tol=1e-12):
            raise ValueError(f"Archived epoch coordinates disagree: {domain}, p={percent}")
        if percent:
            if request["support_sequences"] * evaluation.experiment.SEQ_LEN != pool_tokens:
                raise ValueError(f"Matched pool size differs: {domain}, p={percent}")
            allocated = coordinates[percent]["proxy_allocation"]["starcoder"]
            if not math.isclose(point["epochs"], allocated / request["support_sequences"], abs_tol=1e-12):
                raise ValueError(f"Materialized allocation does not reproduce epochs: {domain}, p={percent}")
    return points


def collect(spec: dict, domain: str, points: dict[int, dict]) -> list[dict]:
    """Keep every requested endpoint, including absent receipts, and retain full measured metrics."""
    rows = []
    population_digest = canonical_sha256(spec["population_counts"])
    for endpoint in spec["endpoints"]:
        request = endpoint["request"]
        result = evaluation.verified_result(spec, endpoint)
        if result is not None and result["population_sha256"] != population_digest:
            raise ValueError(f"Scored population differs: {request['run_name']}")
        rows.append(
            {
                "domain": domain,
                "percent": request["percent"],
                "epochs": points[request["percent"]]["epochs"],
                "status": "verified" if result is not None else "missing",
                "run_name": request["run_name"],
                "shared_zero_fraction_checkpoint": request["percent"] == 0,
                "spec_sha256": spec["spec_sha256"],
                "receipt_uri": f"{evaluation.output_root(spec)}/{request['run_name']}.json",
                "request": request,
                "checkpoint": endpoint["checkpoint"],
                "archived_metrics": points[request["percent"]]["metrics"],
                "result": result,
            }
        )
    return rows


def cross_choice_regret(rows: dict[str, list[dict]], summaries: dict[str, dict], dataset: str) -> dict:
    """Evaluate each domain curve at the fraction selected on the other complete curve."""
    if any(not summaries[domain][dataset].get("complete_grid", False) for domain in DOMAINS):
        return {"status": "incomplete", "comparisons": []}
    comparisons = []
    for domain in DOMAINS:
        other = next(name for name in DOMAINS if name != domain)
        own_minimum = summaries[domain][dataset]["minimum"]
        other_minimum = summaries[other][dataset]["minimum"]
        selected = next(row for row in rows[domain] if row["percent"] == other_minimum["percent"])
        loss = selected["result"]["metrics"][f"eval/{dataset}/loss"]
        perplexity = selected["result"]["perplexity"][dataset]
        comparisons.append(
            {
                "evaluated_training_domain": domain,
                "selection_training_domain": other,
                "selected_percent": selected["percent"],
                "selected_epochs": selected["epochs"],
                "own_best_percent": own_minimum["percent"],
                "own_best_epochs": own_minimum["epochs"],
                "loss_at_other_choice": loss,
                "perplexity_at_other_choice": perplexity,
                "loss_regret": loss - own_minimum["loss"],
                "perplexity_regret": perplexity - own_minimum["perplexity"],
                "relative_perplexity_increase": perplexity / own_minimum["perplexity"] - 1,
            }
        )
    return {"status": "complete", "comparisons": comparisons}


def write_csv(rows: dict[str, list[dict]], destination: Path) -> None:
    flat_rows = []
    for domain_rows in rows.values():
        for row in domain_rows:
            flat = {
                key: value
                for key, value in row.items()
                if key not in {"result", "request", "checkpoint", "archived_metrics"}
            }
            flat["request_json"] = json.dumps(row["request"], sort_keys=True)
            flat["checkpoint_json"] = json.dumps(row["checkpoint"], sort_keys=True)
            flat["archived_metrics_json"] = json.dumps(row["archived_metrics"], sort_keys=True)
            result = row["result"]
            flat["math_receipt_json"] = json.dumps(result, sort_keys=True) if result is not None else None
            for dataset in previous.DATASETS:
                flat[f"{dataset}_loss"] = result["metrics"][f"eval/{dataset}/loss"] if result is not None else None
                flat[f"{dataset}_perplexity"] = result["perplexity"][dataset] if result is not None else None
            flat_rows.append(flat)
    with destination.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)


def plot_results(rows: dict[str, list[dict]], summaries: dict[str, dict], destination: Path) -> list[str]:
    """Overlay raw points on common metric axes; missing measurements remain gaps."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.15), layout="constrained", sharex=True)
    for ax, (dataset, label) in zip(axes, previous.DATASETS.items(), strict=True):
        for domain, domain_rows in rows.items():
            epochs = np.array([row["epochs"] for row in domain_rows])
            values = np.array(
                [row["result"]["perplexity"][dataset] if row["result"] is not None else np.nan for row in domain_rows]
            )
            if not np.isfinite(values).any():
                continue
            ax.plot(epochs, values, "o-", color=COLORS[domain], linewidth=1.5, markersize=4.5, label=DOMAINS[domain])
            minima = summaries[domain][dataset]["tied_minima"]
            ax.scatter(
                [point["epochs"] for point in minima],
                [point["perplexity"] for point in minima],
                marker="*",
                s=145,
                color=COLORS[domain],
                edgecolors="black",
                linewidths=0.7,
                zorder=4,
            )
        baseline = rows["finemath_3plus"][0]["result"]["perplexity"][dataset]
        ax.scatter([0], [baseline], s=34, marker="D", color="#333333", zorder=5, label="Shared zero-fraction checkpoint")
        role = "primary" if dataset == "math500" else "secondary"
        ax.set_title(f"{label} · {role}", fontsize=11)
        ax.set_ylabel("Reference-solution perplexity")
        ax.set_xlabel("Materialized epochs of the varied domain")
        ax.set_xlim(-0.4, 16.3)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.margins(y=0.12)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    verified = {domain: sum(row["result"] is not None for row in domain_rows) for domain, domain_rows in rows.items()}
    fig.suptitle(
        f"Matched TPP-10 proxies · Wikipedia {verified['wikipedia']}/8, "
        f"FineMath {verified['finemath_3plus']}/8 verified",
        fontsize=12,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    paths = []
    for extension in ("png", "pdf"):
        path = destination / f"math_likelihood_comparison.{extension}"
        fig.savefig(path, dpi=190)
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def plot_valley_detail(rows: dict[str, list[dict]], summaries: dict[str, dict], destination: Path) -> list[str]:
    """Show the first seven mixture fractions alongside explicitly retained full-range plots."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), layout="constrained", sharex=True)
    maximum_percent = 70
    for ax, (dataset, label) in zip(axes, previous.DATASETS.items(), strict=True):
        for domain, domain_rows in rows.items():
            selected = [row for row in domain_rows if row["percent"] <= maximum_percent]
            epochs = np.array([row["epochs"] for row in selected])
            values = np.array(
                [row["result"]["perplexity"][dataset] if row["result"] is not None else np.nan for row in selected]
            )
            if not np.isfinite(values).any():
                continue
            ax.plot(epochs, values, "o-", color=COLORS[domain], linewidth=1.5, markersize=4.5, label=DOMAINS[domain])
            minima = summaries[domain][dataset]["tied_minima"]
            for point in minima:
                if point["percent"] > maximum_percent:
                    continue
                ax.scatter(
                    point["epochs"],
                    point["perplexity"],
                    marker="*",
                    s=145,
                    color=COLORS[domain],
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=4,
                )
                ax.annotate(
                    f"{point['epochs']:.2f} epochs",
                    (point["epochs"], point["perplexity"]),
                    xytext=(-18, 20) if domain == "finemath_3plus" else (0, -24),
                    textcoords="offset points",
                    ha="right" if domain == "finemath_3plus" else "center",
                    va="center",
                    fontsize=9,
                    color=COLORS[domain],
                )
        baseline = rows["finemath_3plus"][0]["result"]["perplexity"][dataset]
        ax.scatter([0], [baseline], s=34, marker="D", color="#333333", zorder=5, label="Shared zero-fraction checkpoint")
        role = "primary" if dataset == "math500" else "secondary"
        ax.set_title(f"{label} · {role}", fontsize=11)
        ax.set_ylabel("Reference-solution perplexity")
        ax.set_xlabel("Materialized epochs of the varied domain")
        ax.set_xlim(-0.3, 11.55)
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.margins(y=0.15)
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    complete = all(row["result"] is not None for domain_rows in rows.values() for row in domain_rows)
    state = "both eight-point curves verified" if complete else "partial measurements"
    fig.suptitle(
        f"Valley detail through 70 percent mixture fraction · {state}\n"
        "Stars mark observed grid minima; the full-range companion retains the 100 percent endpoints",
        fontsize=10,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    paths = []
    for extension in ("png", "pdf"):
        path = destination / f"math_likelihood_valley_detail.{extension}"
        fig.savefig(path, dpi=190)
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def plot_log_scale(rows: dict[str, list[dict]], summaries: dict[str, dict], destination: Path) -> list[str]:
    """Retain both complete curves while making their lower-perplexity valleys visible."""
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.7), layout="constrained", sharex=True)
    for ax, (dataset, label) in zip(axes, previous.DATASETS.items(), strict=True):
        available_values = []
        for domain, domain_rows in rows.items():
            epochs = np.array([row["epochs"] for row in domain_rows])
            values = np.array(
                [row["result"]["perplexity"][dataset] if row["result"] is not None else np.nan for row in domain_rows]
            )
            finite = values[np.isfinite(values)]
            if not len(finite):
                continue
            available_values.extend(finite)
            ax.plot(epochs, values, "o-", color=COLORS[domain], linewidth=1.5, markersize=4.5, label=DOMAINS[domain])
            for point in summaries[domain][dataset]["tied_minima"]:
                ax.scatter(
                    point["epochs"],
                    point["perplexity"],
                    marker="*",
                    s=145,
                    color=COLORS[domain],
                    edgecolors="black",
                    linewidths=0.7,
                    zorder=4,
                )
                ax.annotate(
                    f"{point['epochs']:.2f} epochs",
                    (point["epochs"], point["perplexity"]),
                    xytext=(0, 17) if domain == "finemath_3plus" else (0, 24),
                    textcoords="offset points",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=COLORS[domain],
                )
        baseline = rows["finemath_3plus"][0]["result"]["perplexity"][dataset]
        ax.scatter([0], [baseline], s=34, marker="D", color="#333333", zorder=5, label="Shared zero-fraction checkpoint")
        role = "primary" if dataset == "math500" else "secondary"
        ax.set_title(f"{label} · {role}", fontsize=11)
        ax.set_ylabel("Reference-solution perplexity (log scale)")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.set_ylim(min(available_values) * 0.65, max(available_values) * 1.4)
        ax.set_xlabel("Materialized epochs of the varied domain")
        ax.set_xlim(-0.4, 16.3)
        ax.set_xticks([0, 4, 8, 12, 16])
        ax.grid(alpha=0.22)
        ax.spines[["top", "right"]].set_visible(False)
    complete = all(row["result"] is not None for domain_rows in rows.values() for row in domain_rows)
    state = "both eight-point curves verified" if complete else "partial measurements"
    fig.suptitle(
        f"Complete grid on logarithmic perplexity axes · {state}\nStars mark observed grid minima",
        fontsize=10,
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    paths = []
    for extension in ("png", "pdf"):
        path = destination / f"math_likelihood_log_scale.{extension}"
        fig.savefig(path, dpi=190)
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DIRECTORY / "spec.json")
    parser.add_argument("--archive", type=Path, default=previous.ARCHIVE)
    parser.add_argument("--output", type=Path, default=DIRECTORY / "results")
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    spec = previous.read_json(args.spec)
    reference = validate_comparison(spec)
    specs = {"wikipedia": spec, "finemath_3plus": reference}
    rows = {
        domain: collect(current, domain, archived_points(current, domain, args.archive))
        for domain, current in specs.items()
    }
    if any(row["result"] is None for row in rows["finemath_3plus"]):
        raise ValueError("The completed FineMath reference grid has a missing verified receipt")
    source_baseline = rows["finemath_3plus"][0]["result"]
    if canonical_sha256(source_baseline) != spec["baseline_result_canonical_sha256"]:
        raise ValueError("The pinned zero-fraction source receipt changed")
    baseline = rows["wikipedia"][0]["result"]
    expected_baseline = {
        **source_baseline,
        "spec_sha256": spec["spec_sha256"],
        "reused_from": {
            "uri": rows["finemath_3plus"][0]["receipt_uri"],
            "source_spec_sha256": reference["spec_sha256"],
            "source_result_canonical_sha256": canonical_sha256(source_baseline),
        },
    }
    if baseline is not None and baseline != expected_baseline:
        raise ValueError("The reused zero-fraction measurement or its provenance differs from the source receipt")
    for wiki, fine in zip(rows["wikipedia"], rows["finemath_3plus"], strict=True):
        if wiki["percent"] != fine["percent"] or not math.isclose(wiki["epochs"], fine["epochs"], abs_tol=1e-12):
            raise ValueError("The two matched curves do not share the same realized epoch coordinates")
    summaries = {
        domain: {dataset: previous.observed_minimum(domain_rows, dataset) for dataset in previous.DATASETS}
        for domain, domain_rows in rows.items()
    }
    comparisons = {dataset: cross_choice_regret(rows, summaries, dataset) for dataset in previous.DATASETS}
    completed = sum(row["result"] is not None for row in rows["wikipedia"])
    destination = args.output / spec["spec_sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {
        "collected_at": datetime.now(UTC).isoformat(),
        "status": "complete" if completed == len(evaluation.GRID) else "partial",
        "wikipedia_verified_points": completed,
        "wikipedia_expected_points": len(evaluation.GRID),
        "missing_wikipedia_percent": [row["percent"] for row in rows["wikipedia"] if row["result"] is None],
        "spec_sha256": spec["spec_sha256"],
        "reference_spec_sha256": reference["spec_sha256"],
        "shared_scoring": {field: spec[field] for field in SHARED_FIELDS},
        "sources_sha256": {
            str(path): file_sha256(path)
            for path in (args.spec, args.archive / "preflight.json", args.archive / "curves.json")
        },
        "analyzer_sha256": file_sha256(Path(__file__)),
        "rows": rows,
        "summaries": summaries,
        "cross_choice_regrets": comparisons,
        "interpretation": (
            "Single-seed observed grid minima under a common reference-solution likelihood evaluation. "
            "The zero-fraction checkpoint is shared, not an independent replicate. A boundary minimum does not "
            "establish an interior turnover or locate an optimum beyond the sampled range."
        ),
    }
    write_csv(rows, destination / "points.csv")
    receipt["figures"] = plot_results(rows, summaries, destination)
    receipt["figures"].extend(plot_valley_detail(rows, summaries, destination))
    receipt["figures"].extend(plot_log_scale(rows, summaries, destination))
    receipt["figure_views"] = {
        "math_likelihood_comparison": "All eight fractions, including both 100 percent endpoints",
        "math_likelihood_valley_detail": (
            "Fractions through 70 percent, with axes spanning every displayed point; stars use observed grid minima"
        ),
        "math_likelihood_log_scale": "All eight fractions on logarithmic perplexity axes with ordinary numeric ticks",
    }
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    (destination / "summary.json").write_text(
        json.dumps({"summaries": summaries, "cross_choice_regrets": comparisons}, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {"output": str(destination.resolve()), "wikipedia_verified_points": completed, "summaries": summaries}
        )
    )
    if args.require_complete and completed != len(evaluation.GRID):
        raise SystemExit("The prescribed Wikipedia proxy grid is incomplete; partial artifacts were retained")


if __name__ == "__main__":
    main()
