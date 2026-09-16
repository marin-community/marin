# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Compare verified proxy and target math likelihoods without launching jobs."""

import argparse
import json
import math
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.lines import Line2D

from experiments.domain_phase_mix import analyze_tpp10_finemath_math as previous
from experiments.domain_phase_mix import analyze_tpp10_wikipedia_math as proxies
from experiments.domain_phase_mix import evaluate_tpp10_target_math as evaluation
from experiments.domain_phase_mix import evaluate_tpp10_target_math_complete as completion
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

PROXY_SPEC = proxies.DIRECTORY / "spec.json"
SCALES = {"matched": "Epoch-matched proxy", "target": "Target"}
STYLES = {"matched": ("-", "o"), "target": ("--", "s")}


def archived_grid(domain: str, arm: str, archive: Path, requests: list[dict]) -> dict[int, dict]:
    """Resolve complete epoch coordinates, including unreleased target requests."""
    preflight = previous.read_json(archive / "preflight.json")
    curves = previous.read_json(archive / "curves.json")["curves"]
    selected = [curve for curve in curves if curve["domain"] == domain and curve["arm"] == arm]
    if len(selected) != 1:
        raise ValueError(f"Expected one archived curve: {domain}/{arm}")
    observed = {point["percent"]: point for point in selected[0]["points"]}
    if len(observed) != len(selected[0]["points"]):
        raise ValueError(f"Duplicate archived points: {domain}/{arm}")
    coordinates = {point["percent"]: point for point in preflight["allocation"]["coordinates"]}
    by_percent = {request["percent"]: request for request in requests}
    grid = proxies.evaluation.GRID
    if tuple(sorted(by_percent)) != grid or len(requests) != len(grid):
        raise ValueError(f"Expected the complete requested grid: {domain}/{arm}")
    cache = "matched" if arm == "matched" else "parent"
    pool_tokens = preflight["data"]["caches"][f"{domain}/{cache}"]["tokens"]
    allocation = "proxy_allocation" if arm == "matched" else "target_allocation"
    epoch_field = "matched_epochs" if arm == "matched" else "target_epochs"
    points = {}
    for percent in grid:
        request = by_percent[percent]
        epochs = coordinates[percent][epoch_field]
        if percent:
            if request["support_sequences"] * proxies.evaluation.experiment.SEQ_LEN != pool_tokens:
                raise ValueError(f"Pool size differs: {request['run_name']}")
            allocated = coordinates[percent][allocation]["starcoder"]
            if not math.isclose(epochs, allocated / request["support_sequences"], abs_tol=1e-12):
                raise ValueError(f"Epoch allocation differs: {request['run_name']}")
        if percent in observed:
            point = observed[percent]
            expected_name = "shared p0" if percent == 0 else request["run_name"]
            if point["run_name"] != expected_name or not math.isclose(point["epochs"], epochs, abs_tol=1e-12):
                raise ValueError(f"Archived identity or epoch coordinate differs: {request['run_name']}")
            points[percent] = point
        else:
            if percent not in selected[0]["missing_percent"]:
                raise ValueError(f"Unexplained missing archived point: {request['run_name']}")
            points[percent] = {"percent": percent, "epochs": epochs, "run_name": request["run_name"], "metrics": {}}
    return points


def collect_proxies(spec: dict, source: dict, proxy_spec: dict, archive: Path) -> dict[str, list[dict]]:
    """Reuse the complete proxy artifact after checking its frozen protocol and live receipts."""
    reference = proxies.validate_comparison(proxy_spec)
    if source["status"] != "complete" or source["spec_sha256"] != proxy_spec["spec_sha256"]:
        raise ValueError("The proxy comparison artifact is incomplete or has the wrong identity")
    if source["reference_spec_sha256"] != spec["reference_spec"]["spec_sha256"]:
        raise ValueError("The proxy comparison does not use the target release's FineMath reference")
    for field in proxies.SHARED_FIELDS:
        if source["shared_scoring"][field] != spec[field] or proxy_spec[field] != spec[field]:
            raise ValueError(f"Proxy and target scoring differ: {field}")
    rows = {}
    for domain, current in (("wikipedia", proxy_spec), ("finemath_3plus", reference)):
        domain_rows = source["rows"][domain]
        endpoints = {endpoint["request"]["run_name"]: endpoint for endpoint in current["endpoints"]}
        grid = archived_grid(domain, "matched", archive, [row["request"] for row in domain_rows])
        rows[domain] = []
        for row in domain_rows:
            endpoint = endpoints[row["run_name"]]
            result = proxies.evaluation.verified_result(current, endpoint)
            if result is None or result != row["result"]:
                raise ValueError(f"The archived proxy receipt differs from its verified source: {row['run_name']}")
            if row["request"] != endpoint["request"] or row["checkpoint"] != endpoint["checkpoint"]:
                raise ValueError(f"Proxy endpoint identity differs: {row['run_name']}")
            if result["population_sha256"] != canonical_sha256(spec["population_counts"]):
                raise ValueError(f"Proxy scored population differs: {row['run_name']}")
            if not math.isclose(row["epochs"], grid[row["percent"]]["epochs"], abs_tol=1e-12):
                raise ValueError(f"Proxy epoch coordinate differs: {row['run_name']}")
            rows[domain].append({**row, "arm": "matched"})
    return rows


def collect_targets(
    spec: dict, archive: Path, result_reader: Callable[[dict, dict], dict | None]
) -> dict[str, list[dict]]:
    """Keep all target requests while evaluating only released, verified endpoints."""
    baseline = next(endpoint for endpoint in spec["endpoints"] if endpoint["request"]["percent"] == 0)
    results = {endpoint["request"]["run_name"]: result_reader(spec, endpoint) for endpoint in spec["endpoints"]}
    rows = {}
    for domain in proxies.DOMAINS:
        endpoints = [
            baseline,
            *[endpoint for endpoint in spec["endpoints"] if endpoint["request"].get("domain") == domain],
        ]
        omitted = [request for request in spec["omitted_requests"] if request["domain"] == domain]
        requests = sorted([*[endpoint["request"] for endpoint in endpoints], *omitted], key=lambda r: r["percent"])
        points = archived_grid(domain, "target", archive, requests)
        by_percent = {endpoint["request"]["percent"]: endpoint for endpoint in endpoints}
        rows[domain] = []
        for request in requests:
            percent = request["percent"]
            endpoint = by_percent.get(percent)
            result = results[request["run_name"]] if endpoint is not None else None
            if result is not None and result["population_sha256"] != canonical_sha256(spec["population_counts"]):
                raise ValueError(f"Target scored population differs: {request['run_name']}")
            status = (
                "not_released_in_this_spec"
                if endpoint is None
                else "verified" if result is not None else "missing_evaluation"
            )
            rows[domain].append(
                {
                    "domain": domain,
                    "arm": "target",
                    "percent": percent,
                    "epochs": points[percent]["epochs"],
                    "status": status,
                    "run_name": request["run_name"],
                    "shared_zero_fraction_checkpoint": percent == 0,
                    "spec_sha256": spec["spec_sha256"],
                    "receipt_uri": (
                        f"{evaluation.output_root(spec)}/{request['run_name']}.json" if endpoint is not None else None
                    ),
                    "request": request,
                    "checkpoint": endpoint["checkpoint"] if endpoint is not None else None,
                    "archived_metrics": points[percent]["metrics"],
                    "result": result,
                }
            )
    return rows


def summarize(rows: list[dict], dataset: str) -> dict:
    """Distinguish a complete-grid minimum from the lowest available partial observation."""
    summary = previous.observed_minimum(rows, dataset)
    summary["unreleased_percent"] = [row["percent"] for row in rows if row["status"] == "not_released_in_this_spec"]
    summary["missing_evaluation_percent"] = [row["percent"] for row in rows if row["status"] == "missing_evaluation"]
    available = [row for row in rows if row["result"] is not None]
    if available:
        best = summary["minimum"]
        summary["at_observed_range_boundary"] = best["percent"] in (available[0]["percent"], available[-1]["percent"])
        summary["description"] = (
            "observed grid minimum" if summary["complete_grid"] else "lowest observed; incomplete grid"
        )
    if summary["unreleased_percent"]:
        summary["interpretation"] = (
            "The high-fraction target measurements are not part of this evaluation release. The lowest available "
            "value does not locate the target optimum, and the plotted curve stops at the final measured point."
        )
    return summary


def plot_results(rows: dict[str, list[dict]], summaries: dict[str, dict], destination: Path) -> list[str]:
    """Plot measured log-perplexity curves without bridging missing targets."""
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.5), sharex=True, layout="constrained")
    for index, (domain, domain_label) in enumerate(proxies.DOMAINS.items()):
        for ax, (dataset, dataset_label) in zip(axes[index], previous.DATASETS.items(), strict=True):
            observed_values = []
            for arm in SCALES:
                key = f"{domain}/{arm}"
                domain_rows = rows[key]
                epochs = np.array([row["epochs"] for row in domain_rows])
                values = np.array(
                    [
                        row["result"]["perplexity"][dataset] if row["result"] is not None else np.nan
                        for row in domain_rows
                    ]
                )
                finite = values[np.isfinite(values)]
                if not len(finite):
                    continue
                observed_values.extend(finite)
                style, marker = STYLES[arm]
                ax.plot(
                    epochs,
                    values,
                    linestyle=style,
                    marker=marker,
                    color=proxies.COLORS[domain],
                    linewidth=1.7,
                    markersize=4.5,
                    markerfacecolor=proxies.COLORS[domain] if arm == "matched" else "white",
                    markeredgecolor=proxies.COLORS[domain],
                    markeredgewidth=0.8,
                )
                summary = summaries[key][dataset]
                for point in summary["tied_minima"]:
                    complete = summary["complete_grid"]
                    ax.scatter(
                        point["epochs"],
                        point["perplexity"],
                        marker="*" if complete else "v",
                        s=135 if complete else 65,
                        color=proxies.COLORS[domain],
                        edgecolors="black",
                        linewidths=0.65,
                        zorder=4,
                    )
                    text = f"{SCALES[arm]}: {point['epochs']:.2f} epochs" if complete else "Lowest observed; incomplete"
                    alignment = "left" if point["epochs"] < 2 else "right" if point["epochs"] > 13 else "center"
                    ax.annotate(
                        text,
                        (point["epochs"], point["perplexity"]),
                        xytext=(0, 17 if complete else -22),
                        textcoords="offset points",
                        ha=alignment,
                        va="center",
                        fontsize=8.3,
                        color=proxies.COLORS[domain],
                    )
            ax.set_title(f"{domain_label} · {dataset_label}", fontsize=11)
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(ticker.LogLocator(base=10, subs=(1, 2, 5)))
            ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
            ax.yaxis.set_minor_locator(ticker.NullLocator())
            if observed_values:
                ax.set_ylim(min(observed_values) * 0.55, max(observed_values) * 1.65)
            if index == 0:
                ax.set_ylim(top=50)
            ax.set_ylabel("Reference-solution perplexity (log scale)")
            ax.set_xlim(-0.4, 16.3)
            ax.set_xticks([0, 4, 8, 12, 16])
            ax.grid(alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            if index == 1:
                ax.set_xlabel("Materialized epochs of the varied domain")
                target_count = summaries[f"{domain}/target"][dataset]["verified_points"]
                if target_count < len(proxies.evaluation.GRID):
                    ax.text(
                        0.98,
                        0.97,
                        f"Target: {target_count}/8 points available\nRemaining target measurements not yet available",
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8.2,
                        color="#555555",
                    )
    fig.suptitle("Math likelihood across scales · TPP 10", fontsize=12)
    handles = [
        Line2D(
            [],
            [],
            color="#555555",
            linestyle=style,
            marker=marker,
            markerfacecolor="#555555" if arm == "matched" else "white",
            markeredgecolor="#555555",
            markeredgewidth=0.8,
            label=label,
        )
        for arm, label in SCALES.items()
        for style, marker in [STYLES[arm]]
    ]
    handles.append(Line2D([], [], color="#555555", linestyle="none", marker="*", markersize=10, label="Grid minimum"))
    if any(row["result"] is None for current in rows.values() for row in current):
        handles.append(
            Line2D([], [], color="#555555", linestyle="none", marker="v", label="Lowest observed; incomplete")
        )
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    paths = []
    for extension in ("png", "pdf"):
        path = destination / f"math_likelihood_proxy_target.{extension}"
        fig.savefig(path, dpi=190)
        paths.append(str(path.resolve()))
    plt.close(fig)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=evaluation.DIRECTORY / "spec.json")
    parser.add_argument("--complete-spec", type=Path, help="Use the completed fifteen-checkpoint target release")
    parser.add_argument("--proxy-spec", type=Path, default=PROXY_SPEC)
    parser.add_argument("--proxy-receipt", type=Path)
    parser.add_argument("--archive", type=Path, default=previous.ARCHIVE)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Require every released evaluation; omitted requests stay absent",
    )
    args = parser.parse_args()
    spec_path = args.complete_spec or args.spec
    release = completion if args.complete_spec else evaluation
    spec = previous.read_json(spec_path)
    release.validate_spec(spec)
    proxy_spec = previous.read_json(args.proxy_spec)
    proxy_path = args.proxy_receipt or proxies.DIRECTORY / "results" / proxy_spec["spec_sha256"] / "receipt.json"
    proxy_receipt = previous.read_json(proxy_path)
    proxy_rows = collect_proxies(spec, proxy_receipt, proxy_spec, args.archive)
    target_rows = collect_targets(spec, args.archive, release.verified_result)
    rows = {
        f"{domain}/{arm}": current[domain]
        for domain in proxies.DOMAINS
        for arm, current in (("matched", proxy_rows), ("target", target_rows))
    }
    if proxy_rows["wikipedia"][0]["checkpoint"] == target_rows["wikipedia"][0]["checkpoint"]:
        raise ValueError("Proxy and target must use different scale-specific zero-fraction checkpoints")
    summaries = {
        key: {dataset: summarize(current, dataset) for dataset in previous.DATASETS} for key, current in rows.items()
    }
    verified_names = {
        row["run_name"] for current in target_rows.values() for row in current if row["result"] is not None
    }
    missing_names = [
        e["request"]["run_name"] for e in spec["endpoints"] if e["request"]["run_name"] not in verified_names
    ]
    destination = (args.output or release.DIRECTORY / "results") / spec["spec_sha256"]
    destination.mkdir(parents=True, exist_ok=True)
    complete_targets = all(row["result"] is not None for current in target_rows.values() for row in current)
    receipt = {
        "collected_at": datetime.now(UTC).isoformat(),
        "status": "complete_released_evaluations" if not missing_names else "partial_evaluations",
        "all_target_grids_complete": complete_targets,
        "target_verified_checkpoints": len(verified_names),
        "target_released_checkpoints": len(spec["endpoints"]),
        "missing_target_evaluations": missing_names,
        "omitted_target_requests": spec["omitted_requests"],
        "spec_sha256": spec["spec_sha256"],
        "proxy_spec_sha256": proxy_spec["spec_sha256"],
        "reference_spec_sha256": spec["reference_spec"]["spec_sha256"],
        "shared_scoring": {field: spec[field] for field in proxies.SHARED_FIELDS},
        "sources_sha256": {
            str(path): file_sha256(path)
            for path in (
                spec_path,
                args.proxy_spec,
                proxy_path,
                args.archive / "curves.json",
                args.archive / "preflight.json",
            )
        },
        "analyzer_sha256": file_sha256(Path(__file__)),
        "rows": rows,
        "summaries": summaries,
        "interpretation": (
            "Single-seed reference-solution likelihood curves. Each scale has one zero-fraction checkpoint shared "
            "across domains. Stars denote minima only on complete eight-point grids."
            + (
                " Missing target measurements remain absent; no optimum is inferred from a partial curve."
                if not complete_targets
                else ""
            )
        ),
    }
    proxies.write_csv(rows, destination / "points.csv")
    receipt["figures"] = plot_results(rows, summaries, destination)
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    (destination / "summary.json").write_text(json.dumps(summaries, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(destination.resolve()), "status": receipt["status"], "summaries": summaries}))
    if args.require_complete and missing_names:
        raise SystemExit("Released target evaluations are incomplete; partial artifacts were retained")


if __name__ == "__main__":
    main()
