# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10", "numpy>=2"]
# ///
"""Overlay descriptive MARINER fits while retaining measured selections and regret."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator

DIRECTORY = Path(__file__).resolve().parent
COLORS = {"unmatched": "#0072B2", "matched": "#D55E00", "target": "#303030"}


def bracket(axis, x: float, bottom: float, top: float, color: str, cap: float = 0.013) -> None:
    axis.plot([x, x], [bottom, top], color=color, linewidth=1.8, zorder=5)
    axis.hlines([bottom, top], x - cap, x + cap, color=color, linewidth=1.8, zorder=5)


def main() -> None:
    source = DIRECTORY.parent / "consistent_bpb_20260912/analysis.json"
    result = json.loads(source.read_text())
    metric_definition = result.get("metric_definition", {})
    if (
        metric_definition.get("id") != "scored_byte_bpb_from_token_loss_v1"
        or metric_definition.get("schema_version") != 2
        or metric_definition.get("total_records") != 102
    ):
        raise ValueError("The figure requires consistent scored-byte BPB; raw or mixed metric definitions are invalid")
    allocation_path = DIRECTORY / "sources/allocation_audit.json"
    allocations = json.loads(allocation_path.read_text())
    if allocations["status"] != "passed":
        raise ValueError("The materialized epoch audit did not pass")
    epoch_coordinates = {row["percent"]: row for row in allocations["coordinates"]}
    fits_path = DIRECTORY / "fits/summary.json"
    fit_results = json.loads(fits_path.read_text())
    assert fit_results["source_sha256"][str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
    fits = {f["curve"]: f for f in fit_results["curves"]}
    curves = result["curves"]
    target = next(curve for curve in curves if curve["arm"] == "target")
    unmatched = next(curve for curve in curves if curve["arm"] == "unmatched")
    matched = [curve for curve in curves if curve["arm"] == "matched"]
    target_values = {point["percent"]: point["value"] for point in target["points"]}
    reference = result["complete_common_grid_analysis"]
    if reference["verified_artifact_count"] != 102:
        raise ValueError("The complete-grid figure requires all 102 normalized endpoints")
    grid = reference["grid_percent"]
    if result["refinement_complete"] != 45 or result["refinement_planned"] != 45:
        raise ValueError("The figure requires all 45 verified refinements")
    if result["missing_run_names"] or result["verified_but_unplotted"]:
        raise ValueError("Missing or unplotted measurements in the complete-grid figure")
    if grid != [0, 10, 30, 40, 50, 55, 60, 65, 70, 80, 90, 100]:
        raise ValueError("Unexpected complete common grid")
    if any([point["percent"] for point in curve["points"]] != grid for curve in curves):
        raise ValueError("Every plotted curve must cover the complete common grid")
    target_minimum = min(target_values[p] for p in grid)
    target_choice = min(grid, key=lambda p: (target_values[p], p))
    choices = {
        "unmatched": unmatched["selected_percent"],
        "matched": matched[0]["selected_percent"],
    }
    parent_sequences = 92928  # Frozen finite StarCoder parent in the training design.
    selected_epochs = {
        "matched_proxy": epoch_coordinates[choices["matched"]]["matched_epochs"],
        "unmatched_proxy": epoch_coordinates[choices["unmatched"]]["proxy_allocation"]["starcoder"] / parent_sequences,
        "target_at_matched_choice": epoch_coordinates[choices["matched"]]["target_epochs"],
        "target_at_unmatched_choice": epoch_coordinates[choices["unmatched"]]["target_epochs"],
    }
    if any(curve["selected_percent"] != choices["matched"] for curve in matched):
        raise ValueError("The subset means no longer select the same mixture; revise the target annotations")
    if choices["unmatched"] != reference["unmatched_selected_percent"] or any(
        curve["selected_percent"] != choices["matched"] for curve in reference["matched_subsets"]
    ):
        raise ValueError("Available-grid selections differ from the common-grid comparison")
    penalties = {arm: target_values[percent] - target_minimum for arm, percent in choices.items()}
    percentages = {arm: 100 * penalty / target_minimum for arm, penalty in penalties.items()}
    avoided_fraction = (penalties["unmatched"] - penalties["matched"]) / penalties["unmatched"]
    if not np.isclose(penalties["unmatched"], reference["unmatched_target_regret_bpb"], atol=1e-12, rtol=0):
        raise ValueError("Target penalty differs from the verified common-grid analysis")
    if not np.isclose(penalties["matched"], reference["matched_subsets"][0]["target_regret_bpb"], atol=1e-12, rtol=0):
        raise ValueError("Matched target penalty differs from the verified common-grid analysis")

    style = {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.usetex": False,
        "axes.grid": False,
    }
    with plt.rc_context(style):
        figure, (left, right) = plt.subplots(1, 2, figsize=(7.6, 3.3))
        figure.subplots_adjust(left=0.075, right=0.925, bottom=0.17, top=0.74, wspace=0.28)
        for curve in [unmatched, *matched]:
            arm = curve["arm"]
            key = arm if arm != "matched" else f"matched_{curve['subset_seed']}"
            fit = fits[key]
            np.testing.assert_allclose(fit["observed"], [p["value"] for p in curve["points"]])
            left.plot(
                fit["dense_requested_share"],
                fit["dense_prediction"],
                color=COLORS[arm],
                linewidth=1.8 if arm == "unmatched" else 1.05,
                alpha=1 if arm == "unmatched" else 0.78,
                zorder=2,
            )
            x = np.array([p["percent"] / 100 for p in curve["points"]])
            y = np.array([p["value"] for p in curve["points"]])
            left.plot(
                x,
                y,
                color=COLORS[arm],
                linestyle="none",
                marker="o",
                markersize=3.2 if arm == "unmatched" else 2.1,
                alpha=1 if arm == "unmatched" else 0.78,
                zorder=3,
            )
        for arm, selected in choices.items():
            chosen_curves = [unmatched] if arm == "unmatched" else matched
            value = np.mean(
                [next(p["value"] for p in curve["points"] if p["percent"] == selected) for curve in chosen_curves]
            )
            left.plot(
                selected / 100,
                value,
                "*",
                markersize=11,
                color=COLORS[arm],
                markeredgecolor="white",
                markeredgewidth=0.5,
                zorder=6,
            )
            epochs = selected_epochs[f"{arm}_proxy"]
            left.annotate(
                f"{selected}% selected\n{epochs:.2f} epochs",
                (selected / 100, value),
                xytext=(-6, 18) if arm == "unmatched" else (0, 13),
                textcoords="offset points",
                ha="right" if arm == "unmatched" else "center",
                color=COLORS[arm],
                fontsize=8,
            )
        visible_proxy_values = [
            point["value"] for curve in [unmatched, *matched] for point in curve["points"] if point["percent"] >= 10
        ]
        left.set(
            xlim=(0.085, 1.025),
            ylim=(min(visible_proxy_values) - 0.025, max(visible_proxy_values) + 0.07),
            xlabel="StarCoder mixture fraction, $p$",
            ylabel="Programming-language loss (BPB)",
        )
        left.set_title(r"Proxy runs ($2.49\times10^{16}$ FLOPs)", loc="left", y=1.18, pad=5)
        handles = [
            Line2D([], [], color=COLORS[arm], linewidth=1.7, label=label)
            for arm, label in [
                ("unmatched", "Without simulated epoching"),
                ("matched", "With simulated epoching (3 subsets)"),
            ]
        ]
        left.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.50, 1.005),
            frameon=False,
            handlelength=2,
            borderaxespad=0.2,
            labelspacing=0.35,
        )
        left.yaxis.set_major_locator(MultipleLocator(0.1))

        x = np.array([p["percent"] / 100 for p in target["points"]])
        y = np.array([p["value"] for p in target["points"]])
        np.testing.assert_allclose(fits["target"]["observed"], y)
        right.plot(
            fits["target"]["dense_requested_share"],
            fits["target"]["dense_prediction"],
            color=COLORS["target"],
            linewidth=1.65,
            zorder=2,
        )
        right.plot(x, y, color=COLORS["target"], marker="o", markersize=3.2, linestyle="none", zorder=3)
        right.axhline(target_minimum, color="#909090", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
        for arm, selected in choices.items():
            x_selected = selected / 100
            height = target_values[selected]
            color = COLORS[arm]
            right.plot(
                x_selected,
                height,
                marker="s",
                markersize=6.3,
                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.6,
                zorder=6,
            )
            bx = x_selected
            if arm == "matched":
                upper = target_values[choices["unmatched"]]
                avoided_color = "#66717E"
                bracket(right, bx, height, upper, avoided_color)
                right.plot(
                    [bx, choices["unmatched"] / 100],
                    [upper, upper],
                    color=avoided_color,
                    linewidth=0.65,
                    linestyle=(0, (3, 3)),
                    alpha=0.55,
                    zorder=1,
                )
                right.annotate(
                    f"{100 * avoided_fraction:.1f}% less\nexcess loss",
                    (bx, (height + upper) / 2),
                    xytext=(9, 0),
                    textcoords="offset points",
                    va="center",
                    ha="left",
                    fontsize=8.2,
                    color=avoided_color,
                    linespacing=1.15,
                )
            bracket(right, bx, target_minimum, height, color)
            right.annotate(
                f"+{percentages[arm]:.2f}%",
                (bx, (target_minimum + height) / 2),
                textcoords="offset points",
                xytext=(-8, 0) if arm == "matched" else (8, 0),
                ha="right" if arm == "matched" else "left",
                va="center",
                color=color,
                fontsize=8.4,
                annotation_clip=False,
            )
        right.plot(
            target_choice / 100,
            target_minimum,
            "*",
            markersize=11,
            color=COLORS["target"],
            markeredgecolor="white",
            markeredgewidth=0.5,
            zorder=6,
        )
        right.annotate(
            "Observed minimum",
            (target_choice / 100, target_minimum),
            xytext=(0, -14),
            textcoords="offset points",
            ha="center",
            color=COLORS["target"],
            fontsize=8,
        )
        visible_target_maximum = max(target_values[p] for p in grid if p >= 30)
        right.set(
            xlim=(0.285, 1.035),
            ylim=(target_minimum - 0.011, visible_target_maximum + 0.006),
            xlabel="StarCoder mixture fraction, $p$",
            ylabel="Target loss (BPB)",
        )
        right.set_title(r"Target run ($6.66\times10^{18}$ FLOPs)", loc="left", y=1.18, pad=5)
        right.yaxis.set_major_locator(MultipleLocator(0.01))
        top = right.secondary_xaxis("top")
        epoch_ticks = [30, 50, 70, 100]
        top.set_xticks(
            [p / 100 for p in epoch_ticks], [f"{epoch_coordinates[p]['target_epochs']:.2f}" for p in epoch_ticks]
        )
        top.set_xlabel("Materialized StarCoder epochs", fontsize=8.5, labelpad=5)
        top.spines["top"].set_visible(True)
        top.spines["top"].set_linewidth(0.6)
        top.tick_params(labelsize=8, length=3, width=0.6)
        for axis in (left, right):
            axis.xaxis.set_major_locator(MultipleLocator(0.2))
            axis.grid(axis="y", color="#DDE2E7", linewidth=0.6, alpha=0.7)
            axis.tick_params(length=3, width=0.6)
        figure.text(
            0.5,
            0.008,
            "Lines: MARINER fits. Stars and regret: measured-grid selections.",
            ha="center",
            fontsize=7.5,
            color="#555555",
        )
        for suffix in ("pdf", "png"):
            figure.savefig(
                DIRECTORY / f"epoch_matching_tpp10_mariner.{suffix}", dpi=240, bbox_inches="tight", pad_inches=0.045
            )
        plt.close(figure)

    receipt = {
        "source": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "metric_definition": metric_definition,
        "allocation_source_sha256": hashlib.sha256(allocation_path.read_bytes()).hexdigest(),
        "selected_materialized_epochs": selected_epochs,
        "target_epoch_ticks": {str(p): epoch_coordinates[p]["target_epochs"] for p in epoch_ticks},
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "target_reference_grid_percent": grid,
        "target_reference_minimum_bpb": target_minimum,
        "pilot_plan_sha256": result["pilot_plan_sha256"],
        "refinement_plan_sha256": result["refinement_plan_sha256"],
        "matched_subset_selections": reference["matched_subsets"],
        "selected_percent": choices,
        "target_selected_percent": target_choice,
        "target_penalty_bpb": penalties,
        "target_training_seeds": 1,
        "target_excess_percent_of_minimum": percentages,
        "excess_loss_reduction_percent": 100 * avoided_fraction,
        "avoided_percentage_points": percentages["unmatched"] - percentages["matched"],
        "total_target_loss_reduction_percent": (
            100 * (penalties["unmatched"] - penalties["matched"]) / target_values[choices["unmatched"]]
        ),
        "percent_definition": (
            "Both axes use absolute BPB. Lower annotations: 100*(target loss - common-grid minimum)/minimum. "
            "Upper stacked label: fraction of unmatched excess eliminated; all bracket heights are BPB differences."
        ),
        "proxy_trainer_seeds_per_point": 2,
        "matched_subset_curves": len(matched),
        "refinement_complete": result["refinement_complete"],
        "refinement_planned": result["refinement_planned"],
        "displayed_x_ranges": {"proxy": [0.1, 1.0], "target": [0.3, 1.0]},
        "fits_sha256": hashlib.sha256(fits_path.read_bytes()).hexdigest(),
        "note": (
            "Lines are descriptive MARINER fits. "
            "All displayed selections and regret brackets retain observed grid values."
        ),
    }
    (DIRECTORY / "figure3_mariner_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt["target_penalty_bpb"]))


if __name__ == "__main__":
    main()
