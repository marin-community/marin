# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.9", "numpy>=2", "pandas>=2.2"]
# ///
"""Plot the existing and recomputed RegMix paths without implying new measurements."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TARGETS = (("uncheatable", "Uncheatable"), ("table9", "OlmoBaseEval Easy"))
PATHS = (("old_endpoint", "Existing trained RegMix proposal"),
         ("official_objective_endpoint", "Recomputed reference-procedure proposal"))
PREDICTORS = (("mariner", "MARINER (frozen)", "#178A72", "-"),
              ("prior_only", "RegMix (existing fit)", "#666666", (0, (4, 2))),
              ("official_objective", "RegMix (reference objective fit)", "#0072B2", "-"))
OUTPUT = HERE / "official_replay_paths.png"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_inputs():
    completion = json.loads((HERE / "completion.json").read_text())
    for name in ("path_predictions.csv", "cross_predictions.csv", "proposal_comparison.csv", "policy_weights.csv"):
        assert sha256(HERE / name) == completion["outputs"][name]
    historical = json.loads((HERE / "historical_measurements_receipt.json").read_text())
    assert sha256(HERE / "historical_measurements.csv") == historical["output_sha256"]["historical_measurements.csv"]
    paths = pd.read_csv(HERE / "path_predictions.csv")
    cross = pd.read_csv(HERE / "cross_predictions.csv")
    measurements = pd.read_csv(HERE / "historical_measurements.csv")
    shifts = pd.read_csv(HERE / "proposal_comparison.csv")
    assert set(paths.target) == {target for target, _ in TARGETS}
    for target, _ in TARGETS:
        for path, _ in PATHS:
            for predictor, *_ in PREDICTORS:
                series = paths[(paths.target == target) & (paths.path == path) & (paths.predictor == predictor)]
                assert len(series) == 201
                np.testing.assert_allclose(series.fraction, np.linspace(0, 1, 201), rtol=0, atol=1e-15)
                for fraction, policy in ((0, "mariner"), (1, path)):
                    reference = cross[(cross.target == target) & (cross.policy == policy) & (cross.predictor == predictor)]
                    assert len(reference) == 1
                    np.testing.assert_allclose(series.loc[series.fraction == fraction, "prediction"],
                                               reference.prediction, rtol=0, atol=1e-12)
    return paths, measurements, shifts


def main():
    paths, measurements, shifts = read_inputs()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.usetex": False, "figure.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.1), sharey="row", sharex=True)
    observed = measurements[measurements.trainer_seed == 0]
    for row, (target, label) in enumerate(TARGETS):
        included = paths[(paths.target == target) & paths.path.isin([path for path, _ in PATHS])
                         & paths.predictor.isin([predictor for predictor, *_ in PREDICTORS])]
        measured = observed[observed.target == target]
        assert len(measured) == 3 and set(measured.policy) == {"mariner", "regmix", "midpoint"}
        values = np.r_[included.prediction, measured.measured_bpb]
        lower, upper = values.min(), values.max()
        pad = max(0.008, 0.10 * (upper - lower))
        shift = shifts[(shifts.target == target) & (shifts.variant == "official_objective")].iloc[0]
        for column, (path, title) in enumerate(PATHS):
            ax = axes[row, column]
            for predictor, _, color, style in PREDICTORS:
                series = paths[(paths.target == target) & (paths.path == path) & (paths.predictor == predictor)]
                ax.plot(series.fraction, series.prediction, color=color, linestyle=style, linewidth=1.8,
                        zorder=2 if predictor == "prior_only" else 3)
            plotted = measured if column == 0 else measured[measured.policy == "mariner"]
            ax.scatter(plotted.position, plotted.measured_bpb, s=43, c="#E69F00",
                       edgecolor="#242424", linewidth=0.75, zorder=5)
            ax.set_ylim(lower - pad, upper + pad * 1.4)
            ax.set_xlim(-0.03, 1.03)
            ax.set_xticks([0, .25, .5, .75, 1])
            ax.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
            ax.tick_params(labelbottom=True)
            ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
            if column == 0:
                ax.set_ylabel(f"{label}\nLoss (BPB)")
                note = "Measured: MARINER, old midpoint, old RegMix"
            else:
                note = (f"Mass shifted: endpoint {shift.endpoint_tv_from_old:.1%}; "
                        f"midpoint {shift.midpoint_tv_from_old:.1%}")
            ax.set_title(note, loc="left", fontsize=9.2, pad=8, color="#444444")
            if row == 1:
                ax.set_xlabel("Blend toward the panel's RegMix proposal")
    fig.text(.284, .883, PATHS[0][1], ha="center", fontsize=12, fontweight="bold")
    fig.text(.748, .883, PATHS[1][1], ha="center", fontsize=12, fontweight="bold")
    handles = [Line2D([], [], color=color, linewidth=1.8, linestyle=style, label=label)
               for _, label, color, style in PREDICTORS]
    handles.append(Line2D([], [], marker="o", linestyle="none", markerfacecolor="#E69F00",
                          markeredgecolor="#242424", markersize=6, label="Measured (trainer seed 0)"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .995),
               ncol=2, frameon=False, fontsize=10, handlelength=2.7, columnspacing=2.8)
    fig.text(.51, .022, "Right panels: new proposal and midpoint have not been trained. Only the MARINER endpoint is measured.",
             ha="center", fontsize=9.3, color="#333333")
    fig.subplots_adjust(left=.096, right=.985, top=.819, bottom=.123, hspace=.29, wspace=.135)
    fig.savefig(OUTPUT, dpi=180)
    plt.close(fig)
    inputs = [HERE / name for name in ("completion.json", "path_predictions.csv", "cross_predictions.csv",
                                       "proposal_comparison.csv", "historical_measurements.csv")]
    receipt = {"source_sha256": {str(path): sha256(path) for path in inputs},
               "output_sha256": sha256(OUTPUT), "script_sha256": sha256(Path(__file__)),
               "observations": "Only historical trainer seed zero; three points on each old path, MARINER at zero on each new path.",
               "primary_reference_predictor": "official_objective", "components_sensitivity_plotted": False,
               "paper_assets_modified": False, "new_policy_measurements": False,
               "note": "Curves interpolate exact continuous mixture paths; measured midpoints use archived runtime-rounded weights."}
    (HERE / "official_replay_plot_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
