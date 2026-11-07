# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Redraws of the paper's analysis figures: Figures 12, 13 and 15.

- ``fig:domain-repetition`` (Figure 12): two sweeps of one domain against unrepeated web text under simulated
  epoching, Dolmino FLAN scored on SciQ and FineMath-3+ scored on GSM8K. Each panel draws the matched proxy's and the
  target's loss against the domain's share, Olmix's law fitted to the proxy, and the share Olmix selects under each
  epoch cap on the domain's materialized epochs at the target.
- ``fig:mixture-epochs`` (Figure 13): the epochs of each bucket in MARINER's and Olmix's mixtures for both
  objectives, Olmix's deployed mixtures and its proposals without a cap or KL penalty.
- ``fig:r7`` (Figure 15): each surrogate's cross-validated R^2 against the task's signal-to-noise ratio, with the
  noise ceiling 1 - 1/SNR^2 that run noise sets.

The paper fitted Olmix to the domain sweeps with its fitter's default Huber delta, 0.02, where the swarms' procedure
uses 0.01 (``mariner.olmix.HUBER_DELTA``); with 0.02, ``fit_log_linear`` reproduces the dataset's
``domain_sweeps_olmix_fits`` to 7e-8 BPB.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from mariner.data import (
    OBJECTIVES,
    OLMIX_CAP,
    RELIABILITY_COLLAPSED_SOURCE,
    RELIABILITY_SOURCES,
    RELIABILITY_TABLE,
    TRAINED_MARINER,
    TRAINED_OLMIX,
    mixture_weights,
    pool_fractions,
    read_table,
    release_partition,
    reliability_rows,
)
from mariner.figures.layout import (
    METHOD_COLORS,
    METHOD_LABELS,
    NEUTRAL,
    OBJECTIVE_TITLES,
    FigureData,
    Method,
    figure_legend,
    panels,
)
from mariner.figures.starcoder import ARM_COLORS, Arm, parameters_label, tokens_label
from mariner.olmix import LogLinearLaw, fit_log_linear

# ---------------------------------------------------------------------------------------------------------------
# Figure 12: epoch caps against two domains' optimal repetition
# ---------------------------------------------------------------------------------------------------------------


class DomainSweep(NamedTuple):
    """A sweep of Figure 12: its ``domain`` and ``evaluation`` in ``domain_sweeps``, and its panel's title."""

    domain: str
    evaluation: str
    title: str


FLAN = DomainSweep("instruction", "sciq_bpb", "Dolmino FLAN, scored on SciQ")
FINEMATH = DomainSweep("finemath", "gsm8k_bpb", "FineMath-3+, scored on GSM8K")
DOMAIN_SWEEPS = (FLAN, FINEMATH)
EPOCH_CAPS = (4.0, 8.0, 12.0)  # the caps Figure 12 marks, on the domain's materialized epochs at the target
DOMAIN_SWEEP_HUBER_DELTA = 0.02
PROXY, TARGET = "proxy", "target"  # domain_sweeps' scales
FIT_POINTS = 401
CAP_COLOR = "#8A9099"
SCALE_COLUMNS = {"parameters": "model_total_parameters", "tokens": "train_tokens"}
# The styles of Figure 12's series, shared by its panels and its legend.
CURVE_STYLES = {
    PROXY: {"color": ARM_COLORS[Arm.MATCHED], "marker": "o"},
    TARGET: {"color": ARM_COLORS[Arm.TARGET], "marker": "s", "markerfacecolor": "white", "linestyle": "--"},
}
FIT_STYLE = {"color": METHOD_COLORS[Method.OLMIX], "linestyle": ":", "linewidth": 1.6}
MINIMUM_STYLE = {
    "color": METHOD_COLORS[Method.OLMIX],
    "marker": "D",
    "markerfacecolor": "none",
    "markersize": 7.0,
    "markeredgewidth": 1.2,
    "linestyle": "none",
}
SELECTION_STYLE = {
    "color": METHOD_COLORS[Method.OLMIX],
    "marker": "D",
    "markeredgecolor": "white",
    "markersize": 5.0,
    "linestyle": "none",
}
CAP_STYLE = {"color": CAP_COLOR, "linestyle": "--", "linewidth": 0.8}


def sweep_rows(sweeps: pd.DataFrame, sweep: DomainSweep, scale: str) -> pd.DataFrame:
    return sweeps[
        (sweeps["domain"] == sweep.domain) & (sweeps["evaluation"] == sweep.evaluation) & (sweeps["scale"] == scale)
    ]


def sweep_curve(rows: pd.DataFrame) -> pd.DataFrame:
    """A sweep's runs as its curve, by share: the domain's share, its materialized epochs and the loss.

    Raises ``ValueError`` when two runs share a share.
    """
    curve = rows.rename(columns={"domain_weight": "share", "domain_epochs": "epochs"})[["share", "epochs", "bpb"]]
    repeated = curve["share"][curve["share"].duplicated()]
    if len(repeated):
        raise ValueError(f"two runs of one sweep train the share {repeated.iloc[0]:g}")
    return curve.sort_values("share", ignore_index=True)


def matched_proxy(sweeps: pd.DataFrame, sweep: DomainSweep) -> pd.DataFrame:
    """The proxy with simulated epoching: its runs that draw a subset of the domain (``subset_seed``), and the web-only
    run at share 0, which the two proxies share when this one has no run of its own there."""
    proxy = sweep_rows(sweeps, sweep, PROXY)
    matched = proxy[proxy["subset_seed"].notna()]
    if not (matched["domain_weight"] == 0.0).any():
        matched = pd.concat([proxy[proxy["domain_weight"] == 0.0], matched])
    return sweep_curve(matched)


def sweep_olmix_law(proxy: pd.DataFrame) -> LogLinearLaw:
    """Olmix's law of a proxy curve over the mixture (domain share p, web share 1 - p), fitted as the paper's were."""
    shares = proxy["share"].to_numpy(float)
    weights = np.column_stack([shares, 1.0 - shares])
    return fit_log_linear(weights, proxy["bpb"].to_numpy(float), delta=DOMAIN_SWEEP_HUBER_DELTA)


def law_bpb(law: LogLinearLaw, shares: np.ndarray) -> np.ndarray:
    return law.predict(np.column_stack([shares, 1.0 - shares]))


def lowest_share(law: LogLinearLaw, upper: float) -> float:
    """The share in [0, ``upper``] at which ``law`` predicts the least loss.

    The law's exponent is linear in the share, so it is monotone there and an end of the interval is lowest.
    """
    ends = np.array([0.0, upper])
    return float(ends[int(np.argmin(law_bpb(law, ends)))])


def unique_value(rows: pd.DataFrame, column: str, what: str) -> float:
    values = rows[column].unique()
    if len(values) != 1:
        raise ValueError(f"the {what} runs hold {len(values)} values of {column}, not one")
    return float(values[0])


def domain_repetition_values(root: Path) -> dict[str, pd.DataFrame]:
    """Each sweep's matched proxy and target, Olmix's law of the proxy, its minimum and its selection under each cap.

    Under a cap of K epochs the feasible shares run from 0 to K over the target's epochs per unit of share.
    """
    sweeps = read_table(root, "domain_sweeps")
    series: dict[str, list[pd.DataFrame]] = {name: [] for name in (PROXY, TARGET, "fit_curves")}
    fits, minima, selections, scales = [], [], [], []
    for sweep in DOMAIN_SWEEPS:
        for scale in (PROXY, TARGET):
            rows, what = sweep_rows(sweeps, sweep, scale), f"{sweep.domain} {scale}"
            sizes = {name: unique_value(rows, column, what) for name, column in SCALE_COLUMNS.items()}
            scales.append({"domain": sweep.domain, "scale": scale, **sizes})
        target_rows = sweep_rows(sweeps, sweep, TARGET)
        per_share = unique_value(target_rows, "epochs_per_unit_weight", f"{sweep.domain} target")
        proxy, target = matched_proxy(sweeps, sweep), sweep_curve(target_rows)
        law = sweep_olmix_law(proxy)
        shares = np.linspace(0.0, 1.0, FIT_POINTS)
        series[PROXY].append(proxy.assign(domain=sweep.domain))
        series[TARGET].append(target.assign(domain=sweep.domain))
        series["fit_curves"].append(pd.DataFrame({"domain": sweep.domain, "share": shares, "bpb": law_bpb(law, shares)}))
        domain, web = law.coefficients
        fits.append(
            {
                "domain": sweep.domain,
                "log_c": law.log_c,
                "coefficient_domain": domain,
                "coefficient_web": web,
                "huber_loss": law.huber_loss,
                "target_epochs_per_share": per_share,
            }
        )
        share = lowest_share(law, 1.0)
        minima.append({"domain": sweep.domain, "share": share, "bpb": float(law_bpb(law, np.array([share]))[0])})
        for cap in EPOCH_CAPS:
            share = lowest_share(law, min(cap / per_share, 1.0))
            selections.append(
                {
                    "domain": sweep.domain,
                    "cap": cap,
                    "share": share,
                    "epochs": share * per_share,
                    "predicted_bpb": float(law_bpb(law, np.array([share]))[0]),
                }
            )
    return {
        **{name: pd.concat(frames, ignore_index=True) for name, frames in series.items()},
        "fits": pd.DataFrame(fits),
        "fit_minima": pd.DataFrame(minima),
        "selections": pd.DataFrame(selections),
        "scales": pd.DataFrame(scales),
    }


def scale_label(scales: pd.DataFrame, scale: str, name: str) -> str:
    """``name`` with the model size and tokens of the runs at ``scale``, which every sweep must share."""
    rows = scales[scales["scale"] == scale]
    parameters, tokens = unique_value(rows, "parameters", name), unique_value(rows, "tokens", name)
    return f"{name} ({parameters_label(parameters)} parameters, {tokens_label(tokens)} tokens)"


def draw_domain_repetition(root: Path) -> FigureData:
    """Figure 12: each domain's proxy and target loss against its share, with Olmix's fit and its picks under caps."""
    values = domain_repetition_values(root)
    fits = values["fits"].set_index("domain")
    figure, axes = panels(len(DOMAIN_SWEEPS), legend_height=0.9)
    for axis, sweep in zip(axes, DOMAIN_SWEEPS, strict=True):
        per_share = float(fits.loc[sweep.domain, "target_epochs_per_share"])
        for cap in EPOCH_CAPS:
            axis.axvline(cap / per_share, zorder=1, **CAP_STYLE)
            axis.annotate(
                f"cap {cap:g}",
                xy=(cap / per_share, 1.0),
                xycoords=axis.get_xaxis_transform(),
                xytext=(2, -2),
                textcoords="offset points",
                ha="left",
                va="top",
                color=CAP_COLOR,
                fontsize=6.5,
            )
        for name, style in CURVE_STYLES.items():
            curve = values[name][values[name]["domain"] == sweep.domain]
            axis.plot(curve["share"], curve["bpb"], zorder=3, **style)
        fit = values["fit_curves"][values["fit_curves"]["domain"] == sweep.domain]
        axis.plot(fit["share"], fit["bpb"], zorder=4, **FIT_STYLE)
        minimum = values["fit_minima"][values["fit_minima"]["domain"] == sweep.domain]
        axis.plot(minimum["share"], minimum["bpb"], clip_on=False, zorder=5, **MINIMUM_STYLE)
        selected = values["selections"][values["selections"]["domain"] == sweep.domain]
        axis.plot(selected["share"], selected["predicted_bpb"], zorder=6, **SELECTION_STYLE)
        epochs = axis.secondary_xaxis(
            "top", functions=(lambda share, scale=per_share: share * scale, lambda e, scale=per_share: e / scale)
        )
        epochs.set_xlabel("Target's epochs of the domain")
        axis.set_xlim(-0.03, 1.03)
        axis.set_xlabel("Domain's share of the mixture, p")
        axis.set_ylabel("Loss (BPB)")
        axis.set_title(sweep.title)
    handles = [
        Line2D([], [], label=scale_label(values["scales"], PROXY, "matched proxy"), **CURVE_STYLES[PROXY]),
        Line2D([], [], label=scale_label(values["scales"], TARGET, "target"), **CURVE_STYLES[TARGET]),
        Line2D([], [], label="Olmix's law of the matched proxy", **FIT_STYLE),
        Line2D([], [], label="its minimum", **MINIMUM_STYLE),
        Line2D([], [], label="its selection under each cap", **SELECTION_STYLE),
        Line2D([], [], label="epoch cap (excludes shares to its right)", **CAP_STYLE),
    ]
    figure_legend(figure, handles, columns=2)
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figure 13: the epochs of each bucket in the proposed mixtures
# ---------------------------------------------------------------------------------------------------------------


class EpochColumn(NamedTuple):
    """A column of Figure 13: the objective its mixture was proposed for, its printed name, the mixture, and the
    ``method`` and ``variant`` that ``validation_3e18`` records for the mixture's runs."""

    objective: str
    name: str
    mixture_id: str
    method: Method
    variant: str


# The variants of each column's runs, by objective: MARINER's proposal, Olmix's deployed proposal (cap 4 and the KL
# weight of mariner.data.TRAINED_OLMIX), and Olmix's proposal without a cap or KL penalty, which the Olmix cap and KL
# sweep trained.
COLUMN_VARIANTS = {
    "uncheatable": ("lwspu_u_snc_cap06", "olmixq_u_kl0p05_cap04", "olmixq_u_kl0_nocap_cap17"),
    "obe_easy": ("lwspu_t9_snc_cap08", "olmixq_t9_kl0p005_cap04", "olmixq_t9_kl0_nocap_cap17"),
}
UNREGULARIZED_OLMIX = {"uncheatable": "mix_522b7ad988b71710", "obe_easy": "mix_d5cbd56954690512"}
EPOCH_COLUMNS = tuple(
    EpochColumn(objective, name, mixture, method, variant)
    for objective in OBJECTIVES
    for name, mixture, method, variant in zip(
        (
            METHOD_LABELS[Method.MARINER],
            f"Olmix, cap {OLMIX_CAP:g}, KL {TRAINED_OLMIX[objective].kl:g}",
            "Olmix, no cap, no KL",
        ),
        (TRAINED_MARINER[objective], TRAINED_OLMIX[objective].mixture_id, UNREGULARIZED_OLMIX[objective]),
        (Method.MARINER, Method.OLMIX, Method.OLMIX),
        COLUMN_VARIANTS[objective],
        strict=True,
    )
)
# The buckets' sources, in the order of the figure's rows: the Common Crawl topics at two quality tiers, Dolma 3's
# other sources, and Dolmino's.
COMMON_CRAWL, DOLMA3, DOLMINO = "Common Crawl", "Dolma 3, other sources", "Dolmino"
SOURCES = (COMMON_CRAWL, DOLMA3, DOLMINO)
DOLMA3_POOL, DOLMINO_POOL = "dolma3", "dolmino"
# Colors and labels show log2 epochs, from 1/8 to 16, around one epoch.
LOG2_EPOCH_RANGE = (-3.0, 4.0)
EPOCH_COLORMAP = "RdBu_r"
EPOCH_TICKS = {-3.0: "≤1/8", -2.0: "1/4", -1.0: "1/2", 0.0: "1", 1.0: "2", 2.0: "4", 3.0: "8", 4.0: "≥16"}
DARK_CELL = 0.6  # cells whose color lies this far from the center along the scale take white text
SMALLEST_PRINTED_EPOCHS = 0.1
CELL_HEIGHT = 0.17  # inches per row
HEATMAP_WIDTH = 7.4  # inches
HEATMAP_HEADER = 1.8  # inches above and below the rows: column labels and the color bar
OBJECTIVE_HEADER_RISE = 34.0  # points above the rows: each objective's title, over its columns' labels


def epoch_text(epochs: float) -> str:
    """Epochs as Figure 13 prints them: 0 for a bucket left out, <0.1 below a tenth, one decimal up to ten, and whole
    numbers from ten."""
    if epochs == 0.0:
        return "0"
    if epochs < SMALLEST_PRINTED_EPOCHS:
        return f"<{SMALLEST_PRINTED_EPOCHS:g}"
    if epochs >= 10.0:
        return f"{epochs:.0f}"
    return f"{epochs:.1f}"


def bucket_sources(partition: pd.DataFrame) -> pd.Series:
    """Each bucket's source, by bucket id: Common Crawl for the topics at a quality tier, else its pool's."""
    source = np.where(
        partition["quality_tier"].notna(),
        COMMON_CRAWL,
        np.where(partition["pool"] == DOLMINO_POOL, DOLMINO, DOLMA3),
    )
    unknown = sorted(set(partition.loc[~partition["pool"].isin([DOLMA3_POOL, DOLMINO_POOL]), "pool"]))
    if unknown:
        raise ValueError(f"the partition has buckets of the unknown pools {', '.join(map(str, unknown))}")
    return pd.Series(source, index=partition["bucket_id"])


def mixture_epochs(
    validation: pd.DataFrame, column: EpochColumn, bucket_ids: tuple[str, ...], per_weight: np.ndarray
) -> np.ndarray:
    """The epochs of each bucket in ``column``'s mixture: weight times epochs per unit weight, over the pool fraction.

    Raises ``ValueError`` unless the mixture's runs share one mixture and record the column's method, objective and
    variant.
    """
    mixture = column.mixture_id
    runs = validation[validation["mixture_id"] == mixture]
    if runs.empty:
        raise ValueError(f"no validation_3e18 run trains {mixture}")
    recorded = set(zip(runs["method"], runs["proposal_objective"], runs["variant"], strict=True))
    if recorded != {(column.method.value, column.objective, column.variant)}:
        raise ValueError(
            f"the runs of {mixture} record (method, proposal objective, variant) {sorted(recorded)}, not"
            f" {(column.method.value, column.objective, column.variant)}"
        )
    weights = mixture_weights(runs, bucket_ids, "validation_3e18")
    fractions = pool_fractions(runs, bucket_ids, "validation_3e18")
    if not ((weights == weights[0]).all() and (fractions == fractions[0]).all()):
        raise ValueError(f"the runs of {mixture} do not share one mixture")
    return weights[0] * per_weight / fractions[0]


def mixture_epochs_values(root: Path) -> dict[str, pd.DataFrame]:
    """Each column's epochs per bucket, the rows' sources and names, and proportional mixing's epochs."""
    table = read_table(root, "partition")
    partition = release_partition(root)
    validation = read_table(root, "validation_3e18")
    sources = bucket_sources(table)
    names = table.set_index("bucket_id")["paper_name"].str.replace("\\&", "&", regex=False)
    order = [bucket for source in SOURCES for bucket in partition.bucket_ids if sources[bucket] == source]
    rows = []
    for column in EPOCH_COLUMNS:
        epochs = pd.Series(
            mixture_epochs(validation, column, partition.bucket_ids, partition.epochs_per_unit_weight),
            index=partition.bucket_ids,
        )
        rows += [
            {
                "objective": column.objective,
                "column": column.name,
                "mixture_id": column.mixture_id,
                "bucket_id": bucket,
                "bucket": names[bucket],
                "source": sources[bucket],
                "epochs": float(epochs[bucket]),
            }
            for bucket in order
        ]
    proportional = partition.pool_tokens / partition.pool_tokens.sum() * partition.epochs_per_unit_weight
    return {
        "epochs": pd.DataFrame(rows),
        "proportional": pd.DataFrame({"bucket_id": partition.bucket_ids, "epochs": proportional}),
    }


def draw_mixture_epochs(root: Path) -> FigureData:
    """Figure 13: each bucket's epochs in each proposed mixture, the rows grouped by source; colors and numbers show
    the epochs."""
    values = mixture_epochs_values(root)
    epochs = values["epochs"]
    columns = list(dict.fromkeys(zip(epochs["objective"], epochs["column"], strict=True)))
    grid = epochs.pivot_table(index="bucket_id", columns=["objective", "column"], values="epochs", sort=False)
    grid = grid.loc[list(dict.fromkeys(epochs["bucket_id"])), columns]
    rows = epochs.drop_duplicates("bucket_id").set_index("bucket_id").loc[grid.index]
    matrix = grid.to_numpy(float)
    with np.errstate(divide="ignore"):
        logged = np.clip(np.log2(matrix), *LOG2_EPOCH_RANGE)
    norm = TwoSlopeNorm(vmin=LOG2_EPOCH_RANGE[0], vcenter=0.0, vmax=LOG2_EPOCH_RANGE[1])
    figure = Figure(figsize=(HEATMAP_WIDTH, len(grid) * CELL_HEIGHT + HEATMAP_HEADER), layout="constrained")
    axis = figure.subplots()
    image = axis.imshow(np.ma.masked_where(matrix == 0.0, logged), cmap=EPOCH_COLORMAP, norm=norm, aspect="auto")
    for (row, column), value in np.ndenumerate(matrix):
        dark = value > 0.0 and abs(float(norm(logged[row, column])) - 0.5) > DARK_CELL / 2
        axis.text(
            column, row, epoch_text(value), ha="center", va="center", fontsize=6.5, color="white" if dark else "black"
        )
    axis.set_xticks(range(len(columns)), [name.replace(", ", "\n") for _, name in columns])
    axis.xaxis.tick_top()
    per_objective = len(columns) // len(OBJECTIVES)
    for index, objective in enumerate(OBJECTIVES):
        axis.annotate(
            OBJECTIVE_TITLES[objective],
            xy=(index * per_objective + (per_objective - 1) / 2, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, OBJECTIVE_HEADER_RISE),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
        )
    axis.set_yticks(range(len(grid)), rows["bucket"].tolist())
    axis.tick_params(length=0)
    axis.grid(False)
    for side in axis.spines.values():
        side.set_visible(False)
    starts = rows["source"].ne(rows["source"].shift()).to_numpy().nonzero()[0]
    for start in starts[1:]:
        axis.axhline(start - 0.5, color=NEUTRAL, linewidth=1.0)
    for start, source in zip(starts, rows["source"].iloc[starts], strict=True):
        axis.annotate(
            source,
            xy=(1.0, start - 0.5),
            xycoords=("axes fraction", "data"),
            xytext=(4, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=7.5,
            fontweight="bold",
        )
    for index in range(1, len(OBJECTIVES)):
        axis.axvline(index * per_objective - 0.5, color=NEUTRAL, linewidth=1.0)
    colorbar = figure.colorbar(image, ax=axis, location="bottom", shrink=0.6, aspect=40)
    colorbar.set_ticks(list(EPOCH_TICKS), labels=list(EPOCH_TICKS.values()))
    colorbar.set_label("Materialized epochs in the target run (0, uncolored: the bucket is left out)")
    return FigureData(figure, values)


# ---------------------------------------------------------------------------------------------------------------
# Figure 15: cross-validated R^2 against signal-to-noise ratio
# ---------------------------------------------------------------------------------------------------------------

R2_MODELS = (Method.MARINER, Method.OLMIX, Method.REGMIX)
# The OlmoBaseEval Easy components below this SNR, both Basic Skills subtasks, are drawn uncollapsed as well.
UNCOLLAPSED_SNR = 2.0
OLMIX_R2 = f"{Method.OLMIX.value}_r2"
OBE_EASY, UNCHEATABLE = "OlmoBaseEval Easy", "Uncheatable"
EVALUATION_MARKERS = {OBE_EASY: "o", UNCHEATABLE: "D"}
SNR_TICKS = (1, 2, 5, 10, 20, 40)
CEILING_SNR = (1.0, 45.0)
CEILING_POINTS = 200


def r2_points(root: Path) -> pd.DataFrame:
    """Each plotted task's SNR and each surrogate's out-of-fold R^2, 1 - MSE / Var over the swarm; ``labeled`` marks
    the tasks with a negative Olmix R^2, the rule of the paper's caption."""
    reliability = read_table(root, RELIABILITY_TABLE)
    tasks = reliability_rows(reliability, RELIABILITY_COLLAPSED_SOURCE)
    tasks = tasks[tasks["group"].notna()]
    components = reliability_rows(reliability, RELIABILITY_SOURCES["obe_easy"])
    uncollapsed = components[components["task_id"].notna() & (components["snr"] < UNCOLLAPSED_SNR)]
    uncheatable = reliability_rows(reliability, RELIABILITY_SOURCES["uncheatable"])
    uncheatable = uncheatable[uncheatable["task_id"].notna()]
    points = pd.concat(
        [
            tasks.assign(evaluation=OBE_EASY, name=tasks["task"]),
            uncollapsed.assign(evaluation=OBE_EASY, name=uncollapsed["task"].str.cat(uncollapsed["subtask"], sep=": ")),
            uncheatable.assign(evaluation=UNCHEATABLE, name=uncheatable["task"]),
        ],
        ignore_index=True,
    )
    repeated = points["name"][points["name"].duplicated()]
    if len(repeated):
        raise ValueError(f"{RELIABILITY_TABLE} names two plotted tasks {repeated.iloc[0]}")
    columns = [f"{model.value}_r2" for model in R2_MODELS]
    for model, column in zip(R2_MODELS, columns, strict=True):
        points[column] = 1.0 - points[f"{model.value}_mse"] / points["panel_sd"] ** 2
    points["labeled"] = points[OLMIX_R2] < 0.0
    return points[["name", "evaluation", "snr", *columns, "labeled"]]


def draw_r2_vs_snr(root: Path) -> FigureData:
    """Figure 15: each surrogate's cross-validated R^2 against the task's SNR, with the noise ceiling 1 - 1/SNR^2."""
    points = r2_points(root)
    columns = [f"{model.value}_r2" for model in R2_MODELS]
    figure, axes = panels(1, legend_width=1.6, legend_height=0.9)
    axis = axes[0]
    axis.vlines(
        points["snr"],
        points[columns].min(axis=1),
        points[columns].max(axis=1),
        color=NEUTRAL,
        linestyles="--",
        linewidth=0.6,
        alpha=0.6,
        zorder=2,
    )
    snr = np.geomspace(*CEILING_SNR, CEILING_POINTS)
    ceiling = pd.DataFrame({"snr": snr, "r2": 1.0 - 1.0 / snr**2})
    axis.plot(ceiling["snr"], ceiling["r2"], color=NEUTRAL, linewidth=1.0, zorder=1)
    for depth, (model, column) in enumerate(reversed(list(zip(R2_MODELS, columns, strict=True)))):
        color = METHOD_COLORS[model]
        for evaluation, marker in EVALUATION_MARKERS.items():
            rows = points[points["evaluation"] == evaluation]
            axis.plot(
                rows["snr"],
                rows[column],
                marker,
                linestyle="none",
                color=color,
                markerfacecolor="white" if model == Method.OLMIX else color,
                markersize=4.5,
                zorder=3 + depth,
            )
    for row in points[points["labeled"]].itertuples():
        axis.annotate(
            row.name,
            xy=(row.snr, getattr(row, OLMIX_R2)),
            xytext=(4, -2),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=6.5,
        )
    axis.set_xscale("log")
    axis.set_xticks(SNR_TICKS, [str(tick) for tick in SNR_TICKS])
    axis.minorticks_off()
    axis.set_xlim(0.9, CEILING_SNR[1])
    axis.set_xlabel("Signal-to-noise ratio: SD over the swarm / SD over proportional repeats")
    axis.set_ylabel("Cross-validated R²")
    axis.set_title("Task predictability against SNR")
    handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="s",
            color=METHOD_COLORS[model],
            markerfacecolor="white" if model == Method.OLMIX else METHOD_COLORS[model],
            label=METHOD_LABELS[model],
        )
        for model in R2_MODELS
    ]
    handles += [
        Line2D([], [], linestyle="none", marker=marker, color=NEUTRAL, label=f"{evaluation} task")
        for evaluation, marker in EVALUATION_MARKERS.items()
    ]
    handles += [
        Line2D([], [], color=NEUTRAL, linestyle="--", linewidth=0.6, label="one task's three surrogates"),
        Line2D([], [], color=NEUTRAL, linewidth=1.0, label="noise ceiling 1 - 1/SNR²"),
    ]
    figure_legend(figure, handles, columns=2)
    return FigureData(figure, {"points": points, "ceiling": ceiling})
