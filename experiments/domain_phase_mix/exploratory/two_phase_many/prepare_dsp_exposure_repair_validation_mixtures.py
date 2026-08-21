# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas", "plotly"]
# ///
"""Prepare DSP aggregate-exposure repair mixtures for 3e18 validation.

This diagnostic constructs two repaired mixtures per objective:

1. A targeted repair that raises a small set of underexposed buckets to the
   corresponding best one-phase aggregate exposure.
2. A broad repair that raises every underexposed bucket to the corresponding
   best one-phase aggregate exposure.

The phase-aggregated mass is changed first, then phase weights are reconstructed
using the original two-phase contrast vector. A global contrast shrink is
applied only when needed to keep phase weights nonnegative.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
import plotly.io as pio

from plot_dsp_uncheatable_exposure_repair import (
    original_frame,
    plot_repair,
    repair_aggregate_exposure,
)
from plot_one_vs_two_phase_best_mixtures import (
    COMPARISONS,
    OUTPUT_DIR as BEST_MIXTURE_OUTPUT_DIR,
    PHASE_0_FRACTION,
    PHASE_1_FRACTION,
    PLOT_CONFIG,
    comparison_frames,
)


OUTPUT_DIR = (
    BEST_MIXTURE_OUTPUT_DIR.parent / "dsp_exposure_repair_validation_mixtures_20260702"
)
MIXTURE_DIR = OUTPUT_DIR / "mixtures"

TARGETED_REPAIR_DOMAINS = {
    "Uncheatable BPB": [
        "dolmino_synth_code",
        "dolma3_wikipedia",
        "dolmino_stack_edu_fim",
        "dolmino_synth_instruction",
        "dolma3_arxiv",
        "dolmino_stem_heavy_crawl",
        "dolmino_synth_math",
        "dolma3_cc/science_math_and_technology_low",
        "dolma3_cc/science_math_and_technology_high",
        "dolma3_cc/literature_high",
        "dolma3_stack_edu",
        "dolma3_finemath_3plus",
    ],
    "Table-9 Macro BPB": [
        # All dolmino buckets whose aggregate exposure is lower than the
        # best one-phase DSP reference, plus the largest non-dolmino deficits.
        "dolmino_olmocr_pdfs_hq",
        "dolmino_stack_edu_fim",
        "dolmino_stem_heavy_crawl",
        "dolmino_synth_code",
        "dolmino_synth_instruction",
        "dolmino_synth_math",
        "dolmino_synth_qa",
        "dolmino_synth_thinking",
        "dolma3_wikipedia",
        "dolma3_finemath_3plus",
        "dolma3_stack_edu",
        "dolma3_arxiv",
    ],
}
OBJECTIVE_TO_KEY = {
    "Uncheatable BPB": "uncheatable",
    "Table-9 Macro BPB": "table9",
}


@dataclass(frozen=True)
class CandidateSummary:
    mixture_id: str
    objective: str
    repair_type: str
    source_two_phase_label: str
    source_single_phase_label: str
    selected_domain_count: int
    selected_domains: str
    selected_mass_increase: float
    donor_surplus_scale: float
    contrast_scale: float
    max_simulated_epochs: float
    q95_simulated_epochs: float
    max_phase_0_epochs: float
    max_phase_1_epochs: float
    phase0_sum: float
    phase1_sum: float
    aggregate_sum: float
    min_phase_weight: float
    output_csv: str


def q95(values: pd.Series) -> float:
    return float(values.quantile(0.95, interpolation="linear"))


def launch_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "domain",
        "phase_0_weight",
        "phase_1_weight",
        "simulated_epochs",
        "aggregate_weight",
        "phase_0_epoch_multiplier",
        "phase_1_epoch_multiplier",
    ]
    output = frame[columns].copy()
    for phase_column in ["phase_0_weight", "phase_1_weight"]:
        min_weight = float(output[phase_column].min())
        if min_weight < -1e-12:
            raise ValueError(f"{phase_column} has negative weight {min_weight}")
        output[phase_column] = output[phase_column].clip(lower=0.0)
        phase_sum = float(output[phase_column].sum())
        if phase_sum <= 0:
            raise ValueError(f"{phase_column} sums to {phase_sum}")
        output[phase_column] = output[phase_column] / phase_sum

    p = output["aggregate_weight"] / output["simulated_epochs"]
    output["aggregate_weight"] = (
        PHASE_0_FRACTION * output["phase_0_weight"]
        + PHASE_1_FRACTION * output["phase_1_weight"]
    )
    output["simulated_epochs"] = output["aggregate_weight"] / p
    output["phase_0_epoch_multiplier"] = output["phase_0_weight"] / p
    output["phase_1_epoch_multiplier"] = output["phase_1_weight"] / p
    return output


def summarize_candidate(
    *,
    mixture_id: str,
    objective: str,
    repair_type: str,
    selected_domains: list[str],
    repaired,
    source_single_phase_label: str,
    source_two_phase_label: str,
    output_csv: Path,
) -> CandidateSummary:
    frame = repaired.frame
    return CandidateSummary(
        mixture_id=mixture_id,
        objective=objective,
        repair_type=repair_type,
        source_two_phase_label=source_two_phase_label,
        source_single_phase_label=source_single_phase_label,
        selected_domain_count=len(selected_domains),
        selected_domains=";".join(selected_domains),
        selected_mass_increase=repaired.total_selected_mass_increase,
        donor_surplus_scale=repaired.donor_scale,
        contrast_scale=repaired.contrast_scale,
        max_simulated_epochs=float(frame["simulated_epochs"].max()),
        q95_simulated_epochs=q95(frame["simulated_epochs"]),
        max_phase_0_epochs=float(frame["phase_0_epoch_multiplier"].max()),
        max_phase_1_epochs=float(frame["phase_1_epoch_multiplier"].max()),
        phase0_sum=float(frame["phase_0_weight"].sum()),
        phase1_sum=float(frame["phase_1_weight"].sum()),
        aggregate_sum=float(frame["aggregate_weight"].sum()),
        min_phase_weight=float(frame[["phase_0_weight", "phase_1_weight"]].min().min()),
        output_csv=str(output_csv),
    )


def objective_spec(task: str):
    return [spec for spec in COMPARISONS if spec.task == task and spec.method == "DSP"][0]


def objective_repair(task: str) -> tuple[list[CandidateSummary], list[tuple[str, object]]]:
    spec = objective_spec(task)
    _, _, merged = comparison_frames(spec)
    merged = merged.sort_values("domain").reset_index(drop=True)
    merged["exposure_deficit_single_minus_two"] = (
        merged["simulated_epochs_single"] - merged["simulated_epochs_two_phase"]
    )
    targeted_domains = TARGETED_REPAIR_DOMAINS[task]
    all_deficit_domains = merged.loc[
        merged["exposure_deficit_single_minus_two"] > 1e-9, "domain"
    ].tolist()
    objective_key = OBJECTIVE_TO_KEY[task]
    repairs = [
        (
            f"dsp_{objective_key}_exposure_targeted",
            "targeted",
            targeted_domains,
            "targeted exposure repair",
        ),
        (
            f"dsp_{objective_key}_exposure_all_deficits",
            "all_deficits",
            all_deficit_domains,
            "all-deficit exposure repair",
        ),
    ]
    two_original = original_frame(merged, "two_phase")
    single_reference = original_frame(merged, "single")
    summaries: list[CandidateSummary] = []
    figures: list[tuple[str, object]] = []
    deficit_path = OUTPUT_DIR / f"{objective_key}_exposure_deficits.csv"
    merged.sort_values("exposure_deficit_single_minus_two", ascending=False)[
        [
            "domain",
            "domain_short",
            "domain_group",
            "aggregate_weight_single",
            "aggregate_weight_two_phase",
            "simulated_epochs_single",
            "simulated_epochs_two_phase",
            "exposure_deficit_single_minus_two",
            "phase_0_weight_two_phase",
            "phase_1_weight_two_phase",
        ]
    ].to_csv(deficit_path, index=False)

    order_domains = (
        merged.sort_values("exposure_deficit_single_minus_two", ascending=True)["domain"]
        .tolist()
    )
    for mixture_id, repair_type, selected_domains, label in repairs:
        repaired = repair_aggregate_exposure(
            merged,
            selected_domains,
            name="repaired_top3" if repair_type == "targeted" else "repaired_all_deficits",
            label=label,
        )
        output_csv = MIXTURE_DIR / f"{mixture_id}.csv"
        launch_ready_frame(repaired.frame).to_csv(output_csv, index=False)
        repaired.frame.to_csv(OUTPUT_DIR / f"{mixture_id}_diagnostic.csv", index=False)
        summaries.append(
            summarize_candidate(
                mixture_id=mixture_id,
                objective=task,
                repair_type=repair_type,
                selected_domains=selected_domains,
                repaired=repaired,
                source_single_phase_label=spec.single.label,
                source_two_phase_label=spec.two_phase.label,
                output_csv=output_csv,
            )
        )
        long_rows = []
        for name, mixture_label, frame in [
            ("two_phase_original", "original two-phase DSP", two_original),
            (repaired.name, repaired.label, repaired.frame),
            ("single_phase_reference", "single-phase DSP reference", single_reference),
        ]:
            for _, row in frame.iterrows():
                for phase, weight_column, epoch_column in [
                    ("phase_0", "phase_0_weight", "phase_0_epoch_multiplier"),
                    ("phase_1", "phase_1_weight", "phase_1_epoch_multiplier"),
                    ("aggregate", "aggregate_weight", "simulated_epochs"),
                ]:
                    long_rows.append(
                        {
                            "mixture": name,
                            "mixture_label": mixture_label,
                            "phase": phase,
                            "domain": row["domain"],
                            "domain_short": row["domain_short"],
                            "domain_group": row["domain_group"],
                            "weight": float(row[weight_column]),
                            "epoch_multiplier": float(row[epoch_column]),
                        }
                    )
        long_df = pd.DataFrame(long_rows)
        long_df.to_csv(OUTPUT_DIR / f"{mixture_id}_long.csv", index=False)
        figure = plot_repair(
            f"{task} DSP aggregate-exposure repair",
            long_df,
            order_domains,
            (
                f"{label}; selected {len(selected_domains)} buckets; "
                f"contrast scale {repaired.contrast_scale:.3f}, donor scale {repaired.donor_scale:.3f}."
            ),
        )
        figure_path = OUTPUT_DIR / f"{mixture_id}.html"
        figure.write_html(figure_path, include_plotlyjs="cdn", config=PLOT_CONFIG)
        figures.append((f"{task}: {label}", figure))
    return summaries, figures


def write_index(figures: list[tuple[str, object]], manifest: pd.DataFrame) -> None:
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>DSP exposure-repair validation mixtures</title>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:24px;color:#172033}"
        "table{border-collapse:collapse;margin:16px 0;width:100%;font-size:13px}"
        "th,td{border:1px solid #d9e0ea;padding:6px 8px;text-align:left;vertical-align:top}"
        "th{background:#eef3f8} code{background:#eef3f8;padding:2px 4px;border-radius:4px}</style>",
        "</head><body>",
        "<h1>DSP exposure-repair validation mixtures</h1>",
        "<p>Each candidate starts from the current best two-phase DSP candidate for the objective, "
        "raises selected aggregate exposures to the best single-phase DSP reference, removes the required "
        "mass only from non-selected buckets with aggregate surplus above the single-phase reference, and "
        "reconstructs phase weights using the original phase contrast with global shrink only if needed for "
        "nonnegative phase weights.</p>",
        manifest.to_html(index=False, escape=True),
    ]
    include_js: str | bool = "cdn"
    for title, figure in figures:
        parts.append(f"<h2>{title}</h2>")
        parts.append(pio.to_html(figure, include_plotlyjs=include_js, full_html=False, config=PLOT_CONFIG))
        include_js = False
    parts.append("</body></html>")
    (OUTPUT_DIR / "dsp_exposure_repair_validation_mixtures.html").write_text("\n".join(parts))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries: list[CandidateSummary] = []
    all_figures: list[tuple[str, object]] = []
    for task in ["Uncheatable BPB", "Table-9 Macro BPB"]:
        summaries, figures = objective_repair(task)
        all_summaries.extend(summaries)
        all_figures.extend(figures)
    manifest = pd.DataFrame([asdict(summary) for summary in all_summaries])
    manifest.to_csv(OUTPUT_DIR / "validation_mixture_manifest.csv", index=False)
    (OUTPUT_DIR / "validation_mixture_manifest.json").write_text(
        json.dumps([asdict(summary) for summary in all_summaries], indent=2)
    )
    write_index(all_figures, manifest)
    print(manifest.to_string(index=False))
    print(f"wrote {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
