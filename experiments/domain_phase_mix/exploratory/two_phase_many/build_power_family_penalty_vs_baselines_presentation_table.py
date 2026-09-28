# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas"]
# ///
"""Build a presentation-friendly Markdown table for power_family_penalty vs 60M baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_RANKED_CSV = SCRIPT_DIR / "eval_signal_to_noise_ranked.csv"
INPUT_WIDE_CSV = SCRIPT_DIR / "power_family_penalty_vs_baselines_ranked_metrics_common_wide.csv"
OUTPUT_MD = SCRIPT_DIR / "power_family_penalty_vs_baselines_presentation_table.md"
OUTPUT_ALL_EVAL_MD = SCRIPT_DIR / "power_family_penalty_vs_baselines_presentation_table_all_perplexity.md"

ROW_ORDER = [
    "proportional",
    "unimax",
    "uniform_stratified",
    "olmix",
    "power_family_penalty",
]

DISPLAY_ROW_LABELS = {
    "proportional": "Proportional",
    "unimax": "Unimax",
    "uniform_stratified": "Uniform",
    "olmix": "Olmix",
    "power_family_penalty": "Power-Family Penalty",
}

METRIC_DISPLAY_NAMES = {
    "eval/uncheatable_eval/bpb": "Uncheatable Eval",
    "eval/uncheatable_eval/macro_bpb": "Uncheatable Eval Macro",
    "eval/paloma/bpb": "Paloma",
    "eval/paloma/macro_bpb": "Paloma Macro",
    "lm_eval/piqa_5shot/bpb": "PIQA",
    "lm_eval/socialiqa_5shot/choice_logprob": "Social IQA CLP",
    "lm_eval/hellaswag_5shot/bpb": "HellaSwag",
    "lm_eval/arc_challenge_5shot/bpb": "ARC Challenge",
    "lm_eval/lambada_0shot/perplexity": "LAMBADA ppl",
    "lm_eval/arc_easy_5shot/bpb": "ARC Easy",
    "lm_eval/socialiqa_5shot/acc": "Social IQA Acc",
    "lm_eval/winogrande_5shot/bpb": "Winogrande",
    "lm_eval/medmcqa_5shot/acc_norm": "MedMCQA",
    "lm_eval/sciq_5shot/acc_norm": "SciQ",
    "lm_eval/mmlu_sl_verb_5shot/bpb": "MMLU-SL Verb",
    "lm_eval/mmlu_humanities_5shot/bpb": "MMLU Humanities",
    "lm_eval/mmlu_social_sciences_5shot/bpb": "MMLU SocSci",
    "lm_eval/olmo_base_easy_overlap/macro_bpb": "OLMoBase Overlap",
    "lm_eval/csqa_5shot/bpb": "CSQA",
    "lm_eval/mmlu_stem_5shot/bpb": "MMLU STEM",
    "lm_eval/mmlu_5shot/bpb": "MMLU",
    "lm_eval/mmlu_other_5shot/bpb": "MMLU Other",
}

LOWER_IS_BETTER_KINDS = {"bpb", "perplexity"}
HIGHER_IS_BETTER_KINDS = {"acc", "acc_norm", "choice_logprob"}
TOP_LEVEL_EVAL_METRICS = [
    "eval/uncheatable_eval/bpb",
    "eval/paloma/bpb",
    "eval/uncheatable_eval/macro_bpb",
    "eval/paloma/macro_bpb",
]
TOKEN_DISPLAY_NAMES = {
    "ao3": "AO3",
    "bbc": "BBC",
    "c4": "C4",
    "cpp": "C++",
    "gab": "Gab",
    "helm": "HELM",
    "mc4": "mC4",
    "m2d2": "M2D2",
    "ptb": "PTB",
    "s2orc": "S2ORC",
}


def _load_metric_metadata() -> pd.DataFrame:
    ranked = pd.read_csv(INPUT_RANKED_CSV)
    metadata = ranked.drop_duplicates(subset=["metric"]).set_index("metric")
    return metadata


def _benchmark_metrics(metadata: pd.DataFrame) -> list[str]:
    candidates = []
    for metric, row in metadata.iterrows():
        if not metric.startswith("lm_eval/"):
            continue
        candidates.append((metric, float(row["signal_to_noise"])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [metric for metric, _ in candidates]


def _remaining_eval_metrics(metadata: pd.DataFrame) -> list[str]:
    candidates = []
    for metric, row in metadata.iterrows():
        if not metric.startswith("eval/"):
            continue
        if metric in TOP_LEVEL_EVAL_METRICS:
            continue
        candidates.append((metric, float(row["signal_to_noise"])))
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [metric for metric, _ in candidates]


def _focused_column_order(metadata: pd.DataFrame) -> list[str]:
    return TOP_LEVEL_EVAL_METRICS + _benchmark_metrics(metadata)


def _all_eval_column_order(metadata: pd.DataFrame) -> list[str]:
    return TOP_LEVEL_EVAL_METRICS + _remaining_eval_metrics(metadata) + _benchmark_metrics(metadata)


def _metric_kind(metric: str, metadata: pd.DataFrame) -> str:
    return str(metadata.loc[metric, "primary_metric_kind"])


def _format_value(value: float) -> str:
    return f"{value:.3f}"


def _best_mask(frame: pd.DataFrame, metric: str, metadata: pd.DataFrame) -> pd.Series:
    values = frame[metric]
    kind = _metric_kind(metric, metadata)
    if kind in LOWER_IS_BETTER_KINDS:
        target = values.min()
    elif kind in HIGHER_IS_BETTER_KINDS:
        target = values.max()
    else:
        raise ValueError(f"Unknown metric kind for {metric}: {kind}")
    return values.eq(target)


def _display_token(token: str) -> str:
    return TOKEN_DISPLAY_NAMES.get(token.lower(), token.title())


def _titleize_slug(slug: str) -> str:
    pieces: list[str] = []
    for chunk in slug.split("_"):
        if "-" in chunk:
            subpieces = [_display_token(part) for part in chunk.split("-")]
            pieces.append("-".join(subpieces))
        else:
            pieces.append(_display_token(chunk))
    return " ".join(pieces)


def _display_name(metric: str, metadata: pd.DataFrame) -> str:
    if metric in METRIC_DISPLAY_NAMES:
        return METRIC_DISPLAY_NAMES[metric]
    eval_name = str(metadata.loc[metric, "eval_name"])
    if metric.startswith("eval/uncheatable_eval/"):
        suffix = eval_name.split("::", 1)[1]
        return _titleize_slug(suffix)
    if metric.startswith("eval/paloma/"):
        suffix = eval_name.split("::", 1)[1]
        return _titleize_slug(suffix)
    if metric == "eval/macro_bpb":
        return "Eval Macro"
    if metric == "eval/bpb":
        return "Eval"
    return metric


def build_table(*, columns: list[str]) -> str:
    metadata = _load_metric_metadata()
    frame = pd.read_csv(INPUT_WIDE_CSV).set_index("label").loc[ROW_ORDER]

    header_cells = ["Model"]
    for metric in columns:
        display = _display_name(metric, metadata)
        snr = float(metadata.loc[metric, "signal_to_noise"])
        header_cells.append(f"{display} (S/N {snr:.2f}x)")

    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(["---"] * len(header_cells)) + " |",
    ]

    best_by_metric = {metric: _best_mask(frame, metric, metadata) for metric in columns}
    for label, row in frame.iterrows():
        cells = [DISPLAY_ROW_LABELS[label]]
        for metric in columns:
            rendered = _format_value(float(row[metric]))
            if bool(best_by_metric[metric].loc[label]):
                rendered = f"**{rendered}**"
            cells.append(rendered)
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def main() -> None:
    metadata = _load_metric_metadata()
    focused_markdown = build_table(columns=_focused_column_order(metadata))
    all_eval_markdown = build_table(columns=_all_eval_column_order(metadata))
    OUTPUT_MD.write_text(focused_markdown)
    OUTPUT_ALL_EVAL_MD.write_text(all_eval_markdown)
    print(
        json.dumps(
            {
                "output_md": str(OUTPUT_MD),
                "output_all_eval_md": str(OUTPUT_ALL_EVAL_MD),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
