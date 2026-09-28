# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas"]
# ///

"""Signal-to-noise tables for the Table-9 macro and its components, in the Olmix Table 9 layout.

Signal is the standard deviation across the 280 single-phase designs of the Qwen3 360M/1.6B
panel; noise is the standard deviation across the ten proportional repeat runs. Both are
computed for every component, for every Table-9 task (subtasks averaged, as Olmix treats
Minerva MATH, MT MBPP, ARC and Basic Skills) and for the macro. Two tables are written: the
task-level table for the main text and the full component table for the appendix, both in
the Math / Code / QA order of the Olmix paper's Table 9 with the macro on top.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pandas as pd  # noqa: E402

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)

PANEL = "delphi_3e18_39bucket"
TARGET = "table9"
NOISE_MATRIX = (
    SCRIPT_DIR / "reference_outputs" / "delphi_3e18_proportional_noise_floor_20260703" / "noise_component_matrix.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "table9_reliability_20260905"
PANEL_PREFIX = "olmo_base_eval/easy_bpb/"
NOISE_PREFIX = "olmo_base_easy/table9/"
MACRO_LABEL = "Table-9 macro (mean of 51 components)"

# Olmix Table 9 ("Details of the BPB evaluation suite"): group, task, subtasks in paper order.
TABLE9_LAYOUT: tuple[tuple[str, tuple[tuple[str, tuple[tuple[str, str], ...]], ...]], ...] = (
    (
        "Math",
        (
            (
                "Minerva MATH",
                (
                    ("minerva_math_algebra", "Algebra"),
                    ("minerva_math_counting_and_probability", "Counting and Probability"),
                    ("minerva_math_geometry", "Geometry"),
                    ("minerva_math_intermediate_algebra", "Intermediate Algebra"),
                    ("minerva_math_number_theory", "Number Theory"),
                    ("minerva_math_prealgebra", "Prealgebra"),
                    ("minerva_math_precalculus", "Precalculus"),
                ),
            ),
        ),
    ),
    (
        "Code",
        (
            ("HumanEval", (("codex_humaneval", "HumanEval"),)),
            ("MBPP", (("mbpp", "MBPP"),)),
            (
                "MT MBPP",
                (
                    ("mt_mbpp_bash", "Bash"),
                    ("mt_mbpp_c", "C"),
                    ("mt_mbpp_cpp", "C++"),
                    ("mt_mbpp_csharp", "C#"),
                    ("mt_mbpp_go", "Go"),
                    ("mt_mbpp_haskell", "Haskell"),
                    ("mt_mbpp_java", "Java"),
                    ("mt_mbpp_javascript", "JavaScript"),
                    ("mt_mbpp_matlab", "MatLab"),
                    ("mt_mbpp_php", "PHP"),
                    ("mt_mbpp_python", "Python"),
                    ("mt_mbpp_r", "R"),
                    ("mt_mbpp_ruby", "Ruby"),
                    ("mt_mbpp_rust", "Rust"),
                    ("mt_mbpp_scala", "Scala"),
                    ("mt_mbpp_swift", "Swift"),
                    ("mt_mbpp_typescript", "TypeScript"),
                ),
            ),
        ),
    ),
    (
        "QA",
        (
            ("ARC", (("arc_easy", "ARC-Easy"), ("arc_challenge", "ARC-Challenge"))),
            ("MMLU STEM", (("mmlu_stem", "MMLU STEM"),)),
            ("MMLU Humanities", (("mmlu_humanities", "MMLU Humanities"),)),
            ("MMLU Social Sci.", (("mmlu_social_sciences", "MMLU Social Sci."),)),
            ("MMLU Other", (("mmlu_other", "MMLU Other"),)),
            ("CSQA", (("csqa", "CSQA"),)),
            ("HellaSwag", (("hellaswag", "HellaSwag"),)),
            ("WinoGrande", (("winogrande", "WinoGrande"),)),
            ("SocialIQA", (("socialiqa", "SocialIQA"),)),
            ("PiQA", (("piqa", "PiQA"),)),
            ("CoQA", (("coqa", "CoQA"),)),
            ("DROP", (("drop", "DROP"),)),
            ("Jeopardy", (("jeopardy", "Jeopardy"),)),
            ("NaturalQs", (("naturalqs", "NaturalQs"),)),
            ("SQuAD", (("squad", "SQuAD"),)),
            ("SciQ", (("sciq", "SciQ"),)),
            (
                "Basic Skills",
                (
                    ("basic_skills_arithmetic", "Basic Arithmetic"),
                    ("basic_skills_string_operations", "String Manipulation"),
                    ("basic_skills_coding", "Simple Coding"),
                    ("basic_skills_logical_reasoning", "Elementary Logical Reasoning"),
                    ("basic_skills_common_knowledge", "Basic Common Sense"),
                    ("basic_skills_pattern", "Simple Pattern Recognition"),
                ),
            ),
            ("Lambada", (("lambada", "Lambada"),)),
            ("MedMCQA", (("medmcqa", "MedMCQA"),)),
        ),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_matrices() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (panel values 280 x 51, noise values 10 x 51), both keyed by bare task name."""
    panel = benchmark.load_panel(PANEL)
    group = panel.group(TARGET)
    names = [component.removeprefix(PANEL_PREFIX).removesuffix("/bpb") for component in group.components]
    values = pd.DataFrame(group.outcomes, columns=names, index=list(panel.runs))
    noise = pd.read_csv(NOISE_MATRIX, index_col=0)
    noise.columns = [column.removeprefix(NOISE_PREFIX).removesuffix("/bpb") for column in noise.columns]
    missing = set(names) - set(noise.columns)
    if missing:
        raise ValueError(f"Noise matrix lacks components: {sorted(missing)}")
    layout_tasks = [task for _, tasks in TABLE9_LAYOUT for _, subtasks in tasks for task, _ in subtasks]
    if sorted(layout_tasks) != sorted(names):
        raise ValueError("Table 9 layout does not match the panel's 51 components")
    return values, noise[names]


def statistics(panel_values: pd.Series, noise_values: pd.Series) -> dict[str, float]:
    panel_sd = float(panel_values.std(ddof=1))
    noise_sd = float(noise_values.std(ddof=1))
    return {
        "panel_mean": float(panel_values.mean()),
        "panel_sd": panel_sd,
        "panel_range": float(panel_values.max() - panel_values.min()),
        "repeat_sd": noise_sd,
        "snr": panel_sd / noise_sd,
    }


def build_tables(values: pd.DataFrame, noise: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    macro = {"group": "", "task": MACRO_LABEL, "subtasks": 51, **statistics(values.mean(axis=1), noise.mean(axis=1))}
    short_rows = [macro]
    full_rows = [{**macro, "component": ""}]
    for group_name, tasks in TABLE9_LAYOUT:
        for task_name, subtasks in tasks:
            keys = [key for key, _ in subtasks]
            short_rows.append(
                {
                    "group": group_name,
                    "task": task_name,
                    "subtasks": len(keys),
                    **statistics(values[keys].mean(axis=1), noise[keys].mean(axis=1)),
                }
            )
            for key, label in subtasks:
                full_rows.append(
                    {
                        "group": group_name,
                        "task": task_name,
                        "component": key,
                        "subtask": label,
                        "subtasks": 1,
                        **statistics(values[key], noise[key]),
                    }
                )
    return pd.DataFrame(short_rows), pd.DataFrame(full_rows)


def markdown_short(frame: pd.DataFrame) -> str:
    lines = ["| Task | Subtasks | Panel mean | Panel SD | Repeat SD | SNR |", "|---|---|---|---|---|---|"]
    current_group = None
    for _, row in frame.iterrows():
        if row["task"] == MACRO_LABEL:
            lines.append(
                f"| **{row['task']}** | 51 | {row['panel_mean']:.3f} | {row['panel_sd']:.4f} | "
                f"{row['repeat_sd']:.4f} | **{row['snr']:.1f}** |"
            )
            continue
        if row["group"] != current_group:
            current_group = row["group"]
            lines.append(f"| *{current_group}* | | | | | |")
        subtasks = int(row["subtasks"]) if row["subtasks"] > 1 else ""
        lines.append(
            f"| {row['task']} | {subtasks} | {row['panel_mean']:.3f} | {row['panel_sd']:.4f} | "
            f"{row['repeat_sd']:.4f} | {row['snr']:.1f} |"
        )
    return "\n".join(lines)


def markdown_full(frame: pd.DataFrame) -> str:
    lines = ["| Task | Panel mean | Panel SD | Repeat SD | SNR |", "|---|---|---|---|---|"]
    current_group = None
    current_task = None
    for _, row in frame.iterrows():
        if row["task"] == MACRO_LABEL:
            lines.append(
                f"| **{row['task']}** | {row['panel_mean']:.3f} | {row['panel_sd']:.4f} | "
                f"{row['repeat_sd']:.4f} | **{row['snr']:.1f}** |"
            )
            continue
        if row["group"] != current_group:
            current_group = row["group"]
            lines.append(f"| *{current_group}* | | | | |")
        multi = frame.loc[frame["task"].eq(row["task"])].shape[0] > 1
        if multi and row["task"] != current_task:
            lines.append(f"| {row['task']} | | | | |")
        current_task = row["task"]
        name = f"&nbsp;&nbsp;{row['subtask']}" if multi else row["subtask"]
        lines.append(
            f"| {name} | {row['panel_mean']:.3f} | {row['panel_sd']:.4f} | {row['repeat_sd']:.4f} | {row['snr']:.1f} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values, noise = load_matrices()
    short, full = build_tables(values, noise)
    short.to_csv(args.output_dir / "snr_tasks_delphi.csv", index=False)
    full.to_csv(args.output_dir / "snr_components_delphi.csv", index=False)
    (args.output_dir / "snr_tasks_delphi.md").write_text(markdown_short(short) + "\n")
    (args.output_dir / "snr_components_delphi.md").write_text(markdown_full(full) + "\n")
    components = full.loc[full["task"].ne(MACRO_LABEL)]
    print(
        f"macro SNR {short.iloc[0]['snr']:.1f}; task SNR range {short.iloc[1:]['snr'].min():.1f}-"
        f"{short.iloc[1:]['snr'].max():.1f}; components below 3: "
        f"{sorted(components.loc[components['snr'] < 3, 'subtask'].tolist())}"
    )
    print(markdown_short(short))


if __name__ == "__main__":
    main()
