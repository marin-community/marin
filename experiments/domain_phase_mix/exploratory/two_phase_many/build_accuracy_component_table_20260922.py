# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Per-component and per-group OlmoBaseEval Easy BPB and accuracy at 1e21 FLOPs for the paper.

Reads the merged coverage reports and native Table-9 summaries of the four measured mixtures
(``.agents/projects/mariner_table9_accuracy_20260914/olmix_t9_1e21_east1/accuracy_vs_bpb``) and the executable
MT-MBPP pass@1 (``.agents/projects/mt_mbpp_exec_20260926/results/components.csv``), and writes:
``rows.tex``, the body of the appendix table (one row per component with its accuracy example count, grouped by
family, lowest BPB and highest accuracy in bold); ``groups.tex``, the mixture rows of the main-text group table; and
``components.csv`` with the per-component numbers.

usage: uv run --offline --no-sync python build_accuracy_component_table_20260922.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from marin.evaluation.olmo_base_eval.accuracy import MMLU_CATEGORY_WEIGHTS

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
ANALYSIS = REPO_ROOT / ".agents/projects/mariner_table9_accuracy_20260914/olmix_t9_1e21_east1/accuracy_vs_bpb"
MT_MBPP_RESULTS = REPO_ROOT / ".agents/projects/mt_mbpp_exec_20260926/results/components.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "accuracy_component_table_20260922"
NATIVE_PREFIX = "olmo_base_easy/table9/"
MIXTURES = (("proportional", "Proportional"), ("unimax8", "UniMax-8"), ("olmix", "Olmix"), ("mariner", "MARINER"))
MT_MBPP_GROUP = "Code: machine-translated MBPP"
LANGUAGES = {
    "bash": "Bash",
    "c": "C",
    "cpp": "C++",
    "csharp": "C\\#",
    "go": "Go",
    "haskell": "Haskell",
    "java": "Java",
    "javascript": "JavaScript",
    "matlab": "MATLAB",
    "php": "PHP",
    "python": "Python",
    "r": "R",
    "ruby": "Ruby",
    "rust": "Rust",
    "scala": "Scala",
    "swift": "Swift",
    "typescript": "TypeScript",
}
NAMES = {
    "arc_challenge": "ARC Challenge",
    "arc_easy": "ARC Easy",
    "csqa": "CommonsenseQA",
    "hellaswag": "HellaSwag",
    "medmcqa": "MedMCQA",
    "piqa": "PIQA",
    "sciq": "SciQ",
    "socialiqa": "SocialIQA",
    "winogrande": "WinoGrande",
    "lambada": "LAMBADA",
    "coqa": "CoQA",
    "drop": "DROP",
    "jeopardy": "Jeopardy",
    "naturalqs": "Natural Questions",
    "squad": "SQuAD",
    "mmlu_stem": "MMLU STEM",
    "mmlu_humanities": "MMLU humanities",
    "mmlu_social_sciences": "MMLU social sciences",
    "mmlu_other": "MMLU other",
    "codex_humaneval": "HumanEval",
    "mbpp": "MBPP",
}
# (group label, ordered component ids)
GROUPS = (
    (
        "Question answering (lm-eval overlap)",
        [
            "arc_challenge",
            "arc_easy",
            "csqa",
            "hellaswag",
            "lambada",
            "medmcqa",
            "piqa",
            "sciq",
            "socialiqa",
            "winogrande",
        ],
    ),
    ("QA answer selection", ["coqa", "drop", "jeopardy", "naturalqs", "squad"]),
    ("MMLU", ["mmlu_humanities", "mmlu_other", "mmlu_social_sciences", "mmlu_stem"]),
    (
        "Basic Skills",
        [
            "basic_skills_arithmetic",
            "basic_skills_coding",
            "basic_skills_common_knowledge",
            "basic_skills_logical_reasoning",
            "basic_skills_pattern",
            "basic_skills_string_operations",
        ],
    ),
    (
        "Minerva MATH",
        [
            "minerva_math_algebra",
            "minerva_math_counting_and_probability",
            "minerva_math_geometry",
            "minerva_math_intermediate_algebra",
            "minerva_math_number_theory",
            "minerva_math_prealgebra",
            "minerva_math_precalculus",
        ],
    ),
    ("Code", ["codex_humaneval", "mbpp"]),
    (MT_MBPP_GROUP, [f"mt_mbpp_{k}" for k in LANGUAGES]),
)
# Columns of the main-text group table: (source groups, accuracy shown); "All" spans all 51 components and Code
# includes the 17 machine-translated MBPP components.
TABLE_GROUPS = (
    ("All", None, True),
    ("Basic Skills", ["Basic Skills"], True),
    ("Minerva", ["Minerva MATH"], True),
    ("Code", ["Code", MT_MBPP_GROUP], True),
    ("MMLU", ["MMLU"], True),
    ("QA", ["Question answering (lm-eval overlap)", "QA answer selection"], True),
)


def label(component: str) -> str:
    if component in NAMES:
        return NAMES[component]
    if component.startswith("basic_skills_"):
        return component.removeprefix("basic_skills_").replace("_", " ").capitalize()
    if component.startswith("minerva_math_"):
        return component.removeprefix("minerva_math_").replace("_", " ").capitalize()
    if component.startswith("mt_mbpp_"):
        return LANGUAGES[component.removeprefix("mt_mbpp_")]
    raise KeyError(component)


def example_count(report: dict, component: str, mt_counts: dict[str, int]) -> int:
    """Accuracy examples behind a component; an MMLU category sums its subjects; MT-MBPP counts problems with valid tests."""
    if component in mt_counts:
        return mt_counts[component]
    subjects = MMLU_CATEGORY_WEIGHTS.get(component, {component: 1.0})
    return sum(int(report["tasks"][subject]["count"]) for subject in subjects)


def bold_best(values: dict[str, float], fmt: str, best) -> dict[str, str]:
    """Format each value; bold every value tied with the best at the displayed precision."""
    shown = {name: fmt.format(value) for name, value in values.items()}
    winner = best(shown.values(), key=lambda text: float(text.rstrip("\\%")))
    return {name: f"\\textbf{{{text}}}" if text == winner else text for name, text in shown.items()}


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    native = json.loads((ANALYSIS / "native_summaries.json").read_text())
    coverage = json.loads((ANALYSIS / "coverage_merged.json").read_text())
    accuracy, reports = {}, {}
    for name, _ in MIXTURES:
        uri = native[name]["config"]["checkpoint_path"].rstrip("/")
        # A relocated evaluation records the original checkpoint it copied.
        (report,) = [
            r
            for r in coverage
            if r["checkpoint"].get("copied_from", r["checkpoint"]["checkpoint_uri"]).rstrip("/") == uri
        ]
        reports[name] = report
        accuracy[name] = {k: 100 * float(v) for k, v in report["coverage"]["components"].items()}
    keys = {display: name for name, display in MIXTURES}
    mt_counts = {}
    with MT_MBPP_RESULTS.open() as handle:
        for row in csv.DictReader(handle):
            component = f"mt_mbpp_{row['language']}"
            accuracy[keys[row["mixture"]]][component] = 100 * float(row["pass@1"])
            assert mt_counts.setdefault(component, int(row["scored"])) == int(row["scored"]), component
    listed = [c for _, comps in GROUPS for c in comps]
    assert len(listed) == 51 and len(set(listed)) == 51
    bpb = {}
    for name, _ in MIXTURES:
        keys = {
            k.removeprefix(NATIVE_PREFIX).removesuffix("/bpb")
            for k in native[name]["summary"]
            if k.startswith(NATIVE_PREFIX) and k.endswith("/bpb")
        }
        assert keys == set(listed), (name, keys ^ set(listed))
        bpb[name] = {c: float(native[name]["summary"][f"{NATIVE_PREFIX}{c}/bpb"]) for c in listed}
    columns = 2 + 2 * len(MIXTURES)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, lines = [], []
    for group, comps in GROUPS:
        lines.append(f"\\multicolumn{{{columns}}}{{l}}{{\\emph{{{group}}}}} \\\\")
        for c in comps:
            acc = {n: accuracy[n].get(c) for n, _ in MIXTURES}
            scored = acc[MIXTURES[0][0]] is not None
            assert all((v is not None) == scored for v in acc.values()), c
            counts = {example_count(reports[n], c, mt_counts) for n, _ in MIXTURES} if scored else set()
            assert len(counts) <= 1, (c, counts)
            cells = [str(counts.pop()) if scored else "--"]
            cells += bold_best({n: bpb[n][c] for n, _ in MIXTURES}, "{:.3f}", min).values()
            if scored:
                cells += bold_best(acc, "{:.1f}\\%", max).values()
            else:
                cells += ["--"] * len(MIXTURES)
            lines.append(f"\\quad {label(c)} & " + " & ".join(cells) + " \\\\")
            rows.append(
                {
                    "group": group,
                    "component": c,
                    **{f"bpb_{n}": bpb[n][c] for n, _ in MIXTURES},
                    **{f"accuracy_{n}": acc[n] for n, _ in MIXTURES},
                }
            )
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    (args.output_dir / "rows.tex").write_text("\n".join(lines) + "\n")
    with (args.output_dir / "components.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    members = {group: comps for group, comps in GROUPS}
    cells = {n: [] for n, _ in MIXTURES}
    for _, sources, with_accuracy in TABLE_GROUPS:
        comps = (
            [c for c in listed if accuracy[MIXTURES[0][0]].get(c) is not None]
            if sources is None
            else [c for g in sources for c in members[g]]
        )
        if with_accuracy:
            for n, text in bold_best(
                {n: mean([accuracy[n][c] for c in comps]) for n, _ in MIXTURES}, "{:.1f}\\%", max
            ).items():
                cells[n].append(text)
        for n, text in bold_best({n: mean([bpb[n][c] for c in comps]) for n, _ in MIXTURES}, "{:.3f}", min).items():
            cells[n].append(text)
    group_lines = [f"{display} & " + " & ".join(cells[n]) + " \\\\" for n, display in MIXTURES]
    (args.output_dir / "groups.tex").write_text("\n".join(group_lines) + "\n")
    print(f"wrote {args.output_dir / 'rows.tex'} ({len(rows)} components) and {args.output_dir / 'groups.tex'}")


if __name__ == "__main__":
    main()
