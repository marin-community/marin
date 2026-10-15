# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.release import export_dataset, paths, tables
from experiments.domain_phase_mix.release import verify_paper_numbers as v

TEX = r"""
\begin{table}[t]
\caption{Example.}
\label{tab:example}
\begin{tabular}{lrr}
\toprule
Method & Uncheatable & OBE Easy \\
\midrule
MARINER & $0.9814$ & 1.0678 $\pm$ 0.0052 \\
Olmix & 0.9900 & 1.0820 \\
\bottomrule
\end{tabular}
\end{table}
"""

# The other forms the paper's tables take: a wrapped table, a captioned longtable outside any table environment, a
# two-row header that holds numbers, escaped ampersands, multicolumn cells, section rows, comments and `\\*` row ends.
FORMS = r"""
% a comment with & and \\ is not a row
\begin{wraptable}{r}{0.4\linewidth}
\caption{Wrapped.}
\label{tab:wrapped}
\begin{tabular}{lr}
\toprule
Model & BPB \\
\midrule
MARINER & {\boldmath$0.9825 \pm 0.0010$} \\
\quad fully convex & $0.9831 \pm 0.0011$ \\ % 9.99 & 9.99
\bottomrule
\end{tabular}
\end{wraptable}

\begin{table}[H]
\caption{Two panels.}
\label{tab:panels}
\begin{tabular}{lrr@{\hspace{1.2em}}lr}
\toprule
& \multicolumn{2}{c}{Uncheatable, by cap} & & \\
\cmidrule(lr){2-3}
$\lambda$ & 1 & 4 & Bucket & Tokens (B) \\
\midrule
\multicolumn{5}{l}{\emph{Dolma 3}} \\
0 & 1.0401 & \textit{1.0024}\rlap{$^\dagger$} & Finance \& business (H) & 1,317.6 \\
0.5 & \multicolumn{2}{c}{0.9814} & FineMath 3+ & 34.0 \\
\bottomrule
\end{tabular}
\end{table}

\captionof{table}{A long table.}
\label{tab:long}
\begingroup
\begin{longtable}{lrr}
\toprule
Component & $n$ & Accuracy \\
\midrule
\endfirsthead
\toprule
Component & $n$ & Accuracy \\
\midrule
\endhead
\bottomrule
\endlastfoot
\multicolumn{3}{l}{\emph{Code}} \\*
\quad HumanEval & 164 & \textbf{17.1\%} \\*
\quad Qwen3 & 12 & $3.4\times10^{18}$ \\
\end{longtable}
\endgroup
"""
PAPER_LABELS = {
    "tab:ablations",
    "tab:scaling-main",
    "tab:fit",
    "tab:a-snr",
    "tab:a-snr-subset",
    "tab:a-phase-fiber",
    "tab:a-matched-swarms",
    "tab:a-heteroskedastic",
    "tab:a-factorial-predictions",
    "tab:a-unimax-cap",
    "tab:a-scaling-configurations",
    "tab:a-regmix-argmin",
    "tab:a-olmix-cap-kl",
    "tab:a-kl-ablation",
    "tab:a-hyperparameters",
    "tab:a-compute-equivalence",
    "tab:a-buckets",
    "tab:a-baseline-validation",
    "tab:a-accuracy-components",
    "tab:a-ablations-full",
}
SECTIONS = paths.DRIVE / "sections"
# The numbers of the paper that differ from the build, as .agents/projects/mariner_release/AUDIT.md explains them.
KNOWN_DIFFERENCES: set[tuple[str, int, int, int]] = set()  # the two tab:a-unimax-cap cells were corrected on 2026-10-15


def registry_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{"part": 0, "query": "", "decimals": 4, "note": "", **row} for row in rows])


def test_parse_tables_resolves_label_and_body_rows():
    tables = v.parse_tables(TEX)
    assert list(tables) == ["tab:example"]
    body = tables["tab:example"]
    assert body[0][0] == "MARINER" and body[1][2] == "1.0820"
    assert v.numeric(body[0][1]) == 0.9814 and v.numeric(body[0][2]) == 1.0678


def test_parse_tables_reads_every_table_form():
    tables = v.parse_tables(FORMS)
    assert set(tables) == {"tab:wrapped", "tab:panels", "tab:long"}
    assert [row[0] for row in tables["tab:wrapped"]] == ["MARINER", r"\quad fully convex"]
    # The header row's numbers are not body cells, and a section row without `&` is not a row.
    panels = tables["tab:panels"]
    assert [row[0] for row in panels] == ["0", "0.5"]
    assert panels[0][3] == r"Finance \& business (H)" and v.numeric(panels[0][4]) == 1317.6
    assert v.numeric(panels[0][2]) == 1.0024 and v.numeric(panels[1][1]) == 0.9814
    assert v.numeric(panels[1][2]) is None  # a name with a digit is text
    # The longtable's body starts after its repeated heads.
    assert tables["tab:long"] == [
        [r"\quad HumanEval", "164", r"\textbf{17.1\%}"],
        [r"\quad Qwen3", "12", r"$3.4\times10^{18}$"],
    ]


@pytest.mark.parametrize(
    "cell, printed",
    [
        (r"$\mathbf{0.982}\!\pm\!0.001$", [(0.982, 3), (0.001, 3)]),
        (r"{\boldmath$0.9825 \pm 0.0010$}", [(0.9825, 4), (0.001, 4)]),
        (r"$-0.0065 \pm 0.0017$", [(-0.0065, 4), (0.0017, 4)]),
        (r"$3.25\times^\dagger$ (69\%)", [(3.25, 2), (69.0, 0)]),
        (r"$0.497 + 0.638\,c^{-0.150}$", [(0.497, 3), (0.638, 3), (-0.15, 3)]),
        (r"$3.4\times10^{18}$", [(3.4, 1)]),
        (r"\textbf{\textit{0.9958}}", [(0.9958, 4)]),
        (r"6,986.4", [(6986.4, 1)]),
        (r"10\% warmup, 20\% linear decay", [(10.0, 0), (20.0, 0)]),
        (r"--", []),
        (r"Qwen3", []),
        (r"warmup-stable-decay", []),
    ],
)
def test_printed_numbers(cell, printed):
    # A cell is numeric when it starts with a number; the parts follow in print order with their printed decimals.
    assert v.printed_numbers(cell) == printed


def test_evaluate_queries():
    tables = {"ladder": pd.DataFrame({"method": ["mariner"] * 3 + ["olmix"], "bpb": [1.0, 1.1, 1.2, 2.0]})}
    assert v.evaluate("ladder[method=='mariner'].bpb.mean", tables) == pytest.approx(1.1)
    assert v.evaluate("ladder[method=='olmix'].bpb.value", tables) == 2.0
    assert v.evaluate(
        "ladder[method=='olmix'].bpb.value diff: ladder[method=='mariner'].bpb.mean", tables
    ) == pytest.approx(0.9)
    with pytest.raises(ValueError, match="exactly one row"):
        v.evaluate("ladder[method=='mariner'].bpb.value", tables)


def test_evaluate_extensions():
    runs = pd.DataFrame(
        {
            "launch": ["a", "a", "a", "b", "b", "b"],
            "trainer_seed": [0, 1, 2, 2, 0, 1],
            "bpb__x": [1.0, 2.0, 3.0, 2.5, 1.5, 2.0],
            "bpb__y/z": [3.0, 2.0, 1.0, 1.0, 0.5, 0.0],
        }
    )
    tasks = pd.DataFrame({"task_id": ["x", "y/z"], "snr": [9.0, 3.0]})
    tables = {"runs": runs, "tasks": tasks}
    # Arithmetic on scalars, per-row expressions, all rows, medians and constants.
    assert v.evaluate("100 * runs[launch=='a'].bpb__x.mean / 4 - 1", tables) == 49.0
    assert v.evaluate("runs[launch=='a'].{bpb__x + `bpb__y/z`}.unique", tables) == 4.0
    assert v.evaluate("runs[].bpb__x.median", tables) == 2.0
    assert v.evaluate("runs[launch=='a'].bpb__x.std / sqrt(3)", tables) == pytest.approx(1 / np.sqrt(3))
    assert v.evaluate("runs[launch=='a'].bpb__x.se", tables) == pytest.approx(1 / np.sqrt(3))
    assert v.evaluate("runs[bpb__x >= 2 * `bpb__y/z`].bpb__x.count", tables) == 4.0
    # A row-wise mean over the per-task columns another selection names.
    assert v.evaluate("runs[launch=='b'].mean_of(tasks[snr>5].task_id).mean", tables) == pytest.approx(2.0)
    assert v.evaluate("runs[launch=='a' & trainer_seed==0].mean_of(tasks[].task_id).value", tables) == 2.0
    # Differences of runs paired on a key, whatever their order.
    paired = "paired(runs[launch=='b'].bpb__x, runs[launch=='a'].bpb__x, trainer_seed)"
    assert v.evaluate(f"{paired}.mean", tables) == pytest.approx(0.0)
    assert v.evaluate(f"{paired}.se", tables) == pytest.approx(np.std([0.5, 0.0, -0.5], ddof=1) / np.sqrt(3))
    # Selections of several tables combine into one, which can also be paired.
    extra = "concat(tasks[task_id=='x'].snr, runs[launch=='a' & trainer_seed>0].bpb__x)"
    assert v.evaluate(f"{extra}.std", tables) == pytest.approx(np.std([9.0, 2.0, 3.0], ddof=1))
    combined = "concat(runs[launch=='a' & trainer_seed==0].bpb__x, runs[launch=='b' & trainer_seed>0].bpb__x)"
    assert v.evaluate(f"paired({combined}, runs[launch=='a'].bpb__x, trainer_seed).sum", tables) == pytest.approx(-0.5)
    # A null is never averaged away.
    runs.loc[0, "bpb__x"] = np.nan
    with pytest.raises(ValueError, match="null"):
        v.evaluate("runs[launch=='a'].bpb__x.mean", tables)


def test_untraced_cell_fails(tmp_path):
    registry = pd.DataFrame(
        [
            {
                "label": "tab:example",
                "row": 0,
                "col": 1,
                "query": "ladder[method=='mariner'].bpb.mean",
                "decimals": 4,
                "note": "",
            }
        ]
    )
    tables = {"ladder": pd.DataFrame({"method": ["mariner"], "bpb": [0.9814]})}
    differences, untraced = v.compare(TEX, registry, tables)
    assert differences == []
    assert {(u.label, u.row, u.col) for u in untraced} == {
        ("tab:example", 0, 2),
        ("tab:example", 1, 1),
        ("tab:example", 1, 2),
    }


def test_difference_is_reported_at_the_papers_rounding():
    registry = pd.DataFrame(
        [
            {
                "label": "tab:example",
                "row": 1,
                "col": 1,
                "query": "ladder[method=='olmix'].bpb.value",
                "decimals": 4,
                "note": "",
            }
        ]
    )
    tables = {"ladder": pd.DataFrame({"method": ["olmix"], "bpb": [0.99004]})}
    assert v.compare(TEX, registry, tables)[0] == []  # rounds to 0.9900
    tables["ladder"].loc[0, "bpb"] = 0.9906
    assert len(v.compare(TEX, registry, tables)[0]) == 1


def test_parts_and_registered_skips():
    runs = {"runs": pd.DataFrame({"m": ["a", "a", "b"], "bpb": [1.0673, 1.0683, 1.08204]})}
    registry = registry_frame(
        [
            {"label": "tab:example", "row": 0, "col": 2, "query": "runs[m=='a'].bpb.mean"},
            {"label": "tab:example", "row": 0, "col": 2, "part": 1, "query": "runs[m=='a'].bpb.std"},
            {"label": "tab:example", "row": 0, "col": 1, "note": "constant"},
            {"label": "tab:example", "row": 1, "col": 1, "note": "untraced: the source is absent"},
            {"label": "tab:example", "row": 1, "col": 2, "query": "runs[m=='b'].bpb.value"},
        ]
    )
    differences, untraced = v.compare(TEX, registry, runs)
    # The std of 1.0673 and 1.0683 is 0.0007, not the printed 0.0052.
    assert [(d.row, d.col, d.part, d.paper_value) for d in differences] == [(0, 2, 1, 0.0052)]
    assert {(u.row, u.col, u.kind) for u in untraced} == {(0, 1, "constant"), (1, 1, "untraced")}


@pytest.mark.parametrize(
    "row, problem",
    [
        ({"row": 0, "col": 0, "query": "runs[].bpb.value"}, "not a printed number"),  # the stub's text
        ({"row": 0, "col": 2, "part": 2, "query": "runs[].bpb.value"}, "not a printed number"),
        ({"row": 1, "col": 1, "decimals": 3, "query": "runs[].bpb.value"}, "prints 4 decimals"),
        ({"row": 1, "col": 1, "note": "constant", "query": "runs[].bpb.value"}, "skipped cell with a query"),
    ],
)
def test_registry_rows_must_name_a_printed_number(row, problem):
    # A row that no longer points at the number it was written for would check the wrong cell or none.
    runs = {"runs": pd.DataFrame({"bpb": [0.99]})}
    with pytest.raises(ValueError, match=problem):
        v.compare(TEX, registry_frame([{"label": "tab:example", **row}]), runs)


def test_cli_exit_codes(tmp_path):
    build = tmp_path / "build"
    (build / "data/ladder").mkdir(parents=True)
    ladder = pd.DataFrame({"method": ["mariner", "olmix"], "bpb": [0.98141, 0.99]})
    ladder.to_csv(build / "data/ladder/ladder.csv", index=False)
    sections = tmp_path / "sections"
    sections.mkdir()
    (sections / "results.tex").write_text(TEX)
    traced = [
        {"label": "tab:example", "row": 0, "col": 1, "query": "ladder[method=='mariner'].bpb.value"},
        {"label": "tab:example", "row": 1, "col": 1, "query": "ladder[method=='olmix'].bpb.value"},
    ]
    skipped = [
        {"label": "tab:example", "row": 0, "col": 2, "note": "deferred_two_phase"},
        {"label": "tab:example", "row": 0, "col": 2, "part": 1, "note": "deferred_two_phase"},
        {"label": "tab:example", "row": 1, "col": 2, "note": "constant"},
    ]

    def run(rows: list[dict], *flags: str) -> int:
        registry = tmp_path / "paper_cells.csv"
        registry_frame(rows).to_csv(registry, index=False)
        return v.main(["--build", str(build), "--sections", str(sections), "--registry", str(registry), *flags])

    assert run(traced, "--fail-on-untraced") == 1  # three numbers unregistered
    assert run(traced + skipped, "--fail-on-untraced") == 1  # registered skips fail unless allowed
    # Constants and deferred two-phase numbers are the only skips the flag allows; a registered untraced number fails.
    assert run(traced + skipped, "--fail-on-untraced", "--allow-registered-skips") == 0
    unchecked = [*skipped[:2], {"label": "tab:example", "row": 1, "col": 2, "note": "untraced: the source is absent"}]
    assert run(traced + unchecked, "--fail-on-untraced", "--allow-registered-skips") == 1
    assert run(traced) == 0  # untraced numbers fail only on request
    mismatch = [{**traced[0], "query": "ladder[method=='olmix'].bpb.value"}, traced[1]]
    assert run(mismatch + skipped, "--allow-registered-skips") == 1  # a difference always fails


def test_figure_index_lists_every_config_the_registry_reads():
    # figures_tables.csv names the configs behind each paper table; the registry's queries read only those.
    index = tables.read_csv(export_dataset.FIGURES_TABLES, keep_default_na=False).set_index("label").configs
    for label, queries in v.read_registry(v.REGISTRY).groupby("label")["query"]:
        read = {export_dataset.config_of(t) for query in queries for t in re.findall(r"([A-Za-z_]\w*)\[", query)}
        assert read <= set(index[label].split(";")), (label, sorted(read - set(index[label].split(";"))))


@pytest.mark.skipif(not SECTIONS.exists(), reason="paper folder absent")
def test_every_paper_table_is_parsed():
    tables = v.parse_tables(v.read_sections(SECTIONS))
    assert set(tables) == PAPER_LABELS
    assert all(tables[label] for label in PAPER_LABELS)


@pytest.mark.timeout(300)
@pytest.mark.skipif(not (SECTIONS.exists() and paths.BUILD.exists()), reason="paper folder or build absent")
def test_registry_checks_every_number_of_the_paper():
    # Every registry row names a printed number and evaluates; every number is checked or a registered skip.
    differences, untraced = v.audit(paths.BUILD, SECTIONS)
    assert {(d.label, d.row, d.col, d.part) for d in differences} == KNOWN_DIFFERENCES
    assert {u.kind for u in untraced} <= set(v.SKIP_KINDS)
