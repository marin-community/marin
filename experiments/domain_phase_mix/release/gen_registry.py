# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Regenerate paper_cells.csv: `uv run python -m experiments.domain_phase_mix.release.gen_registry [--sections DIR]`."""

import argparse
import dataclasses
import re
import sys
from pathlib import Path

import pandas as pd

from experiments.domain_phase_mix.release import paths
from experiments.domain_phase_mix.release import verify_paper_numbers as v

V = "validation_3e18"
U, O = "bpb__uncheatable", "bpb__obe_easy"
MARINER_U = "launch.isin(['frozen_procedure','fairness_repeats']) & variant=='lwspu_u_snc_cap06'"
MARINER_O = "launch.isin(['frozen_procedure','fairness_repeats']) & variant=='lwspu_t9_snc_cap08'"
PRED = "fits_proposal_predictions"
REL = "fits_reliability"
COMPONENTS = "source_table=='snr_fit_components_delphi' & task_id.notna()"
METHODS = {"Proportional": "proportional", "UniMax-8": "unimax8", "Olmix": "olmix", "MARINER": "mariner"}
RUNGS = ["3e18", "2e19", "3e20", "1e21"]
# Paper row of the baseline tables -> (Uncheatable selection, OBE Easy selection) of validation_3e18.
BASELINE_ROWS = {
    "MARINER": (MARINER_U, MARINER_O),
    "fully convex objective": (
        "launch=='convex_additive' & variant=='cmp_u_cvx_cap06'",
        "launch=='convex_additive' & variant=='cmp_t9_cvx_cap08'",
    ),
    "additive response": (
        "launch=='convex_additive' & variant=='cmp_u_add_cap08'",
        "launch=='convex_additive' & variant=='cmp_t9_add_cap08'",
    ),
    "per-bucket threshold": (
        "launch=='per_bucket_threshold' & variant=='cmp_u_pbt_cap06'",
        "launch=='per_bucket_threshold' & variant=='cmp_t9_pbt_cap08'",
    ),
    "Olmix": (
        "launch=='matched_olmix' & variant=='olmixq_u_kl0p05_cap04'",
        "launch=='matched_olmix' & variant=='olmixq_t9_kl0p005_cap04'",
    ),
    "Quadratic": (
        "launch=='comparator_proposals' & variant=='cmp_u_quad_cap08'",
        "launch=='comparator_proposals' & variant=='cmp_t9_quad_cap12'",
    ),
    "Natural cubic spline": (
        "launch.isin(['comparator_proposals','table5_repeats']) & variant=='cmp_u_spline_cap08'",
        "launch.isin(['comparator_proposals','table5_repeats']) & variant=='cmp_t9_spline_cap12'",
    ),
    "RegMix (tuned, sampled proposal)": (
        "launch.isin(['comparator_proposals','table5_repeats']) & variant=='cmp_u_lgbm_cap08'",
        "launch.isin(['comparator_proposals','table5_repeats']) & variant=='cmp_t9_lgbm_cap12'",
    ),
    "RegMix (released)": (
        "launch.isin(['regmix_reference','table5_repeats']) & variant=='rgref_u_endpoint_cap64'",
        "launch.isin(['regmix_reference','table5_repeats']) & variant=='rgref_t9_endpoint_cap64'",
    ),
    "Hellinger kernel ridge": (
        "launch.isin(['comparator_proposals','table5_repeats']) & variant=='cmp_u_krr_cap12'",
        "launch.isin(['comparator_proposals','table5_repeats']) & variant=='cmp_t9_krr_cap16'",
    ),
}
# The swarm's best mixture: the swarm's own run (trainer seed 0) and its two retrains at the same data seed.
BEST_SWARM = (
    "concat(swarm_qwen3_360m_1p6b[run_label=='run_00125'].{c}, "
    "validation_3e18[launch=='table5_repeats' & variant=='swarm_run00125_cap08'].{c})"
)
LAMBDAS = {
    "0": "0",
    "0.005": "0p005",
    "0.01": "0p01",
    "0.025": "0p025",
    "0.05": "0p05",
    "0.075": "0p075",
    "0.1": "0p1",
    "0.2": "0p2",
    "0.5": "0p5",
}
OLMIX_CAPS = ["01", "04", "08", "12", "none"]
SNR_COLUMNS = ["panel_mean", "panel_sd", "proportional_mean", "repeat_sd", "snr", "mariner_spearman", "olmix_spearman"]
FIT_GROUPS = {
    "OlmoBaseEval Easy, all": COMPONENTS,
    "math": COMPONENTS + " & group=='Math'",
    "code": COMPONENTS + " & group=='Code'",
    "question answering": COMPONENTS + " & group=='QA'",
    "Uncheatable": "source_table=='snr_fit_uncheatable_delphi' & task_id.notna()",
}
SUBSET_RUNS = {
    "Full mean (MARINER)": MARINER_O,
    "SNR geq 5 subset": "launch=='table9_snr_subset' & variant=='snr5_t9_cap09'",
    "SNR geq 8 subset": "launch=='table9_snr_subset' & variant=='snr8_t9_cap07'",
}
# Paper column -> (column of the selection, objective of the predictions).
SUBSET_COLUMNS = {
    1: (O, "obe_easy"),
    3: (f"mean_of({REL}[{COMPONENTS} & snr >= 5].task_id)", "obe_easy_snr5"),
    5: (f"mean_of({REL}[{COMPONENTS} & snr >= 8].task_id)", "obe_easy_snr8"),
}
SUBSET_VARIANTS = {"SNR geq 5 subset": "snr5_t9_cap09", "SNR geq 8 subset": "snr8_t9_cap07"}
CENTRE_REPEATS = "heldout[proposal_method=='center_control' & proposal_objective=='obe_easy']"
SWARMS = {1: "qwen3_360m_1p6b", 2: "llama_160m_1p2b", 3: "llama_200m_6b"}
QWEN = "ladder[compute_flops==3e18]"  # the Qwen3 swarm trains the ladder's 3e18 model on its token budget
SIZES = {
    "Total parameters": ("total_parameters", "/ 1e6", "millions"),
    "Non-embedding parameters": ("non_embedding_parameters", "/ 1e6", "millions"),
    "Training tokens": ("train_tokens", "/ 1e9", "billions"),
}
ACCURACY_METHODS = ["proportional", "unimax8", "olmix", "mariner"]
UNIMAX_GROUPS = {
    3: "tasks[task_id.str.startswith('mt_mbpp_')].task_id",
    4: "tasks[task_id.str.startswith('minerva_math_')].task_id",
    5: "tasks[task_id.str.startswith('basic_skills_')].task_id",
    6: "tasks[task_id.isin(['codex_humaneval','mbpp'])].task_id",
    7: "tasks[task_id.str.startswith('mmlu_')].task_id",
    8: (
        "tasks[family=='Question answering' & not task_id.str.startswith('mmlu_') "
        "& not task_id.str.startswith('basic_skills_')].task_id"
    ),
}
UNIMAX_LADDER_RUNS = {
    "Proportional": "ladder[method=='proportional' & compute_flops==3e18 & data_seed==660700]",
    "8": "ladder[method=='unimax8' & compute_flops==3e18 & trainer_seed==0]",
}
MV = "weibull_softplus_unscaled@kappa_floor_link_flat15_nocap"
# Rows of tab:a-ablations-full and the fits_surrogate_benchmark model of each.
BENCHMARK_ROWS = [
    ("MARINER", MV),
    ("exponential benefit", MV + "_kappa1"),
    ("per-bucket shapes", MV + "_per_bucket_shape"),
    ("per-bucket threshold", MV + "_per_bucket_threshold"),
    ("per-bucket rate", MV + "_per_bucket_rate"),
    ("per-bucket power", MV + "_per_bucket_power"),
    ("finer harm-threshold grid", MV + "_fine_threshold"),
    ("signed amplitudes", MV + "_signed_head"),
    ("additive response", "weibull_softplus_unscaled"),
    ("fully convex objective", MV + "_raw_epoch_hinge"),
    ("fully convex objective, squared harm", MV + "_raw_epoch_harm"),
    ("fully convex objective, per-bucket threshold", MV + "_raw_epoch_hinge_per_bucket_threshold"),
    ("fully convex objective, per-bucket shapes", MV + "_raw_epoch_hinge_per_bucket_shape"),
    ("without the harm term", MV + "_no_harm"),
    ("harm reads scrambled exposures", MV + "_row_scrambled_harm"),
    ("weights in place of epochs", MV + "_weight_coordinate"),
    ("permuted bucket sizes", MV + "_permuted_inventory"),
    ("one pool size for every bucket", MV + "_common_inventory"),
    ("outcome permutation", MV + "_outcome_permutation"),
    ("Olmix", "olmix_loglinear_taskwise"),
    ("on materialized epochs", "olmix_loglinear_taskwise_exposure"),
    ("Linear in the weights", "linear_weight"),
    ("on materialized epochs", "linear_exposure"),
    ("Quadratic", "quadratic_log_epoch@kappa_floor_link_flat15_nocap"),
    ("Natural cubic spline", "spline_log_epoch@kappa_floor_link_flat15_nocap"),
    ("Hellinger kernel ridge", "hellinger_krr"),
    ("RegMix (tuned)", "lightgbm_regmix"),
    ("RegMix (released)", "lightgbm_regmix_released"),
]
BENCHMARK_COLUMNS = [
    ("qwen3_360m_1p6b", "uncheatable"),
    ("qwen3_360m_1p6b", "obe_easy"),
    ("llama_200m_6b", "uncheatable"),
    ("llama_200m_6b", "obe_easy"),
    ("llama_160m_1p2b", "uncheatable"),
    ("llama_160m_1p2b", "obe_easy"),
]


@dataclasses.dataclass
class _Registry:
    """The registry rows of the parsed paper tables, each with the decimals the paper prints."""

    parsed: dict[str, list[list[str]]]
    rows: list[dict] = dataclasses.field(default_factory=list)

    def parts(self, label: str, row: int, col: int) -> range:
        return range(len(v.printed_numbers(self.parsed[label][row][col])))

    def stub(self, label: str, row: int) -> str:
        """A row's label without LaTeX indentation, bold or math mode."""
        return " ".join(re.sub(r"\\quad\s*|\\textbf\{|\}|\$", "", self.parsed[label][row][0]).split())

    def add(self, label: str, row: int, col: int, part: int, query: str, note: str) -> None:
        printed = v.printed_numbers(self.parsed[label][row][col])
        if part >= len(printed):
            raise ValueError(f"{label} r{row} c{col} prints no number {part}: {self.parsed[label][row][col]}")
        entry = {"label": label, "row": row, "col": col, "part": part, "query": query}
        self.rows.append({**entry, "decimals": printed[part][1], "note": note})

    def mean_std(self, label: str, row: int, col: int, selection: str, note: str) -> None:
        self.add(label, row, col, 0, f"{selection}.mean", f"{note}: mean")
        if len(self.parts(label, row, col)) > 1:
            self.add(label, row, col, 1, f"{selection}.std", f"{note}: std")

    def mean_se(self, label: str, row: int, col: int, a: str, b: str, note: str) -> None:
        """A paired difference on trainer seeds and its standard error."""
        self.add(label, row, col, 0, f"paired({a}, {b}, trainer_seed).mean", f"{note}: mean paired difference")
        self.add(label, row, col, 1, f"paired({a}, {b}, trainer_seed).se", f"{note}: SE of the paired difference")

    def skip_all(self, label: str, note: str) -> None:
        for i, cells in enumerate(self.parsed[label]):
            for j in range(1, len(cells)):
                for k in self.parts(label, i, j):
                    self.add(label, i, j, k, "", note)


def _task_of(build: dict[str, pd.DataFrame], name: str) -> str:
    hits = build["tasks"].task_id[build["tasks"].paper_name == name]
    if len(hits) != 1:
        raise ValueError(f"no single task has the paper name {name!r}")
    return hits.iloc[0]


def _selection(name: str, column: str) -> str:
    """The validation runs of a baseline-table row, as `<table>[<filter>].<column>`."""
    if name == "Best swarm mixture, retrained":
        return BEST_SWARM.format(c=column)
    uncheatable, obe_easy = BASELINE_ROWS[name]
    return f"{V}[{uncheatable if column == U else obe_easy}].{column}"


def _scaling_main(reg: _Registry) -> None:
    label = "tab:scaling-main"
    for i in range(len(reg.parsed[label])):
        method = METHODS[reg.stub(label, i)]
        for j in range(1, 9):
            objective, column = ("uncheatable", U) if j <= 4 else ("obe_easy", O)
            rung = RUNGS[(j - 1) % 4]
            target = "none" if method in ("proportional", "unimax8") else objective
            sel = f"ladder[method=='{method}' & proposal_objective=='{target}' & compute_flops=={rung}].{column}"
            what = f"{reg.stub(label, i)} {objective} at {rung}"
            if rung == "3e18":
                reg.mean_std(label, i, j, sel, what)
            else:
                reg.add(label, i, j, 0, f"{sel}.value", what)
        query = f"ladder_accuracy[method=='{method}'].accuracy.mean"
        reg.add(label, i, 9, 0, query, "mean accuracy of the 51 components at 1e21")


def _baselines(reg: _Registry) -> None:
    label = "tab:ablations"
    for i in range(len(reg.parsed[label])):
        name = reg.stub(label, i)
        reg.mean_std(label, i, 1, _selection(name, U), f"{name}, Uncheatable, trainer seeds 0-2")
    label = "tab:a-baseline-validation"
    for i in range(len(reg.parsed[label])):
        name = reg.stub(label, i)
        for col, column in ((1, U), (3, O)):
            objective = "Uncheatable" if column == U else "OBE Easy"
            reg.mean_std(label, i, col, _selection(name, column), f"{name}, {objective}, trainer seeds 0-2")
            if name != "MARINER":
                mariner = _selection("MARINER", column)
                reg.mean_se(label, i, col + 1, _selection(name, column), mariner, f"{name} minus MARINER, {objective}")


def _regmix_argmin(reg: _Registry) -> None:
    label = "tab:a-regmix-argmin"
    proposals = {  # paper row -> (column, selection or BASELINE_ROWS name, variant)
        0: (U, "MARINER", "lwspu_u_snc_cap06"),
        1: (U, "RegMix (tuned, sampled proposal)", "cmp_u_lgbm_cap08"),
        2: (U, "launch=='regmix_argmin' & variant=='rgx_u_argmin_center_cap08'", "rgx_u_argmin_center_cap08"),
        3: (O, "MARINER", "lwspu_t9_snc_cap08"),
        4: (O, "RegMix (tuned, sampled proposal)", "cmp_t9_lgbm_cap12"),
        5: (O, "launch=='regmix_argmin' & variant=='rgx_t9_argmin_center_cap12'", "rgx_t9_argmin_center_cap12"),
    }
    for i, (column, name, variant) in proposals.items():
        objective = "uncheatable" if column == U else "obe_easy"
        sel = _selection(name, column) if name in BASELINE_ROWS else f"{V}[{name}].{column}"
        row = reg.stub(label, i)
        reg.mean_std(label, i, 1, sel, f"{row}, {objective}, trainer seeds 0-2")
        if row != "MARINER":
            reg.mean_se(label, i, 2, sel, _selection("MARINER", column), f"{row} minus MARINER, {objective}")
        for col, predictor in ((3, "regmix"), (4, "mariner")):
            filters = f"variant=='{variant}' & objective=='{objective}' & predictor=='{predictor}'"
            reg.add(label, i, col, 0, f"{PRED}[{filters}].predicted_bpb.unique", f"{predictor} prediction for {variant}")


def _kl_ablation(reg: _Registry) -> None:
    label = "tab:a-kl-ablation"
    for i in range(len(reg.parsed[label])):
        lam = reg.stub(label, i)
        for col, column, objective, variant in (
            (1, U, "uncheatable", "lwspu_u_snc_cap06"),
            (3, O, "obe_easy", "lwspu_t9_snc_cap08"),
        ):
            if lam != "0 (control)":
                variant = variant.replace("snc", f"kl{LAMBDAS[lam]}")
            filters = f"variant=='{variant}' & objective=='{objective}' & predictor=='mariner'"
            reg.add(label, i, col, 0, f"{PRED}[{filters}].predicted_bpb.unique", f"MARINER's prediction for {variant}")
            if lam == "0 (control)":
                note = f"zero-penalty control, {objective}, trainer seeds 0-2"
                reg.mean_std(label, i, col + 1, _selection("MARINER", column), note)
            else:
                query = f"{V}[launch=='kl_ablation' & variant=='{variant}'].{column}.value"
                reg.add(label, i, col + 1, 0, query, f"{variant}, {objective}")


def _olmix_cap_kl(reg: _Registry, build: dict[str, pd.DataFrame]) -> None:
    label = "tab:a-olmix-cap-kl"
    trained = set(build[V].variant[build[V].launch == "olmix_cap_kl_sweep"])

    def sweep_variant(objective: str, kl: str, cap: str) -> str | None:
        prefix = f"olmixq_{objective}_kl{kl}_"
        if cap == "none":
            uncapped = sorted(x for x in trained if x.startswith(prefix + "nocap_"))
            return uncapped[0] if uncapped else None
        return f"{prefix}cap{cap}" if f"{prefix}cap{cap}" in trained else None

    for i in range(len(reg.parsed[label]) - 1):
        kl = LAMBDAS[reg.stub(label, i)]
        for j in range(1, 11):
            objective, column = ("u", U) if j <= 5 else ("t9", O)
            cap = OLMIX_CAPS[(j - 1) % 5]
            variant = sweep_variant(objective, kl, cap)
            note = f"lambda {reg.stub(label, i)}, cap {cap}"
            if variant is None:  # italic: repeats a trained policy within total variation 0.01
                if cap == "01":  # every cap-1 policy repeats the unpenalized one
                    variant = sweep_variant(objective, "0", "01")
                else:  # the policy of the next lower trained cap at this KL weight
                    lower = OLMIX_CAPS[: OLMIX_CAPS.index(cap)][::-1]
                    variant = next(name for name in (sweep_variant(objective, kl, c) for c in lower) if name)
                note += f" (italic: repeats {variant})"
            query = f"{V}[launch=='olmix_cap_kl_sweep' & variant=='{variant}'].{column}.value"
            reg.add(label, i, j, 0, query, note)
    last = len(reg.parsed[label]) - 1
    mariner_u = f"{V}[launch=='frozen_procedure' & variant=='lwspu_u_snc_cap06'].{U}.value"
    mariner_o = f"{V}[launch=='frozen_procedure' & variant=='lwspu_t9_snc_cap08'].{O}.value"
    reg.add(label, last, 1, 0, mariner_u, "MARINER's Uncheatable seed-0 run")
    reg.add(label, last, 2, 0, mariner_o, "MARINER's OBE Easy seed-0 run")


def _snr(reg: _Registry, build: dict[str, pd.DataFrame]) -> None:
    label = "tab:a-snr"
    for i, cells in enumerate(reg.parsed[label]):
        if not any(v.printed_numbers(cell) for cell in cells[1:]):
            continue  # a task heading such as `Minerva MATH (7)`
        name = reg.stub(label, i)
        if name == "OlmoBaseEval Easy mean":
            sel = f"{REL}[source_table=='snr_fit_components_delphi' & task_id.isna()]"
        else:
            sel = f"{REL}[source_table=='snr_fit_components_delphi' & task_id=='{_task_of(build, name)}']"
        for j, column in enumerate(SNR_COLUMNS, start=1):
            reg.add(label, i, j, 0, f"{sel}.{column}.value", f"{name}: {column}")


def _fit(reg: _Registry) -> None:
    label = "tab:fit"
    for i in range(len(reg.parsed[label])):
        name = reg.stub(label, i)
        base = FIT_GROUPS[name]
        reg.add(label, i, 1, 0, f"{REL}[{base}].task_id.count", f"{name}: components")
        reg.add(label, i, 2, 0, f"{REL}[{base}].snr.median", f"{name}: median SNR")
        for j, model in enumerate(("mariner", "olmix", "regmix")):
            r2 = f"{REL}[{base}].{{1 - {model}_mse / panel_sd**2}}.median"
            reg.add(label, i, 3 + j, 0, r2, f"{name}: median {model} R2 = 1 - MSE / swarm variance")
            noise = f"{REL}[{base}].{{100 * repeat_sd**2 / {model}_mse}}.median"
            reg.add(label, i, 6 + j, 0, noise, f"{name}: median {model} noise share, percent of MSE")
        count = f"{REL}[{base} & mariner_rmse <= 2 * repeat_sd].task_id.count"
        reg.add(label, i, 9, 0, count, f"{name}: MARINER RMSE within twice the proportional std")


def _snr_subset(reg: _Registry) -> None:
    label = "tab:a-snr-subset"
    proposal = ""  # the proposal row that a predicted row follows
    for i in range(len(reg.parsed[label])):
        name = reg.stub(label, i).replace("\\geq", "geq")
        if name.startswith("predicted"):  # the predicted differences of the proposal above
            for col, (_, objective) in SUBSET_COLUMNS.items():
                a = f"{PRED}[variant=='{SUBSET_VARIANTS[proposal]}' & objective=='{objective}' & predictor=='mariner']"
                b = f"{PRED}[variant=='lwspu_t9_snc_cap08' & objective=='{objective}' & predictor=='mariner']"
                query = f"{a}.predicted_bpb.unique diff: {b}.predicted_bpb.unique"
                note = f"predicted {objective} of {SUBSET_VARIANTS[proposal]} minus MARINER's"
                reg.add(label, i, col + 1, 0, query, note)
            continue
        proposal = name
        for col, (column, objective) in SUBSET_COLUMNS.items():
            sel = f"{V}[{SUBSET_RUNS[name]}].{column}"
            reg.mean_std(label, i, col, sel, f"{name}, {objective} mean, trainer seeds 0-2")
            if name != "Full mean (MARINER)":
                mariner = f"{V}[{MARINER_O}].{column}"
                reg.mean_se(label, i, col + 1, sel, mariner, f"{name} minus MARINER, {objective} mean")


def _factorial(reg: _Registry) -> None:
    label = "tab:a-factorial-predictions"
    for i in range(len(reg.parsed[label])):
        factor = reg.stub(label, i)
        for col, column, objective in ((1, O, "obe_easy"), (3, U, "uncheatable")):
            predicted = f"objective=='{objective}' & predictor=='mariner'"
            if i == 0:
                centre = f"{V}[launch=='factorial' & variant.str.startswith('centre_')].{column}.mean"
                reg.add(label, i, col, 0, centre, f"centre, two trainer seeds, {objective}")
                note = f"centre SE: std of the 20 data-seed repeats of the centre over sqrt(2), {objective}"
                reg.add(label, i, col, 1, f"{CENTRE_REPEATS}.{column}.std / sqrt(2)", note)
                query = (
                    f"{PRED}[launch=='factorial' & variant.str.startswith('centre_') & {predicted}].predicted_bpb.unique"
                )
                reg.add(label, i, col + 1, 0, query, f"MARINER's prediction for the centre, {objective}")
                continue
            up, down = ("fac_" + "." * (i - 1) + level for level in "pm")  # the factor's level in the variant name
            measured = [f"{V}[launch=='factorial' & variant.str.match('{x}')].{column}.mean" for x in (up, down)]
            forecast = [
                f"{PRED}[launch=='factorial' & variant.str.match('{x}') & {predicted}].predicted_bpb.mean"
                for x in (up, down)
            ]
            reg.add(
                label, i, col, 0, f"{measured[0]} diff: {measured[1]}", f"{factor}: measured main effect, {objective}"
            )
            note = f"{factor}: SE, half the std of the centre's 20 data-seed repeats, {objective}"
            reg.add(label, i, col, 1, f"{CENTRE_REPEATS}.{column}.std / 2", note)
            reg.add(
                label,
                i,
                col + 1,
                0,
                f"{forecast[0]} diff: {forecast[1]}",
                f"{factor}: MARINER's predicted main effect, {objective}",
            )


def _buckets(reg: _Registry, build: dict[str, pd.DataFrame]) -> None:
    label = "tab:a-buckets"
    partition = build["partition"]
    for i, cells in enumerate(reg.parsed[label]):
        for name_col in (0, 3):  # the table's two panels
            if name_col + 2 >= len(cells) or not v.printed_numbers(cells[name_col + 1]):
                continue
            if cells[name_col].startswith("\\multicolumn"):  # the total row
                continue
            bucket = partition.bucket_id[partition.paper_name == cells[name_col]]
            if len(bucket) != 1:
                raise ValueError(f"no single bucket has the paper name {cells[name_col]!r}")
            b = bucket.iloc[0]
            tokens = f"partition[bucket_id=='{b}'].pool_tokens.value"
            reg.add(label, i, name_col + 1, 0, f"{tokens} / 1e9", f"{b}: pool tokens in billions")
            share = f"100 * {tokens} / partition[].pool_tokens.sum"
            reg.add(label, i, name_col + 2, 0, share, f"{b}: percent of all pool tokens")
    total = len(reg.parsed[label]) - 1
    reg.add(label, total, 2, 0, "partition[].pool_tokens.sum / 1e9", "total pool tokens in billions")
    reg.add(
        label, total, 3, 0, "100 * partition[].pool_tokens.sum / partition[].pool_tokens.sum", "total share, percent"
    )


def _hyperparameters(reg: _Registry) -> None:
    label = "tab:a-hyperparameters"
    for i in range(len(reg.parsed[label])):
        name = reg.stub(label, i)
        for col, swarm in SWARMS.items():
            table = f"swarm_{swarm}"
            meta = QWEN if swarm.startswith("qwen3") else f"{table}[]"
            if name in SIZES:
                column, scale, unit = SIZES[name]
                reg.add(label, i, col, 0, f"{meta}.{column}.unique {scale}", f"{swarm}: {column} in {unit}")
            elif name == "Tokens per parameter":
                query = f"{meta}.train_tokens.unique / {meta}.non_embedding_parameters.unique"
                reg.add(label, i, col, 0, query, f"{swarm}: tokens per non-embedding parameter")
            elif name == "FLOPs per run" and swarm.startswith("qwen3"):
                query = f"6 * {meta}.total_parameters.unique * {meta}.train_tokens.unique / 1e18"
                reg.add(label, i, col, 0, query, f"{swarm}: 6 N D in units of 1e18")
            elif name == "FLOPs per run":
                query = f"{meta}.train_flops.unique / 1e18"
                reg.add(label, i, col, 0, query, f"{swarm}: train_flops (6 N D) in units of 1e18")
            elif name == "Fit swarm (runs)":
                reg.add(label, i, col, 0, f"{table}[fit_inclusion == True].observation_id.count", f"{swarm}: fit rows")
            elif name == "Proportional repeats":
                query = f"{table}[design_kind=='repeat'].observation_id.count"
                reg.add(label, i, col, 0, query, f"{swarm}: proportional repeats")
            elif name == "Held-out runs (frozen)":
                query = f"heldout[swarm=='{swarm}' & cohort=='frozen'].observation_id.count"
                reg.add(label, i, col, 0, query, f"{swarm}: frozen held-out runs")
            else:  # architecture and optimizer settings, which no build table records
                for k in reg.parts(label, i, col):
                    reg.add(label, i, col, k, "", "constant")


def _scaling_configurations(reg: _Registry) -> None:
    label = "tab:a-scaling-configurations"
    for i, rung in enumerate(RUNGS):
        meta = f"ladder[compute_flops=={rung}]"
        for col, column in ((1, "total_parameters"), (2, "non_embedding_parameters"), (3, "train_tokens")):
            scale, unit = ("/ 1e9", "billions") if reg.parsed[label][i][col].endswith("B") else ("/ 1e6", "millions")
            reg.add(label, i, col, 0, f"{meta}.{column}.unique {scale}", f"{rung}: {column} in {unit}")
        reg.add(label, i, 4, 0, f"{meta}.tpp.unique", f"{rung}: tokens per non-embedding parameter")


def _accuracy_components(reg: _Registry, build: dict[str, pd.DataFrame]) -> None:
    label = "tab:a-accuracy-components"
    for i in range(len(reg.parsed[label])):
        task = _task_of(build, reg.stub(label, i))
        reg.add(label, i, 1, 0, f"ladder_accuracy[task_id=='{task}'].n_examples.unique", f"{task}: accuracy examples")
        for k, method in enumerate(ACCURACY_METHODS):
            run = f"ladder_accuracy[method=='{method}' & task_id=='{task}']"
            reg.add(label, i, 2 + k, 0, f"{run}.bpb.value", f"{task}: {method} BPB at 1e21")
            reg.add(label, i, 6 + k, 0, f"{run}.accuracy.value", f"{task}: {method} accuracy, percent")


def _unimax_cap(reg: _Registry) -> None:
    label = "tab:a-unimax-cap"
    for i in range(len(reg.parsed[label])):
        cap = reg.stub(label, i)
        if cap in UNIMAX_LADDER_RUNS:
            run = UNIMAX_LADDER_RUNS[cap]
        else:  # the sweep's runs at caps 1, 4 and 12
            run = f"ladder_unimax_cap_sweep[epoch_cap=={cap}]"
        reg.add(label, i, 1, 0, f"{run}.{U}.value", f"{cap}: Uncheatable")
        reg.add(label, i, 2, 0, f"{run}.{O}.value", f"{cap}: OBE Easy")
        for col, group in UNIMAX_GROUPS.items():
            reg.add(label, i, col, 0, f"{run}.mean_of({group}).value", f"{cap}: equal-weight group mean")


def _heteroskedastic(reg: _Registry, build: dict[str, pd.DataFrame]) -> None:
    label = "tab:a-heteroskedastic"
    panel = build["starcoder_repeat_noise"].groupby("mixture_label")[["phase_0_starcoder", "phase_1_starcoder"]].first()
    for i, cells in enumerate(reg.parsed[label]):
        printed = (v.numeric(cells[0]), v.numeric(cells[1]))  # a row's mixture is the one with its printed shares
        (mixture,) = [m for m, (p0, p1) in panel.iterrows() if (round(p0, 2), round(p1, 2)) == printed]
        sel = f"starcoder_repeat_noise[mixture_label=='{mixture}']"
        reg.add(label, i, 0, 0, f"{sel}.phase_0_starcoder.unique", f"{mixture}: StarCoder share, first phase")
        reg.add(label, i, 1, 0, f"{sel}.phase_1_starcoder.unique", f"{mixture}: StarCoder share, second phase")
        reg.add(label, i, 2, 0, f"{sel}.pl_bpb.mean", f"{mixture}: mean programming-languages BPB of five data seeds")
        reg.add(label, i, 3, 0, f"{sel}.pl_bpb.std", f"{mixture}: repeat std of programming-languages BPB")


def _compute_equivalence(reg: _Registry) -> None:
    label = "tab:a-compute-equivalence"
    for i in range(len(reg.parsed[label])):
        objective = "uncheatable" if i < 3 else "obe_easy"
        baseline = METHODS[reg.stub(label, i)]
        what = f"{baseline}, {objective}"
        law = f"ladder_compute_equivalence[objective=='{objective}' & baseline=='{baseline}']"
        reg.add(label, i, 1, 0, f"{law}.law_e.unique", f"{what}: fitted E")
        reg.add(label, i, 1, 1, f"{law}.law_a.unique", f"{what}: fitted A")
        reg.add(label, i, 1, 2, f"- {law}.law_alpha.unique", f"{what}: the printed exponent, minus alpha")
        reg.add(label, i, 2, 0, f"{law}.law_rmse.unique", f"{what}: residual of the fit over its four rungs")
        for col, rung in enumerate(RUNGS, start=3):
            match = (
                f"ladder_compute_equivalence[objective=='{objective}' & baseline=='{baseline}' & compute_flops=={rung}]"
            )
            reg.add(label, i, col, 0, f"{match}.multiplier.value", f"{what}, MARINER at {rung}: compute multiple")
            note = f"{what}, MARINER at {rung}: compute saved, 100 (1 - 1 / multiple)"
            reg.add(label, i, col, 1, f"100 * (1 - 1 / {match}.multiplier.value)", note)


def _ablations_full(reg: _Registry) -> None:
    label = "tab:a-ablations-full"
    names = [reg.stub(label, i) for i in range(len(reg.parsed[label]))]
    if names != [name for name, _ in BENCHMARK_ROWS]:
        raise ValueError(f"{label}: the rows are not those of BENCHMARK_ROWS: {names}")
    for i, (name, model) in enumerate(BENCHMARK_ROWS):
        for k in reg.parts(label, i, 1):  # Params: nominal response parameters per task, a property of the model form
            reg.add(label, i, 1, k, "", "constant")
        for col, (swarm, objective) in enumerate(BENCHMARK_COLUMNS, start=2):
            filters = f"model=='{model}' & swarm=='{swarm}' & objective=='{objective}'"
            note = f"{name}: out-of-fold Spearman, {swarm}, {objective}"
            reg.add(label, i, col, 0, f"fits_surrogate_benchmark[{filters}].spearman.value", note)


def registry(parsed: dict[str, list[list[str]]], build: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per printed number of the paper's tables, in the order of the parsed tables."""
    reg = _Registry(parsed)
    _scaling_main(reg)
    _baselines(reg)
    _regmix_argmin(reg)
    _kl_ablation(reg)
    _olmix_cap_kl(reg, build)
    _snr(reg, build)
    _fit(reg)
    _snr_subset(reg)
    _factorial(reg)
    _buckets(reg, build)
    _hyperparameters(reg)
    _scaling_configurations(reg)
    _accuracy_components(reg, build)
    _unimax_cap(reg)
    _heteroskedastic(reg, build)
    _compute_equivalence(reg)
    _ablations_full(reg)
    reg.skip_all("tab:a-matched-swarms", "deferred_two_phase")
    reg.skip_all("tab:a-phase-fiber", "deferred_two_phase")
    order = {label: n for n, label in enumerate(parsed)}
    frame = pd.DataFrame(reg.rows)
    return frame.sort_values(
        ["label", "row", "col", "part"], key=lambda s: s.map(order) if s.name == "label" else s
    ).reset_index(drop=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", type=Path, default=paths.BUILD)
    parser.add_argument("--sections", type=Path, default=paths.DRIVE / "sections")
    parser.add_argument("--out", type=Path, default=v.REGISTRY)
    args = parser.parse_args(argv)
    frame = registry(v.parse_tables(v.read_sections(args.sections)), v.load_build(args.build))
    frame.to_csv(args.out, index=False)
    print(f"wrote {len(frame)} rows to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
