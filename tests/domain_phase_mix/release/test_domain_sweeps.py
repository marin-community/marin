# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re

import numpy as np
import pandas as pd
import pytest

from experiments.domain_phase_mix.olmix_loglinear_fit import DEFAULT_HUBER_DELTA, OlmixLoglinearFit
from experiments.domain_phase_mix.release import context, paths
from experiments.domain_phase_mix.release.configs import domain_sweeps

SWEEPS = paths.AG / "starcoder_tpp10"
pytestmark = pytest.mark.skipif(not SWEEPS.exists(), reason="domain sweep sources absent")
# The files the epoch-cap figure's builder chain reads for the four curves it draws (build_preview.load_curves).
FLAN = SWEEPS / "instruction_sweep"
MATH = (
    SWEEPS
    / "domain_sweeps/target_math_complete_20260912/results"
    / "c788d33ddd3b357a7f80de6ed1d5ecb632104e4beabd5cc0934d580ecac77d39/receipt.json"
)
# Each sweep's launch plans, which the export does not read.
PLANS = {
    "instruction": [
        "instruction_sweep/plan_proxy.json",
        "instruction_sweep/plan_target.json",
        "instruction_sweep/plan_unmatched.json",
    ],
    "rcqa": ["rcqa_sweep/plan_proxy_r2.json"],
    "arxiv": ["arxiv_sweep/plan_proxy.json"],
    "finemath": ["domain_sweeps/plan.json", "domain_sweeps/finemath_unmatched/plan.json"],
}
SOURCE_DOMAIN = {
    "instruction": "dolmino_flan",
    "rcqa": "wiki_to_rcqa",
    "arxiv": "arxiv_papers",
    "finemath": "finemath_3plus",
}
# FineMath's weight-0 points are the original TPP-10 experiment's web-only runs, launched from its design.
FINEMATH_CONTROLS = {"tpp10_unmatched_p000_s20260910", "tpp10_target_p000_s20260910"}
# The realized epochs each sweep's analysis lists for its matched-proxy or target grid.
ANALYSES = {
    ("instruction", "proxy"): "instruction_sweep/results/analysis.json",
    ("instruction", "target"): "instruction_sweep/results_target/analysis.json",
    ("rcqa", "proxy"): "rcqa_sweep/results/analysis.json",
    ("arxiv", "proxy"): "arxiv_sweep/results/analysis.json",
}
QA = {"arc_easy_bpb", "arc_challenge_bpb", "openbookqa_bpb", "qasc_bpb", "sciq_bpb"}
UNCHEATABLE = [
    f"uncheatable/{c}"
    for c in (
        "wikipedia_english",
        "github_python",
        "github_cpp",
        "bbc_news",
        "arxiv_physics",
        "arxiv_computer_science",
        "ao3_english",
    )
]
EVALUATIONS = {
    "instruction": {"dolmino_flan_heldout_bpb", *QA},
    "rcqa": {"wiki_to_rcqa_heldout_bpb", *QA},
    "arxiv": {"s2orc_bpb"},
    "finemath": {"gsm8k_bpb", "math500_bpb"},
}


@pytest.fixture(scope="module")
def ctx(tmp_path_factory):
    return context.build(private_dir=tmp_path_factory.mktemp("private"))


@pytest.fixture(scope="module")
def out(ctx):
    return domain_sweeps.export(ctx)


@pytest.fixture(scope="module")
def table(out):
    return out["domain_sweeps"]


def _figure_points() -> list[tuple[str, str, str, float, float]]:
    """(domain, scale, evaluation, epochs, BPB) of the points the epoch-cap figure draws: each panel's matched-proxy and
    target curves, read as build_preview.load_curves reads them."""
    points = []
    for scale, folder in (("proxy", "results"), ("target", "results_target")):
        # The builder parses with float(), exactly; pandas' default parser is one ulp off for some values.
        runs = pd.read_csv(FLAN / folder / "measurements.csv", float_precision="round_trip").sort_values("percent")
        epochs = json.loads((FLAN / folder / "analysis.json").read_text())["metrics"]["sciq_bpb"]["epochs"]
        points += [("instruction", scale, "sciq_bpb", e, b) for e, b in zip(epochs, runs.sciq_bpb, strict=True)]
    rows = json.loads(MATH.read_text())["rows"]
    for scale, arm in (("proxy", "matched"), ("target", "target")):
        points += [
            ("finemath", scale, "gsm8k_bpb", row["epochs"], row["result"]["metrics"]["eval/gsm8k/bpb"])
            for row in rows[f"finemath_3plus/{arm}"]
        ]
    return points


def test_every_figure_point_is_present(table):
    # The figure reads the instruction sweep's results and the FineMath math receipt; plots_20260912's
    # completed_points.csv is an earlier Wikipedia and FineMath snapshot on the Uncheatable mean.
    points = _figure_points()
    assert len(points) == 32
    for domain, scale, evaluation, epochs, bpb in points:
        rows = table[(table.domain == domain) & (table.scale == scale) & (table.evaluation == evaluation)]
        match = rows[(rows.domain_epochs - epochs).abs() < 1e-6]
        assert (match.bpb == bpb).any(), (domain, scale, evaluation, epochs)


def test_domains_scales_and_sources(table):
    assert set(table.domain) == {"instruction", "rcqa", "arxiv", "finemath"}
    assert set(table.scale) == {"proxy", "target"}
    sources = json.loads(
        (paths.REPO / "experiments/domain_phase_mix/release/configs/domain_sweeps_sources.json").read_text()
    )
    assert set(sources) == set(table.domain)
    assert table.bpb.gt(0).all() and table.domain_epochs.ge(0).all()
    assert table.observation_id.notna().all()


def _launch_records() -> dict[str, pd.DataFrame]:
    """Each domain's launched runs from its sweep plans, with FineMath's controls from the TPP-10 design."""
    design = json.loads(domain_sweeps.DESIGN.read_text())
    controls = [  # both draw from the whole pool
        {**run, "tokens": run["total_steps"] * run["batch_size"] * 2048, "support_sequences": design["parent_sequences"]}
        for run in design["runs"]
        if run["run_name"] in FINEMATH_CONTROLS
    ]
    records = {}
    for domain, plans in PLANS.items():
        runs = [run for plan in plans for run in json.loads((SWEEPS / plan).read_text())["runs"]]
        runs = [run for run in runs if run["domain"] == SOURCE_DOMAIN[domain]]
        records[domain] = pd.DataFrame(runs + (controls if domain == "finemath" else []))
    return records


def test_runs_match_their_launch_plans(ctx, table):
    runs = table.drop_duplicates("observation_id").set_index("observation_id")
    for domain, launched in _launch_records().items():
        launched.index = launched.run_name.map(ctx.registry.id_for)
        assert set(runs.index[runs.domain == domain]) == set(launched.index), domain
        mine = runs.loc[launched.index]
        assert (mine.scale == launched.arm.map({"matched": "proxy", "unmatched": "proxy", "target": "target"})).all()
        assert (mine.train_tokens == launched.tokens).all()
        per_unit_weight = launched.tokens / (launched.support_sequences * 2048)
        assert np.allclose(mine.epochs_per_unit_weight, per_unit_weight, rtol=0, atol=1e-12)
        assert np.allclose(mine.domain_weight, launched.percent / 100, rtol=0, atol=1e-12)
        assert (mine.trainer_seed == launched.trainer_seed).all()
        assert mine.subset_seed.fillna(-1).astype(int).tolist() == launched.subset_seed.fillna(-1).astype(int).tolist()
    sizes = runs.groupby("scale").model_total_parameters.unique()
    assert sizes.to_dict() == {"proxy": [16_587_008], "target": [301_241_344]}


def test_epochs_match_each_sweeps_analysis(table):
    # The export derives epochs from the design's allocation audit; each sweep's analysis records them per grid point.
    runs = table.drop_duplicates("observation_id")
    for (domain, scale), path in ANALYSES.items():
        grid = runs[(runs.domain == domain) & (runs.scale == scale) & (runs.epochs_per_unit_weight > 1)]
        grid = grid.set_index("domain_weight")
        for metric in json.loads((SWEEPS / path).read_text())["metrics"].values():
            epochs = grid.domain_epochs.loc[np.array(metric["grid_percent"]) / 100]
            assert np.allclose(epochs, metric["epochs"], rtol=0, atol=1e-12), (domain, scale)


def test_every_run_reports_its_sweeps_evaluations(table):
    assert list(table.columns) == [
        "observation_id",
        "domain",
        "scale",
        "model_total_parameters",
        "tokenizer",
        "train_tokens",
        "epochs_per_unit_weight",
        "domain_weight",
        "domain_epochs",
        "evaluation",
        "bpb",
        "trainer_seed",
        "subset_seed",
    ]
    # One tokenizer for every run: the TPP-10 design's pin, which the FineMath math receipt also records.
    pinned = json.loads(MATH.read_text())["shared_scoring"]["tokenizer_pins"]["tokenizer_repo"]
    assert table.tokenizer.notna().all() and set(table.tokenizer) == {pinned}
    assert not table.duplicated(["observation_id", "evaluation"]).any()
    for domain, own in EVALUATIONS.items():
        per_run = table[table.domain == domain].groupby("observation_id").evaluation.apply(frozenset)
        assert set(per_run) == {frozenset({*own, "macro_bpb", "paloma_bpb", *UNCHEATABLE})}, domain
    # macro_bpb is the equal-weight mean of the seven Uncheatable components (Levanter's float32 mean for FineMath's
    # matched proxy and target).
    wide = table.pivot(index="observation_id", columns="evaluation", values="bpb")
    assert np.allclose(wide.macro_bpb, wide[UNCHEATABLE].mean(axis=1), rtol=0, atol=2e-7)


def test_olmix_fits_are_fits_to_the_matched_proxies(out):
    table, fits = out["domain_sweeps"], out["domain_sweeps_olmix_fits"]
    assert sorted(zip(fits.domain, fits.evaluation, strict=True)) == [
        ("finemath", "gsm8k_bpb"),
        ("instruction", "sciq_bpb"),
    ]
    for fit in fits.itertuples():
        rows = table[(table.domain == fit.domain) & (table.scale == fit.scale) & (table.evaluation == fit.evaluation)]
        # The matched proxy's runs and the web-only control, which sees no domain data (FineMath's is shared).
        same_pool = np.isclose(rows.epochs_per_unit_weight, fit.epochs_per_unit_weight, rtol=0, atol=1e-12)
        curve = rows[same_pool | rows.domain_weight.eq(0)]
        curve = curve.sort_values("epochs_per_unit_weight", ascending=False).drop_duplicates("domain_weight")
        assert len(curve) == 8
        head = OlmixLoglinearFit(fit.log_c, (fit.coefficient_domain, fit.coefficient_web), fit.huber_loss)
        p = curve.domain_weight.to_numpy()
        residual = np.abs(head.predict(np.column_stack([p, 1 - p])) - curve.bpb.to_numpy())
        delta = DEFAULT_HUBER_DELTA
        huber = np.where(residual <= delta, 0.5 * residual**2, delta * (residual - 0.5 * delta)).sum()
        assert huber == pytest.approx(fit.huber_loss, rel=0, abs=1e-12)
    # Panel A's fit rises with FLAN and panel B's falls with FineMath: Olmix picks no FLAN and as much FineMath as a cap
    # allows.
    fit = fits.set_index("domain")
    assert fit.coefficient_domain["instruction"] > fit.coefficient_web["instruction"]
    assert fit.coefficient_domain["finemath"] < fit.coefficient_web["finemath"]


def test_no_run_names_or_internal_strings(out):
    sources = json.loads(domain_sweeps.SOURCES.read_text())
    names = set()
    for entry in [entry for entries in sources.values() for entry in entries if entry["reader"] == "measurements"]:
        names |= set(pd.read_csv(paths.REPO / entry["original"]).run_name)
    names |= {row["run_name"] for rows in json.loads(MATH.read_text())["rows"].values() for row in rows}
    patterns = [
        re.compile(r"wandb|gs://|http|/Users/|pinlin_calvin_xu"),
        re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),  # email addresses
        re.compile(r"[0-9a-f]{64}"),
        re.compile(r"(?:exp|run|job)_01[a-z0-9]{24}"),  # Fieldbook ids
    ]
    for name, t in out.items():
        strings = set(t.columns)
        for column in t.columns:
            if not pd.api.types.is_numeric_dtype(t[column]):
                strings |= set(t[column].dropna().astype(str))
        assert not strings & names, (name, sorted(strings & names)[:3])
        text = "\n".join(sorted(strings))
        for pattern in patterns:
            assert not pattern.search(text), (name, pattern.pattern)
