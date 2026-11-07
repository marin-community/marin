# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mariner.cli import main
from mariner.data import FIGURES_INDEX, OBJECTIVES, read_table
from mariner.figures import FIGURES, write_figures
from mariner.figures.epochs import Track
from mariner.figures.layout import Method
from mariner.figures.pipeline import scaling_values
from mariner.figures.starcoder import EPOCH_MATCHING_TABLE, epoch_matching

# figures_tables.csv marks configuration constants and the two-phase data the release omits among a figure's configs.
INDEX_MARKERS = {"constant", "deferred"}
TERMS = "fig:mariner-terms"  # the one figure that reads no table: mariner.model's terms at the paper's values
# Figure 3's inset tables print each sweep's training FLOPs and fitted optimum in materialized StarCoder epochs, by
# path and scale; the paths match compute at each scale (its caption: "within 3% across columns").
PRINTED_FLOPS = ["8.8e+17", "1.7e+18", "3.4e+18", "6.5e+18"]
PRINTED_OPTIMUM_EPOCHS = {
    Track.FIXED_TOKENS: [8.8, 8.3, 8.0, 7.7],
    Track.FIXED_PARAMETERS: [8.8, 11.1, 14.0, 17.5],
    Track.FIXED_TPP: [9.2, 9.7, 10.9, 12.5],
}
COMPUTE_MATCH = 0.03
# And each path's optimum-share law, p* = coefficient (q / q0)^exponent with q0 its smallest scale to three digits.
PRINTED_SHARE_LAWS = {
    Track.FIXED_TOKENS: (0.332, -0.078),
    Track.FIXED_PARAMETERS: (0.335, 0.338),
    Track.FIXED_TPP: (0.340, 0.326),
}
# Figure 5's harm legend: each threshold's activation, e^tau - 1 epochs, to two significant digits.
PRINTED_ACTIVATIONS = {1.0: "1.7", 2.0: "6.4", 3.0: "19"}
SOFTPLUS_AT_ZERO = math.log(2.0)
# The plotted curve, linearly interpolated between its 1,500 log-spaced epochs, at an activation (measured 1.7e-6).
HARM_CURVE_TOLERANCE = 1e-5
# BPB by which the 1.00B-token sweep falls from its smallest share (sections/surrogate_properties.tex:22).
LOWEST_COMPUTE_DIP = 0.22
# The dataset recomputes the ladder's Uncheatable aggregate from its components, up to 7.7e-8 BPB from the means the
# paper fitted, which moves the fitted laws by up to 3.1e-7 (measured); the module's fit of the paper's own means gives
# the released laws exactly.
LAW_TOLERANCE = 1e-6
# Figure 1's compute multiples of MARINER over Olmix and over proportional mixing at 3e18, 2e19, 3e20 and 1e21 FLOPs.
PRINTED_MULTIPLES = {"olmix": [1.3, 1.4, 1.5, 1.6], "proportional": [2.0, 2.4, 2.8, 3.2]}
# Mean OlmoBaseEval Easy accuracy at 1e21 FLOPs: the abstract's 30.1% and 36.4%, and Figure 1's MARINER bar labels.
PRINTED_ACCURACY = {("All", "proportional"): "30.1", ("All", "mariner"): "36.4"}
PRINTED_MARINER_GROUPS = {"Basic": "78.5", "Math": "9.9", "Code": "15.9", "MMLU": "36.8", "QA": "57.8"}
# The shares that Figures 1 and 2 mark with diamonds and a star, positions rather than printed text: the matched
# proxies' mean selection, 8/15, which the target's follow-up run trains in starcoder_epoch_matching; the unmatched
# proxy's pure StarCoder (sections/simulated_epoching.tex:17: "The unmatched proxy selects pure StarCoder"); and the
# target's optimum.
MARKED_SHARES = {"matched": 8 / 15, "unmatched": 1.0}
MARKED_OPTIMUM = 0.7
SHARE_TOLERANCE = 1e-12
PRINTED_EXCESS_AVOIDED = "75"  # Figure 1: "Excess loss avoided: 75%"
# Figure 2's labels: each selection's materialized epochs and target loss, the target's optimum loss, the share of
# excess loss avoided, the epochs its top axes print at p = 0.6, 0.8 and 1.0, and its titles' FLOPs.
PRINTED_SELECTIONS = {"matched": ("8.5", "0.774"), "unmatched": ("0.87", "0.801")}
PRINTED_OPTIMUM_BPB = "0.766"
PRINTED_AVOIDED = "74.7"
PRINTED_EPOCH_ROWS = {
    "matched": ["9.5", "12.7", "15.8"],
    "unmatched": ["0.52", "0.70", "0.87"],
    "target": ["9.5", "12.7", "15.8"],
}
PRINTED_COMPUTE = ("2.49e+16", "6.66e+18")
PROPOSAL_PREDICTION_GAP = 0.005  # BPB: "Its proposed mixtures measure within 0.005 BPB of its predictions" (abstract)
# Figure 1's surrogate panel shows MARINER's predicted loss falling and then rising (sections/surrogate.tex:40: "so a
# task's predicted loss can fall and then rise as a bucket is upweighted"), with its interior optimum at p = 0.578 of the
# 400-point grid (predicted_minimum_fraction 0.5781954887 in figure1_real_data_drafts/v40/
# figure1_real_data_verification.json); Olmix's monotone fit picking pure StarCoder (sections/abstract.tex:1: "common
# monotone surrogates cannot represent a task loss that first falls and then rises"); and boosted trees that "reach the
# 90% mean but stay flat past it, below the measured 100% mean" (figure1_real_data_drafts/v40/build_figure.py:14-15).
MARKED_MARINER_OPTIMUM = "0.578"


def epoch_label(value: float) -> str:
    return f"{value:.2f}" if value < 1 else f"{value:.1f}"


def test_write_figures_writes_the_named_figure_reproducibly(tmp_path):
    out = tmp_path / "out"
    paths = write_figures(tmp_path, out, [TERMS])
    assert paths == [out / f"{FIGURES[TERMS].stem}.pdf"]
    assert [path.name for path in out.iterdir()] == [paths[0].name]
    written = paths[0].read_bytes()
    assert written.startswith(b"%PDF")
    assert b"CreationDate" not in written
    # The PDFs carry no creation date, so the committed figures can be regenerated byte for byte.
    write_figures(tmp_path, out, [TERMS])
    assert paths[0].read_bytes() == written


def test_write_figures_names_an_unknown_label(tmp_path):
    # fig:b5 comes first and would fail to read the empty snapshot if anything were drawn before the check.
    with pytest.raises(ValueError, match="fig:nope"):
        write_figures(tmp_path, tmp_path, ["fig:b5", "fig:nope"])
    assert list(tmp_path.iterdir()) == []


def test_cli_figures_exits_nonzero_on_unknown_label(tmp_path, capsys):
    assert main(["figures", "--data", str(tmp_path), "--out", str(tmp_path), "--only", "fig:nope"]) != 0
    assert "fig:nope" in capsys.readouterr().err


def test_mariner_terms_mark_each_harm_activation_at_the_papers_epochs(tmp_path):
    """Figure 5's caption: "dots mark activations at E = e^tau - 1 epochs" on each harm curve."""
    values = FIGURES[TERMS].draw(tmp_path).values
    harm = values["harm"]
    for activation in values["activations"].itertuples():
        assert f"{activation.epochs:.2g}" == PRINTED_ACTIVATIONS[activation.tau]
        # The squared softplus activates where its argument log(1 + E) - tau crosses zero.
        assert activation.harm == pytest.approx(SOFTPLUS_AT_ZERO**2, rel=1e-12)
        curve = harm[harm["tau"] == activation.tau]
        on_curve = np.interp(np.log(activation.epochs), np.log(curve["epochs"]), curve["harm"])
        assert on_curve == pytest.approx(activation.harm, abs=HARM_CURVE_TOLERANCE)


def small_epoch_matching() -> pd.DataFrame:
    """A two-share sweep: two trainer seeds of each proxy and one matched subset, whose share-0 runs are the unmatched
    proxy's, and the target at both shares."""
    rows = []
    for seed in (1, 2):
        rows.append({"observation_id": f"u0_{seed}", "arm": "unmatched", "trainer_seed": seed, "starcoder_weight": 0.0})
        rows.append({"observation_id": f"u1_{seed}", "arm": "unmatched", "trainer_seed": seed, "starcoder_weight": 1.0})
        rows.append(
            {
                "observation_id": f"m1_{seed}",
                "arm": "matched",
                "subset_seed": 7,
                "trainer_seed": seed,
                "starcoder_weight": 1.0,
            }
        )
    rows += [
        {"observation_id": f"t{share:g}", "arm": "target", "trainer_seed": 1, "starcoder_weight": share}
        for share in (0.0, 1.0)
    ]
    runs = pd.DataFrame(rows)
    return runs.assign(
        starcoder_epochs=runs["starcoder_weight"],
        programming_languages_bpb=1.0 + runs.index / 100,
        train_tokens=100,
        starcoder_subset_tokens=100,
        compute_flops=1e16,
    )


def write_epoch_matching(root: Path, runs: pd.DataFrame) -> None:
    table = Path("data/starcoder") / f"{EPOCH_MATCHING_TABLE}.csv"
    (root / table).parent.mkdir(parents=True)
    runs.to_csv(root / table, index=False)
    entry = {"table": EPOCH_MATCHING_TABLE, "path": table.as_posix(), "format": "csv"}
    (root / "MANIFEST.json").write_text(json.dumps({"tables": [entry]}))


@pytest.mark.parametrize(
    ("edit", "message"),
    [
        (lambda runs: pd.concat([runs, runs.iloc[[2]].assign(observation_id="m1_1_again")]), "repeats runs"),
        (lambda runs: runs[runs["observation_id"] != "m1_2"], "lacks a run"),
    ],
    ids=["repeated", "missing"],
)
def test_epoch_matching_refuses_repeated_or_missing_runs(tmp_path, edit, message):
    write_epoch_matching(tmp_path, edit(small_epoch_matching()))
    with pytest.raises(ValueError, match=message):
        epoch_matching(tmp_path)


def check_pipeline(values: dict[str, pd.DataFrame]) -> None:
    """Figure 1: its repetition panel's "Excess loss avoided: 75%" and the shares it marks, its surrogate panel's
    optima and trees, its compute multiples, and the abstract's accuracies (sections/abstract.tex:1: "from
    proportional mixing's 30.1\\% to 36.4\\%")."""
    assert f"{100 * values['summary']['excess_loss_avoided'].iloc[0]:.0f}" == PRINTED_EXCESS_AVOIDED
    selections = values["selections"].set_index("arm")["share"]
    for arm, share in MARKED_SHARES.items():
        assert selections[arm] == pytest.approx(share, abs=SHARE_TOLERANCE)
    assert values["target_optimum"]["share"].iloc[0] == MARKED_OPTIMUM
    optima = values["surrogate_optima"].set_index("model")["share"]
    grid = values["surrogate_curves"]["share"]
    assert grid.min() < optima["mariner"] < grid.max()  # an interior optimum
    assert f"{optima['mariner']:.3f}" == MARKED_MARINER_OPTIMUM
    assert optima["olmix"] == MARKED_SHARES["unmatched"]
    plateaus = values["surrogate_trees"]
    means = values["surrogate_means"].set_index("share")["bpb"]
    last = plateaus.iloc[-1]  # the trees' last plateau runs to pure StarCoder
    assert last["share_left"] < max(share for share in means.index if share < 1.0)  # flat from below the 90% mean on
    assert last["bpb"] < means[1.0]
    multiples = values["multiples"].sort_values("compute_flops")
    for baseline, printed in PRINTED_MULTIPLES.items():
        assert multiples.loc[multiples["baseline"] == baseline, "multiple"].round(1).tolist() == printed
    accuracy = values["accuracy"].set_index(["group", "method"])["accuracy"]
    for key, printed in PRINTED_ACCURACY.items():
        assert f"{accuracy.loc[key]:.1f}" == printed
    for group, printed in PRINTED_MARINER_GROUPS.items():
        assert f"{accuracy.loc[(group, Method.MARINER.value)]:.1f}" == printed


def check_epoch_matching(values: dict[str, pd.DataFrame]) -> None:
    """Figure 2: the shares it marks, each selection's printed epochs and target loss ("0.774", "0.801"), the target's
    optimum ("0.766"), "74.7% less excess loss", the epochs its top axes print, and its titles' FLOPs."""
    for selection in values["selections"].itertuples():
        assert selection.share == pytest.approx(MARKED_SHARES[selection.arm], abs=SHARE_TOLERANCE)
        assert (epoch_label(selection.epochs), f"{selection.target_bpb:.3f}") == PRINTED_SELECTIONS[selection.arm]
    optimum = values["target_optimum"].iloc[0]
    assert optimum["share"] == MARKED_OPTIMUM
    assert f"{optimum['bpb']:.3f}" == PRINTED_OPTIMUM_BPB
    summary = values["summary"].iloc[0]
    assert f"{100 * summary['excess_loss_avoided']:.1f}" == PRINTED_AVOIDED
    assert (f"{summary['proxy_flops']:.3g}", f"{summary['target_flops']:.3g}") == PRINTED_COMPUTE
    for arm, rows in values["epochs"].groupby("arm"):
        assert [epoch_label(epochs) for epochs in rows.sort_values("share")["epochs"]] == PRINTED_EPOCH_ROWS[arm]


def check_matched_compute_scaling(values: dict[str, pd.DataFrame]) -> None:
    """Figure 3's caption (sections/simulated_epoching.tex:37): "Increasing training tokens shifts the optimum toward
    more epochs; increasing model size shifts it toward fewer. Scaling both at fixed TPP produces a milder shift toward
    more epochs. Compute is matched within 3% across columns.\" """
    minima = values["minima"]
    epochs = {track: rows["epochs"].to_numpy() for track, rows in minima.groupby("track")}
    assert (np.diff(epochs[Track.FIXED_PARAMETERS]) > 0).all()  # more tokens at one model size
    assert (np.diff(epochs[Track.FIXED_TOKENS]) < 0).all()  # a larger model at one token budget
    assert (np.diff(epochs[Track.FIXED_TPP]) > 0).all()
    tpp_shift = epochs[Track.FIXED_TPP][-1] - epochs[Track.FIXED_TPP][0]
    tokens_shift = epochs[Track.FIXED_PARAMETERS][-1] - epochs[Track.FIXED_PARAMETERS][0]
    assert tpp_shift < tokens_shift
    # The paths' k-th sweeps train at matched compute: the columns of the paper's panels.
    flops = np.stack([rows["compute_flops"].to_numpy(float) for _, rows in minima.groupby("track")])
    assert np.isfinite(flops).all()
    assert (flops.max(axis=0) / flops.min(axis=0) - 1 <= COMPUTE_MATCH).all()


def check_olmix_capacity_failure(values: dict[str, pd.DataFrame]) -> None:
    """Figure 4: in every sweep, Olmix's pick is an endpoint and MARINER's fitted minimum is interior. And
    sections/surrogate_properties.tex:22: "the lowest-compute curve dips by more than 0.22 BPB".
    """
    observations = values["observations"]
    picks = values["olmix_picks"].set_index("curve")
    minima = values["mariner_minima"].set_index("curve")
    for curve, rows in observations.groupby("curve"):
        low, high = rows["share"].min(), rows["share"].max()
        assert picks.loc[curve, "share"] in (low, high)
        assert low < minima.loc[curve, "share"] < high
    lowest = observations[observations["train_tokens"] == observations["train_tokens"].min()].sort_values("share")
    assert lowest["bpb"].iloc[0] - lowest["bpb"].min() > LOWEST_COMPUTE_DIP


def check_learning_curves(values: dict[str, pd.DataFrame]) -> None:
    """sections/surrogate.tex:106: "MARINER's selections have lower regret than those of Olmix and both RegMix variants
    at essentially every swarm size (Figure~\\ref{fig:r6})."

    Of the 24 swarm sizes and objectives, MARINER's mean ratio is the lowest at all but one: OlmoBaseEval Easy at 240
    runs, where tuned RegMix's is 0.002 lower.
    """
    ratios = values["regret_ratios"].pivot_table(index=["objective", "swarm_size"], columns="model", values="ratio")
    assert ratios.notna().all().all()
    others = ratios[[Method.OLMIX.value, Method.REGMIX.value, Method.REGMIX_RELEASED.value]].min(axis=1)
    assert len(ratios) == 24
    assert (ratios[Method.MARINER.value] >= others).sum() <= 1


def check_proposal_predictions(values: dict[str, pd.DataFrame]) -> None:
    """sections/abstract.tex:1: "Its proposed mixtures measure within 0.005 BPB of its predictions." The redraw raises
    unless the snapshot holds all four models' predictions of every proposal."""
    measured = values["measured_summary"].set_index(["objective", "proposal"])["mean_bpb"]
    predicted = values["predictions"].set_index(["objective", "proposal", "predictor"])["predicted_bpb"]
    mariner = Method.MARINER.value
    for objective in OBJECTIVES:
        gap = measured.loc[(objective, mariner)] - predicted.loc[(objective, mariner, mariner)]
        assert abs(gap) <= PROPOSAL_PREDICTION_GAP


PAPER_CHECKS: dict[str, Callable[[dict[str, pd.DataFrame]], None]] = {
    "fig:pipeline": check_pipeline,
    "fig:epoch-matching": check_epoch_matching,
    "fig:b5": check_matched_compute_scaling,
    "fig:b6": check_olmix_capacity_failure,
    "fig:r6": check_learning_curves,
    "fig:proposal-predictions": check_proposal_predictions,
}


@pytest.mark.data_integration
@pytest.mark.timeout(600)
@pytest.mark.parametrize("label", [label for label, spec in FIGURES.items() if spec.tables])
def test_redraw_reproduces_the_paper(snapshot, label):
    PAPER_CHECKS[label](FIGURES[label].draw(snapshot).values)


@pytest.mark.data_integration
@pytest.mark.timeout(60)
def test_registry_matches_the_dataset_index(snapshot):
    index = pd.read_csv(snapshot / FIGURES_INDEX).set_index("label")
    for label, spec in FIGURES.items():
        configs = [config for config in index.loc[label, "configs"].split(";") if config not in INDEX_MARKERS]
        assert (spec.stem, sorted(spec.tables)) == (index.loc[label, "file"], sorted(configs))


@pytest.mark.data_integration
@pytest.mark.timeout(300)
def test_starcoder_fits_reproduce_the_papers_printed_optima_and_compute(snapshot):
    values = FIGURES["fig:b5"].draw(snapshot).values
    for track, rows in values["minima"].groupby("track"):
        assert [round(epochs, 1) for epochs in rows["epochs"]] == PRINTED_OPTIMUM_EPOCHS[Track(track)]
        assert [f"{flops:.2g}" for flops in rows["compute_flops"]] == PRINTED_FLOPS
    for law in values["path_laws"].itertuples():
        rounded = float(f"{law.reference:.3g}")
        coefficient = law.share_coefficient * (rounded / law.reference) ** law.share_exponent
        assert (round(coefficient, 3), round(law.share_exponent, 3)) == PRINTED_SHARE_LAWS[Track(law.track)]


@pytest.mark.data_integration
@pytest.mark.timeout(120)
def test_pipeline_scaling_laws_match_the_released_compute_equivalence(snapshot):
    laws = scaling_values(snapshot)["laws"].set_index("method")
    equivalence = read_table(snapshot, "ladder_compute_equivalence")
    released = equivalence[equivalence["objective"] == "uncheatable"].drop_duplicates("baseline").set_index("baseline")
    for baseline in released.index:
        fitted = laws.loc[baseline, ["e", "a", "alpha"]].to_numpy(float)
        expected = released.loc[baseline, ["law_e", "law_a", "law_alpha"]].to_numpy(float)
        np.testing.assert_allclose(fitted, expected, rtol=0, atol=LAW_TOLERANCE)
