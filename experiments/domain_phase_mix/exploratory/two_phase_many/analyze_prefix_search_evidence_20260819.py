# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "scipy"]
# ///
"""The evidence the prefix-search design rests on, in one auditable place (ATOM-029).

Five claims decide the design. Each was first established in a throwaway script, which is not good enough
for something about to spend three hundred TPU-hours, so they are reproduced here with their data-quality
exclusions and noise corrections explicit rather than buried.

1. ORDERING CHANNEL. Within aggregate-matched cells, moving the code family into the decay phase lowers
   code loss and raises text loss. Both macros benefit on net, Table 9 far more.
2. SELECTION VALUE. Ranking held-out policies by an out-of-fold prediction and reporting what they
   ACTUALLY scored, bucket-level features are worth about twelve times the family-level summary.
3. THE ORACLE IS NOT A CEILING. The best observed contrast is a minimum over noisy outcomes. Corrected for
   that, the model already selects better than outcome-selection does in expectation, so there is no
   headroom left in the existing cloud and the next gain must come from new policies.
4. THE ARCHIVE READOUT IS SUFFICIENT FOR PREFIX SELECTION. It sits at 83% of phase 0 rather than at
   the boundary, and the archive cannot say whether a boundary readout would be better -- readout
   position is perfectly confounded with horizon on the atomic panel and near-constant on this one. What
   it can say is whether the scalar readout already carries the phase-0 policy's endpoint-relevant
   content, which is all the design asks of it: the new runs get boundary readouts by construction.
5. THE PANEL NEVER CROSSES PHASES. Of 1825 distinct phase-0 mixtures, none appears with more than one
   phase-1 mixture, which is why the crossed structure has to be manufactured.

Three traps this file is written to avoid, each having already produced a wrong answer once:

- **Contaminated rows.** 66 of 1930 heldout rows carry a W&B summary macro disagreeing with the panel's
  authoritative value by up to 190 times the run noise. Left in they dominate every least-squares fit and
  flipped one coefficient's t-statistic from -0.1 to -8.0.
- **Rank deficiency.** Family shares sum to one and the 39 phase deltas sum to zero, so centred designs
  built from them are exactly singular. A plain least-squares fit returned -1.7e8 on one subsample.
- **Selection on noise.** Any "how much better could we have done" claim must correct for the fact that a
  minimum over n noisy candidates is optimistically biased; here that bias exceeds the entire real effect.

Usage: ``uv run python ... [--claims 1,2,3,4,5]``
"""

import argparse
import collections
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402

DELPHI = SCRIPT_DIR / "reference_outputs" / "delphi_3e18_append_only_heldouts_20260714"
CONTAMINATION_TOLERANCE = 0.01  # BPB; ten times the run noise, far below the 0.177 seen on bad rows
CODE_FAMILY = "tech_code"


def panel():
    """Heldout rows joined to their per-component metrics, with the contaminated rows removed.

    The exclusion is not cosmetic. Those 66 rows disagree with the panel's authoritative macro by up to
    0.177 BPB against a run noise of 0.00091, so they are 190-sigma outliers whose summary values cannot
    be trusted; including them changes conclusions rather than merely adding variance.
    """
    heldout = pd.read_csv(DELPHI / "heldout_current.csv")
    heldout = heldout[heldout["fit_panel_overlap"] == "coordinate_disjoint"]
    merged = heldout.merge(pd.read_csv(DELPHI / "endpoint_components.csv"), on="heldout_id", how="inner")
    # `<=` also drops rows where either macro is missing, since NaN fails the comparison. That is
    # deliberate and is why the dropped count exceeds the 66 rows that disagree numerically.
    clean = (merged["uncheatable_bpb"] - merged["eval/uncheatable_eval/bpb"]).abs() <= CONTAMINATION_TOLERANCE
    return merged[clean].reset_index(drop=True), int((~clean).sum())


def with_phase0_readout(frame):
    """Join the pre-boundary readout. It is a scalar compression of a 39-dimensional phase-0 policy."""
    readouts = pd.read_csv(DELPHI / "phase0_readouts.csv")
    columns = ["heldout_id", "phase0_step", "phase0_fraction", "phase0_uncheatable_bpb"]
    # Renaming rather than relying on merge suffixes, which only fire on collisions and so produce
    # different column names depending on what the left frame happens to carry.
    renamed = readouts[columns].rename(columns={name: f"readout_{name}" for name in columns[1:]})
    return frame.merge(renamed, on="heldout_id", how="left")


def geometry(frame):
    """Mixtures, aggregate, phase delta and the signed code-late coordinate."""
    fit, _held = swarm39.load_scale("delphi_3e18")

    def vector(payload: str) -> np.ndarray:
        mapping = json.loads(payload)
        return np.asarray([float(mapping.get(bucket, 0.0)) for bucket in fit.buckets], dtype=float)

    phase_0 = np.stack([vector(x) for x in frame["phase_0_weights_json"]])
    phase_1 = np.stack([vector(x) for x in frame["phase_1_weights_json"]])
    code = fit.family_index == list(fit.family_names).index(CODE_FAMILY)
    return {
        "fit": fit,
        "phase_0": phase_0,
        "phase_1": phase_1,
        "aggregate": fit.alpha * phase_0 + (1.0 - fit.alpha) * phase_1,
        "delta": phase_1 - phase_0,
        "separation": 0.5 * np.abs(phase_1 - phase_0).sum(axis=1),
        "code_late": phase_1[:, code].sum(axis=1) - phase_0[:, code].sum(axis=1),
        "code": code,
    }


def matched_contrasts(geo, values):
    """Each untied policy minus the mean of the tied policies sharing its aggregate.

    Holding the aggregate fixed is what isolates ORDER from dose: two policies in a cell differ only in
    how the same aggregate mixture is split across the phases.
    """
    cells = collections.defaultdict(list)
    for index, row in enumerate(np.round(geo["aggregate"], 6)):
        cells[tuple(row)].append(index)
    untied = geo["separation"] > 1e-9
    observed = np.isfinite(values)
    rows = []
    for members in cells.values():
        members = np.array([i for i in members if observed[i]], dtype=int)
        if len(members) == 0:
            continue
        tied = members[~untied[members]]
        if len(tied) == 0:
            continue
        for index in members[untied[members]]:
            rows.append((values[index] - values[tied].mean(), index))
    if not rows:
        return np.array([]), np.array([], dtype=int)
    return np.array([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=int)


def run_noise(frame, geo, column):
    """Run-to-run noise from groups sharing BOTH phase mixtures exactly, differing only by seed."""
    groups = collections.defaultdict(list)
    for index, (early, late) in enumerate(zip(np.round(geo["phase_0"], 8), np.round(geo["phase_1"], 8), strict=True)):
        groups[(tuple(early), tuple(late))].append(index)
    values = frame[column].to_numpy(float)
    spreads = [
        float(np.std(values[np.array(members)], ddof=1))
        for members in groups.values()
        if len(members) > 1 and np.isfinite(values[np.array(members)]).all()
    ]
    return float(np.median(spreads)) if spreads else float("nan")


def out_of_fold(design, response, folds=5, strengths=(1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0), seed=0):
    """Ridge predictions with the strength chosen inside each training fold.

    Regularisation is not optional here: the phase deltas sum to zero, so the design is rank deficient and
    an unregularised solve returns whichever minimum-norm direction the pseudo-inverse happens to pick.
    """
    generator = np.random.default_rng(seed)
    order = generator.permutation(len(response))
    assignment = np.zeros(len(response), dtype=int)
    assignment[order] = np.arange(len(response)) % folds
    predicted = np.empty(len(response))
    for fold in range(folds):
        test, train = assignment == fold, assignment != fold
        inner = np.arange(train.sum()) % 3
        best, chosen = np.inf, strengths[0]
        for strength in strengths:
            error = 0.0
            for split in range(3):
                a, b = inner != split, inner == split
                left = design[train][a]
                weights = np.linalg.solve(
                    left.T @ left + strength * np.eye(design.shape[1]), left.T @ response[train][a]
                )
                error += float(np.sum((design[train][b] @ weights - response[train][b]) ** 2))
            if error < best:
                best, chosen = error, strength
        left = design[train]
        weights = np.linalg.solve(left.T @ left + chosen * np.eye(design.shape[1]), left.T @ response[train])
        predicted[test] = design[test] @ weights
    return predicted


def claim_ordering_channel(frame, geo) -> None:
    """1. Does moving code late help, and on which objectives?"""
    print("\n=== 1. ORDERING CHANNEL: coefficient on the signed code-late shift, at fixed aggregate ===")
    print(f"  {'objective':40s} {'n':>5s} {'signed b':>10s} {'t':>7s} {'even b':>9s} {'win@sep>.3':>11s}")
    components = sorted(c for c in frame.columns if c.startswith("eval/uncheatable_eval/") and c.endswith("/bpb"))
    columns = ["table9_macro_bpb", "uncheatable_bpb", *components]
    for column in columns:
        contrast, index = matched_contrasts(geo, frame[column].to_numpy(float))
        if len(contrast) < 50:
            continue
        signed, separation = geo["code_late"][index], geo["separation"][index]
        design = np.column_stack([np.ones(len(contrast)), signed, separation**2])
        weights, *_ = np.linalg.lstsq(design, contrast, rcond=None)
        residual = contrast - design @ weights
        variance = np.sum(residual**2) / (len(contrast) - design.shape[1])
        error = float(np.sqrt(variance * np.linalg.pinv(design.T @ design)[1, 1]))
        far = separation > 0.3
        print(
            f"  {column[:40]:40s} {len(contrast):5d} {weights[1]:+10.5f} {weights[1] / error:+7.1f} "
            f"{weights[2]:+9.4f} {(contrast[far] < 0).mean() if far.any() else float('nan'):10.0%}"
        )


def claim_selection_value(frame, geo) -> None:
    """2. How much does bucket-level structure buy over the family summary?"""
    print("\n=== 2. SELECTION VALUE: actual score of the top-10 policies chosen out-of-fold ===")
    families = geo["fit"].family_index
    for column, noise_column in (("table9_macro_bpb", "table9_macro_bpb"), ("uncheatable_bpb", "uncheatable_bpb")):
        contrast, index = matched_contrasts(geo, frame[column].to_numpy(float))
        noise = run_noise(frame, geo, noise_column)
        delta, separation = geo["delta"][index], geo["separation"][index]
        print(f"  --- {column} (n={len(contrast)}, run noise {noise:.5f})")
        designs = {
            "code-family share": delta[:, geo["code"]].sum(axis=1)[:, None],
            "3 family shares": np.column_stack([delta[:, families == f].sum(axis=1) for f in range(3)]),
            "full 39-bucket delta": delta,
        }
        for name, block in designs.items():
            design = np.column_stack([np.ones(len(contrast)), block, separation**2])
            predicted = out_of_fold(design, contrast)
            top = np.argsort(predicted)[:10]
            print(f"      {name:22s} {contrast[top].mean():+.5f} = {contrast[top].mean() / noise:+.1f} sigma")
        print(f"      {'random policy':22s} {contrast.mean():+.5f}")


def claim_oracle_is_not_a_ceiling(frame, geo, draws: int = 4000) -> None:
    """3. The best observed contrast is a minimum over noise, so it is not an achievable target."""
    print("\n=== 3. THE ORACLE IS NOT A CEILING: best observed, corrected for selection on noise ===")
    generator = np.random.default_rng(0)
    for column in ("table9_macro_bpb", "uncheatable_bpb"):
        contrast, _index = matched_contrasts(geo, frame[column].to_numpy(float))
        noise = run_noise(frame, geo, column) * np.sqrt(1.5)  # a contrast subtracts a tied mean
        pure = np.sort(generator.normal(0.0, noise, size=(draws, len(contrast))), axis=1)[:, :10].mean(axis=1)
        print(
            f"  {column:22s} observed {np.sort(contrast)[:10].mean():+.5f}  "
            f"from pure noise {pure.mean():+.5f}  net {np.sort(contrast)[:10].mean() - pure.mean():+.5f}"
        )


def claim_panel_never_crosses(geo) -> None:
    """5. The archive holds no state fixed while varying the continuation."""
    print("\n=== 5. THE PANEL NEVER CROSSES PHASES ===")
    for label, held, varied in (
        ("phase-0 fixed", geo["phase_0"], geo["phase_1"]),
        ("phase-1 fixed", geo["phase_1"], geo["phase_0"]),
    ):
        groups = collections.defaultdict(list)
        for index, row in enumerate(np.round(held, 8)):
            groups[tuple(row)].append(index)
        crossing = [
            members
            for members in groups.values()
            if len(members) > 1 and len({tuple(np.round(varied[i], 8)) for i in members}) > 1
        ]
        print(f"  {label:16s} {len(groups):5d} distinct mixtures, {len(crossing):3d} with the other phase varying")


def claim_readout_is_sufficient(frame, geo) -> None:
    """4. Does the 83% readout carry what the full phase-0 policy carries, for predicting the endpoint?

    The stronger claim -- that a readout at 83% equals one at 100% -- is NOT measurable here and is not
    asserted. On the atomic panel the readout position takes exactly four values, one per horizon, so
    position and horizon are perfectly confounded; on this panel 1841 of 1847 runs share one position.
    Neither supports a position contrast.

    What is measurable, and is what the design actually leans on, is whether the scalar readout is a
    LOSSY summary of the phase-0 policy. If adding the 39 phase-0 weights on top of the readout does not
    improve out-of-fold prediction of the endpoint, the readout has already extracted the phase-0
    information the endpoint responds to, and a readout taken later could add little. If the weights DO
    add a lot, the scalar is lossy and moving it later might genuinely help -- which would be an argument
    for rerunning the swarm rather than for the prefix search.
    """
    print("\n=== 4. IS THE 83% READOUT A SUFFICIENT SUMMARY OF THE PHASE-0 POLICY? ===")
    joined = with_phase0_readout(frame)
    readout = joined["readout_phase0_uncheatable_bpb"].to_numpy(float)
    position = joined["readout_phase0_fraction"].to_numpy(float)
    print(
        f"  readout position: median {np.nanmedian(position):.3f}, "
        f"{np.mean(position >= 0.83 - 1e-9):.0%} of rows at 0.833, {int(np.sum(position < 0.8))} earlier"
    )
    for column in ("table9_macro_bpb", "uncheatable_bpb"):
        response = joined[column].to_numpy(float)
        usable = np.isfinite(response) & np.isfinite(readout)
        target = response[usable]
        centred = target - target.mean()
        blocks = {
            "readout alone": readout[usable, None],
            "39 phase-0 weights": geo["phase_0"][usable],
            "both": np.column_stack([readout[usable, None], geo["phase_0"][usable]]),
        }
        print(f"  --- {column} (n={usable.sum()})")
        for name, block in blocks.items():
            design = np.column_stack([np.ones(usable.sum()), block])
            predicted = out_of_fold(design, target)
            score = 1.0 - np.sum((target - predicted) ** 2) / np.sum(centred**2)
            print(f"      {name:22s} out-of-fold R2 {score:+.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims", default="1,2,3,4,5")
    args = parser.parse_args()
    frame, dropped = panel()
    geo = geometry(frame)
    print(f"delphi 3e18 heldout: {len(frame)} rows after dropping {dropped} contaminated or incomplete")
    wanted = set(args.claims.split(","))
    if "1" in wanted:
        claim_ordering_channel(frame, geo)
    if "2" in wanted:
        claim_selection_value(frame, geo)
    if "3" in wanted:
        claim_oracle_is_not_a_ceiling(frame, geo)
    if "4" in wanted:
        claim_readout_is_sufficient(frame, geo)
    if "5" in wanted:
        claim_panel_never_crosses(geo)


if __name__ == "__main__":
    main()
