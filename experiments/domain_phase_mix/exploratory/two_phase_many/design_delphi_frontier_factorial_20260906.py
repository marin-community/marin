# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A resolution-V two-level factorial around the replicated Table-9 frontier centre (design only, no launch).

Five factors move mass between the frontier centre's key sources and the 26 Common Crawl cells: code
(Stack-Edu and Stack-Edu FIM together), synthetic QA, Dolmino Common Crawl HQ, synthetic reasoning
(thinking, instruction, math together) and PDFs/arXiv (olmOCR and arXiv together). The 2^(5-1) design with
E = ABCD estimates every main effect and every two-factor interaction without aliasing between them; with the
Table-9 repeat SD of 0.0038 each effect has a standard error of about 0.0019 BPB. Two centre replicates at the
same data seed give a pure-error check. Weights are rounded to the 2048-count runtime grid and checked against
the swarm's 16-epoch range (the centre itself repeats Wikipedia 11.8 times). The output table uses the launcher
schema of the epoch-cap sweeps.

usage: uv run python design_delphi_frontier_factorial_20260906.py [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_delphi_selection_20260906 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    materialize_delphi_one_phase_surrogate_challengers_20260831 as grid,
)

FROZEN = benchmark.DEFAULT_OUTPUT
DEFAULT_OUTPUT = SCRIPT_DIR / "reference_outputs" / "delphi_frontier_factorial_design_20260906"
# The 26-run replicated Table-9 frontier centre (measured mean 1.0639, SD 0.0041) in the frozen bank.
CENTRE_ID = "delphi_3e18_39bucket:a1a917b1981fc2cad2c2759dccf963b51fb941fd9816c49087f7b119545d4161"
CC_PREFIX = "dolma3_cc/"
# The centre already repeats synthetic math 7.9 times and the STEM crawl 7.7 times, so the design's rows are
# checked against 16 epochs (the swarm's range) rather than the 8-epoch policy cap of the validations.
EPOCH_CAP = 16.0
BLOCK_SIZE = grid.MIXTURE_BLOCK_SIZE
# Factor name -> (bucket weight deltas at the +1 level); the -1 level is the negative. Mass is balanced
# against the Common Crawl cells in proportion to their centre weights.
FACTORS: dict[str, dict[str, float]] = {
    "code": {"dolma3_stack_edu": 0.02, "dolmino_stack_edu_fim": 0.02},
    "synth_qa": {"dolmino_synth_qa": 0.04},
    "cc_hq": {"dolmino_common_crawl_hq": 0.04},
    "synth_reasoning": {
        "dolmino_synth_thinking": 0.006,
        "dolmino_synth_instruction": 0.006,
        "dolmino_synth_math": 0.006,
    },
    "pdf_arxiv": {"dolmino_olmocr_pdfs_hq": 0.015, "dolma3_arxiv": 0.005},
}
FACTOR_LETTERS = "ABCDE"
CENTRE_REPLICATES = 2


def half_fraction(levels: int) -> list[tuple[int, ...]]:
    """2^(k-1) design with the last factor equal to the product of the others (resolution V for k = 5)."""
    rows = []
    for signs in itertools.product((-1, 1), repeat=levels - 1):
        rows.append((*signs, int(np.prod(signs))))
    return rows


def perturb(centre: pd.Series, signs: tuple[int, ...]) -> pd.Series:
    weights = centre.copy()
    moved = 0.0
    for sign, deltas in zip(signs, FACTORS.values(), strict=True):
        for bucket, delta in deltas.items():
            weights[bucket] += sign * delta
            moved += sign * delta
    cc = [bucket for bucket in weights.index if bucket.startswith(CC_PREFIX)]
    share = centre[cc] / centre[cc].sum()
    weights[cc] = weights[cc] - moved * share
    if (weights < 0).any():
        raise ValueError(f"Negative weight for signs {signs}: {weights[weights < 0].to_dict()}")
    if abs(weights.sum() - 1) > 1e-9:
        raise ValueError("Perturbed weights do not sum to one")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = benchmark.read_npz(FROZEN / "inputs" / "panel.npz")
    buckets = [str(b) for b in data["buckets"]]
    inventory = pd.Series(data["inventory"], index=buckets)
    bank = benchmark.read_npz(FROZEN / "inputs" / "table9_bank_features.npz")
    labels = pd.read_csv(FROZEN / "inputs" / "table9_bank_labels.csv").set_index("coordinate_id")
    frame = pd.DataFrame(bank["weights"], columns=buckets, index=bank["coordinate_id"].astype(str))
    centre = frame.loc[CENTRE_ID]
    centre_row = labels.loc[CENTRE_ID]
    print("centre weights of the factor buckets (weight, epochs):")
    for deltas in FACTORS.values():
        for bucket in deltas:
            print(f"  {bucket:32s} {centre[bucket]:.4f}  {centre[bucket] * inventory[bucket]:.2f}")
    maximum = np.floor(np.minimum(1.0, EPOCH_CAP / inventory.to_numpy()) * BLOCK_SIZE + 1e-12).astype(np.int64)
    rows, table = [], []
    # Candidate ids end with the cap suffix the sweep loader requires.
    suffix = f"_cap{int(EPOCH_CAP):02d}"
    designs = [((0,) * len(FACTORS), f"centre_r{i}{suffix}") for i in range(CENTRE_REPLICATES)]
    designs += [
        (signs, "fac_" + "".join("p" if s > 0 else "m" for s in signs) + suffix) for signs in half_fraction(len(FACTORS))
    ]
    for signs, name in designs:
        weights = perturb(centre, signs) if any(signs) else centre.copy()
        counts = grid.prefix_materializer.constrained_counts(weights.to_numpy(), maximum)
        if int(counts.sum()) != BLOCK_SIZE or np.any(counts > maximum):
            raise ValueError(f"{name}: runtime counts violate the grid or the cap")
        runtime = counts / BLOCK_SIZE
        epochs = runtime * inventory.to_numpy()
        if not any(signs) and np.abs(runtime - weights.to_numpy()).sum() / 2 > 0.002:
            moved = pd.Series(runtime - weights.to_numpy(), index=buckets)
            print("centre rounding moved:", moved[moved.abs() > 1e-4].round(4).to_dict())
        table.append(
            {
                "candidate_id": name,
                **{
                    f"factor_{letter}_{factor}": sign
                    for letter, factor, sign in zip(FACTOR_LETTERS, FACTORS, signs, strict=True)
                },
                "tv_to_centre": float(np.abs(runtime - centre.to_numpy()).sum() / 2),
                "max_materialized_epoch": float(epochs.max()),
                "max_epoch_bucket": buckets[int(np.argmax(epochs))],
            }
        )
        for bucket, count, epoch in zip(buckets, counts, epochs, strict=True):
            rows.append(
                {
                    "candidate_id": name,
                    "target": "table9",
                    "target_label": "Table-9 macro",
                    "epoch_cap": int(EPOCH_CAP),
                    "domain": bucket,
                    "runtime_count": int(count),
                    "weight": float(count / BLOCK_SIZE),
                    "materialized_epochs": float(epoch),
                }
            )
    design = pd.DataFrame(table)
    design.to_csv(args.output_dir / "design.csv", index=False)
    # The sweep loader aliases identical mixtures, so the second centre replicate (same mixture, different
    # trainer seed) gets its own table.
    weights_table = pd.DataFrame(rows)
    replicate_id = f"centre_r1{suffix}"
    weights_table[weights_table.candidate_id.ne(replicate_id)].to_csv(
        args.output_dir / "candidate_weights.csv", index=False
    )
    weights_table[weights_table.candidate_id.eq(replicate_id)].to_csv(
        args.output_dir / "candidate_weights_replicate.csv", index=False
    )
    aliases = {
        "A*B": "C*D*E",
        "resolution": (
            "V (E = ABCD): main effects and all ten two-factor interactions are estimable and unaliased with each other"
        ),
    }
    factor_lines = "\n".join(
        f"- {letter} `{factor}`: {json.dumps(deltas)}"
        for letter, (factor, deltas) in zip(FACTOR_LETTERS, FACTORS.items(), strict=True)
    )
    worst = design.loc[design.max_materialized_epoch.idxmax()]
    readme = [
        "# Frontier factorial design (not launched)",
        "",
        f"Centre: `{CENTRE_ID}` ({int(centre_row.run_count)} runs, measured mean {centre_row.measured_mean_bpb:.4f}).",
        "",
        "Factors and +1-level weight deltas (mass balanced against the 26 Common Crawl cells in proportion",
        "to their centre weights):",
        "",
        factor_lines,
        "",
        f"Design: 2^(5-1) half fraction, {aliases['resolution']}, plus {CENTRE_REPLICATES} centre replicates;",
        f"{len(design)} runs at 3e18 (about {len(design) * 3.4:.0f}e18 FLOPs). Effect standard error at the Table-9",
        f"repeat SD of 0.0038: about {0.0038 * 0.5:.4f} BPB for main effects and interactions.",
        "",
        f"Largest materialized epochs: {worst.max_materialized_epoch:.2f} ({worst.max_epoch_bucket}); largest TV from",
        f"the centre: {design.tv_to_centre.max():.3f}.",
        "",
        "The candidate table is in the epoch-cap sweep launcher schema; a launcher would bind all rows to one data",
        "seed (662009, the Table-9 validation seed) so that the centre replicates measure seed-free noise.",
        "",
    ]
    (args.output_dir / "README.md").write_text("\n".join(readme))
    pd.set_option("display.width", 220)
    print(design.round(4).to_string(index=False))
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
