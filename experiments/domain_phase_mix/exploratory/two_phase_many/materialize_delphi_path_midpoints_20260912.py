# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Freeze Figure 10 midpoint mixtures on the endpoint runs' sampler grid.

Run as a module with uv from the Marin checkout. No models are fitted and no
training is submitted. Existing outputs must match byte for byte on rerun.
"""

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_comparator_proposals_20260909 as proposals,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    plot_comparator_proposal_diagnostics_20260909 as paths,
)

OUTPUT = proposals.SCRIPT_DIR / "reference_outputs/delphi_path_midpoints_3e18_20260912"
BASELINES = ("olmix", "quad", "spline", "lgbm", "krr")
TARGETS = (("uncheatable", "u", "Uncheatable", 666200), ("table9", "t9", "OlmoBaseEval Easy", 662009))
BLOCK_SIZE = 2048
METADATA_CAP = 64


def write_frozen(path: Path, content: str) -> None:
    """Create a frozen artifact, refusing to replace a different existing value."""
    encoded = content.encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"Frozen artifact differs: {path}")
        return
    path.write_bytes(encoded)


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def main() -> None:
    bucket_table = pd.read_csv(proposals.ms.DATA / "buckets.csv")
    buckets = tuple(bucket_table.bucket)
    assert len(buckets) == 39 and len(set(buckets)) == 39
    assert buckets == tuple(sorted(buckets)), "Alphabetical bucket order breaks rounding ties"
    inventory = bucket_table.epochs_per_unit_weight.to_numpy(float)
    endpoints = paths.proposal_weights(buckets)
    weights_rows, exact_rows, records = [], [], []
    coordinates = set()
    for target, target_key, target_label, data_seed in TARGETS:
        for baseline in BASELINES:
            start = endpoints[target]["mariner"]
            end = endpoints[target][baseline]
            for endpoint in (start, end):
                assert np.all(endpoint >= 0) and endpoint.sum() == 1
                assert np.all(endpoint * BLOCK_SIZE == np.rint(endpoint * BLOCK_SIZE))
            exact = (start + end) / 2
            # Largest residual first; equal residuals follow alphabetical bucket
            # order. No model prediction is used to choose among rounding ties.
            counts = proposals.ms.constrained_counts(exact, np.full(len(buckets), BLOCK_SIZE))
            runtime = counts / BLOCK_SIZE
            assert counts.sum() == BLOCK_SIZE and np.all(counts >= 0)
            coordinate = tuple(int(count) for count in counts)
            assert coordinate not in coordinates, "Duplicate training mixture"
            coordinates.add(coordinate)
            epochs = runtime * inventory
            assert epochs.max() < METADATA_CAP
            candidate_id = f"mp50_{target_key}_{baseline}_cap{METADATA_CAP:02d}"
            for i, bucket in enumerate(buckets):
                weights_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "target": target,
                        "target_label": target_label,
                        "epoch_cap": METADATA_CAP,
                        "surrogate": baseline,
                        "domain": bucket,
                        "runtime_count": int(counts[i]),
                        "weight": float(runtime[i]),
                        "materialized_epochs": float(epochs[i]),
                    }
                )
                exact_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "target": target,
                        "baseline": baseline,
                        "domain": bucket,
                        "mariner_weight": float(start[i]),
                        "baseline_weight": float(end[i]),
                        "exact_weight": float(exact[i]),
                        "runtime_weight": float(runtime[i]),
                    }
                )
            records.append(
                {
                    "candidate_id": candidate_id,
                    "target": target,
                    "baseline": baseline,
                    "position": 0.5,
                    "data_seed": data_seed,
                    "trainer_seed": 0,
                    "rounding_tv": float(np.abs(runtime - exact).sum() / 2),
                    "max_coordinate_rounding_error": float(np.abs(runtime - exact).max()),
                    "max_materialized_epochs": float(epochs.max()),
                    "active_buckets": int(np.count_nonzero(counts)),
                    "mariner_active_buckets": int(np.count_nonzero(start)),
                    "baseline_active_buckets": int(np.count_nonzero(end)),
                }
            )
    sources = [
        Path(__file__),
        Path(paths.__file__),
        Path(proposals.ms.__file__),
        proposals.ms.DATA / "buckets.csv",
        proposals.ms.DATA / "reference_policies.csv",
        paths.PROPOSAL_DIR / "solutions.csv",
        paths.OLMIX_DIR / "solutions.csv",
    ]
    manifest = {
        "sampler_block_size": BLOCK_SIZE,
        "rounding": "Neutral largest remainder, ties in alphabetical bucket order; no objective-based refinement",
        "epoch_cap_metadata": "64 is an inactive loader bound; no cap is optimized or applied",
        "source_hashes": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        "runs": records,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_frozen(OUTPUT / "candidate_weights.csv", csv_text(weights_rows))
    write_frozen(OUTPUT / "exact_midpoints.csv", csv_text(exact_rows))
    write_frozen(OUTPUT / "midpoint_plan.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "runs": len(records),
                "candidate_sha256": hashlib.sha256((OUTPUT / "candidate_weights.csv").read_bytes()).hexdigest(),
                "max_rounding_tv": max(record["rounding_tv"] for record in records),
            }
        )
    )


if __name__ == "__main__":
    main()
