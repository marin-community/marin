# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "pandas"]
# ///
"""Materialize the completed observed-prefix rows for branch-model development.

These 21 fit rows and four proportional repeats are an implementation and
hyperparameter-development panel. They are not used to estimate the KL0.05
branch optimum or to report its selection performance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import fsspec
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    materialize_delphi_phase1_common_branches_20260824 as base,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DESIGN_MANIFEST = (
    SCRIPT_DIR / "reference_outputs" / "delphi_phase1_common_branches_20260824" / "launch_dry_run" / "manifest.json"
)
DESIGN_MANIFEST_SHA256 = "7baf0bf5070195c4d210ae748121fd77bd919c2dde7780592c2b84a91a5e0453"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase1_observed_prefix_development_20260825"
EXPERIMENT_ROOT = "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_3e18_phase1_common_branches_v6e8_20260825"
EXPERIMENT_NAME = "pinlin_calvin_xu/data_mixture/delphi_3e18_phase1_common_branches_v6e8_20260825"
CANDIDATE_SHA256 = "fef07d4188ef05f4df4a43d1eda6a12f7d2daf69a1ae1eb777863fd20db732b6"
CONTINUATION_SHA256 = "9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355"
PREFIX_REPLAY_CODE_COMMIT = "2659c1bf8e7dbb0830b4476bb763a90a35d71837"
BRANCH_CODE_COMMIT = "d016caa0fbd0f1f50e29ffa0c9dea5d40f5438e2"
CONTINUATION_HARDWARE = base.TpuHardware(tpu_type="v6e-8", region="us-east5", zone="us-east5-b")
TARGET_PREFIX = "observed_cap10_best"
FIT_ORDERS = tuple(range(21))
NOISE_ORDERS = (228, 229, 230, 231)


def load_design_rows() -> list[dict[str, object]]:
    manifest_bytes = DESIGN_MANIFEST.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != DESIGN_MANIFEST_SHA256:
        raise ValueError("Frozen branch design manifest changed")
    manifest = json.loads(manifest_bytes)
    if manifest.get("candidate_weights_sha256") != CANDIDATE_SHA256:
        raise ValueError("Candidate weights changed")
    if manifest.get("continuation_weights_sha256") != CONTINUATION_SHA256:
        raise ValueError("Continuation weights changed")
    rows = manifest.get("branch_rows")
    if not isinstance(rows, list):
        raise ValueError("Frozen branch rows are malformed")
    selected = [row for row in rows if isinstance(row, dict) and int(row["run_order"]) in (*FIT_ORDERS, *NOISE_ORDERS)]
    if [int(row["run_order"]) for row in selected] != [*FIT_ORDERS, *NOISE_ORDERS]:
        raise ValueError("Observed-prefix development row identities changed")
    if any(row["prefix"]["candidate_id"] != TARGET_PREFIX or int(row["prefix"]["repeat_seed"]) != 0 for row in selected):
        raise ValueError("Development rows do not share the observed seed-0 prefix")
    if any(bool(row["fit_budget"]) != (int(row["run_order"]) in FIT_ORDERS) for row in selected):
        raise ValueError("Development fit-budget labels changed")
    return selected


def main() -> None:
    fs, root = fsspec.core.url_to_fs(EXPERIMENT_ROOT)
    results = []
    metrics = []
    for design_row in load_design_rows():
        row, row_metrics = base.materialize_design_row(
            fs,
            root,
            design_row,
            candidate_sha256=CANDIDATE_SHA256,
            continuation_sha256=CONTINUATION_SHA256,
            prefix_replay_code_commit=PREFIX_REPLAY_CODE_COMMIT,
            branch_code_commit=BRANCH_CODE_COMMIT,
            expected_experiment_name=EXPERIMENT_NAME,
            continuation_hardware=CONTINUATION_HARDWARE,
        )
        results.append(row)
        metrics.extend(row_metrics)

    frame = pd.DataFrame(results).sort_values("run_order").reset_index(drop=True)
    fit = frame[frame.fit_budget]
    noise = frame[frame.branch_role.eq("same_prefix_branch_noise")]
    if len(frame) != 25 or len(fit) != 21 or len(noise) != 4:
        raise ValueError("Development coverage changed")
    phase_columns = [column for column in frame if column.startswith(("phase_0_", "phase_1_"))]
    if noise.data_seed.nunique() != 4 or noise.trainer_seed.nunique() != 1:
        raise ValueError("Noise-control seeds changed")
    if any(noise[column].nunique() != 1 for column in phase_columns):
        raise ValueError("Noise controls do not repeat one policy")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "branch_results.csv", index=False)
    fit.to_csv(OUTPUT_DIR / "branch_fit_matrix.csv", index=False)
    pd.DataFrame(metrics).sort_values(["run_name", "metric"]).to_csv(
        OUTPUT_DIR / "uncheatable_metrics_long.csv", index=False
    )
    noise_summary = {
        "uncheatable_bpb_mean": float(noise.uncheatable_bpb.mean()),
        "uncheatable_bpb_sample_sd": float(noise.uncheatable_bpb.std(ddof=1)),
        "github_cpp_bpb_mean": float(noise.github_cpp_bpb.mean()),
        "github_cpp_bpb_sample_sd": float(noise.github_cpp_bpb.std(ddof=1)),
    }
    coverage = {
        "purpose": "implementation_and_hyperparameter_development_only",
        "design_manifest": str(DESIGN_MANIFEST),
        "design_manifest_sha256": DESIGN_MANIFEST_SHA256,
        "candidate_weights_sha256": CANDIDATE_SHA256,
        "continuation_weights_sha256": CONTINUATION_SHA256,
        "prefix_replay_code_commit": PREFIX_REPLAY_CODE_COMMIT,
        "branch_code_commit": BRANCH_CODE_COMMIT,
        "continuation_hardware": asdict(CONTINUATION_HARDWARE),
        "result_rows": len(frame),
        "fit_rows": len(fit),
        "noise_rows": len(noise),
        "noise": noise_summary,
    }
    (OUTPUT_DIR / "coverage.json").write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n")
    print(json.dumps(coverage, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
