# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Collect a scaling ladder from GCS: MARINER's (`launch_delphi_frozen_procedure_scaling.py`) or the matched Olmix
policies' (`launch_delphi_matched_olmix_scaling.py`), which train at the same seeds, sizes and hardware.

For every rung that has finished training, reads the final `eval_metrics.jsonl` (byte-weighted Uncheatable BPB and its
components) from the run's checkpoint directory and the native Table-9 evaluator's `olmo_base_eval_table9_results.json`
from the evaluation step's output, then writes `measured_results.csv` beside the launch manifests. Rungs still training
are listed with status `training:<state>` so the figure and table builders skip them.

usage: uv run --offline --no-sync python collect_delphi_frozen_procedure_scaling_20260908.py [--ladder matched_olmix]
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path

import fsspec
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_ROOT = "gs://marin-us-east5/evaluation/olmo_base_eval_table9"
SCALES = ("2e19", "3e20", "1e21")
RUN_PATTERN = re.compile(r"^(?P<policy>[a-z0-9_]+)_(?P<scale>[0-9]e[0-9]{2})_seed(?P<seed>[0-9]+)-[0-9a-f]{6}$")
UNCHEATABLE_KEY = "eval/uncheatable_eval/bpb"


@dataclass(frozen=True)
class Ladder:
    """One scaling launch: where its runs train, which policies it trains, and how its evaluations are named."""

    output_dir: Path
    training_root: str
    policies: dict[str, str]  # policy candidate id -> target
    eval_stem: dict[str, str]  # target -> evaluation name stem before `_<scale>_s<seed>`


LADDERS = {
    "mariner": Ladder(
        SCRIPT_DIR / "reference_outputs" / "delphi_frozen_procedure_scaling_v6e_20260908",
        "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_frozen_procedure_scaling_v6e_20260908",
        {"lwspu_u_snc_cap06": "uncheatable", "lwspu_t9_snc_cap08": "table9"},
        {"uncheatable": "t9_fpu", "table9": "t9_fpt"},
    ),
    "matched_olmix": Ladder(
        SCRIPT_DIR / "reference_outputs" / "delphi_matched_olmix_scaling_v6e_20260910",
        "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_matched_olmix_scaling_v6e_20260910",
        {"olmixq_u_kl0p05_cap04": "uncheatable", "olmixq_t9_kl0p005_cap04": "table9"},
        {"uncheatable": "t9_mou", "table9": "t9_mot"},
    ),
}


def collect(ladder: Ladder) -> pd.DataFrame:
    fs = fsspec.filesystem("gs")
    rows = []
    for run_dir in sorted(fs.ls(ladder.training_root)):
        name = run_dir.rstrip("/").split("/")[-1]
        match = RUN_PATTERN.match(name)
        if match is None or match.group("policy") not in ladder.policies:
            continue
        policy, scale, seed = match.group("policy"), match.group("scale"), int(match.group("seed"))
        target = ladder.policies[policy]
        row: dict[str, object] = {
            "run_name": name,
            "policy": policy,
            "target": target,
            "target_flops": float(scale),
            "data_seed": seed,
            "status": "pending",
        }
        status_path = f"{run_dir}/.executor_status"
        status = fs.cat(status_path).decode().strip() if fs.exists(status_path) else "missing"
        eval_path = f"{run_dir}/checkpoints/eval_metrics.jsonl"
        if status == "SUCCESS" and fs.exists(eval_path):
            final = json.loads(fs.cat(eval_path).decode().strip().splitlines()[-1])
            row["status"] = "measured"
            row["final_step"] = int(final["step"])
            row["uncheatable_bpb"] = float(final[UNCHEATABLE_KEY])
            row["uncheatable_macro_bpb"] = float(final["eval/uncheatable_eval/macro_bpb"])
            for key, value in final.items():
                component = re.fullmatch(r"eval/uncheatable_eval/([a-z0-9_]+)/bpb", key)
                if component:
                    row[f"uncheatable_{component.group(1)}_bpb"] = float(value)
            row["eval_metrics_uri"] = eval_path
        else:
            row["status"] = f"training:{status}"
        prefix = f"{ladder.eval_stem[target]}_{scale}_s{seed}-"
        eval_dirs = [d for d in fs.ls(EVAL_ROOT) if d.rstrip("/").split("/")[-1].startswith(prefix)]
        for eval_dir in eval_dirs:
            result_path = f"{eval_dir}/olmo_base_eval_table9_results.json"
            if fs.exists(result_path):
                row["table9_macro_bpb"] = float(json.loads(fs.cat(result_path).decode())["table9_macro_bpb"])
                row["table9_results_uri"] = result_path
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["target", "target_flops"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ladder", choices=tuple(LADDERS), default="mariner")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to the ladder's reference directory")
    args = parser.parse_args()
    ladder = LADDERS[args.ladder]
    output_dir = args.output_dir or ladder.output_dir
    frame = collect(ladder)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "measured_results.csv", index=False)
    shown = ("run_name", "target", "target_flops", "status", "uncheatable_bpb", "table9_macro_bpb")
    columns = [c for c in shown if c in frame]
    print(frame[columns].to_string())


if __name__ == "__main__":
    main()
