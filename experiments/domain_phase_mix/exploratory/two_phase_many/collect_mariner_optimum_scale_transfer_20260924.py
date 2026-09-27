# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["fsspec", "gcsfs", "pandas"]
# ///

"""Collect the deployed-optimum runs of the scale-transfer figure into `deployed_optima.csv`.

The four Uncheatable optima of the paper's Table 5 (MARINER, matched Olmix, tuned and released RegMix) were
trained at Llama 160M/1.2B and 200M/6B by `launch_mariner_optimum_scale_transfer.py` (Iris parent
`/calvinxu/dm-mopt-scale-transfer-20260924-retry3`). For every run whose executor status is SUCCESS, the final
line of `checkpoints/eval_metrics.jsonl` gives the byte-weighted Uncheatable BPB (`eval/uncheatable_eval/bpb`),
the metric the swarm panels of `plot_scale_transfer_results_figure_20260905.py` use. The Qwen3 360M/1.6B values
are the measured Table 5 entries (seed means for MARINER and Olmix). Runs not yet landed leave NaN, and the
figure draws a pair only when both of its settings are measured.
"""

import argparse
import json
import logging
from pathlib import Path

import fsspec
import pandas as pd

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "mariner_optimum_scale_transfer_20260924"
TRAINING_ROOT = "gs://marin-us-east5/checkpoints/pinlin_calvin_xu/data_mixture"
UNCHEATABLE_KEY = "eval/uncheatable_eval/bpb"
SCALES = {"60m_1p2b": ("bpb_60m", 4577), "300m_6b": ("bpb_300m", 22888)}
RUN_LABELS = {
    "mariner_u": "MARINER",
    "olmix_u": "Olmix",
    "regmix_tun_u": "RegMix (tuned)",
    "regmix_rel_u": "RegMix (released)",
}
# Paper Table 5, Uncheatable column (results.tex); MARINER and Olmix are means over training seeds.
QWEN3_3E18_BPB = {"mariner_u": 0.9825, "olmix_u": 1.0022, "regmix_tun_u": 1.0003, "regmix_rel_u": 1.0271}


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    """The deployed table (one row per mixture) and a provenance table (one row per run)."""
    fs = fsspec.filesystem("gs")
    provenance: list[dict[str, object]] = []
    for scale, (column, expected_steps) in SCALES.items():
        root = f"{TRAINING_ROOT}/mopt_{scale}"
        run_dirs = [d for d in fs.ls(root)] if fs.exists(root) else []
        for run_name in RUN_LABELS:
            matches = [d for d in run_dirs if d.rstrip("/").split("/")[-1].startswith(f"{run_name}-")]
            if len(matches) > 1:
                raise ValueError(f"{root}: several directories for {run_name}: {matches}")
            row: dict[str, object] = {"run_name": run_name, "scale": scale, "column": column, "status": "missing"}
            if matches:
                run_dir = matches[0].rstrip("/")
                status_path = f"{run_dir}/.executor_status"
                status = fs.cat(status_path).decode().strip() if fs.exists(status_path) else "no status"
                eval_path = f"{run_dir}/checkpoints/eval_metrics.jsonl"
                row["run_dir"] = f"gs://{run_dir}" if not run_dir.startswith("gs://") else run_dir
                row["status"] = f"training:{status}"
                if status == "SUCCESS" and fs.exists(eval_path):
                    final = json.loads(fs.cat(eval_path).decode().strip().splitlines()[-1])
                    # Levanter's last in-training eval is logged at step num_train_steps - 1.
                    if int(final["step"]) != expected_steps - 1:
                        raise ValueError(f"{run_dir}: final eval step {final['step']} != {expected_steps - 1}")
                    row["status"] = "measured"
                    row["final_step"] = int(final["step"])
                    row["uncheatable_bpb"] = float(final[UNCHEATABLE_KEY])
                    row["eval_metrics_uri"] = eval_path
            provenance.append(row)
    provenance_frame = pd.DataFrame(provenance)
    deployed = pd.DataFrame({"run_name": list(RUN_LABELS), "label": list(RUN_LABELS.values())})
    for _, (column, _) in SCALES.items():
        measured = provenance_frame.loc[
            provenance_frame["column"].eq(column) & provenance_frame["status"].eq("measured")
        ]
        values = measured.set_index("run_name")["uncheatable_bpb"] if not measured.empty else pd.Series(dtype=float)
        deployed[column] = deployed["run_name"].map(values).astype(float)
    deployed["bpb_3e18"] = deployed["run_name"].map(QWEN3_3E18_BPB)
    return deployed, provenance_frame


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    deployed, provenance = collect()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    provenance.to_csv(args.output_dir / "deployed_optima_provenance.csv", index=False)
    measured = provenance["status"].eq("measured").sum()
    if measured == 0:
        logger.info("No run measured yet; deployed_optima.csv not written")
    else:
        deployed.to_csv(args.output_dir / "deployed_optima.csv", index=False)
        logger.info("Wrote %s with %d measured runs", args.output_dir / "deployed_optima.csv", measured)
    print(
        provenance[
            ["run_name", "scale", "status"] + [c for c in ("final_step", "uncheatable_bpb") if c in provenance]
        ].to_string(index=False)
    )
    print(deployed.to_string(index=False))


if __name__ == "__main__":
    main()
