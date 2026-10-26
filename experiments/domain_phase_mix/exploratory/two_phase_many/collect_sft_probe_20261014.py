# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Collect the SFT midtraining probe: every fine-tune's final evals, the grid's LR choice, and the arms' Table-9.

Reads ``results.json`` under gs://marin-us-east5/sft_midtraining_probe_20261014/<panel>/... (one per fine-tune) and
the native Table-9 results of the fifteen new arm runs, joins the base-model metrics from the checkpoint inventory,
and writes under ``reference_outputs/sft_midtraining_probe_20261014/``:

- ``sft_results.csv``: one row per fine-tune (panel, base run, arm/design, policy class, seeds, set, lr, sft seed,
  final validation loss, Uncheatable BPB with the paper's fixed weights, Paloma C4 BPB, base-model BPBs);
- ``grid_selection.json``: the peak learning rate per set (lowest final validation loss on the proportional
  checkpoint at SFT seed 0) and the full grid table ``grid_results.csv``;
- ``arms_table9.csv``: Table-9 macro BPB of every arm run with the mean per arm and the pre-registered B choice.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import fsspec
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from uncheatable_objective import fixed_uncheatable_bpb  # noqa: E402

logger = logging.getLogger(__name__)

ROOT = "gs://marin-us-east5/sft_midtraining_probe_20261014"
TABLE9_ROOT = "gs://marin-us-east5/evaluation/olmo_base_eval_table9"
INVENTORY = SCRIPT_DIR.parents[3] / ".agents/projects/sft_midtraining_probe/checkpoint_inventory.csv"
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "sft_midtraining_probe_20261014"
ARMS_PANEL = OUTPUT_DIR / "arms_source_panel.csv"
DATASETS = ("gsm8k", "sciq", "pycode", "norobots")
ARM_A = "lwspu_t9_snc_cap08"
FINAL_PRETRAIN_STEP = 3006


def _read_json(fs, path: str) -> dict:
    with fs.open(path, "r") as handle:
        return json.load(handle)


def collect_runs(fs) -> pd.DataFrame:
    paths = [
        p
        for p in fs.glob(f"{ROOT.removeprefix('gs://')}/**/results.json")
        if "/tokenized/" not in p and "/data/" not in p
    ]
    with ThreadPoolExecutor(32) as pool:
        payloads = list(pool.map(lambda p: (p, _read_json(fs, p)), paths))
    rows = []
    for path, payload in payloads:
        if "run" not in payload:  # a bundle summary, not a fine-tune
            continue
        run, base, final = payload["run"], payload["base"], payload["final"]
        uncheatable = (
            fixed_uncheatable_bpb(final)
            if all(f"eval/uncheatable_eval/{c}/bpb" in final for c in ("ao3_english", "wikipedia_english"))
            else float("nan")
        )
        rows.append(
            {
                "panel": base["panel"],
                "base_run_name": base["run_name"],
                "arm": base.get("arm", ""),
                "design_id": base.get("design_id", ""),
                "policy_class": base["policy_class"],
                "data_seed": base["data_seed"],
                "trainer_seed": base["trainer_seed"],
                "dataset": run["dataset"],
                "learning_rate": run["learning_rate"],
                "sft_seed": run["seed"],
                "num_train_steps": run["num_train_steps"],
                "final_step": final.get("step"),
                "sft_val_loss": final.get(f"eval/{run['dataset']}/loss"),
                "uncheatable_bpb": uncheatable,
                "c4_bpb": final.get("eval/paloma/c4_en/bpb"),
                "result_path": "gs://" + path,
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    inventory = pd.read_csv(INVENTORY)[
        [
            "run_name",
            "objective_bpb_table9",
            "objective_bpb_uncheatable",
            "phase_tv",
            "contrast_family",
            "anchor_or_design_id",
            "seed_block",
        ]
    ]
    inventory = inventory.rename(
        columns={
            "run_name": "base_run_name",
            "objective_bpb_table9": "base_table9_bpb",
            "objective_bpb_uncheatable": "base_uncheatable_bpb",
        }
    )
    return frame.merge(inventory, on="base_run_name", how="left").sort_values(
        ["panel", "base_run_name", "dataset", "learning_rate", "sft_seed"]
    )


def grid_selection(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """The learning rate per set from the first complete grid panel (the grid may be raced on several pools)."""
    grid = frame[frame["panel"].str.startswith("grid")].copy()
    choice: dict[str, float] = {}
    complete = [
        panel
        for panel, group in grid.groupby("panel")
        if group["dataset"].nunique() == len(DATASETS) and len(group) >= 5 * len(DATASETS)
    ]
    if not complete:
        return grid, choice
    chosen = grid[grid["panel"] == sorted(complete)[0]]
    for dataset, group in chosen.groupby("dataset"):
        best = group.sort_values("sft_val_loss").iloc[0]
        choice[dataset] = float(best["learning_rate"])
    return grid, choice


def collect_arms_table9(fs) -> pd.DataFrame:
    panel = pd.read_csv(ARMS_PANEL)[["candidate_id", "arm", "trainer_seed", "data_seed"]]
    rows = []
    # One listing of the evaluator root, then a prefix match per run (a glob per run relists thousands of dirs).
    eval_dirs = fs.ls(TABLE9_ROOT.removeprefix("gs://"))
    for _, row in panel.iterrows():
        prefix = f"{TABLE9_ROOT.removeprefix('gs://')}/t9_{row['candidate_id']}-"
        hits = [f"{d}/olmo_base_eval_table9_results.json" for d in eval_dirs if d.startswith(prefix)]
        hits = [h for h in hits if fs.exists(h)]
        macro = float("nan")
        if hits:
            macro = float(_read_json(fs, hits[0])["table9_macro_bpb"] or "nan")
        rows.append(
            {
                "run_name": row["candidate_id"],
                "arm": row["arm"],
                "trainer_seed": row["trainer_seed"],
                "data_seed": row["data_seed"],
                "table9_macro_bpb": macro,
            }
        )
    inventory = pd.read_csv(INVENTORY)
    for _, row in inventory[inventory["anchor_or_design_id"] == ARM_A].iterrows():
        rows.append(
            {
                "run_name": row["run_name"],
                "arm": "A",
                "trainer_seed": row["trainer_seed"],
                "data_seed": row["data_seed"],
                "table9_macro_bpb": row["objective_bpb_table9"],
            }
        )
    return pd.DataFrame(rows).sort_values(["arm", "trainer_seed"])


def choose_b(arms: pd.DataFrame) -> dict:
    """B is B1 by content (Calvin, 2026-10-14 10:40 PT); the three-seed Table-9 means are reported, not used."""
    means = arms.groupby("arm")["table9_macro_bpb"].agg(["mean", "std", "count"])
    return {
        "means": means.reset_index().to_dict(orient="records"),
        "B": "B1",
        "rule": "fixed by content: B1's late phase raises code, math and synthetic QA and cuts web; "
        "B1-B3 differ by less than the SE of a three-seed Table-9 mean (~0.003)",
    }


def collect_arms_base_metrics(fs, arms: pd.DataFrame) -> pd.DataFrame:
    """Base-model Uncheatable (fixed weights) and Paloma C4 of every arm run, from its final pretraining evaluation."""
    inventory = pd.read_csv(INVENTORY).set_index("run_name")
    rows = []
    for _, row in arms.iterrows():
        name = row["run_name"]
        if name in inventory.index:
            uri = str(inventory.loc[name, "checkpoint_uri"]).split("/hf/")[0]
        else:
            hits = fs.glob(
                f"marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_sft_probe_arms_3e18_20261014/{name}-*/checkpoints/eval_metrics.jsonl"
            )
            uri = "gs://" + hits[0].split("/checkpoints/")[0] if hits else None
        unch = c4 = float("nan")
        if uri:
            path = uri.removeprefix("gs://") + "/checkpoints/eval_metrics.jsonl"
            if fs.exists(path):
                lines = [json.loads(l) for l in fs.open(path).read().decode().splitlines() if l.strip()]
                finals = [l for l in lines if int(l.get("step", -1)) >= FINAL_PRETRAIN_STEP]
                if finals:
                    final = finals[-1]
                    if "eval/uncheatable_eval/ao3_english/bpb" in final:
                        unch = fixed_uncheatable_bpb(final)
                    c4 = float(final.get("eval/paloma/c4_en/bpb", float("nan")))
        rows.append({**row.to_dict(), "base_uncheatable_bpb": unch, "base_c4_bpb": c4})
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-table9", action="store_true")
    args = parser.parse_args()
    fs = fsspec.filesystem("gcs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = collect_runs(fs)
    frame.to_csv(OUTPUT_DIR / "sft_results.csv", index=False)
    logger.info(
        "Collected %d fine-tunes: %s", len(frame), frame.groupby("panel").size().to_dict() if not frame.empty else {}
    )
    if not frame.empty and frame["panel"].str.startswith("grid").any():
        grid, choice = grid_selection(frame)
        grid.to_csv(OUTPUT_DIR / "grid_results.csv", index=False)
        if choice:
            (OUTPUT_DIR / "grid_selection.json").write_text(json.dumps(choice, indent=2, sort_keys=True) + "\n")
        logger.info("Grid choice: %s (%d grid fine-tunes)", choice, len(grid))
        print(
            grid.pivot_table(index="learning_rate", columns=["panel", "dataset"], values="sft_val_loss")
            .round(4)
            .to_string()
        )
    if not args.skip_table9:
        arms = collect_arms_table9(fs)
        arms = collect_arms_base_metrics(fs, arms)
        arms.to_csv(OUTPUT_DIR / "arms_table9.csv", index=False)
        pick = choose_b(arms)
        (OUTPUT_DIR / "arms_b_choice.json").write_text(json.dumps(pick, indent=2, sort_keys=True) + "\n")
        logger.info("Arm Table-9 means: %s; B = %s", pick["means"], pick["B"])


if __name__ == "__main__":
    main()
