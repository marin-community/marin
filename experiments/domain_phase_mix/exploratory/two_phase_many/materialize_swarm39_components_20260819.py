# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "wandb"]
# ///
"""Per-component endpoint metrics for the delphi 3e18 heldout panel (ATOM-028).

The odd/even decomposition of the two-phase contrast was run on the uncheatable MACRO and found the order
channel negligible against a large quadratic separation cost. But the only confirmed two-phase gain in this
programme was entirely a CODE-target phenomenon, and a macro over many evaluations is exactly what dilutes
a bucket-specific effect -- that dilution already produced one false blanket negative on the damage term.

So the decomposition has to be redone per component before the macro's null is believed. The W&B run
summary carries seven uncheatable components and seventeen paloma domains, including
`dolma_100_programing_languages`, which is the precise target the WSD80 gain was measured on. Fetching the
summary is one lightweight call per run, no history scan needed.

Usage: ``uv run python ... [--workers 16] [--limit N]``
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import pandas as pd  # noqa: E402
import wandb  # noqa: E402

DELPHI = SCRIPT_DIR / "reference_outputs" / "delphi_3e18_append_only_heldouts_20260714"
OUTPUT = DELPHI / "endpoint_components.csv"


def components(api, row) -> dict:
    identifier = f"{row.wandb_entity}/{row.wandb_project}/{row.wandb_run_id}"
    try:
        summary = api.run(identifier).summary
        values = {key: float(summary[key]) for key in summary.keys() if key.endswith("/bpb")}
    except Exception as error:
        return {"heldout_id": row.heldout_id, "error": type(error).__name__}
    return {"heldout_id": row.heldout_id, "error": None, **values}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    panel = pd.read_csv(DELPHI / "heldout_current.csv")
    panel = panel[panel["fit_panel_overlap"] == "coordinate_disjoint"].reset_index(drop=True)
    done = pd.read_csv(OUTPUT) if OUTPUT.exists() else pd.DataFrame(columns=["heldout_id"])
    pending = panel[~panel["heldout_id"].isin(set(done["heldout_id"]))]
    if args.limit:
        pending = pending.head(args.limit)
    print(f"{len(panel)} rows in scope, {len(done)} cached, {len(pending)} to fetch", flush=True)

    if len(pending):
        api = wandb.Api(timeout=60)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(lambda row: components(api, row), pending.itertuples()))
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(OUTPUT, index=False)

    metrics = [c for c in done.columns if c.endswith("/bpb")]
    print(f"rows with any component: {int(done[metrics].notna().any(axis=1).sum())}/{len(done)}")
    print(f"component columns retrieved: {len(metrics)}")
    if "error" in done:
        failures = done["error"].dropna()
        if len(failures):
            print("failures:", failures.value_counts().to_dict())


if __name__ == "__main__":
    main()
