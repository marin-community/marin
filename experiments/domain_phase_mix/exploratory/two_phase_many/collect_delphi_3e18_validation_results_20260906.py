# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "wandb>=0.21",
# ]
# ///
"""Collect the measured endpoints of the 2026-09-06 overnight 3e18 launches into one table per launch.

Launches: the bounded-link validation (three runs), the link-plus-hub validation (two runs), Calvin's coupling
validation (nine runs, the links' matched-seed controls), and the frontier factorial (eighteen runs). Uncheatable comes from each run's step-3006 `eval_metrics.jsonl` on GCS; the Table-9
mean comes from the finished native evaluation runs in W&B, matched by the `source_run` tag or the candidate id
in the run name, and reconstructed from the 51 components as a check. Runs that have not finished are listed
with status `pending` and NaN values, so the collector can be re-run until every row is measured.

usage: uv run collect_delphi_3e18_validation_results_20260906.py [--launch link|hub|coupling|factorial|all]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any

import gcsfs
import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "lib/marin/src"))

from marin.evaluation.olmo_base_eval.aggregate import table9_macro  # noqa: E402
from marin.evaluation.olmo_base_eval.components import table9_components  # noqa: E402

REFERENCE = REPO_ROOT / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs"
BUCKET = "marin-us-east5/pinlin_calvin_xu/data_mixture"
FINAL_STEP = 3006
WANDB_PROJECT = "marin-community/marin-eval"


@dataclasses.dataclass(frozen=True)
class Launch:
    name: str
    experiment_root: str
    table9_group: str
    candidate_ids: tuple[str, ...]
    output_dir: Path
    candidate_tables: tuple[Path, ...]


LAUNCHES = {
    "link": Launch(
        "link",
        "delphi_link_validation_3e18_20260906",
        "olmo_base_eval_table9_delphi_3e18_one_phase_wspu_link_validation",
        ("lwspu_u_bl_cap06", "lwspu_t9_bl_cap06", "lwspu_t9_bl_cap08"),
        REFERENCE / "delphi_link_validation_3e18_20260906",
        (REFERENCE / "delphi_link_validation_3e18_20260906" / "runtime_materialization" / "candidate_weights.csv",),
    ),
    "hub": Launch(
        "hub",
        "delphi_link_hub_validation_3e18_20260906",
        "olmo_base_eval_table9_delphi_3e18_one_phase_wspu_link_hub_validation",
        ("lwspu_t9_bh_cap06", "lwspu_t9_bh_cap08"),
        REFERENCE / "delphi_link_hub_validation_3e18_20260906",
        (REFERENCE / "delphi_link_hub_validation_3e18_20260906" / "runtime_materialization" / "candidate_weights.csv",),
    ),
    "coupling": Launch(
        "coupling",
        "delphi_wspu_coupling_validation_3e18_20260906",
        "olmo_base_eval_table9_delphi_3e18_one_phase_wspu_coupling_validation",
        (
            "cwspu_u_k025_cap06",
            "cwspu_u_k05_cap06",
            "cwspu_u_k1_cap06",
            "cwspu_t9_k025_cap06",
            "cwspu_t9_k05_cap06",
            "cwspu_t9_k1_cap06",
            "cwspu_t9_k025_cap08",
            "cwspu_t9_k05_cap08",
            "cwspu_t9_k1_cap08",
        ),
        REFERENCE / "delphi_coupling_validation_3e18_20260906",
        (REFERENCE / "delphi_coupling_validation_3e18_20260906" / "runtime_materialization" / "candidate_weights.csv",),
    ),
    "factorial": Launch(
        "factorial",
        "delphi_frontier_factorial_3e18_20260906",
        "olmo_base_eval_table9_delphi_3e18_one_phase_frontier_factorial",
        (
            "centre_r0_cap16",
            "centre_r1_cap16",
            *(
                "fac_" + "".join("p" if sign > 0 else "m" for sign in (a, b, c, d, a * b * c * d)) + "_cap16"
                for a in (-1, 1)
                for b in (-1, 1)
                for c in (-1, 1)
                for d in (-1, 1)
            ),
        ),
        REFERENCE / "delphi_frontier_factorial_design_20260906",
        (
            REFERENCE / "delphi_frontier_factorial_design_20260906" / "candidate_weights.csv",
            REFERENCE / "delphi_frontier_factorial_design_20260906" / "candidate_weights_replicate.csv",
        ),
    ),
}


def _read_text(filesystem: gcsfs.GCSFileSystem, path: str) -> str:
    with filesystem.open(path, "rt") as handle:
        return handle.read()


def training_endpoint(filesystem: gcsfs.GCSFileSystem, launch: Launch, candidate_id: str) -> dict[str, Any]:
    """The step-3006 inline evaluation of one run, or a pending marker."""
    pattern = f"{BUCKET}/{launch.experiment_root}/*/*_{candidate_id}-*/checkpoints/eval_metrics.jsonl"
    matches = filesystem.glob(pattern)
    if not matches:
        return {"status": "pending", "eval_metrics_uri": ""}
    if len(matches) > 1:
        raise ValueError(f"{candidate_id}: {len(matches)} endpoint files match {pattern}")
    eval_path = matches[0]
    run_dir = eval_path.removesuffix("/checkpoints/eval_metrics.jsonl")
    try:
        status = _read_text(filesystem, f"{run_dir}/.executor_status").strip()
    except FileNotFoundError:
        status = "RUNNING"
    rows = [json.loads(line) for line in _read_text(filesystem, eval_path).splitlines() if line.strip()]
    endpoints = [row for row in rows if int(row.get("step", -1)) == FINAL_STEP]
    if not endpoints:
        return {"status": f"training:{status}", "eval_metrics_uri": f"gs://{eval_path}"}
    endpoint = endpoints[-1]
    return {
        "status": "measured" if status == "SUCCESS" else f"endpoint:{status}",
        "eval_metrics_uri": f"gs://{eval_path}",
        "uncheatable_bpb": float(endpoint["eval/uncheatable_eval/bpb"]),
        "uncheatable_macro_bpb": float(endpoint["eval/uncheatable_eval/macro_bpb"]),
        **{
            f"uncheatable_{name}_bpb": float(endpoint[f"eval/uncheatable_eval/{name}/bpb"])
            for name in (
                "ao3_english",
                "arxiv_computer_science",
                "arxiv_physics",
                "bbc_news",
                "github_cpp",
                "github_python",
                "wikipedia_english",
            )
            if f"eval/uncheatable_eval/{name}/bpb" in endpoint
        },
    }


def _candidate_id(run: Any, candidate_ids: tuple[str, ...]) -> str | None:
    tag_map = {tag.split("=", 1)[0]: tag.split("=", 1)[1] for tag in run.tags if "=" in tag}
    tagged = tag_map.get("source_run")
    if tagged in candidate_ids:
        return tagged
    return next((candidate_id for candidate_id in candidate_ids if candidate_id in run.name), None)


def table9_results(launch: Launch) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(WANDB_PROJECT, filters={"group": launch.table9_group}, per_page=200))
    finished: dict[str, dict[str, Any]] = {}
    components: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: item.created_at):
        candidate_id = _candidate_id(run, launch.candidate_ids)
        summary = dict(run.summary)
        macro = summary.get("olmo_base_easy/table9_macro_bpb")
        if candidate_id is None or run.state != "finished" or macro is None:
            continue
        values = {
            component: float(summary[f"olmo_base_easy/table9/{component}/bpb"]) for component in table9_components()
        }
        if not math.isclose(float(macro), table9_macro(values), rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{candidate_id}: Table-9 component reconstruction mismatch")
        finished[candidate_id] = {"table9_macro_bpb": float(macro), "table9_wandb_url": run.url}
        components.extend(
            {"candidate_id": candidate_id, "component": component, "bpb": values[component]}
            for component in table9_components()
        )
    return finished, pd.DataFrame(components)


def collect(launch: Launch) -> pd.DataFrame:
    filesystem = gcsfs.GCSFileSystem()
    tables = pd.concat([pd.read_csv(path) for path in launch.candidate_tables], ignore_index=True)
    meta = tables.groupby("candidate_id").first()[["target", "epoch_cap"]]
    table9, components = table9_results(launch)
    rows = []
    for candidate_id in launch.candidate_ids:
        row = {"launch": launch.name, "candidate_id": candidate_id, **meta.loc[candidate_id].to_dict()}
        row.update(training_endpoint(filesystem, launch, candidate_id))
        row.update(table9.get(candidate_id, {"table9_macro_bpb": math.nan, "table9_wandb_url": ""}))
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(launch.output_dir / "measured_results.csv", index=False)
    if len(components):
        components.to_csv(launch.output_dir / "measured_table9_components.csv", index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--launch", choices=(*LAUNCHES, "all"), default="all")
    args = parser.parse_args()
    names = list(LAUNCHES) if args.launch == "all" else [args.launch]
    pd.set_option("display.width", 220)
    for name in names:
        frame = collect(LAUNCHES[name])
        columns = [
            c
            for c in ("candidate_id", "target", "epoch_cap", "status", "uncheatable_bpb", "table9_macro_bpb")
            if c in frame
        ]
        print(f"== {name}: {int(frame.status.eq('measured').sum())} of {len(frame)} measured")
        print(frame[columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
