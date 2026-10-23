# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reconcile Structure-Closure mode records and rerank the union tournament."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from iris.cli.connect import open_iris_client
from iris.cluster.types import JobName
from marin.evaluation.records import CW_RECORDS_PREFIX, RunStatus, TaskCoverage, read_record, record_path
from rigging.config_discovery import find_project_root
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.launch import EVALUATION_CONTROLLER_CLUSTER

campaign = importlib.import_module(
    "experiments.evaluation.campaigns.eval-policy.structure_closure_tournament"
)
pairwise = importlib.import_module(
    "experiments.evaluation.campaigns.eval-policy.analysis.snowball_tournament.pairwise"
)


def _cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _standard(path: Path) -> tuple[list[str], list[str], dict[str, list[str]]]:
    lines = [line for line in path.read_text().splitlines() if line.startswith("|")]
    header = _cells(lines[0])
    metrics = _cells(lines[2])
    rows = {}
    for line in lines[3:]:
        cells = _cells(line)
        if len(cells) == len(header) and pairwise.is_candidate(cells[0]):
            rows[cells[0]] = cells[1:]
    if len(rows) != 14:
        raise ValueError(f"expected 14 Snowball/Grug candidates, found {len(rows)}")
    return header[1:], metrics[1:], rows


def _submissions(path: Path) -> dict[tuple[str, str], dict]:
    latest = {}
    if not path.exists():
        return latest
    for line in path.read_text().splitlines():
        entry = json.loads(line)
        for evaluation in entry["evaluations"]:
            latest[(entry["model"], evaluation["eval_name"])] = entry | {"run_id": evaluation["run_id"]}
    return latest


def _status(state: str) -> str:
    if state in {"running", "building", "pending"}:
        return "QUEUED"
    return "CRASHED"


def _score_text(score: float) -> str:
    return f"{score:.3f}"


def _acceptable_coverage(coverage: TaskCoverage | None) -> bool:
    return (
        coverage is not None
        and coverage.n_benchmark == campaign.CASES_PER_MODE
        and coverage.n_attempted == campaign.CASES_PER_MODE
        and coverage.n_scored * 10 >= campaign.CASES_PER_MODE * 9
    )


def _write_table(path: Path, header: list[str], metric: list[str], rows: dict[str, list[str]]) -> None:
    lines = [
        "| " + " | ".join(["Model", *header]) + " |",
        "| " + " | ".join(["---"] * (len(header) + 1)) + " |",
        "| " + " | ".join(["Metric", *metric]) + " |",
    ]
    lines.extend("| " + " | ".join([model, *values]) + " |" for model, values in rows.items())
    path.write_text("\n".join(lines) + "\n")


def refresh(standard: Path, manifest: Path, output: Path) -> None:
    """Use completed records as scores and Iris states for unsealed mode cells."""
    configure_coreweave_s3()
    standard_header, standard_metrics, standard_rows = _standard(standard)
    modes = campaign._mode_inventory()
    mode_names = [f"structure-closure-v6-{family}-{control}" for family, control in modes]
    submissions = _submissions(manifest)
    tournament_rows = {}
    state_by_job = {}
    active_mode_by_job = set()
    with open_iris_client(cluster_name=EVALUATION_CONTROLLER_CLUSTER, workspace=find_project_root()) as client:
        for model in standard_rows:
            cells = []
            for name, (family, control) in zip(mode_names, modes, strict=True):
                entry = submissions.get((model, name))
                if entry is None:
                    cells.append("CRASHED")
                    continue
                path = record_path(CW_RECORDS_PREFIX, entry["run_id"])
                if StoragePath(path).exists():
                    record = read_record(path)
                    task = f"{family}/{control}"
                    score = record.metrics.get(task, {}).get("syntax_complete")
                    if (
                        record.status is RunStatus.SUCCEEDED
                        and score is not None
                        and _acceptable_coverage(record.coverage.get(task))
                    ):
                        cells.append(f"{_score_text(score)} ({record.results_path})")
                    else:
                        cells.append("CRASHED")
                    continue
                job = entry["job"]
                if job not in state_by_job:
                    state_by_job[job] = client.job_status(JobName.from_string(job)).state.value
                state = state_by_job[job]
                if state == "running" and job not in active_mode_by_job:
                    cells.append("RUNNING")
                    active_mode_by_job.add(job)
                else:
                    cells.append(_status(state))
            tournament_rows[model] = cells
    output.mkdir(parents=True, exist_ok=True)
    _write_table(output / "TRACKER.md", mode_names, ["syntax closure"] * len(modes), tournament_rows)
    union_rows = {model: standard_rows[model] + tournament_rows[model] for model in standard_rows}
    union = output / "UNION_TRACKER.md"
    _write_table(union, standard_header + mode_names, standard_metrics + ["syntax closure"] * len(modes), union_rows)
    benchmarks, candidates, scores = pairwise.tracker_scores(union)
    pairs, summary = pairwise.pairwise_rows(benchmarks, candidates, scores)
    pairwise.write_csv(output / "UNION_PAIRWISE.csv", pairs)
    pairwise.write_csv(output / "UNION_RANKING.csv", summary)
    sealed = sum(cell.startswith("0.") or cell.startswith("1.") for row in tournament_rows.values() for cell in row)
    (output / "STATUS.json").write_text(
        json.dumps(
            {
                "models": len(standard_rows),
                "modes": len(modes),
                "sealed_mode_cells": sealed,
                "total_mode_cells": len(standard_rows) * len(modes),
                "ranking_provisional": sealed < len(standard_rows) * len(modes),
                "top_model": summary[0]["model"],
                "standard_tracker": str(standard),
                "dataset": campaign.DATASET,
                "dataset_revision": campaign.DATASET_REVISION,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"sealed {sealed}/{len(standard_rows) * len(modes)} mode cells; leader {summary[0]['model']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standard", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    refresh(args.standard, args.manifest, args.output)


if __name__ == "__main__":
    main()
