# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score retained native checkpoints on one canonical assistant-target holdout."""

import argparse
import json
import math
from dataclasses import dataclass, replace

import draccus
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, create_environment
from iris.rpc.proto_display import priority_band_value
from levanter.main.eval_lm import EvalLmConfig
from levanter.main.eval_lm import main as eval_main
from levanter.tracker.json_file import JsonFileTrackerConfig
from marin.training.run_environment import extras_for_resources
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.post_training.snowball_isoflop.budget import budgeted_map
from experiments.post_training.snowball_isoflop.config import (
    PREFIX,
    SEQ_LEN,
    SPECIAL_IDS,
    TASK_TIMEOUT,
    VERSION,
    Cell,
    Mask,
    build_pod,
    evaluation_data_config,
    primary_cells,
    resources,
)
from experiments.post_training.snowball_isoflop.measurements import read_json, training_history


@dataclass(frozen=True)
class Endpoint:
    cell: Cell
    updates: int

    @property
    def name(self) -> str:
        return f"score-{self.cell.name}-u{self.updates}"

    @property
    def checkpoint(self) -> str:
        return f"{PREFIX}/{VERSION}/{self.cell.name}/checkpoints/step-{self.updates - 1}"

    @property
    def output(self) -> str:
        return f"{PREFIX}/{VERSION}/{self.name}"


def scoring_config(endpoint: Endpoint) -> EvalLmConfig:
    baseline = build_pod(endpoint.cell).train_config
    # Always score with the canonical CP4/data2 Sonic layout, independent of
    # the training objective. Learned heads must never be reseeded on reload.
    trainer = replace(
        baseline.trainer,
        id=f"{VERSION}-{endpoint.name}",
        seed=0,
        max_eval_batches=4,
        tracker=JsonFileTrackerConfig(output_path=endpoint.output),
    )
    return EvalLmConfig(
        checkpoint_path=endpoint.checkpoint,
        trainer=trainer,
        data=evaluation_data_config(),
        max_eval_length=SEQ_LEN,
        model=baseline.model,
        audit_token_ids=SPECIAL_IDS,
        result_path=f"{endpoint.output}/result.json",
    )


def score_endpoint(endpoint: Endpoint) -> dict:
    root = StoragePath(endpoint.output)
    if (root / "consumed-config.json").exists():
        raise FileExistsError(f"Scoring artifacts already exist: {root}")
    metadata = read_json(f"{endpoint.checkpoint}/metadata.json")
    if metadata["step"] != endpoint.updates - 1 or metadata["is_temporary"]:
        raise ValueError(f"Scoring requires a committed permanent checkpoint: {endpoint.checkpoint}")
    config = scoring_config(endpoint)
    (root / "consumed-config.json").write_text(json.dumps(draccus.encode(config), indent=2))
    environment = resolve_training_env(
        {**(build_pod(endpoint.cell).env_vars or {}), "JAX_PLATFORMS": "cuda,cpu"}, resources()
    )
    job = current_client().submit(
        JobRequest(
            name=config.trainer.id,
            entrypoint=Entrypoint.from_callable(eval_main, args=[config]),
            resources=resources(),
            environment=create_environment(extras=extras_for_resources(resources()), env_vars=environment),
            timeout=Duration.from_seconds(TASK_TIMEOUT),
            processes_per_task=8,
            priority=priority_band_value("interactive"),
            max_retries_preemption=0,
        )
    )
    (root / "dispatch.json").write_text(json.dumps({"job_id": job.job_id}))
    job.wait(raise_on_failure=True)
    result = read_json(f"{endpoint.output}/result.json")
    if any(not math.isfinite(result["metrics"][key]) for key in ("eval/loss", "eval/macro_loss")):
        raise ValueError(f"Nonfinite endpoint evaluation: {endpoint.name}")
    return {"checkpoint": endpoint.checkpoint, "output": endpoint.output, **result}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-size", type=int, choices=(512, 768, 1024))
    parser.add_argument("--max-concurrent", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--h100-hour-limit", type=float, required=True)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    endpoints = tuple(Endpoint(cell, step) for cell in primary_cells() for step in cell.checkpoint_steps)
    if args.calibration_size is not None:
        endpoints = (Endpoint(Cell(args.calibration_size, Mask.ASSISTANT, 0, 20, (20,), "calibration"), 20),)
    ceiling = 3 if args.calibration_size is not None else 48
    if not 0 < args.h100_hour_limit <= ceiling:
        parser.error(f"Scoring must fit its {ceiling}-H100-hour envelope inside A")
    if not args.run:
        print(json.dumps({endpoint.name: draccus.encode(scoring_config(endpoint)) for endpoint in endpoints}))
        return
    manager = StoragePath(f"{PREFIX}/{VERSION}/manager-scoring-{args.calibration_size or 'A'}")
    results = budgeted_map(
        {endpoint.name: endpoint for endpoint in endpoints},
        score_endpoint,
        manager,
        args.h100_hour_limit,
        args.max_concurrent,
    )
    audits = [result["input_audit"] for result in results.values()]
    if any(audit != audits[0] for audit in audits):
        raise ValueError("Endpoint scoring inputs differ across arms/seeds/budgets")
    if args.calibration_size is not None:
        endpoint = endpoints[0]
        state, rows = training_history(endpoint.cell.name, endpoint.updates)
        final = [row for row in rows if row.get("global_step") == endpoint.updates - 1 and "eval/loss" in row][-1]
        result = results[endpoint.name]
        differences = {key: result["metrics"][key] - final[key] for key in ("eval/loss", "eval/macro_loss")}
        comparison = {"training_state": state, "checkpoint": endpoint.checkpoint, "differences": differences}
        (manager / "reload-comparison.json").write_text(json.dumps(comparison, indent=2))
        if state != "finished" or any(abs(value) > 0.001 for value in differences.values()):
            raise ValueError("Independent checkpoint scoring does not match native final validation")
    (manager / "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
