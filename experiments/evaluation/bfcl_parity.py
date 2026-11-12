# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Repeat an explicit BFCL Harbor policy and bootstrap complete native run means.

Launch uses the standard evaluation framework, with no custom agent/parser or
whole-run replacement policy. Analysis reads a pinned JSON manifest::

    {"task_ids": ["task-a", "task-b"], "reference": "reference", "arms": {
      "reference": [{"logical_run": "1", "record": "record.json", "record_sha256": "...",
                     "samples": "samples.json", "samples_sha256": "..."}]}}

List every physical attempt, including incomplete attempts. For a terminal
native record with no sample artifact, use samples:null and samples_sha256:null.
The reader verifies supplied artifacts; it does not discover omitted jobs. Use the same
logical_run for a whole-run replacement. Five complete logical runs per arm are
required for a primary estimate. Rewards are fractions, not percentage points.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import click
import fsspec
from iris.cli.connect import open_iris_client
from iris.rpc.proto_display import priority_band_value
from marin.evaluation.bfcl_parity import admit_native_run, compare_run_clusters
from marin.evaluation.hardware import default_platform
from marin.evaluation.model_config import load_model_config
from rigging.config_discovery import find_project_root

from experiments.evaluation.evals import HarborDefinition
from experiments.evaluation.launch import (
    EVALUATION_CONTROLLER_CLUSTER,
    LaunchSpec,
    launch_group,
    prepare_evaluation_batch,
)


def repeat_specs(spec: LaunchSpec, *, run_prefix: str, runs: int = 5) -> tuple[LaunchSpec, ...]:
    """Fresh logical runs preserve the explicit evaluator/model method unchanged."""
    if runs < 5 or spec.resume_results_path or spec.cache_key or spec.retry_unscored_harbor_trials:
        raise ValueError("At least five fresh whole runs are required; no partial resume or cache key")
    return tuple(replace(spec, run_id=f"{run_prefix}-run-{index:02d}") for index in range(1, runs + 1))


@click.group()
def main():
    """BFCL parity repeats and run-level analysis."""


@main.command()
@click.option("--model-config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--harbor-config", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--run-prefix", required=True)
@click.option("--runs", type=click.IntRange(min=5), default=5, show_default=True)
@click.option("--federated-cluster", required=True)
@click.option("--records-prefix", required=True)
@click.option("--accelerator", default=None)
@click.option("--dry-run", is_flag=True)
def launch(model_config, harbor_config, run_prefix, runs, federated_cluster, records_prefix, accelerator, dry_run):
    """Submit >=5 independent standard Harbor runs at production priority."""
    model = load_model_config(model_config)
    spec = LaunchSpec(
        model=model,
        evals=(),
        evalchemy_definitions=(),
        harbor_definitions=(HarborDefinition(name="bfcl-parity", config_path=harbor_config),),
        platform=default_platform(model),
        accelerator=accelerator,
        limit=None,
        records_prefix=records_prefix,
        submission_cluster=EVALUATION_CONTROLLER_CLUSTER,
        federated_cluster=federated_cluster,
        priority_band=priority_band_value("production"),
    )
    specs = repeat_specs(spec, run_prefix=run_prefix, runs=runs)
    if dry_run:
        for item in specs:
            batch = prepare_evaluation_batch(item)
            click.echo(
                json.dumps(
                    {
                        "run_id": item.run_id,
                        "priority_band": batch.priority_band,
                        "records_prefix": batch.records_prefix,
                        "evaluations": [
                            {"run_id": e.identity.run_id, "output": e.identity.output_dir} for e in batch.evaluations
                        ],
                    }
                )
            )
        return
    with open_iris_client(cluster_name=EVALUATION_CONTROLLER_CLUSTER, workspace=find_project_root()) as client:
        # Prepare+submit each pair before the next. Do not retain mutable launcher
        # environment from a later namespace while serializing an earlier batch.
        for item in specs:
            submitted = launch_group(prepare_evaluation_batch(item), client)
            click.echo(f"{item.run_id}: {submitted.job}")


def _read_pinned(path: str, expected: str):
    with fsspec.open(path, "rb") as source:
        raw = source.read()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Native artifact SHA256 differs: {path}")
    return json.loads(raw)


@main.command()
@click.option("--manifest", required=True)
@click.option("--output", required=True)
@click.option("--seed", type=int, default=297, show_default=True)
@click.option("--iterations", type=click.IntRange(min=100), default=10000, show_default=True)
def analyze(manifest, output, seed, iterations):
    """Read native records/samples only; perform no model or scoring requests."""
    with fsspec.open(manifest, "rb") as source:
        manifest_bytes = source.read()
    inputs = json.loads(manifest_bytes)
    tasks = tuple(inputs["task_ids"])
    arms = {}
    for name, attempts in inputs["arms"].items():
        arms[name] = []
        for attempt in attempts:
            if attempt["samples"] is None:
                if attempt.get("samples_sha256") is not None:
                    raise ValueError("Absent sample artifact cannot carry a readback SHA256")
                samples = []
            else:
                samples = _read_pinned(attempt["samples"], attempt["samples_sha256"])
            arms[name].append(
                admit_native_run(
                    _read_pinned(attempt["record"], attempt["record_sha256"]),
                    samples,
                    logical_run=attempt["logical_run"],
                    task_ids=tasks,
                )
            )
    result = compare_run_clusters(arms, reference=inputs["reference"], seed=seed, iterations=iterations)
    result.update(manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(), input_manifest=inputs, task_ids=tasks)
    with fsspec.open(output, "xt") as target:
        json.dump(result, target, indent=2, sort_keys=True)
        target.write("\n")
    if any(not arm["primary_available"] for arm in result["arms"].values()):
        raise click.ClickException("Fewer than five complete logical runs in an arm; primary estimates remain NULL")


if __name__ == "__main__":
    main()
