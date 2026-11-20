# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Host the frozen Micro210 judge in its own regional Iris job."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import click
from fray.client import wait_all
from fray.iris_backend import convert_constraints
from iris.cluster.client.job_info import get_job_info
from iris.cluster.constraints import cluster_directive
from marin.execution.lazy import StepContext
from marin.inference.iris import remote_inference
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.micro210 import Micro210JudgeEndpoint, micro210_judge_inference, micro210_judge_sha256
from experiments.post_training.long_loop.recipe_bundle import require_clean_revision
from experiments.post_training.long_loop.round_runner import adopted_round_evaluation, load_round


@click.command(help=__doc__)
@click.option("--round", "round_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--source-revision", required=True)
@click.option("--run", "do_run", is_flag=True)
def main(round_path: Path, source_revision: str, do_run: bool) -> None:
    info = get_job_info()
    if info is None and do_run:
        raise RuntimeError("Launch the judge inside its regional Iris root job")
    if info is None:
        require_clean_revision(Path(__file__).resolve().parents[2], source_revision)
    configure_coreweave_s3()
    spec = load_round(round_path)
    terminal = adopted_round_evaluation(spec)
    context = StepContext.for_run(
        terminal.path(spec.storage.output_prefix),
        spec.storage.output_prefix,
        runtime_args=terminal.runtime_args,
        deps=terminal.deps,
    )
    config = terminal.build_config(replace(context, is_fingerprint=True))
    inference = micro210_judge_inference(config)
    resources = inference.iris.worker_resources
    if cluster_directive(convert_constraints(resources)) is not None:
        raise ValueError("Regional judge children must not carry a federation cluster pin")
    if not do_run:
        click.echo(
            json.dumps(
                {
                    "source_revision": source_revision,
                    "root_cluster": spec.serving.remote_judge.target_cluster,
                    "resources": asdict(resources),
                    "geometry": asdict(inference.iris.serving_geometry),
                    "child_cluster_pin": None,
                },
                indent=2,
            )
        )
        return
    if info is None or list(resources.regions or ()) != [info.worker_region]:
        raise ValueError("Judge root must run in the configured serving region")
    receipt = StoragePath(spec.serving.remote_judge.endpoint_receipt)
    if receipt.exists():
        raise ValueError("Judge endpoint receipt already exists; use a fresh round identity")
    terminal_receipt = StoragePath(spec.serving.remote_judge.endpoint_receipt + ".terminal.json")
    try:
        with remote_inference(inference) as served:
            if served.tensor_parallel_size != inference.model.tensor_parallel_size:
                raise ValueError("Running judge tensor parallelism differs from the frozen judge")
            endpoint = Micro210JudgeEndpoint(
                config.evaluation.record().canonical_spec_sha256,
                micro210_judge_sha256(config),
                served.tensor_parallel_size,
                served.model.endpoint,
                tuple(str(job.job_id) for job in served.jobs),
            )
            receipt.write_text(json.dumps(asdict(endpoint)))
            wait_all(served.jobs)
    finally:
        terminal_receipt.write_text(json.dumps({"terminated": True}))


if __name__ == "__main__":
    main()
