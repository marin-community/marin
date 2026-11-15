# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan or submit a bounded Snowball TPU SFT initialization smoke."""

import hashlib
import json
import os
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import click
import draccus
import yaml
from fray.iris_backend import FrayIrisClient
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.cli.connect import open_iris_client
from iris.cluster.client.job_info import get_job_info
from iris.rpc.proto_display import priority_band_value
from marin.training.training import resolve_training_env
from rigging.filesystem.storage_path import StoragePath
from rigging.timing import Duration

from experiments.grug.moe.optimizer import GrugMoeMuonHConfig  # noqa: F401 -- optimizer registration
from experiments.post_training.long_loop.sft import SMOKE_CONFIG
from experiments.post_training.long_loop.sft_runtime.train import GrugRunConfig, _run_grug_local

BASE = (
    "gs://marin-us-central2/grug/"
    "moe_67b_a2b_d2560_ep1_rep1_ctx4_bs256_seq262144_ctxext_step156k_qk175_longctx_skew8-c06695/"
    "checkpoints/step-157000"
)
OUTPUT_PREFIX = "gs://marin-us-central2/tmp/ttl=3d/long-loop-tpu-smoke/"
BASE_METADATA_SHA256 = "5adc83178049bce2b0c6d1ed90d3210f88e21801b8fb555fc7bdb9bd39456f72"


def build_config(data: dict, *, output: str, run_id: str) -> GrugRunConfig:
    """Build a one-update, 8k-context, 64-chip fix-validation smoke."""
    if not output.startswith(OUTPUT_PREFIX):
        raise ValueError("TPU smoke outputs must use the dedicated us-central2 TTL prefix")
    if data.get("auto_build_caches") is not False:
        raise ValueError("TPU smoke must reuse existing caches")

    def validate_region(value):
        if isinstance(value, dict):
            for child in value.values():
                validate_region(child)
        elif isinstance(value, list):
            for child in value:
                validate_region(child)
        elif isinstance(value, str) and "://" in value and not value.startswith("gs://marin-us-central2/"):
            raise ValueError(f"Nonregional TPU input: {value}")

    validate_region(data)
    if not str(data.get("tokenizer", "")).startswith("gs://marin-us-central2/"):
        raise ValueError("TPU smoke tokenizer must use existing regional GCS input")
    recipe = yaml.safe_load(SMOKE_CONFIG.read_text())
    recipe["data"] = data
    recipe["model"].update(max_seq_len=8192, attention_implementation="tpu_splash", moe_implementation="ring")
    resources = ResourceConfig.with_tpu("v4-128", zone="us-central2-b", preemptible=False)
    recipe.update(
        processes_per_task=1,
        stop_after_steps=1,
        first_checkpoint_restore_step=None,
        record_compiled_memory=True,
        native_timing_root=f"{output}/timing/native-sft",
        eval=None,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        task_timeout_seconds=3600,
    )
    grug = recipe["trainer"]
    grug.update(
        context_axis_size=4,
        replica_axis_size=1,
        expert_axis_size=1,
        gradient_accumulation_steps=1,
        offload_opt_state=False,
        save_checkpoints=False,
        gc_interval=None,
        sharding_dump_path=f"{output}/state-sharding.json",
    )
    trainer = grug["trainer"]
    trainer.update(
        id=run_id,
        initialize_from=BASE,
        load_checkpoint=False,
        load_checkpoint_path=[],
        train_batch_size=16,
        mp="compute=bfloat16,params=float32,output=bfloat16",
        log_dir=f"logs/{run_id}",
        log_jaxprs=False,
        log_xla_hlo=False,
        shutdown_at_exit=False,
    )
    trainer["mesh"] = {"axes": {"data": -1, "replica": 1, "model": 1, "context": 4}}
    trainer["tracker"].update(id=run_id, name=run_id, mode="disabled")
    trainer["checkpointer"].update(
        base_path=f"{output}/checkpoints", temporary_base_path=f"{output}/temporary-checkpoints"
    )
    return replace(draccus.decode(GrugRunConfig, recipe), resources=resources)


def tpu_worker(config: GrugRunConfig, output: str) -> None:
    """Preserve each rank's original exception before accelerator introspection."""
    info = get_job_info()
    if info is None:
        raise RuntimeError("TPU worker requires an Iris task")
    if os.environ.get("JAX_PLATFORMS") != "tpu,cpu":
        raise RuntimeError("TPU worker platform binding changed")
    try:
        checkpoint = config.trainer.trainer.initialize_from
        if checkpoint is None:
            raise ValueError("TPU smoke requires its pinned native checkpoint")
        metadata = (StoragePath(checkpoint) / "metadata.json").read_bytes()
        if hashlib.sha256(metadata).hexdigest() != BASE_METADATA_SHA256:
            raise ValueError("Pinned native base metadata changed")
        if info.task_index == 0:
            (StoragePath(output) / "consumed-config.json").write_text(json.dumps(draccus.encode(config), indent=2))
        _run_grug_local(config)
        if info.task_index == 0:
            (StoragePath(output) / "success.json").write_text(json.dumps({"updates": config.stop_after_steps}))
    except BaseException:
        report = {
            "task_index": info.task_index,
            "utc": datetime.now(UTC).isoformat(),
            "traceback": traceback.format_exc(),
        }
        traceback.print_exc()
        try:
            local = Path(os.environ["IRIS_OUTPUT_DIR"])
            local.mkdir(parents=True, exist_ok=True)
            (local / "initialization-exception.json").write_text(json.dumps(report, indent=2))
        except Exception:
            traceback.print_exc()
        try:
            (StoragePath(output) / f"failures/task-{info.task_index}.json").write_text(json.dumps(report, indent=2))
        except Exception:
            traceback.print_exc()
        raise


def build_request(config: GrugRunConfig, output: str) -> JobRequest:
    env = resolve_training_env(
        {
            "JAX_PLATFORMS": "tpu,cpu",
            "JAX_TRACEBACK_FILTERING": "off",
            "JAX_COMPILATION_CACHE_DIR": f"{output}/compilation-cache",
        },
        config.resources,
    )
    return JobRequest(
        name=config.trainer.trainer.id,
        entrypoint=Entrypoint.from_callable(tpu_worker, args=[config, output]),
        resources=config.resources,
        environment=create_environment(env_vars=env, extras=["tpu"]),
        priority=priority_band_value("production"),
        timeout=Duration.from_hours(1),
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        processes_per_task=1,
    )


@click.command(help=__doc__)
@click.option("--data-config", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--data-sha256", required=True)
@click.option("--output", required=True)
@click.option("--run-id", required=True)
@click.option("--plan-output", required=True, type=click.Path(path_type=Path))
@click.option("--run", is_flag=True)
def main(data_config: Path, data_sha256: str, output: str, run_id: str, plan_output: Path, run: bool) -> None:
    raw = data_config.read_bytes()
    if hashlib.sha256(raw).hexdigest() != data_sha256:
        raise ValueError("GCS data configuration hash changed")
    config = build_config(json.loads(raw), output=output, run_id=run_id)
    request = build_request(config, output)
    plan_output.write_text(
        json.dumps(
            {
                "config": draccus.encode(config),
                "env": request.environment.env_vars,
                "priority": request.priority,
                "chips": config.resources.chip_count(),
                "replicas": config.resources.replicas,
            },
            indent=2,
        )
    )
    click.echo(f"Plan: {plan_output}; {config.resources.chip_count()} chips; production; retries 0")
    if run:
        with open_iris_client(config_file=Path("lib/iris/config/marin.yaml")) as client:
            job = FrayIrisClient.from_iris_client(client).submit(request, adopt_existing=False)
        click.echo(f"Submitted: {job.job_id}")


if __name__ == "__main__":
    main()
