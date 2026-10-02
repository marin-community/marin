# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Submit a bounded current-Hero pipeline diagnostic inside one GB200 NVLink rack."""

import argparse
import json

from iris.cli.job import resolve_multinode_defaults
from iris.client.client import Job, iris_ctx
from iris.cluster.setup_scripts import cuda_toolchain_setup_script, default_setup_script
from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec, gpu_device
from iris.jax.multigpu import MultiGpuHook
from iris.resources.state import is_job_finished
from iris.rpc import job_pb2
from rigging.timing import Duration, ExponentialBackoff

from experiments.grug.moe_hero_pipeline.runtime.apply_overlay import JAXPP_COMMIT

PJRT_WHEEL = "https://github.com/marin-community/xla/releases/download/marin-xla-pjrt-20260915-708c3a4ec79c/jax_cuda13_pjrt-0.11.1%2Bmarin.708c3a4ec79c-py3-none-manylinux_2_27_aarch64.whl"
_GPUS_PER_NODE = 4
_TASK_TIMEOUT = 7200
_QUEUE_TIMEOUT = 43200
_FINALIZATION_GRACE = 600
_COORDINATOR_TIMEOUT = 54000


def wait_for_gang(job: Job) -> None:
    """Wait for admission and execution with separate budgets; cancel on expiry."""

    def started_or_finished() -> bool:
        status = job.status()
        # BUILDING includes Kueue admission waits. Only execution timestamps
        # distinguish allocated workers from SchedulingGated pods.
        return is_job_finished(status.state) or any(task.started_at is not None for task in status.tasks)

    try:
        ExponentialBackoff(initial=1, maximum=30).wait_until_or_raise(
            started_or_finished,
            timeout=Duration.from_seconds(_QUEUE_TIMEOUT),
            error_message=f"GB200 gang {job.job_id} did not start within {_QUEUE_TIMEOUT} seconds",
        )
        job.wait(timeout=_TASK_TIMEOUT + _FINALIZATION_GRACE, raise_on_failure=True)
    except TimeoutError:
        job.cancel()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--size", choices=("gate", "rack"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    stages, experts, microbatches, batch = (2, 4, 2, 8) if args.size == "gate" else (8, 8, 16, 128)
    nodes = stages * experts // _GPUS_PER_NODE
    command = MultiGpuHook(nproc=_GPUS_PER_NODE, devices_per_proc=1).wrap(
        [
            "python",
            "-u",
            "-m",
            "experiments.grug.moe_hero_pipeline.pipeline_smoke",
            "--main-hero-recipe",
            "--processes-per-task",
            str(_GPUS_PER_NODE),
            "--optimizer",
            "muonh",
            "--schedule",
            "standard_1f1b",
            "--stages",
            str(stages),
            "--expert-axis-size",
            str(experts),
            "--microbatches",
            str(microbatches),
            "--batch-size",
            str(batch),
            "--sequence-length",
            "4096",
            "--steps",
            "2" if args.size == "gate" else "5",
            "--offload-opt-state",
            "--offload-activations",
            "--park-state-during-warmup",
            "--synchronize-devices-after-step",
            "--run-id",
            args.run_id,
            *(["--diagnostic-layers", "2"] if args.size == "gate" else []),
        ]
    )
    replicas, coscheduling = resolve_multinode_defaults(None, f"GB200x{_GPUS_PER_NODE}", nodes)
    contract = dict(
        run_id=args.run_id,
        gpu_count=nodes * _GPUS_PER_NODE,
        nodes=nodes,
        priority="BATCH",
        topology=coscheduling.group_by,
        command=command,
        main_base="89ac0d7705",
        source_model="moe_hero_ep.hero_recipe.HERO_MODEL_CONFIG",
        runtime="JAX 0.11.1 + Marin ARM PJRT; JAXPP 328f75a + pinned host/startup overlay",
        validation="experimental: JAXPP's declared JAX <=0.11.0 cap is overridden",
        queue_timeout_seconds=_QUEUE_TIMEOUT,
        task_timeout_seconds=_TASK_TIMEOUT,
        finalization_grace_seconds=_FINALIZATION_GRACE,
        coordinator_timeout_seconds=_COORDINATOR_TIMEOUT,
    )
    print(json.dumps(contract), flush=True)
    if args.dry_run:
        return
    install = (
        'set -e\nuv pip install --python "$IRIS_VENV/bin/python" '
        "jax==0.11.1 jaxlib==0.11.1 jax-cuda13-plugin==0.11.1 "
        '"nvidia-nccl-cu13>=2.31.2" "nvidia-cublas>=13.2.0.9" '
        '"flash-attn-4[cu13]==4.0.0b28" "nvidia-cutlass-dsl[cu13]==4.6.2" '
        '"quack-kernels[cu13]==0.6.4"\n'
        'uv pip install --python "$IRIS_VENV/bin/python" --index-url https://download.pytorch.org/whl/cu128 '
        '"torch==2.11.0+cu128"\n'
        f'uv pip install --python "$IRIS_VENV/bin/python" --no-deps "{PJRT_WHEEL}"\n'
        'uv pip install --python "$IRIS_VENV/bin/python" --no-deps --reinstall '
        f'"jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@{JAXPP_COMMIT}"\n'
        "uv run --no-sync python experiments/grug/moe_hero_pipeline/runtime/apply_overlay.py\n"
        "uv run --no-sync python -c 'import torch; assert torch.cuda.is_available(), "
        '"Current Hero Quack kernels require CUDA-enabled Torch"; '
        "print(torch.__version__, torch.cuda.get_device_capability())'\n"
    )
    environment = EnvironmentSpec(
        extras=["pipeline"],
        env_vars={
            "WANDB_MODE": "disabled",
            "JAX_ENABLE_PGLE": "false",
            "XLA_PYTHON_CLIENT_ALLOCATOR": "cuda_async",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.85",
            "JAXPP_DISABLE_SCHEDULE_TASK_FUSION": "1",
            "CUDA_MODULE_LOADING": "EAGER",
            "PYTHONFAULTHANDLER": "1",
            "JAX_PLATFORMS": "cuda",
        },
        setup_scripts=[default_setup_script(extras=["pipeline"]), install, cuda_toolchain_setup_script()],
    )
    job = iris_ctx().client.submit(
        entrypoint=Entrypoint.from_command(*command),
        name=args.run_id,
        resources=ResourceSpec(cpu=120, memory="890g", disk="128g", device=gpu_device("GB200", _GPUS_PER_NODE)),
        environment=environment,
        replicas=replicas,
        coscheduling=coscheduling,
        ports=["jax"],
        priority_band=job_pb2.PRIORITY_BAND_BATCH,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=1,
        scheduling_timeout=Duration.from_seconds(_QUEUE_TIMEOUT),
        timeout=Duration.from_seconds(_TASK_TIMEOUT),
    )
    print("SUBMITTED", job.job_id, flush=True)
    wait_for_gang(job)


if __name__ == "__main__":
    main()
