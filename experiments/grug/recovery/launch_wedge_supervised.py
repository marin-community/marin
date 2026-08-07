# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the #7344 wedge reproducer under GPUHangSupervisor on multiple GB200 racks.

WHY THIS EXISTS: NCCL 2.28.9 on aarch64 wedges this cross-rack collective within
seconds, while versions with the upstream proxy-slot fix should complete. No
native NCCL timeout fires on an affected build (see the standalone repro's
README). This launcher runs it as a supervised child so we can show that our framework's
primary detector — XLA's per-execution deadman, armed by the supervisor in the
child's ``XLA_FLAGS`` — catches that wedge where nothing else does. Repeated
``--ablation`` options run environment arms sequentially on one allocation, with
a fresh child process for every arm.

One task per node runs a ``GPUHangSupervisor``. The supervisor makes no JAX calls;
it spawns the reproducer via ``levanter.recovery.child``, which brings up
``jax.distributed`` and joins the ``16 * dp_racks``-process mesh. When the
collective wedges, the deadman ends every wedged process in ``LOG(FATAL)`` and each
supervisor records the crash. The dispatch path is identical to ``minrepro_launch``
and ``moe_hero_fsdp`` so the image, dependency set, and mesh match the production
job.

Recovery is intentionally out of scope here: no snapshot, no restart budget
(``--max-restarts 0``). The goal is to reproduce or exclude the wedge and confirm detection.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import click
from fray.cluster import ResourceConfig
from levanter.recovery.detection import DetectionConfig
from levanter.recovery.supervisor import GPUHangSupervisor
from levanter.recovery.types import AblationSpec, RunOutcome
from marin.execution.artifact import Artifact
from marin.execution.build_context import resolve_version
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.cli import build_options
from marin.experiment.namespacing import user_namespaced_name

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.recovery.ablation_catalog import (
    environment_ablation_names,
    environment_ablations,
    selected_ablations,
)
from experiments.grug.recovery.wedge_entrypoint import WedgeReproConfig, run_wedge_repro

logger = logging.getLogger(__name__)

HERO_NODES_PER_RACK = 16  # matches moe_hero_fsdp / minrepro_launch
# The deadman must sit well above a healthy execution (sub-second here) so it only
# fires on the wedge, and well below the reproducer's time-to-wedge budget so
# detection is sub-minute.
DEFAULT_EXECUTION_TERMINATE_TIMEOUT = 60.0
WEDGE_PROVENANCE_ENV = {
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,BOOTSTRAP,ENV,NET,GRAPH,TUNING,RAS",
}


@dataclass(frozen=True)
class SupervisedWedgeConfig:
    run_id: str
    resources: ResourceConfig
    dp_racks: int
    ablations: tuple[AblationSpec, ...]
    execution_terminate_timeout: float
    max_restarts: int


class SupervisedWedgeResult(Artifact):
    """Marker artifact for the supervised wedge-detection run."""


def _run_supervised(config: SupervisedWedgeConfig) -> None:
    """Run the selected arms in supervised children and report what detection fired."""
    detection = DetectionConfig(
        execution_terminate_timeout_seconds=config.execution_terminate_timeout,
        enable_recoverability=False,
    )
    with GPUHangSupervisor(
        detection=detection,
        # Inert here (the reproducer writes no heartbeat), but must be positive; the
        # XLA execution deadman is the detector, with startup_timeout as the backstop.
        deadman_timeout=120.0,
        max_restarts_per_run=config.max_restarts,
    ) as supervisor:
        for ablation in config.ablations:
            if ablation.num_steps is None:
                raise ValueError(f"wedge ablation {ablation.name!r} has no num_steps")
            logger.info("=== wedge ablation %s: %s ===", ablation.name, ablation.notes)
            result = supervisor.run(
                run_wedge_repro,
                WedgeReproConfig(dp_racks=config.dp_racks, num_steps=ablation.num_steps),
                label=f"{config.run_id}-{ablation.name}",
                env={**WEDGE_PROVENANCE_ENV, **ablation.env},
            )

            faults = ", ".join(
                f"attempt={fault.attempt} class={fault.fault_class} returncode={fault.returncode} "
                f"detail={fault.detail!r}"
                for fault in result.faults
            )
            if result.faults:
                logger.warning(
                    "WEDGE DETECTED for ablation %s: outcome=%s attempts=%d faults=[%s]",
                    ablation.name,
                    result.outcome,
                    result.attempts,
                    faults,
                )
            elif result.outcome is RunOutcome.COMPLETED:
                logger.warning("no wedge reproduced for ablation %s through %d steps", ablation.name, ablation.num_steps)
            else:
                logger.warning("ablation %s ended with outcome=%s and no recorded fault", ablation.name, result.outcome)


def build_supervised_wedge_run(
    *,
    run_id: str,
    dp_racks: int,
    num_steps: int,
    ablation_names: tuple[str, ...],
    execution_terminate_timeout: float,
    max_restarts: int,
    version: str | None = None,
) -> ArtifactStep:
    ablations = tuple(selected_ablations(environment_ablations(num_steps=num_steps), ablation_names))
    resources = ResourceConfig.with_gpu(
        "GB200", count=4, cpu=120, ram="850g", disk="1t", replicas=HERO_NODES_PER_RACK * dp_racks
    )
    name = f"grug/{run_id}"
    version = resolve_version(name, version)

    def build_config(ctx: StepContext) -> SupervisedWedgeConfig:
        return SupervisedWedgeConfig(
            run_id=run_id,
            resources=ctx.runtime_arg("train_resources"),
            dp_racks=dp_racks,
            ablations=ablations,
            execution_terminate_timeout=execution_terminate_timeout,
            max_restarts=max_restarts,
        )

    return ArtifactStep(
        name=user_namespaced_name(name, version),
        version=version,
        artifact_type=SupervisedWedgeResult,
        run=_run_dispatch,
        build_config=build_config,
        deps=(),
        runtime_args={"train_resources": resources},
    )


def _run_dispatch(config: SupervisedWedgeConfig) -> None:
    # Iris does not retry the gang on failure: the supervisor owns recovery, and a
    # retry would just re-wedge an expensive 128-GPU gang.
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_supervised,
        resources=config.resources,
        max_retries_failure=0,
        processes_per_task=1,
    )


@click.command()
@click.option("--run-id", required=True)
@click.option("--dp-racks", type=click.IntRange(min=2), required=True, help="Rack count; >=2 to span racks.")
@click.option("--num-steps", type=int, default=20000, show_default=True)
@click.option(
    "--ablation",
    "ablation_names",
    type=click.Choice(environment_ablation_names()),
    multiple=True,
    default=("baseline",),
    show_default=True,
    help="Environment arm to run; repeat the option to sweep several arms on one allocation.",
)
@click.option(
    "--execution-terminate-timeout",
    type=float,
    default=DEFAULT_EXECUTION_TERMINATE_TIMEOUT,
    show_default=True,
    help="XLA per-execution deadman budget, in seconds.",
)
@click.option("--max-restarts", type=int, default=0, show_default=True, help="Supervisor warm-restart budget.")
@build_options
def main(
    run_id: str,
    dp_racks: int,
    num_steps: int,
    ablation_names: tuple[str, ...],
    execution_terminate_timeout: float,
    max_restarts: int,
):
    return build_supervised_wedge_run(
        run_id=run_id,
        dp_racks=dp_racks,
        num_steps=num_steps,
        ablation_names=ablation_names,
        execution_terminate_timeout=execution_terminate_timeout,
        max_restarts=max_restarts,
    )


if __name__ == "__main__":
    main()
