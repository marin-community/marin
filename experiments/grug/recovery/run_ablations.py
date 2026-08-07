# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Supervisor-driven ablation sweep for the grug hang-recovery system.

Demonstrates the two things the framework buys us on a single warm pod:

1. Recovery: inject a fault mid-run; the supervisor detects it, warm-restarts the
   trainer from the host snapshot, and the run completes.
2. Cheap ablation iteration: sweep NCCL/CUDA/XLA env vars back-to-back without a
   scheduler round-trip or a cold compile (the compilation cache stays warm on
   the node), each ablation getting a fresh subprocess so process-start env vars
   actually take effect.

Run on a reserved GPU node::

    python -m experiments.grug.recovery.run_ablations
    python -m experiments.grug.recovery.run_ablations --smoke   # one direct run, no supervisor
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import tempfile

from levanter.recovery.detection import ENV_DISABLE_WATCHDOG
from levanter.recovery.faults import ENV_FAULT_KIND, ENV_FAULT_STEP, FaultKind
from levanter.recovery.supervisor import ENV_RUN_ID, ENV_SNAPSHOT_TMPFS_BASE, GPUHangSupervisor
from levanter.recovery.types import DEFAULT_TMPFS_DIR, AblationSpec, RunResult

from experiments.grug.recovery.ablation_catalog import environment_ablations, selected_ablations
from experiments.grug.recovery.config import RecoveryRunConfig, grug_model_preset
from experiments.grug.recovery.train import recovery_train

logger = logging.getLogger(__name__)


def build_ablations(*, fault_step: int, sweep_steps: int) -> list[AblationSpec]:
    """Recovery drills (one per detection path) followed by an NCCL/CUDA env sweep."""
    return [
        AblationSpec(
            name="recover-spin-host-deadman",
            env={ENV_FAULT_KIND: FaultKind.SPIN.value, ENV_FAULT_STEP: str(fault_step)},
            notes="silent GPU wedge; supervisor heartbeat deadman detects, warm-restarts",
        ),
        AblationSpec(
            name="recover-sticky-sentinel",
            env={ENV_FAULT_KIND: FaultKind.STICKY.value, ENV_FAULT_STEP: str(fault_step)},
            notes="poisoned-context (#7956) sentinel exits 125, warm-restarts",
        ),
        AblationSpec(
            name="xla-execution-deadman-probe",
            env={
                ENV_FAULT_KIND: FaultKind.SPIN.value,
                ENV_FAULT_STEP: str(fault_step),
                "XLA_FLAGS": "--xla_gpu_execution_terminate_timeout=60s --xla_gpu_execution_progress_tracking=8",
                ENV_DISABLE_WATCHDOG: "1",
            },
            deadman_timeout=180.0,
            notes="isolate the JAX-level deadman: CRASH => XLA killed the wedge; STALL => host deadman had to",
        ),
        *environment_ablations(num_steps=sweep_steps),
    ]


def _summary(results: list[RunResult]) -> str:
    header = f"{'ablation':<28} {'outcome':<14} {'attempts':>8} {'faults':>7} {'final_step':>11} {'wall_s':>8}"
    lines = [header, "-" * len(header)]
    for r in results:
        fault_classes = ",".join(f.fault_class.value for f in r.faults) or "-"
        lines.append(
            f"{r.label:<28} {r.outcome.value:<14} {r.attempts:>8} {fault_classes:>7} "
            f"{r.final_step!s:>11} {r.total_wall_time:>8.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="10b", choices=["tiny", "1b", "3b", "10b"])
    parser.add_argument("--num-steps", type=int, default=24)
    parser.add_argument("--sweep-steps", type=int, default=8)
    parser.add_argument("--fault-step", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--model-axis-size", type=int, default=1)
    parser.add_argument("--snapshot-every", type=int, default=4)
    parser.add_argument("--deadman-timeout", type=float, default=45.0)
    parser.add_argument("--tmpfs-base", default=DEFAULT_TMPFS_DIR, help="host-RAM snapshot tier")
    parser.add_argument("--compile-cache", default=None, help="JAX_COMPILATION_CACHE_DIR (defaults under $TMPDIR)")
    parser.add_argument("--smoke", action="store_true", help="run recovery_train once directly, no supervisor")
    parser.add_argument("--only", nargs="*", default=None, help="subset of ablation names to run")
    args = parser.parse_args()

    config = RecoveryRunConfig(
        model=grug_model_preset(args.model, max_seq_len=args.seq_len),
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        snapshot_every_steps=args.snapshot_every,
        model_axis_size=args.model_axis_size,
    )

    if args.smoke:
        os.environ[ENV_SNAPSHOT_TMPFS_BASE] = args.tmpfs_base
        os.environ[ENV_RUN_ID] = "smoke"
        recovery_train(config)
        return

    compile_cache = args.compile_cache or os.path.join(tempfile.gettempdir(), "levanter-recovery-jax-cache")
    common_env = {"JAX_COMPILATION_CACHE_DIR": compile_cache}

    ablations = build_ablations(fault_step=args.fault_step, sweep_steps=args.sweep_steps)
    ablations = selected_ablations(ablations, args.only)

    results: list[RunResult] = []
    with GPUHangSupervisor(
        detection=config.detection,
        deadman_timeout=args.deadman_timeout,
        snapshot_tmpfs_base=args.tmpfs_base,
    ) as supervisor:
        for ablation in ablations:
            run_config = (
                config if ablation.num_steps is None else dataclasses.replace(config, num_steps=ablation.num_steps)
            )
            logger.info("=== ablation %s: %s ===", ablation.name, ablation.notes)
            result = supervisor.run(
                recovery_train,
                run_config,
                label=ablation.name,
                env={**common_env, **ablation.env},
                deadman_timeout=ablation.deadman_timeout,
            )
            results.append(result)
            logger.info("ablation %s -> %s", ablation.name, result.outcome.value)

    print("\n" + _summary(results) + "\n")


if __name__ == "__main__":
    main()
