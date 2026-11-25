# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Guarded staging and measured rollout of the existing reconciler on TPU v4."""

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

from finestore.rl.difficulty import Solver
from iris.cluster.types import JobName
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.task_curation.difficulty.campaign import identity_bundle
from experiments.post_training.task_curation.difficulty.catalog import DEFAULT_APPLET
from experiments.post_training.task_curation.difficulty.controller import controller_environment, launch_controller
from experiments.post_training.task_curation.difficulty.launch import cpu_job_command
from experiments.post_training.task_curation.difficulty.ledger import digest
from experiments.post_training.task_curation.difficulty.tpu import CHIP_BUDGET, REGION, REGIONAL_ROOT, SHAPES, ZONE


def source_digest() -> str:
    paths = sorted(
        {
            *Path("experiments/post_training/task_curation/difficulty").glob("*.py"),
            *Path("experiments/post_training/task_curation").glob("difficulty_*.py"),
            *Path("lib/marin/src/marin/inference").glob("*.py"),
            *Path("vendor/harbor/src/harbor/trial").glob("*.py"),
            *Path("lib/taskcompendium/src/taskcompendium/harbor").glob("*.py"),
            Path("config/external/vllm/tpu.toml"),
            Path("scripts/ops/difficulty_failure_watch.py"),
        }
    )
    checksum = hashlib.sha256()
    for path in paths:
        checksum.update(str(path).encode() + b"\0" + path.read_bytes())
    return checksum.hexdigest()


def regional_cpu_command(name: str, *, memory: str = "8GB", cpu: str = "4") -> list[str]:
    command = cpu_job_command(name)
    command.remove("--target-cluster=cw-rno2a")
    command[command.index("--max-retries") + 1] = "3"
    command[command.index("--memory") + 1] = memory
    command[command.index("--cpu") + 1] = cpu
    command[command.index("--disk") + 1] = "64GB"
    command.extend(["--region", REGION, "--zone", ZONE, "--env-vars", "MARIN_TEMP_PREFIX", "gs://marin-us-central2"])
    for name in ("CW_KEY_ID", "CW_KEY_SECRET", "HF_TOKEN"):
        command.extend(["--secret", name, "env:" + name])
    return command


def commands_for_phase(
    phase: str,
    root: str,
    generation: str,
    manifest: str,
    sha: str,
    qualification: str,
    solver_only: Solver | None = None,
) -> list[list[str]]:
    commands = []
    module = "experiments.post_training.task_curation.difficulty.tpu"
    if phase == "stage":
        for solver in SHAPES:
            command = regional_cpu_command(f"atlas-difficulty-stage-{solver.value}-v4-{generation}")
            commands.append(
                [*command, "--", "python", "-m", module, "stage-solver", "--root", root, "--solver", solver.value]
            )
        command = regional_cpu_command(f"atlas-difficulty-stage-inputs-v4-{generation}")
        commands.append(
            [
                *command,
                "--",
                "python",
                "-m",
                module,
                "stage-inputs",
                "--root",
                root,
                "--sealed-manifest",
                manifest,
                "--sealed-sha256",
                sha,
                "--judge-qualification",
                qualification,
            ]
        )
    elif phase in {"pilot", "full"}:
        for solver, shape in SHAPES.items():
            if solver_only is not None and solver != solver_only:
                continue
            count = 1 if phase == "pilot" else shape.replicas
            command = regional_cpu_command(
                f"atlas-difficulty-engines-{solver.value}-v4-{generation}-{phase}", cpu="2", memory="4GB"
            )
            command[command.index("--timeout") + 1] = "0"
            commands.append(
                [
                    *command,
                    "--",
                    "python",
                    "-m",
                    "experiments.post_training.task_curation.difficulty.engine_pool",
                    "--solver",
                    solver.value,
                    "--registry",
                    root + "/pools",
                    "--tpu-root",
                    root,
                    "--replicas",
                    str(count),
                ]
            )
    else:
        raise ValueError(f"Unknown launch phase {phase}")
    return commands


def validate_gate(path: Path) -> None:
    gate = json.loads(path.read_text())
    if gate.get("passed") is not True or gate.get("source_digest") != source_digest():
        raise ValueError("Local E2E gate is missing, failed or stale; validate all guards before submission")


def validate_staging(root: str, solvers: tuple[Solver, ...] = tuple(SHAPES)) -> None:
    for solver in solvers:
        receipt = json.loads((StoragePath(root) / "staging" / f"{solver.value}.json").read_text())
        if not receipt["weights"].startswith("gs://marin-us-central2/"):
            raise ValueError("Staged solver weights are outside the target region")
    if not (StoragePath(root) / "ledger/v1/campaign.json").exists():
        raise ValueError("Regional frozen task inputs are not ready")


def validate_pilot(root: str, solvers: tuple[Solver, ...] = tuple(SHAPES)) -> None:
    for solver in solvers:
        receipt = json.loads((StoragePath(root) / "pools" / f"{solver.value}.json").read_text())
        if receipt["platform"] != "tpu" or receipt["chip_count"] != SHAPES[solver].chips:
            raise ValueError("Every solver needs one qualified pilot slice before the full allocation")
        if any(replica["warm_throughput"]["output_tokens_per_second"] <= 0 for replica in receipt["replicas"]):
            raise ValueError("A pilot has no completed throughput measurement")
    if sum(shape.chips * shape.replicas for shape in SHAPES.values()) != CHIP_BUDGET:
        raise ValueError("Full layout does not fit the 256-chip allocation")


def await_pilot_teardown(
    root: str, timeout_seconds: float = 600, *, solvers: tuple[Solver, ...] = tuple(SHAPES)
) -> None:
    """Release selected pilot chips before expanding toward the 256-chip allocation."""
    children = []
    for solver in solvers:
        receipt = json.loads((StoragePath(root) / "pools" / f"{solver.value}.json").read_text())
        children.extend(job for replica in receipt["replicas"] for job in replica["engine_jobs"])
    deadline = time.monotonic() + timeout_seconds
    while True:
        active = []
        for child in children:
            result = subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "iris",
                    "--cluster",
                    "marin",
                    "rpc",
                    "controller",
                    "get-job-status",
                    "--job-id",
                    child,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            status = json.loads(result.stdout)["job"]
            if any(
                count and state not in {"succeeded", "failed", "killed"}
                for state, count in status["task_state_counts"].items()
            ):
                active.append(child)
        if not active:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Pilot tasks still hold TPU chips; refusing full allocation: {active}")
        print(json.dumps({"waiting_for_pilot_task_teardown": active}), flush=True)
        time.sleep(5)


def pilot_owners(root: str, solvers: tuple[Solver, ...] = tuple(SHAPES)) -> list[str]:
    owners = set()
    for solver in solvers:
        receipt = json.loads((StoragePath(root) / "pools" / f"{solver.value}.json").read_text())
        for replica in receipt["replicas"]:
            for job in replica["engine_jobs"]:
                owner = str(JobName.from_string(job).root_job)
                if not owner.startswith(f"/benfeuer/atlas-difficulty-engines-{solver.value}-v4-") or not owner.endswith(
                    "-pilot"
                ):
                    raise ValueError(f"Receipt does not identify an owned pilot: {owner}")
                owners.add(owner)
    return sorted(owners)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "pilot", "full", "controller"))
    parser.add_argument("--generation", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--secrets-env", type=Path, required=True)
    parser.add_argument("--sealed-manifest", required=True)
    parser.add_argument("--sealed-sha256", required=True)
    parser.add_argument("--judge-qualification", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--solver", type=Solver, choices=tuple(SHAPES))
    args = parser.parse_args()
    if args.solver is not None and args.phase not in {"pilot", "full"}:
        parser.error("--solver selects a staged pilot or a qualified solver's full rollout")
    root = REGIONAL_ROOT + "/" + digest(identity_bundle())
    validate_gate(args.gate)
    if args.phase != "stage":
        validate_staging(root, (args.solver,) if args.solver is not None else tuple(SHAPES))
    if args.phase == "full":
        validate_pilot(root, (args.solver,) if args.solver is not None else tuple(SHAPES))
    if args.phase == "controller":
        if args.dry_run:
            print(json.dumps({"ledger": root + "/ledger", "serving_pool": root + "/pools", "budget": None}))
            return
        launch_controller(
            root + "/ledger",
            DEFAULT_APPLET,
            False,
            frozenset(Solver),
            args.generation,
            None,
            args.secrets_env,
            root + "/pools",
            excluded_sources=frozenset(),
        )
        return
    commands = commands_for_phase(
        args.phase,
        root,
        args.generation,
        args.sealed_manifest,
        args.sealed_sha256,
        args.judge_qualification,
        args.solver,
    )
    print(json.dumps({"phase": args.phase, "root": root, "commands": commands}, indent=2), flush=True)
    if args.dry_run:
        return
    environment = controller_environment(args.secrets_env)
    if args.phase == "full":
        # Replace each selected pilot with its final allocation. Unselected
        # solvers retain either a pilot or their final share of the 256 chips.
        selected = (args.solver,) if args.solver is not None else tuple(SHAPES)
        for name in pilot_owners(root, selected):
            subprocess.run(
                ["iris-cancel-own", "--cluster", "marin", name],
                check=True,
                env={**environment, "IRIS_AGENT_PREFIX": "/benfeuer/atlas-difficulty-engines-"},
            )
        await_pilot_teardown(root, solvers=selected)
    for command in commands:
        subprocess.run(command, check=True, env=environment)


if __name__ == "__main__":
    main()
