# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Agentic canary driver: validate the agent's experiment, launch it under a clamped
consumption envelope, and check that it trained.

An agent authors a grug-base experiment from the repo + its skills; CI runs it, but never
trusts the agent's consumption choices. Two gates bracket the run:

  Gate A (pre-launch, CPU, no accelerator) — "is the authored experiment usable and safe to
    launch?" It must import and expose a valid `launch`, and the pinned dataset must be present
    in-region. A Gate-A failure is an ergonomics/safety problem, so no accelerator is spent.
  Gate B (post-run) — "did the launched run actually train?" It must have produced enough
    finite loss steps. A Gate-B failure points at infra/training, not the agent.

Subcommands:
  check  --module M   Gate A's code half (agent self-check, no cloud): M imports + exposes a valid `launch`.
  gate-a --module M   check + assert the pinned dataset exists in-region. Fails closed.
  launch --module M   (Iris worker) import M's `launch`, REPLACE consumption (resources/data/steps)
                      with the clamp, assert it held, submit via train_grug. Agent's picks are recorded, not used.
  gate-b --run-id R   read R's train/loss from W&B; assert it reached enough finite steps. Fails closed.

Safety (cost + egress) comes from the clamp + preflight here — never from trusting the agent.
The agent's freedom is the authoring (model / optimizer / trainer); the accelerator, dataset,
region, and step count are CI's.
"""

import argparse
import dataclasses
import importlib
import math
import os
import subprocess

import wandb
from fray.cluster import ResourceConfig
from levanter.tracker.wandb import WandbConfig
from marin.execution.executor import unwrap_versioned_value
from marin.execution.types import this_output_path, versioned

from experiments.grug.base.launch import GrugBaseLaunchConfig, train_grug
from experiments.pretraining_datasets import nemotron_mix

# --- The clamp: the fixed run envelope CI forces, ignoring whatever the agent wrote —
# one small TPU slice in a pinned region, plus a pinned dataset and a step cap (below). ---
ALLOWED_TPU_VARIANTS = ("v6e-8", "v5p-8", "v4-8")  # single-VM TPU variants (1 slice == 1 replica)
CLAMP_TPU_VARIANT = "v6e-8"
CLAMP_REGION = "us-east5"
CLAMP_SLICE_COUNT = 1
STEP_CAP = 500  # hard ceiling on steps, even if env asks for more

# Pinned, region-local training mixture (verified present in gs://marin-us-east5).
# The agent's `data` choice is recorded but never used — this is what actually runs,
# so reads stay in-region (no egress). Relative paths resolve under gs://marin-<region>.
PINNED_DATA = nemotron_mix
PINNED_DATA_COMPONENT_PATHS = (
    "tokenized/nemotron_cc/hq_actual-5af4cc",
    "tokenized/nemotron_cc/hq_synth-3525e2",
    "tokenized/nemotron_cc/low_actual-cb3f2c",
    "tokenized/nemotron_cc/low_synth-3c57b3",
    "tokenized/nemotron_cc/medium-d86506",
    "tokenized/nemotron_cc/medium_high-d21701",
    "tokenized/nemotron_cc/medium_low-0fdb07",
    "tokenized/starcoderdata-12f018",
    "tokenized/proofpile_2-4a35c7",
)

# --- Gate B: post-run health check (did it train at all — not how well) ---
TRAIN_LOSS_KEY = "train/loss"
MIN_STEPS = 100  # must reach this many logged steps

CANARY_NAME = "grug/agentic-canary"
WANDB_GROUP = "agentic-canary"
# Training env forwarded from the launcher into the worker.
CHILD_ENV_KEYS = ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT", "HF_TOKEN", "MARIN_PREFIX")


def _load_launch(module_path: str) -> GrugBaseLaunchConfig:
    """Import the agent's module and return its `launch`, asserting the contract."""
    module = importlib.import_module(module_path)
    launch = getattr(module, "launch", None)
    if launch is None:
        raise SystemExit(f"GATE A FAILED: {module_path} has no module-level `launch`.")
    if not isinstance(launch, GrugBaseLaunchConfig):
        raise SystemExit(
            f"GATE A FAILED: {module_path}.launch is {type(launch).__name__}, "
            "expected experiments.grug.base.launch.GrugBaseLaunchConfig."
        )
    return launch


def _clamped_resources() -> ResourceConfig:
    return ResourceConfig.with_tpu(CLAMP_TPU_VARIANT, slice_count=CLAMP_SLICE_COUNT, regions=[CLAMP_REGION])


def _assert_in_envelope(resources: ResourceConfig) -> None:
    """Defense-in-depth: verify the clamped resources match the envelope — an allowed
    single-VM variant, one slice, pinned region."""
    variant = resources.device.variant
    if variant not in ALLOWED_TPU_VARIANTS:
        raise SystemExit(f"CLAMP FAILED: variant {variant!r} not in {ALLOWED_TPU_VARIANTS}.")
    if resources.device_alternatives:
        raise SystemExit(f"CLAMP FAILED: unexpected device_alternatives {resources.device_alternatives}.")
    if resources.regions != [CLAMP_REGION]:
        raise SystemExit(f"CLAMP FAILED: regions {resources.regions} != [{CLAMP_REGION!r}].")
    if resources.replicas > CLAMP_SLICE_COUNT:
        raise SystemExit(f"CLAMP FAILED: replicas {resources.replicas} > {CLAMP_SLICE_COUNT}.")


def _data_preflight(region: str) -> None:
    """Assert every pinned data component exists in the target region (no egress / no miss)."""
    missing = []
    for path in PINNED_DATA_COMPONENT_PATHS:
        url = f"gs://marin-{region}/{path}/"
        result = subprocess.run(
            ["gcloud", "storage", "ls", url],
            capture_output=True,
            text=True,
        )
        status = "ok" if result.returncode == 0 else "MISSING"
        print(f"  [{status}] {url}")
        if result.returncode != 0:
            missing.append(url)
    if missing:
        raise SystemExit(f"GATE A FAILED: {len(missing)} pinned data component(s) absent in {region}: {missing}")


def _wandb_target() -> tuple[str | None, str]:
    """(entity, project) for W&B reads and writes, from the environment."""
    return os.environ.get("WANDB_ENTITY") or None, os.environ.get("WANDB_PROJECT", "marin")


def _tracker() -> WandbConfig:
    entity, project = _wandb_target()
    return WandbConfig(
        entity=entity,
        project=project,
        tags=["grug", "agentic-canary"],
        group=WANDB_GROUP,
        name=None,  # the grug launcher sets the W&B run name from the launch run_id
        replicate_path=this_output_path(),
    )


def validate_module(module_path: str) -> GrugBaseLaunchConfig:
    """Import the agent's module, assert the contract, report its (untrusted) picks.

    The code half of Gate A — no cloud access — so the authoring agent can run it
    itself (`check`) to validate before CI does.
    """
    launch = _load_launch(module_path)
    print(f"OK: {module_path}.launch is a valid GrugBaseLaunchConfig.")

    # Record (do not trust) what the agent chose for the clamped dimensions. Best-effort:
    # the values are overridden at launch, so a weird shape must not fail validation.
    chosen = unwrap_versioned_value(launch.resources)
    variant = getattr(getattr(chosen, "device", None), "variant", "?")
    components = getattr(getattr(launch, "data", None), "components", {})
    print(
        f"  agent chose: accelerator={variant} replicas={getattr(chosen, 'replicas', '?')} "
        f"steps={unwrap_versioned_value(launch.steps)} data_components={len(components)} "
        "(all overridden by the clamp at launch)"
    )
    return launch


def check(module_path: str) -> None:
    """Agent self-check: the module imports and exposes a valid launch (no cloud, no launch)."""
    validate_module(module_path)
    print("CHECK PASSED.")


def gate_a(module_path: str, region: str) -> None:
    validate_module(module_path)
    print(f"GATE A: data preflight for the pinned mixture in {region}:")
    _data_preflight(region)
    print("GATE A PASSED.")


def launch_cmd(module_path: str) -> None:
    launch = _load_launch(module_path)
    run_id = os.environ["RUN_ID"]
    steps = min(int(os.environ.get("CANARY_STEPS", STEP_CAP)), STEP_CAP)

    clamped = dataclasses.replace(
        launch,
        resources=versioned(_clamped_resources()),
        data=PINNED_DATA,
        steps=versioned(steps),
        output_path=this_output_path(),
        run_id=run_id,
        tracker=_tracker(),
        eval=None,
    )
    _assert_in_envelope(unwrap_versioned_value(clamped.resources))
    if clamped.data is not PINNED_DATA:
        raise SystemExit("CLAMP FAILED: data is not the pinned in-region mixture.")

    env_vars = {key: os.environ[key] for key in CHILD_ENV_KEYS if os.environ.get(key)}
    print(f"LAUNCH: run_id={run_id} steps={steps} variant={CLAMP_TPU_VARIANT} region={CLAMP_REGION}")
    train_grug(name=CANARY_NAME, launch=clamped, env_vars=env_vars)


def _loss_points(run_id: str) -> list[tuple[int, float]]:
    entity, project = _wandb_target()
    run_path = f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}"
    run = wandb.Api(timeout=60).run(run_path)
    points: list[tuple[int, float]] = []
    for row in run.scan_history(keys=["_step", TRAIN_LOSS_KEY]):
        step, loss = row.get("_step"), row.get(TRAIN_LOSS_KEY)
        if not isinstance(loss, (int, float)):
            continue
        if not math.isfinite(loss):
            raise SystemExit(f"GATE B FAILED: non-finite {TRAIN_LOSS_KEY}={loss} at step {step}.")
        if isinstance(step, (int, float)):
            points.append((int(step), float(loss)))
    return points


def gate_b(run_id: str) -> None:
    # _loss_points already fails closed on any non-finite loss (NaN/Inf = divergence/crash).
    points = _loss_points(run_id)
    if not points:
        raise SystemExit(f"GATE B FAILED: no {TRAIN_LOSS_KEY} points for run {run_id}.")

    max_step = max(s for s, _ in points)
    print(f"GATE B: {len(points)} finite loss points, max_step={max_step}, final={points[-1][1]:.3f}")
    if max_step < MIN_STEPS:
        raise SystemExit(f"GATE B FAILED: only reached step {max_step} (< {MIN_STEPS}).")
    print("GATE B PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_c = sub.add_parser("check", help="Import + type-check the agent's module (no cloud; agent self-check).")
    p_c.add_argument("--module", required=True, help="Dotted path to the agent's module.")

    p_a = sub.add_parser("gate-a", help="Validate the agent's module + pinned-data preflight (pre-launch).")
    p_a.add_argument("--module", required=True, help="Dotted path to the agent's module.")
    p_a.add_argument("--region", default=CLAMP_REGION)

    p_l = sub.add_parser("launch", help="Clamp consumption and submit the run (Iris worker entrypoint).")
    p_l.add_argument("--module", required=True, help="Dotted path to the agent's module.")

    p_b = sub.add_parser("gate-b", help="Check loss health from W&B (post-run).")
    p_b.add_argument("--run-id", required=True)

    args = parser.parse_args()
    if args.command == "check":
        check(args.module)
    elif args.command == "gate-a":
        gate_a(args.module, args.region)
    elif args.command == "launch":
        launch_cmd(args.module)
    elif args.command == "gate-b":
        gate_b(args.run_id)


if __name__ == "__main__":
    main()
