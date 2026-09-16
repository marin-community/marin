# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Plan, collect, or explicitly release the reviewed TPP10 experiment."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import tomllib
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path

import fsspec
from fray.types import ResourceConfig
from levanter.optim.muonh import MuonHConfig
from marin.execution.artifact import read_record
from marin.execution.lazy import ArtifactStep, StepContext, run
from marin.execution.remote import remote
from marin.execution.step_status import STATUS_SUCCESS, StatusFile
from marin.experiment.train import train_lm
from marin.training.training import LevanterCheckpoint, TrainLmOnPodConfig, run_levanter_train_lm

from experiments.domain_phase_mix import prepare_starcoder_tpp10 as preparation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.launch_starcoder_epoch_matching import pending_training_steps, persist_submission_plan
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

REPO = Path(__file__).resolve().parents[2]
TPU = ResourceConfig.with_tpu("v5p-8", regions=(experiment.REGION,), zone=experiment.ZONE)


@dataclass(frozen=True)
class TrainingRecipe:
    pod: TrainLmOnPodConfig
    design_sha256: str
    code_sha256: dict[str, str]
    runtime_versions: dict[str, str]


def code_pins() -> dict[str, str]:
    paths = (
        "experiments/domain_phase_mix/starcoder_tpp10.py",
        "experiments/domain_phase_mix/prepare_starcoder_tpp10.py",
        "experiments/domain_phase_mix/launch_starcoder_tpp10.py",
        "experiments/domain_phase_mix/launch_starcoder_epoch_matching.py",
        "experiments/domain_phase_mix/starcoder_epoch_matching.py",
        "lib/levanter/src/levanter/data/text/datasets.py",
        "lib/levanter/src/levanter/data/text/formats.py",
        "lib/levanter/src/levanter/data/text/cache.py",
        "lib/levanter/src/levanter/store/cache.py",
        "lib/levanter/src/levanter/data/mixture.py",
        "lib/levanter/src/levanter/data/dataset.py",
        "lib/levanter/src/levanter/tokenizers.py",
        "lib/levanter/src/levanter/models/qwen.py",
        "lib/levanter/src/levanter/models/llama.py",
        "lib/levanter/src/levanter/optim/muonh.py",
        "lib/levanter/src/levanter/main/train_lm.py",
        "lib/levanter/src/levanter/trainer.py",
        "lib/marin/src/marin/experiment/train.py",
        "lib/marin/src/marin/training/training.py",
        "uv.lock",
    )
    assets = tuple(str(p.relative_to(REPO)) for p in sorted(experiment.ASSETS.iterdir()) if p.is_file())
    return {path: file_sha256(REPO / path) for path in (*paths, *assets)}


def runtime_versions() -> dict[str, str]:
    lock = tomllib.loads((REPO / "uv.lock").read_text())
    versions = {}
    for name in ("jax", "jaxlib", "numpy", "tokenizers"):
        values = {package["version"] for package in lock["package"] if package["name"] == name}
        if len(values) != 1:
            raise ValueError(f"Ambiguous lock version: {name}")
        versions[name] = values.pop()
    return versions


def verified_training(recipe: TrainingRecipe) -> None:
    # Do not initialize a JAX backend here: Levanter initializes the TPU distributed runtime.
    experiment.require_central1()
    if code_pins() != recipe.code_sha256:
        raise ValueError("Child code/lock differs from the reviewed training recipe")
    for name, digest in json.loads((experiment.ASSETS / "pins.json").read_text())["files_sha256"].items():
        if file_sha256(experiment.ASSETS / name) != digest:
            raise ValueError(f"Child asset mismatch: {name}")
    experiment.verified_tokenizer()
    observed = {name: version(name) for name in recipe.runtime_versions}
    if observed != recipe.runtime_versions:
        raise ValueError(f"Child dependency versions differ: {observed}")
    persist_submission_plan(
        {"design_sha256": recipe.design_sha256, "versions": observed, "code_sha256": recipe.code_sha256},
        f"{recipe.pod.output_path}/verified_runtime.json",
    )
    run_levanter_train_lm(recipe.pod)


def dispatch_training(recipe: TrainingRecipe) -> None:
    remote(verified_training, resources=recipe.pod.resources, env_vars={"MARIN_PREFIX": experiment.PREFIX})(recipe)


def training_step(design: dict, request: experiment.RunSpec, caches: dict) -> ArtifactStep[LevanterCheckpoint]:
    code = code_pins()
    versions = runtime_versions()
    selected = caches[f"subset_{request.subset_seed}"] if request.arm == experiment.Arm.MATCHED else caches["starcoder"]
    weights = experiment.mixture_weights(request.percent)
    datasets = {caches[name]: weights[name] for name in experiment.WEB_COUNTS}
    datasets[selected] = weights["starcoder"]
    end_stable = int(0.8 * request.total_steps)
    optimizer = MuonHConfig(
        learning_rate=0.02,
        adam_lr=0.008,
        min_lr_ratio=0.0,
        momentum=0.95,
        beta1=0.9,
        beta2=0.98,
        epsilon=1e-15,
        muon_epsilon=1e-5,
        max_grad_norm=1.0,
        warmup=int(0.01 * request.total_steps),
        decay=request.total_steps - end_stable,
        rewarmup=0.0,
        lr_schedule="cosine",
        cycle_length=None,
    )
    base = train_lm(
        name=f"checkpoints/starcoder_tpp10/{request.run_name}",
        version=experiment.VERSION,
        model=experiment.model_config(request.arm),
        optimizer=optimizer,
        datasets=datasets,
        validation=(caches["evaluation"],),
        batch_size=request.batch_size,
        seq_len=experiment.SEQ_LEN,
        num_train_steps=request.total_steps,
        z_loss_weight=None,
        evals=None,
        resources=TPU,
        steps_per_eval=max(1, request.total_steps // 5),
        wandb_project="marin",
        wandb_group="starcoder_tpp10_20260909",
        run_id=request.run_name,
        tags=("starcoder_tpp10", request.arm),
    )

    def config(ctx: StepContext) -> TrainingRecipe:
        pod = base.build_config(ctx)
        data = pod.train_config.data
        source_names = {caches[name].name: name for name in experiment.WEB_COUNTS}
        source_names[selected.name] = "starcoder"
        components = {source_names.get(name, name): component for name, component in data.components.items()}
        # Stable named keys preserve web order when StarCoder or other zero-weight sources disappear.
        keys = {
            name: (
                int(canonical_sha256({"component": name, "seed": experiment.DATA_SEED})[:8], 16),
                experiment.DATA_SEED,
            )
            for name in weights
        }
        data = replace(
            data,
            components=components,
            train_weights={**weights, caches["evaluation"].name: 0.0},
            mixture_block_size=experiment.BLOCK_SIZE,
            train_component_shuffle_keys=keys,
            auto_build_caches=False,
        )
        trainer = replace(
            pod.train_config.trainer,
            seed=request.trainer_seed,
            checkpointer=replace(pod.train_config.trainer.checkpointer, keep=None),
        )
        pod = replace(
            pod, train_config=replace(pod.train_config, data=data, data_seed=experiment.DATA_SEED, trainer=trainer)
        )
        return TrainingRecipe(pod, design["design_sha256"], code, versions)

    step = replace(base, build_config=config, run=dispatch_training)
    return replace(step, expected_fingerprint=step.fingerprint())


def build_plan(design: dict, stage: str) -> tuple[dict, tuple[ArtifactStep[LevanterCheckpoint], ...]]:
    caches = preparation.data_steps(design)
    rows = experiment.select_runs(design, stage)
    steps = tuple(training_step(design, row, caches) for row in rows)
    plan = {
        "design_sha256": design["design_sha256"],
        "stage": stage,
        "primary_metric": experiment.PRIMARY_METRIC,
        "code_sha256": code_pins(),
        "runtime_versions": runtime_versions(),
        "marin_prefix": experiment.PREFIX,
        "region": experiment.REGION,
        "zone": experiment.ZONE,
        "training_flops": sum(
            design["models"]["target" if row.arm == experiment.Arm.TARGET else "unmatched"]["training_flops"]
            for row in rows
        ),
        "runs": [
            {
                **asdict(row),
                "tokens": row.tokens,
                "support_sequences": row.support_sequences,
                "fingerprint": step.fingerprint(),
                "output_path": step.path(experiment.PREFIX),
            }
            for row, step in zip(rows, steps, strict=True)
        ],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan, steps


def collect_results(plan: dict, output: Path) -> dict[str, float]:
    """Collect against an archived plan, without rebuilding identities from today's checkout."""
    experiment.validate_plan(plan)
    rows = []
    measurements: dict[str, float] = {}
    for request in plan["runs"]:
        path = request["output_path"]
        if StatusFile(path, worker_id="tpp10-collection").status != STATUS_SUCCESS:
            raise ValueError(f"Training incomplete: {request['run_name']}")
        record = read_record(path)
        if record is None or record.fingerprint != request["fingerprint"]:
            raise ValueError(f"Artifact fingerprint differs from the archived plan: {request['run_name']}")
        with fsspec.open(path + "/verified_runtime.json", "rt") as handle:
            runtime = json.load(handle)
        if runtime != {
            "design_sha256": plan["design_sha256"],
            "versions": plan["runtime_versions"],
            "code_sha256": plan["code_sha256"],
        }:
            raise ValueError(f"Unverified child environment: {request['run_name']}")
        expected_step = request["total_steps"] - 1
        values = []
        with fsspec.open(LevanterCheckpoint(path=path).checkpoint_dir + "/eval_metrics.jsonl", "rt") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("step") == expected_step and plan["primary_metric"] in item:
                    values.append(float(item[plan["primary_metric"]]))
        if not values or not all(math.isfinite(v) and v == values[0] for v in values):
            raise ValueError(f"Missing, conflicting or nonfinite endpoint: {request['run_name']}")
        measurements[request["run_name"]] = values[0]
        rows.append(
            {
                "run_name": request["run_name"],
                "step": expected_step,
                "value": values[0],
                "metric": plan["primary_metric"],
                "config_fingerprint": request["fingerprint"],
                "plan_sha256": plan["plan_sha256"],
                "status": "succeeded",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return measurements


def validate_release(plan: dict, release: dict, cache_audit: dict) -> None:
    expected = {
        "approved": True,
        "plan_sha256": plan["plan_sha256"],
        "cache_audit_sha256": canonical_sha256(cache_audit),
        "stage": plan["stage"],
    }
    if any(release.get(k) != v for k, v in expected.items()) or not release.get("reviewer"):
        raise ValueError("Release must record a human-reviewed plan and current cache audit for this stage")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibration", "canary", "pilot", "dense"), default="calibration")
    parser.add_argument("--plan-path", type=Path, required=True)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true")
    action.add_argument("--collect-results", type=Path)
    parser.add_argument("--release", type=Path)
    parser.add_argument("--max-concurrent", type=int, default=8)
    args = parser.parse_args(argv)
    if args.collect_results:
        collect_results(json.loads(args.plan_path.read_text()), args.collect_results)
        return
    if args.max_concurrent < 1:
        raise ValueError("--max-concurrent must be positive")
    design = experiment.load_design()
    plan, steps = build_plan(design, args.stage)
    args.plan_path.parent.mkdir(parents=True, exist_ok=True)
    args.plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    if args.submit:
        if args.release is None:
            raise ValueError("--submit requires a recorded --release")
        experiment.require_central1()
        cache_audit = preparation.verify_caches(design, preparation.data_steps(design), experiment.PREFIX)
        validate_release(plan, json.loads(args.release.read_text()), cache_audit)
        asyncio.run(experiment.audit_allocations(design))
        previous = {"canary": "calibration", "pilot": "canary", "dense": "pilot"}.get(args.stage)
        if previous:
            previous_plan, _ = build_plan(design, previous)
            values = collect_results(previous_plan, args.plan_path.parent / f"{previous}_gate_metrics.csv")
            if (
                previous == "calibration"
                and not experiment.calibration_summary(previous_plan, values)["batch32_loss_screen_passed"]
            ):
                raise ValueError("Batch-32 calibration failed the prespecified loss screen; review a new recipe")
        uri = f"{experiment.PREFIX}/experiments/starcoder_tpp10/{plan['plan_sha256']}/plan.json"
        persist_submission_plan(plan, uri)
        pending = pending_training_steps(steps, marin_prefix=experiment.PREFIX)
        if pending:
            run(*pending, max_concurrent=args.max_concurrent, force_run_failed=True)
        collect_results(plan, args.plan_path.parent / f"{args.stage}_metrics.csv")


if __name__ == "__main__":
    main()
