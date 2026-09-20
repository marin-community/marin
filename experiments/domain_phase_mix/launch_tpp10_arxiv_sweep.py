# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Release the arXiv-papers TPP-10 sweep, one arm at a time, with on-target evaluations.

Proxy stage first: eight matched-proxy runs over the domain-share grid, including p=0, each
scoring the seven Uncheatable components, Paloma S2ORC and the frozen Paloma programming-
languages set at its final step. The target stage reuses the same code path once the proxy
curve is judged promising.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import fsspec
from haliax import Axis
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config, run
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import analyze_tpp10_arxiv_sweep as analysis
from experiments.domain_phase_mix import evaluate_starcoder_tpp10_uncheatable as uncheatable
from experiments.domain_phase_mix import launch_starcoder_tpp10 as original
from experiments.domain_phase_mix import launch_tpp10_domain_sweeps as previous
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as original_data
from experiments.domain_phase_mix import prepare_tpp10_arxiv_sweep as preparation
from experiments.domain_phase_mix import prepare_tpp10_domain_sweeps as previous_data
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix import tpp10_execution_check as execution_check
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
GRID = (0, 5, 10, 20, 30, 50, 70, 100)
STAGES = {"proxy": experiment.Arm.MATCHED, "target": experiment.Arm.TARGET}
TPU = previous.TPU
ROOT = f"{experiment.PREFIX}/experiments/tpp10_arxiv_sweep"
TRACKER_GROUP = "tpp10_arxiv_sweep_20260918"


@dataclass(frozen=True)
class ArxivTrainingRecipe:
    training: original.TrainingRecipe
    domain: str
    extension_code_sha256: dict[str, str]


def code_pins() -> dict[str, str]:
    paths = [
        Path(__file__),
        Path(execution_check.__file__),
        Path(preparation.__file__),
        Path(analysis.__file__),
        Path(previous_data.__file__),
        Path(uncheatable.__file__),
        *sorted(preparation.ASSETS.iterdir()),
        previous_data.ASSETS / "evaluation.json",
    ]
    return {str(path.relative_to(original.REPO)): file_sha256(path) for path in paths if path.is_file()}


def verified_training(recipe: ArxivTrainingRecipe) -> None:
    experiment.require_central1()
    if code_pins() != recipe.extension_code_sha256:
        raise ValueError("arXiv sweep source/assets differ from the release")
    if recipe.training.pod.output_path is None:
        raise ValueError("Training output must be materialized before dispatch")
    original.persist_submission_plan(
        {"domain": recipe.domain, "extension_code_sha256": recipe.extension_code_sha256},
        recipe.training.pod.output_path + "/domain_runtime.json",
    )
    original.verified_training(recipe.training)


def dispatch_training(recipe: ArxivTrainingRecipe) -> None:
    remote(
        verified_training,
        name=recipe.training.pod.train_config.trainer.id,
        resources=recipe.training.pod.resources,
        env_vars={"MARIN_PREFIX": experiment.PREFIX},
    )(recipe)


def training_step(design: dict, request: experiment.RunSpec, caches: dict) -> ArtifactStep:
    """Frozen optimizer, allocator and source keys; held-out evaluations added at zero weight."""
    if request.arm not in STAGES.values():
        raise ValueError("Release includes only target and matched arms")
    base = original.training_step(design, request, caches)
    extension_code = code_pins()
    eval_paths = preparation.evaluation_paths(design)

    def config(ctx: StepContext) -> ArxivTrainingRecipe:
        training = base.build_config(ctx)
        pod = training.pod
        data = pod.train_config.data
        components = {**data.components, **{name: DatasetComponent(cache_dir=path) for name, path in eval_paths.items()}}
        data = replace(
            data, components=components, train_weights={**data.train_weights, **{name: 0.0 for name in eval_paths}}
        )
        tracker = replace(
            pod.train_config.trainer.tracker,
            group=TRACKER_GROUP,
            tags=["tpp10_arxiv_sweep", preparation.DOMAIN, request.arm.value],
        )
        trainer = replace(pod.train_config.trainer, tracker=tracker, max_eval_batches=None)
        pod = replace(pod, resources=TPU, train_config=replace(pod.train_config, data=data, trainer=trainer))
        return ArxivTrainingRecipe(replace(training, pod=pod), preparation.DOMAIN, extension_code)

    step = replace(
        base,
        name=f"checkpoints/tpp10_domain_sweeps/{request.run_name}",
        version=preparation.VERSION,
        build_config=config,
        run=dispatch_training,
        expected_fingerprint=None,
    )
    return replace(step, expected_fingerprint=step.fingerprint())


def build_plan(stage: str) -> tuple[dict, tuple[ArtifactStep, ...]]:
    previous.validate_evaluation_population()
    arm = STAGES[stage]
    design = experiment.load_design()
    old = original_data.data_steps(design)
    new = preparation.data_steps(design)
    # The internal focus alias deliberately stays "starcoder": it retains the frozen
    # allocator ordering and the name-derived random key, as in the two earlier domains.
    caches = {
        **old,
        "starcoder": new[f"{preparation.DOMAIN}/parent"],
        f"subset_{experiment.SUBSET_SEEDS[0]}": new[f"{preparation.DOMAIN}/matched"],
    }
    model = design["models"]["target" if arm == experiment.Arm.TARGET else "unmatched"]
    rows, steps = [], []
    for percent in GRID:
        name = f"tpp10_{preparation.DOMAIN}_{arm.value}_p{percent:03d}_s20260910"
        request = experiment.RunSpec(
            name,
            arm,
            percent,
            experiment.TRAINER_SEEDS[0],
            experiment.SUBSET_SEEDS[0] if arm == experiment.Arm.MATCHED else None,
            model["steps"],
            model["batch_size"],
        )
        step = training_step(design, request, caches)
        steps.append(step)
        rows.append(
            {
                **asdict(request),
                "domain": preparation.DOMAIN,
                "tokens": request.tokens,
                "support_sequences": request.support_sequences,
                "fingerprint": step.fingerprint(),
                "output_path": step.path(experiment.PREFIX),
            }
        )
    plan = {
        "schema_version": 1,
        "stage": f"arxiv_{stage}",
        "arm": arm.value,
        "design_sha256": design["design_sha256"],
        "primary_metric": experiment.PRIMARY_METRIC,
        "on_target_metrics": list(preparation.ON_TARGET_METRICS),
        "code_sha256": original.code_pins(),
        "extension_code_sha256": code_pins(),
        "runtime_versions": original.runtime_versions(),
        "region": experiment.REGION,
        "zone": experiment.ZONE,
        "marin_prefix": experiment.PREFIX,
        "grid": list(GRID),
        "runs": rows,
        "evaluation_paths": preparation.evaluation_paths(design),
        "training_flops": len(rows) * model["training_flops"],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan, tuple(steps)


def validate_evaluation_tags(data: LmDataConfig, design: dict) -> dict[str, int]:
    """Resolve real finite evaluation sets, S2ORC included, before spending training compute."""
    populations = {}
    for dataset, tags in data.tagged_eval_sets(Axis("position", experiment.SEQ_LEN)):
        count = asyncio.run(dataset.async_len())
        for tag in tags:
            populations[tag] = count
    expected = {
        experiment.PRIMARY_METRIC.removeprefix("eval/").removesuffix("/bpb"),
        *preparation.evaluation_paths(design),
    }
    if not expected <= populations.keys() or any(populations[name] <= 0 for name in expected):
        raise ValueError(f"Missing or empty evaluation sets: {expected - populations.keys()}")
    return populations


def metric_keys() -> list[str]:
    return [
        experiment.PRIMARY_METRIC,
        *[f"eval/uncheatable_eval/{name}/bpb" for name in uncheatable.COMPONENTS],
        preparation.S2ORC_METRIC,
    ]


def final_metrics(request: dict) -> dict[str, float]:
    """Require consistent finite final-step scores for every prescribed evaluation."""
    keys = metric_keys()
    values = {}
    path = LevanterCheckpoint(path=request["output_path"]).checkpoint_dir + "/eval_metrics.jsonl"
    with fsspec.open(path, "rt") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("step") != request["total_steps"] - 1:
                continue
            for key in keys:
                if key not in item:
                    continue
                value = float(item[key])
                if not math.isfinite(value) or (key in values and values[key] != value):
                    raise ValueError(f"Invalid or conflicting final metric: {request['run_name']} {key}")
                values[key] = value
    if set(values) != set(keys):
        raise ValueError(f"Missing final evaluations: {request['run_name']} {set(keys) - values.keys()}")
    return values


def collect(plan: dict, output: Path) -> list[dict]:
    original.collect_results(plan, output.with_suffix(".paloma.csv"))
    rows = []
    for request in plan["runs"]:
        receipt = uncheatable.read_json(request["output_path"] + "/domain_runtime.json")
        if receipt != {"domain": request["domain"], "extension_code_sha256": plan["extension_code_sha256"]}:
            raise ValueError(f"Domain code receipt differs: {request['run_name']}")
        checkpoint = uncheatable.checkpoint_metadata(request)
        metrics = final_metrics(request)
        scores = uncheatable.component_scores(metrics)
        rows.append(
            {
                "run_name": request["run_name"],
                "domain": request["domain"],
                "arm": request["arm"],
                "percent": request["percent"],
                "trainer_seed": request["trainer_seed"],
                "subset_seed": request["subset_seed"],
                "macro_bpb": math.fsum(scores.values()) / len(scores),
                "paloma_bpb": metrics[experiment.PRIMARY_METRIC],
                "s2orc_bpb": metrics[preparation.S2ORC_METRIC],
                **scores,
                "fingerprint": request["fingerprint"],
                "checkpoint_sha256": checkpoint["metadata_sha256"],
                "plan_sha256": plan["plan_sha256"],
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def submit(plan: dict, steps: tuple[ArtifactStep, ...], release: dict, output: Path) -> None:
    experiment.require_central1()
    if (
        release.get("approved") is not True
        or release.get("plan_sha256") != plan["plan_sha256"]
        or not release.get("reviewer")
    ):
        raise ValueError("Submission requires the exact user-authorized plan")
    previous.validate_evaluation_population()
    design = experiment.load_design()
    frozen_audit = original_data.verify_caches(design, original_data.data_steps(design), experiment.PREFIX)
    data_steps = preparation.data_steps(design)
    evaluation = preparation.evaluation_step(design)
    pending = original.pending_training_steps((*data_steps.values(), evaluation), marin_prefix=experiment.PREFIX)
    if pending:
        run(*pending, max_concurrent=len(pending), force_run_failed=True)
    data_audit = previous_data.verify_caches(design, data_steps)
    evaluation_audit = preparation.verify_evaluation_cache(design, evaluation)
    paths = uncheatable.prepare_caches(json.loads((previous_data.ASSETS / "evaluation.json").read_text()))
    paths[preparation.S2ORC] = evaluation_audit["path"]
    if any(paths[key] != value for key, value in plan["evaluation_paths"].items()):
        raise ValueError("Evaluation population/cache identity changed")
    evaluation_tags = validate_evaluation_tags(
        materialized_config(steps[0], experiment.PREFIX).training.pod.train_config.data, design
    )
    allocation = asyncio.run(experiment.audit_allocations(design))
    if allocation["maximum_epoch_relative_error"] >= 0.001:
        raise ValueError("Target/proxy epoch exposure differs by at least 0.1%")
    audit = {
        "frozen_data": frozen_audit,
        "data": data_audit,
        "evaluation_cache": evaluation_audit,
        "allocation": allocation,
        "evaluation_paths": paths,
        "evaluation_populations": evaluation_tags,
    }
    uri = f"{ROOT}/{plan['plan_sha256']}"
    original.persist_submission_plan(plan, uri + "/plan.json")
    original.persist_submission_plan(audit, uri + "/preflight.json")
    original.persist_submission_plan(release, uri + "/release.json")
    selected = list(zip(plan["runs"], steps, strict=True))
    canaries = [(row, step) for row, step in selected if row["percent"] == 100]
    remaining = [(row, step) for row, step in selected if row["percent"] != 100]
    if len(canaries) != 1 or len(remaining) != len(GRID) - 1:
        raise ValueError("Release must contain exactly one p=100 execution check and the remaining grid points")
    canary_step = canaries[0][1]
    execution_check.run_with_early_release(
        original.pending_training_steps((canary_step,), marin_prefix=experiment.PREFIX),
        original.pending_training_steps(tuple(s for _, s in remaining), marin_prefix=experiment.PREFIX),
        release_ready=lambda: execution_check.committed_checkpoint_exists(canary_step.path(experiment.PREFIX)),
    )
    collect(previous.subplan(plan, [r for r, _ in canaries]), output / "execution_checks.csv")
    results = collect(plan, output / "measurements.csv")
    summary = analysis.analyze(
        results, {row["percent"]: row for row in allocation["coordinates"]}, plan["grid"], plan["arm"]
    )
    (output / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    original.persist_submission_plan(
        {"status": "succeeded", "plan_sha256": plan["plan_sha256"], "results": results, "analysis": summary},
        uri + "/results.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), default="proxy")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--submit", action="store_true")
    action.add_argument("--collect", type=Path)
    parser.add_argument("--release", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.collect:
        plan = json.loads(args.plan.read_text())
        rows = collect(plan, args.collect)
        allocation = asyncio.run(experiment.audit_allocations(experiment.load_design()))
        summary = analysis.analyze(
            rows, {row["percent"]: row for row in allocation["coordinates"]}, plan["grid"], plan["arm"]
        )
        args.collect.with_name("analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
        return
    plan, steps = build_plan(args.stage)
    if args.plan.exists() and json.loads(args.plan.read_text()) != plan:
        raise ValueError("Archived arXiv plan differs; do not overwrite its identities")
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "runs": len(steps), "training_flops": plan["training_flops"]}))
    if args.submit:
        if args.release is None:
            raise ValueError("--submit requires --release")
        submit(plan, steps, json.loads(args.release.read_text()), args.plan.parent)


if __name__ == "__main__":
    main()
