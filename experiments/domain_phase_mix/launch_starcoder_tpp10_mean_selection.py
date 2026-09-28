# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Validate one TPP10 target mixture at the mean of six proxy selections."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import fsspec
import jax
from levanter.data.mixture import MixtureDataset
from marin.execution.lazy import ArtifactStep, StepContext, materialized_config
from marin.execution.remote import remote
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import launch_starcoder_tpp10 as launcher
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as preparation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

RUN_NAME = "tpp10_target_p8of15_s20260910"
FRACTION = 8 / 15
SELECTIONS = (50, 50, 65, 50, 50, 55)
SCORED_TOKENS = 11_612_631
SCORED_BYTES = 28_741_166
TARGET_REFERENCE_MINIMUM_BPB = 0.7655645236000794
UNMATCHED_TARGET_LOSS_BPB = 0.8006188043853639
LOSS_METRIC = experiment.PRIMARY_METRIC.removesuffix("bpb") + "loss"
EXTRA_PIN_PATHS = (
    "experiments/domain_phase_mix/launch_starcoder_tpp10_mean_selection.py",
    "lib/levanter/src/levanter/eval.py",
)


@dataclass(frozen=True)
class MeanSelectionRecipe:
    training: launcher.TrainingRecipe
    extra_code_sha256: dict[str, str]


def extra_code_pins() -> dict[str, str]:
    return {path: file_sha256(launcher.REPO / path) for path in EXTRA_PIN_PATHS}


def training_weights() -> dict[str, float]:
    """Preserve the original web proportions at the exact nominal fraction 8/15."""
    assert math.isclose(sum(SELECTIONS) / len(SELECTIONS) / 100, FRACTION, abs_tol=1e-15)
    total = sum(experiment.WEB_COUNTS.values())
    return {
        **{name: (1 - FRACTION) * count / total for name, count in experiment.WEB_COUNTS.items()},
        "starcoder": FRACTION,
    }


def verified_training(recipe: MeanSelectionRecipe) -> None:
    experiment.require_central1()
    if extra_code_pins() != recipe.extra_code_sha256:
        raise ValueError("Mean-selection launcher or BPB evaluator differs from its frozen recipe")
    assert recipe.training.pod.output_path is not None
    launcher.persist_submission_plan(
        {"extra_code_sha256": recipe.extra_code_sha256},
        recipe.training.pod.output_path + "/verified_mean_selection.json",
    )
    launcher.verified_training(recipe.training)


def dispatch_training(recipe: MeanSelectionRecipe) -> None:
    remote(
        verified_training,
        resources=recipe.training.pod.resources,
        env_vars={"MARIN_PREFIX": experiment.PREFIX},
    )(recipe)


def build_point(design: dict) -> tuple[ArtifactStep[LevanterCheckpoint], experiment.RunSpec]:
    """Reuse the target recipe while keeping the original frozen grid unchanged."""
    reference = next(
        row for row in experiment.select_runs(design, "pilot") if row.arm == experiment.Arm.TARGET and row.percent == 50
    )
    named_reference = replace(reference, run_name=RUN_NAME)
    base = launcher.training_step(design, named_reference, preparation.data_steps(design))
    pins = extra_code_pins()

    def config(ctx: StepContext) -> MeanSelectionRecipe:
        inherited = base.build_config(ctx)
        data = inherited.pod.train_config.data
        # The baseline constructor accepts only frozen grid coordinates. Replace its
        # weights after construction; every other training/data setting is inherited.
        weights = {**data.train_weights, **training_weights()}
        pod = replace(
            inherited.pod,
            train_config=replace(inherited.pod.train_config, data=replace(data, train_weights=weights)),
        )
        return MeanSelectionRecipe(replace(inherited, pod=pod), pins)

    step = replace(base, build_config=config, run=dispatch_training, expected_fingerprint=None)
    return replace(step, expected_fingerprint=step.fingerprint()), reference


async def allocation(design: dict) -> dict:
    weights = training_weights()
    n = design["models"]["target"]["steps"] * experiment.BATCH_SIZE
    key, _ = jax.random.split(jax.random.PRNGKey(experiment.DATA_SEED))
    sources = {name: experiment.TaggedIndices(name, n + experiment.BLOCK_SIZE) for name in weights}
    mixture = MixtureDataset(sources, weights, experiment.BLOCK_SIZE, key=key)
    full, remainder = divmod(n, experiment.BLOCK_SIZE)
    block = Counter(name for name, _ in await mixture.get_batch(range(experiment.BLOCK_SIZE)))
    counts = Counter({name: count * full for name, count in block.items()})
    counts.update(name for name, _ in await mixture.get_batch(range(full * experiment.BLOCK_SIZE, n)))
    assert sum(counts.values()) == n
    assert all(counts[name] <= design["web_sequences"][name] for name in experiment.WEB_COUNTS)
    return {
        "full_blocks": full,
        "remainder_sequences": remainder,
        "sequences_per_block": dict(block),
        "total_sequences": n,
        "sequences_by_component": dict(counts),
        "realized_starcoder_fraction": counts["starcoder"] / n,
        "starcoder_epochs": counts["starcoder"] / experiment.PARENT_SEQUENCES,
    }


def build_plan(design: dict) -> tuple[dict, ArtifactStep[LevanterCheckpoint]]:
    step, reference = build_point(design)
    recipe = materialized_config(step, experiment.PREFIX)
    reference_step = launcher.training_step(
        design, replace(reference, run_name=RUN_NAME), preparation.data_steps(design)
    )
    reference_recipe = materialized_config(reference_step, experiment.PREFIX)
    actual = recipe.training.pod.train_config
    expected = reference_recipe.pod.train_config
    actual_dict, expected_dict = asdict(actual), asdict(expected)
    actual_dict["data"]["train_weights"] = expected_dict["data"]["train_weights"]
    assert actual_dict == expected_dict, "Target settings changed beyond mixture and output identity"
    plan = {
        "design_sha256": design["design_sha256"],
        "stage": "mean-selection-target",
        "primary_metric": experiment.PRIMARY_METRIC,
        "code_sha256": launcher.code_pins(),
        "extra_code_sha256": extra_code_pins(),
        "runtime_versions": launcher.runtime_versions(),
        "marin_prefix": experiment.PREFIX,
        "region": experiment.REGION,
        "zone": experiment.ZONE,
        "training_flops": design["models"]["target"]["training_flops"],
        "nominal_starcoder_fraction": {"numerator": 8, "denominator": 15},
        "proxy_track_selections_percent": list(SELECTIONS),
        "weights": training_weights(),
        "allocation": asyncio.run(allocation(design)),
        "model": design["models"]["target"],
        "reference_losses": {
            "target_grid_minimum_bpb": TARGET_REFERENCE_MINIMUM_BPB,
            "target_grid_minimum_percent": 70,
            "unmatched_choice_target_bpb": UNMATCHED_TARGET_LOSS_BPB,
            "unmatched_choice_percent": 100,
        },
        "runs": [
            {
                **asdict(reference),
                "run_name": RUN_NAME,
                "percent": 100 * FRACTION,
                "tokens": reference.tokens,
                "support_sequences": reference.support_sequences,
                "fingerprint": step.fingerprint(),
                "output_path": step.path(experiment.PREFIX),
            }
        ],
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan, step


def collect(plan: dict, output: Path) -> None:
    """Retain native measurements and score regret with the paper's BPB definition."""
    native = launcher.collect_results(plan, output.with_suffix(".csv"))[RUN_NAME]
    request = plan["runs"][0]
    path = request["output_path"]
    with fsspec.open(path + "/verified_mean_selection.json", "rt") as handle:
        assert json.load(handle)["extra_code_sha256"] == plan["extra_code_sha256"]
    checkpoint = LevanterCheckpoint(path=path).checkpoint_dir
    with fsspec.open(checkpoint + "/eval_metrics.jsonl", "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    rows = [row for row in rows if row.get("step") == request["total_steps"] - 1 and LOSS_METRIC in row]
    assert rows and all(row[LOSS_METRIC] == rows[0][LOSS_METRIC] for row in rows)
    metric = rows[0]
    assert metric["eval/bpb_schema_version"] == 2
    bpb = metric[LOSS_METRIC] * SCORED_TOKENS / (SCORED_BYTES * math.log(2))
    assert math.isfinite(bpb) and abs(native - bpb) <= 5e-5
    regret = bpb - TARGET_REFERENCE_MINIMUM_BPB
    result = {
        "plan_sha256": plan["plan_sha256"],
        "run_name": RUN_NAME,
        "final_native_record": metric,
        "scored_tokens": SCORED_TOKENS,
        "scored_bytes": SCORED_BYTES,
        "normalized_bpb": bpb,
        "target_regret_bpb": regret,
        "excess_percent": 100 * regret / TARGET_REFERENCE_MINIMUM_BPB,
        "regret_reduction_percent": 100 * (1 - regret / (UNMATCHED_TARGET_LOSS_BPB - TARGET_REFERENCE_MINIMUM_BPB)),
        "reference": "Existing twelve-point target grid; this follow-up does not redefine its minimum.",
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    launcher.persist_submission_plan(result, path + "/mean_selection_result.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--collect", type=Path)
    args = parser.parse_args()
    if args.collect:
        collect(json.loads(args.plan.read_text()), args.collect)
        return
    design = experiment.load_design()
    plan, step = build_plan(design)
    if args.plan.exists() and json.loads(args.plan.read_text()) != plan:
        raise ValueError("Frozen follow-up plan differs; do not replace a submitted recipe")
    args.plan.parent.mkdir(parents=True, exist_ok=True)
    args.plan.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps({"plan_sha256": plan["plan_sha256"], "run": plan["runs"][0], "allocation": plan["allocation"]}))
    if not args.submit:
        return
    experiment.require_central1()
    audit = preparation.verify_caches(design, preparation.data_steps(design), experiment.PREFIX)
    uri = f"{experiment.PREFIX}/experiments/starcoder_tpp10/{plan['plan_sha256']}"
    launcher.persist_submission_plan(plan, uri + "/plan.json")
    launcher.persist_submission_plan(audit, uri + "/cache_audit.json")
    pending = launcher.pending_training_steps((step,), marin_prefix=experiment.PREFIX)
    if pending:
        launcher.run(*pending, max_concurrent=1, force_run_failed=True)
    collect(plan, args.plan.with_name("result.json"))


if __name__ == "__main__":
    main()
