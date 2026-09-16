# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Score completed target checkpoints with the frozen proxy math protocol."""

import argparse
import asyncio
import json
import logging
import math
from pathlib import Path

import equinox as eqx
import haliax as hax
import jax
import jmp
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.data.dataset import ListAsyncDataset
from levanter.eval import TaggedEvaluator, eval_model
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from marin.execution.remote import remote

from experiments.domain_phase_mix import evaluate_tpp10_finemath_math as scoring
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
DIRECTORY = scoring.DIRECTORY.parent / "target_math_eval_20260912"
TARGET_GRID = {"wikipedia": (5, 10, 20, 30, 50, 70, 100), "finemath_3plus": (5, 10, 20)}
OMITTED_GRID = (30, 50, 70, 100)
SHARED_FIELDS = (
    "sources",
    "tokenizer_pins",
    "primary",
    "secondary",
    "metric",
    "format",
    "context",
    "stride",
    "eos_policy",
    "boundary_policy",
    "aggregation",
    "paloma_path",
    "batch_size",
    "code_sha256",
    "population_counts",
)


def validate_spec(spec: dict) -> None:
    """Keep scoring identical and release only the eleven completed target checkpoints."""
    if spec["spec_sha256"] != canonical_sha256({k: v for k, v in spec.items() if k != "spec_sha256"}):
        raise ValueError("Target evaluation specification changed")
    reference = spec["reference_spec"]
    scoring.validate_spec(reference)
    if spec["wrapper_code_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("Target evaluation source changed")
    for key in SHARED_FIELDS:
        if spec[key] != reference[key]:
            raise ValueError(f"Target and proxy scoring differ: {key}")
    expected = ["tpp10_target_p000_s20260910"] + [
        f"tpp10_{domain}_target_p{p:03d}_s20260910" for domain, grid in TARGET_GRID.items() for p in grid
    ]
    requests = [endpoint["request"] for endpoint in spec["endpoints"]]
    if [request["run_name"] for request in requests] != expected:
        raise ValueError("Expected the eleven completed target checkpoints in frozen order")
    if any(request["arm"] != "target" for request in requests):
        raise ValueError("This worker requires target model geometry for every checkpoint")
    omitted = spec["omitted_requests"]
    if [r["run_name"] for r in omitted] != [f"tpp10_finemath_3plus_target_p{p:03d}_s20260910" for p in OMITTED_GRID]:
        raise ValueError("The four canceled FineMath targets must remain omitted")


def build_spec(reference_path: Path) -> dict:
    """Verify existing final checkpoints; never inspect or resume incomplete training states."""
    reference = scoring.repair.read_json(str(reference_path))
    scoring.validate_spec(reference)
    survey = scoring.repair.read_json(str(scoring.SURVEY))
    repair_plan = scoring.repair.read_json(str(scoring.REPAIR))
    controls = [a for a in repair_plan["audits"] if a["request"]["run_name"] == "tpp10_target_p000_s20260910"]
    if len(controls) != 1:
        raise ValueError("Expected the unique frozen target p0 checkpoint")
    control = controls[0]
    requests = [control["request"]]
    for domain, grid in TARGET_GRID.items():
        selected = sorted(
            (r for r in survey["runs"] if r["domain"] == domain and r["arm"] == "target" and r["percent"] in grid),
            key=lambda r: r["percent"],
        )
        if tuple(r["percent"] for r in selected) != grid:
            raise ValueError(f"Completed target request inventory differs: {domain}")
        requests.extend(selected)
    endpoints = []
    for request in requests:
        evidence = scoring.repair.training_evidence(
            request, control["training_plan"] if request["percent"] == 0 else survey
        )
        endpoints.append(
            {
                "request": request,
                "checkpoint": scoring.repair.checkpoint_snapshot(request),
                "paloma_loss": evidence["metrics"][f"eval/{scoring.repair.PALOMA}/loss"],
            }
        )
    spec = {
        **reference,
        "purpose": (
            "Math likelihood on eleven completed target checkpoints; canceled FineMath points omitted; no training"
        ),
        "endpoints": endpoints,
        "reference_spec": reference,
        "wrapper_code_sha256": file_sha256(Path(__file__)),
        "omitted_requests": sorted(
            (
                r
                for r in survey["runs"]
                if r["domain"] == "finemath_3plus" and r["arm"] == "target" and r["percent"] in OMITTED_GRID
            ),
            key=lambda r: r["percent"],
        ),
        "contamination_status": (
            "No benchmark decontamination audit of training subsets; exploratory likelihood diagnostic"
        ),
    }
    spec.pop("spec_sha256")
    spec["spec_sha256"] = canonical_sha256(spec)
    validate_spec(spec)
    return spec


def output_root(spec: dict) -> str:
    return scoring.output_root(spec)


def verified_result(spec: dict, endpoint: dict) -> dict | None:
    result = scoring.verified_result(spec, endpoint)
    if result is not None and result["population_sha256"] != canonical_sha256(spec["population_counts"]):
        raise ValueError("Target result has a different scored population")
    return result


def evaluate(spec: dict) -> None:
    """Restore eleven target models, sharing compilation and skipping verified receipts."""
    scoring.experiment.require_central1()
    validate_spec(spec)
    pending = [endpoint for endpoint in spec["endpoints"] if verified_result(spec, endpoint) is None]
    if not pending:
        return
    populations, counts = scoring.prepare_math(spec)
    if counts != spec["population_counts"]:
        raise ValueError("Math population changed")
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=spec["batch_size"],
        per_device_parallelism=-1,
        per_device_eval_parallelism=-1,
        allow_nondivisible_batch_size=True,
        max_eval_batches=None,
        log_jaxprs=False,
        log_xla_hlo=False,
        mesh=MeshConfig(
            axes={"data": -1, "replica": 1, "model": 1},
            compute_mapping={
                "token": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
                "token_repeat": (ResourceAxis.REPLICA_DCN, ResourceAxis.REPLICA, ResourceAxis.DATA),
            },
        ),
    )
    trainer.initialize()
    tokenizer = scoring.experiment.verified_tokenizer()
    config = scoring.experiment.model_config(scoring.experiment.Arm.TARGET)
    with trainer.use_device_mesh():
        sets = [(ListAsyncDataset(windows).map(scoring.make_example), [name]) for name, windows in populations.items()]
        sets += list(
            scoring.repair.evaluation_data({scoring.repair.PALOMA: spec["paloma_path"]}).tagged_eval_sets(config.max_Pos)
        )
        evaluator = TaggedEvaluator(
            EvalBatch=trainer.EvalBatch,
            tagged_eval_sets=sets,
            loss_fn=scoring.original.eval_loss,
            tokenizer=tokenizer,
            device_mesh=trainer.device_mesh,
            axis_mapping=trainer.compute_axis_mapping,
        )
        for endpoint in pending:
            request = endpoint["request"]
            if scoring.repair.checkpoint_snapshot(request) != endpoint["checkpoint"]:
                raise ValueError(f"Target checkpoint changed: {request['run_name']}")
            with use_cpu_device():
                model = eqx.filter_eval_shape(config.build, hax.Axis("vocab", len(tokenizer)), key=jax.random.PRNGKey(0))
                model = load_checkpoint(model, endpoint["checkpoint"]["path"], subpath="model")
            model = hax.shard_with_axis_mapping(model, trainer.parameter_axis_mapping)
            metrics = {k: float(v) for k, v in eval_model(evaluator, model, prefix="eval").items()}
            result = {
                "spec_sha256": spec["spec_sha256"],
                "endpoint": endpoint,
                "metrics": metrics,
                "perplexity": {name: math.exp(metrics[f"eval/{name}/loss"]) for name in scoring.SOURCES},
                "population_sha256": canonical_sha256(counts),
            }
            if jax.process_index() == 0:
                scoring.repair.write_json(f"{output_root(spec)}/diagnostics/{request['run_name']}.json", result)
            scoring.repair.check_close(
                metrics[f"eval/{scoring.repair.PALOMA}/loss"], endpoint["paloma_loss"], "Restored PALOMA loss"
            )
            if jax.process_index() == 0:
                scoring.repair.write_json(f"{output_root(spec)}/{request['run_name']}.json", result)
                logger.info("Completed %s: %s", request["run_name"], result["perplexity"])
            del model


async def submit(spec: dict) -> None:
    scoring.experiment.require_central1()
    validate_spec(spec)
    scoring.repair.write_json(output_root(spec) + "/spec.json", spec)
    if all(verified_result(spec, endpoint) is not None for endpoint in spec["endpoints"]):
        return
    worker = remote(
        evaluate,
        name="target-math-checkpoints",
        resources=scoring.original.TPU,
        env_vars={"MARIN_PREFIX": scoring.experiment.PREFIX},
    )
    await asyncio.to_thread(worker, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DIRECTORY / "spec.json")
    parser.add_argument("--reference", type=Path, default=scoring.DIRECTORY / "spec.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.build:
        spec = build_spec(args.reference)
        args.spec.parent.mkdir(parents=True, exist_ok=True)
        args.spec.write_text(json.dumps(spec, indent=2) + "\n")
        print(json.dumps({"spec_sha256": spec["spec_sha256"], "checkpoints": len(spec["endpoints"])}))
    else:
        asyncio.run(submit(scoring.repair.read_json(str(args.spec))))


if __name__ == "__main__":
    main()
