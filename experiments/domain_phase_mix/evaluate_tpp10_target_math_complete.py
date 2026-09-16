# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Complete target math evaluation by reusing eleven receipts and scoring four checkpoints."""

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
from experiments.domain_phase_mix import evaluate_tpp10_target_math as first_stage
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
DIRECTORY = scoring.DIRECTORY.parent / "target_math_complete_20260912"
DOMAINS = ("wikipedia", "finemath_3plus")


def validate_spec(spec: dict) -> None:
    """Require the complete target grid and exact preservation of the first evaluation release."""
    if spec["spec_sha256"] != canonical_sha256({k: v for k, v in spec.items() if k != "spec_sha256"}):
        raise ValueError("Completed target evaluation specification changed")
    initial = spec["first_stage_spec"]
    first_stage.validate_spec(initial)
    if spec["wrapper_code_sha256"] != file_sha256(Path(__file__)):
        raise ValueError("Completed target evaluation source changed")
    changed = {
        "purpose",
        "endpoints",
        "spec_sha256",
        "wrapper_code_sha256",
        "omitted_requests",
        "first_stage_spec",
        "first_stage_result_canonical_sha256",
        "completion_training_plan",
    }
    if {k: v for k, v in spec.items() if k not in changed} != {k: v for k, v in initial.items() if k not in changed}:
        raise ValueError("The completion release changed the frozen scoring protocol")
    expected = ["tpp10_target_p000_s20260910"] + [
        f"tpp10_{domain}_target_p{percent:03d}_s20260910" for domain in DOMAINS for percent in scoring.GRID[1:]
    ]
    endpoints = spec["endpoints"]
    if [endpoint["request"]["run_name"] for endpoint in endpoints] != expected or spec["omitted_requests"]:
        raise ValueError("Expected all fifteen unique target checkpoints, with no omitted requests")
    by_name = {endpoint["request"]["run_name"]: endpoint for endpoint in endpoints}
    for endpoint in initial["endpoints"]:
        if endpoint != by_name[endpoint["request"]["run_name"]]:
            raise ValueError("A previously evaluated target endpoint changed")
    for request in initial["omitted_requests"]:
        if request != by_name[request["run_name"]]["request"]:
            raise ValueError("A completing target request differs from the original survey")
    hashes = spec["first_stage_result_canonical_sha256"]
    if set(hashes) != {endpoint["request"]["run_name"] for endpoint in initial["endpoints"]}:
        raise ValueError("The eleven first-stage result hashes are not fully pinned")
    if any(len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest) for digest in hashes.values()):
        raise ValueError("Invalid first-stage result hash")
    survey = spec["completion_training_plan"]
    if survey["plan_sha256"] != canonical_sha256({k: v for k, v in survey.items() if k != "plan_sha256"}):
        raise ValueError("The completion training plan changed")
    planned = {request["run_name"]: request for request in survey["runs"]}
    for endpoint in endpoints:
        request = endpoint["request"]
        if request["arm"] != "target":
            raise ValueError("Every checkpoint requires target model geometry")
        if request["percent"] and request != planned[request["run_name"]]:
            raise ValueError("Target request differs from the frozen training plan")
        checkpoint = endpoint["checkpoint"]
        step = request["total_steps"] - 1
        if checkpoint["path"] != f"{request['output_path']}/checkpoints/step-{step}":
            raise ValueError("Target checkpoint path is not the requested final step")
        if checkpoint["metadata"]["step"] != step or checkpoint["metadata"]["is_temporary"]:
            raise ValueError("Target checkpoint is not final and permanent")


def build_spec(first_stage_path: Path) -> dict:
    """Release only after eleven verified math results and four completed training runs exist."""
    initial = scoring.repair.read_json(str(first_stage_path))
    first_stage.validate_spec(initial)
    hashes = {}
    for endpoint in initial["endpoints"]:
        result = first_stage.verified_result(initial, endpoint)
        if result is None:
            raise ValueError(f"First-stage math evaluation is incomplete: {endpoint['request']['run_name']}")
        hashes[endpoint["request"]["run_name"]] = canonical_sha256(result)
    survey = scoring.repair.read_json(str(scoring.SURVEY))
    by_name = {endpoint["request"]["run_name"]: endpoint for endpoint in initial["endpoints"]}
    for request in initial["omitted_requests"]:
        evidence = scoring.repair.training_evidence(request, survey)
        by_name[request["run_name"]] = {
            "request": request,
            "checkpoint": scoring.repair.checkpoint_snapshot(request),
            "paloma_loss": evidence["metrics"][f"eval/{scoring.repair.PALOMA}/loss"],
        }
    names = ["tpp10_target_p000_s20260910"] + [
        f"tpp10_{domain}_target_p{percent:03d}_s20260910" for domain in DOMAINS for percent in scoring.GRID[1:]
    ]
    spec = {
        **initial,
        "purpose": "Math likelihood on the full target grid; eleven verified measurements reused and four added",
        "endpoints": [by_name[name] for name in names],
        "omitted_requests": [],
        "first_stage_spec": initial,
        "first_stage_result_canonical_sha256": hashes,
        "completion_training_plan": survey,
        "wrapper_code_sha256": file_sha256(Path(__file__)),
    }
    spec.pop("spec_sha256")
    spec["spec_sha256"] = canonical_sha256(spec)
    validate_spec(spec)
    return spec


def output_root(spec: dict) -> str:
    return scoring.output_root(spec)


def reuse_lineage(spec: dict, endpoint: dict) -> dict:
    name = endpoint["request"]["run_name"]
    initial = spec["first_stage_spec"]
    return {
        "uri": f"{first_stage.output_root(initial)}/{name}.json",
        "source_spec_sha256": initial["spec_sha256"],
        "source_result_canonical_sha256": spec["first_stage_result_canonical_sha256"][name],
    }


def verified_result(spec: dict, endpoint: dict) -> dict | None:
    """Check likelihoods, checkpoint control, and exact payload lineage for reused measurements."""
    result = scoring.verified_result(spec, endpoint)
    if result is None:
        return None
    if result["population_sha256"] != canonical_sha256(spec["population_counts"]):
        raise ValueError("Target result has a different scored population")
    name = endpoint["request"]["run_name"]
    source_hash = spec["first_stage_result_canonical_sha256"].get(name)
    if source_hash is None:
        if "reused_from" in result:
            raise ValueError("A newly scored target cannot carry reused measurement lineage")
        return result
    if result.get("reused_from") != reuse_lineage(spec, endpoint):
        raise ValueError("Reused target measurement has incorrect source lineage")
    source = {key: value for key, value in result.items() if key != "reused_from"}
    source["spec_sha256"] = spec["first_stage_spec"]["spec_sha256"]
    if canonical_sha256(source) != source_hash:
        raise ValueError("Reused target measurement differs from its pinned source payload")
    return result


def reuse_first_stage(spec: dict) -> None:
    """Preserve the eleven verified measurements with explicit immutable source lineage."""
    initial = spec["first_stage_spec"]
    for endpoint in initial["endpoints"]:
        name = endpoint["request"]["run_name"]
        source = first_stage.verified_result(initial, endpoint)
        if source is None or canonical_sha256(source) != spec["first_stage_result_canonical_sha256"][name]:
            raise ValueError(f"Pinned first-stage measurement is missing or changed: {name}")
        if "reused_from" in source:
            raise ValueError("First-stage measurements must be original evaluations")
        result = {**source, "spec_sha256": spec["spec_sha256"], "reused_from": reuse_lineage(spec, endpoint)}
        existing = verified_result(spec, endpoint)
        if existing is not None:
            if existing != result:
                raise ValueError(f"Stored target reuse differs from its source: {name}")
            continue
        scoring.repair.write_json(f"{output_root(spec)}/{name}.json", result)
        if verified_result(spec, endpoint) != result:
            raise ValueError(f"Failed to preserve the first-stage measurement: {name}")


def evaluate(spec: dict) -> None:
    """Score only the four newly completed targets using the frozen native evaluation setup."""
    scoring.experiment.require_central1()
    validate_spec(spec)
    reuse_first_stage(spec)
    pending = [endpoint for endpoint in spec["endpoints"] if verified_result(spec, endpoint) is None]
    if not pending:
        return
    if any(endpoint["request"]["run_name"] in spec["first_stage_result_canonical_sha256"] for endpoint in pending):
        raise ValueError("A first-stage checkpoint was not reused")
    populations, counts = scoring.prepare_math(spec)
    if counts != spec["population_counts"]:
        raise ValueError("Math population changed")
    # The submitted first-stage wrapper is immutable; retain its native setup here under a new source pin.
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
            evidence = scoring.repair.training_evidence(request, spec["completion_training_plan"])
            if evidence["metrics"][f"eval/{scoring.repair.PALOMA}/loss"] != endpoint["paloma_loss"]:
                raise ValueError(f"Final training control changed: {request['run_name']}")
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
    reuse_first_stage(spec)
    if all(verified_result(spec, endpoint) is not None for endpoint in spec["endpoints"]):
        return
    worker = remote(
        evaluate,
        name="target-math-completion",
        resources=scoring.original.TPU,
        env_vars={"MARIN_PREFIX": scoring.experiment.PREFIX},
    )
    await asyncio.to_thread(worker, spec)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DIRECTORY / "spec.json")
    parser.add_argument("--first-stage", type=Path, default=first_stage.DIRECTORY / "spec.json")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.build:
        spec = build_spec(args.first_stage)
        args.spec.parent.mkdir(parents=True, exist_ok=True)
        args.spec.write_text(json.dumps(spec, indent=2) + "\n")
        print(json.dumps({"spec_sha256": spec["spec_sha256"], "new_evaluations": 4, "reused_results": 11}))
    else:
        asyncio.run(submit(scoring.repair.read_json(str(args.spec))))


if __name__ == "__main__":
    main()
