# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate the frozen Figure 5 checkpoints on seven-component Uncheatable loss."""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import equinox as eqx
import fsspec
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
from fray.types import ResourceConfig
from google.cloud import storage
from haliax.partitioning import ResourceAxis
from levanter.checkpoint import load_checkpoint
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.eval import LossFnOutput, TaggedEvaluator, eval_model
from levanter.models.lm_model import LmExample, LmHeadModel
from levanter.store.cache import CacheMetadata
from levanter.tracker.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.jax_utils import use_cpu_device
from levanter.utils.mesh import MeshConfig
from levanter.utils.tree_utils import inference_mode
from marin.execution.artifact import read_record
from marin.execution.remote import remote

from experiments.domain_phase_mix import analyze_starcoder_tpp10 as analysis
from experiments.domain_phase_mix import launch_starcoder_tpp10 as launcher
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as preparation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment
from experiments.domain_phase_mix.launch_starcoder_epoch_matching import persist_submission_plan
from experiments.domain_phase_mix.starcoder_epoch_matching import canonical_sha256, file_sha256

logger = logging.getLogger(__name__)
COMPONENTS = (
    "wikipedia_english",
    "github_python",
    "github_cpp",
    "bbc_news",
    "arxiv_physics",
    "arxiv_computer_science",
    "ao3_english",
)
METRIC = "eval/uncheatable_eval/macro_bpb"
VERSION = "2026.09.11"
# Float32 reductions can change slightly when PALOMA shares batches with other sets.
PALOMA_TOLERANCE = 5e-5
TPU = ResourceConfig.with_tpu("v5p-8", cpu=8, ram="32g", disk="32g", regions=(experiment.REGION,), zone=experiment.ZONE)


def read_json(path: str) -> dict:
    with fsspec.open(path, "rt") as handle:
        return json.load(handle)


def code_pins() -> dict[str, str]:
    paths = (
        Path(__file__),
        Path(preparation.__file__),
        Path(experiment.__file__),
        launcher.REPO / "lib/levanter/src/levanter/eval.py",
        launcher.REPO / "lib/levanter/src/levanter/checkpoint.py",
        launcher.REPO / "lib/levanter/src/levanter/trainer.py",
        launcher.REPO / "lib/levanter/src/levanter/data/text/datasets.py",
        launcher.REPO / "lib/levanter/src/levanter/data/text/formats.py",
        launcher.REPO / "lib/levanter/src/levanter/tokenizers.py",
        launcher.REPO / "lib/levanter/src/levanter/models/qwen.py",
        launcher.REPO / "lib/levanter/src/levanter/models/llama.py",
        launcher.REPO / "uv.lock",
    )
    return {str(path.relative_to(launcher.REPO)): file_sha256(path) for path in paths}


def checkpoint_metadata(request: dict) -> dict:
    path = f"{request['output_path']}/checkpoints/step-{request['total_steps'] - 1}"
    with fsspec.open(path + "/metadata.json", "rb") as handle:
        encoded = handle.read()
    metadata = json.loads(encoded)
    if metadata["step"] != request["total_steps"] - 1 or metadata["is_temporary"]:
        raise ValueError(f"Not the permanent final checkpoint: {path}")
    record = read_record(request["output_path"])
    if record is None or record.fingerprint != request["fingerprint"]:
        raise ValueError(f"Training artifact identity changed: {path}")
    return {"path": path, "metadata_sha256": hashlib.sha256(encoded).hexdigest(), "metadata": metadata}


def build_spec(plan: dict, output: Path) -> dict:
    """Freeze data generations, training identities, and the secondary evaluation."""
    experiment.validate_plan(plan)
    values = launcher.collect_results(plan, output.parent / "uncheatable_primary_recheck.csv")
    bucket = storage.Client().bucket("marin-us-central1")
    sources = {}
    for component in COMPONENTS:
        blobs = list(bucket.list_blobs(prefix=f"raw/uncheatable_eval/2026.06.28/{component}_"))
        if len(blobs) != 1 or not blobs[0].name.endswith(".jsonl.gz"):
            raise ValueError(f"Expected one frozen raw object for {component}")
        blob = blobs[0]
        sources[component] = asdict(
            preparation.Source(f"{experiment.PREFIX}/{blob.name}", str(blob.generation), blob.size, blob.crc32c)
        )
    with ThreadPoolExecutor(max_workers=8) as pool:
        checkpoints = dict(
            zip((r["run_name"] for r in plan["runs"]), pool.map(checkpoint_metadata, plan["runs"]), strict=True)
        )
    spec = {
        "schema_version": 1,
        "purpose": "Secondary common-objective evaluation of the completed Figure 5 pilot",
        "training_plan": plan,
        "sources": sources,
        "checkpoints": checkpoints,
        "primary_bpb": values,
        "metric": METRIC,
        "aggregation": "Equal arithmetic mean of the seven component BPBs",
        "seq_len": experiment.SEQ_LEN,
        "max_eval_batches": None,
        "code_sha256": code_pins(),
        "tokenizer_pins": read_json(str(experiment.ASSETS / "pins.json")),
        "paloma_tolerance": PALOMA_TOLERANCE,
    }
    spec["spec_sha256"] = canonical_sha256(spec)
    output.write_text(json.dumps(spec, indent=2, allow_nan=False) + "\n")
    return spec


def validate_spec(spec: dict) -> None:
    if spec["spec_sha256"] != canonical_sha256({k: v for k, v in spec.items() if k != "spec_sha256"}):
        raise ValueError("Evaluation specification hash mismatch")
    experiment.validate_plan(spec["training_plan"])
    if spec["code_sha256"] != code_pins():
        raise ValueError("Evaluation code differs from the frozen specification")
    for name, digest in spec["tokenizer_pins"]["files_sha256"].items():
        if file_sha256(experiment.ASSETS / name) != digest:
            raise ValueError(f"Frozen tokenizer asset changed: {name}")


def result_path(spec: dict, request: dict) -> str:
    root = f"{experiment.PREFIX}/experiments/starcoder_tpp10/uncheatable/{spec['spec_sha256']}"
    return f"{root}/{request['run_name']}.json"


def component_scores(metrics: dict) -> dict[str, float]:
    scores = {name: float(metrics[f"eval/uncheatable_eval/{name}/bpb"]) for name in COMPONENTS}
    if not all(math.isfinite(value) for value in scores.values()):
        raise ValueError("Non-finite Uncheatable component")
    return scores


def verified_result(spec: dict, request: dict) -> dict | None:
    path = result_path(spec, request)
    fs, _ = fsspec.core.url_to_fs(path)
    if not fs.exists(path):
        return None
    result = read_json(path)
    expected = {
        "spec_sha256": spec["spec_sha256"],
        "run_name": request["run_name"],
        "fingerprint": request["fingerprint"],
        "checkpoint": spec["checkpoints"][request["run_name"]],
    }
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f"Evaluation identity mismatch: {path}")
    scores = component_scores(result["metrics"])
    if not math.isclose(result["macro_bpb"], sum(scores.values()) / len(COMPONENTS), abs_tol=1e-12):
        raise ValueError(f"Component average mismatch: {path}")
    if abs(result["metrics"][experiment.PRIMARY_METRIC] - spec["primary_bpb"][request["run_name"]]) > PALOMA_TOLERANCE:
        raise ValueError(f"Restored PALOMA control differs from the training endpoint: {path}")
    return result


def prepare_caches(spec: dict) -> dict[str, str]:
    """Tokenize each complete, pinned held-out set on the regional coordinator."""
    experiment.require_central1()
    paths = {}
    tokenizer = experiment.verified_tokenizer()
    metadata = CacheMetadata(preparation.FORMAT.build_preprocessor(tokenizer).metadata)
    for name in COMPONENTS:
        source = preparation.Source(**spec["sources"][name])
        identity = {
            "source": asdict(source),
            "tokenizer": spec["tokenizer_pins"],
            "preparer": file_sha256(Path(preparation.__file__)),
            "layout": "one-source-flat-validation-v1",
        }
        cache_identity = canonical_sha256(identity)
        path = f"{experiment.PREFIX}/tokenized/uncheatable_eval/{name}-tpp10/{VERSION}/{cache_identity}"
        documents = preparation.remote_documents(source, None)
        # Each component is one source file; write its cache directly without a distributed merge.
        try:
            preparation.write_part(
                preparation.token_records(documents, tokenizer_path=experiment.TOKENIZER, quota=None),
                path + "/validation",
                metadata=metadata,
                identity=identity,
                expected_tokens=None,
            )
        finally:
            documents.close()
        paths[f"uncheatable_eval/{name}"] = path
    paths["paloma/dolma_100_programing_languages-tpp10"] = preparation.data_steps(experiment.load_design())[
        "evaluation"
    ].path(experiment.PREFIX)
    return paths


def eval_loss(model: LmHeadModel, batch: LmExample) -> LossFnOutput:
    """Use the native eval_lm loss and byte accounting without retraining."""
    model = jmp.get_policy("p=f32,c=bfloat16").cast_to_compute(inference_mode(model, True))
    losses = model.compute_next_token_loss(batch, reduction=None, reduction_axis=()).array
    return losses, batch.loss_weight.array, jnp.roll(batch.tokens.array, -1, axis=-1)


def evaluate_checkpoints(spec: dict, paths: dict[str, str], requests: list[dict]) -> None:
    """Reuse one compiled evaluator per model size and resume per checkpoint."""
    experiment.require_central1()
    validate_spec(spec)
    pending = [request for request in requests if verified_result(spec, request) is None]
    if not pending:
        return
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=f32,c=bfloat16"),
        train_batch_size=pending[0]["batch_size"],
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
    tokenizer = experiment.verified_tokenizer()
    model_config = experiment.model_config(experiment.Arm(pending[0]["arm"]))
    data = LmDataConfig(
        tokenizer=experiment.TOKENIZER,
        cache_dir=None,
        auto_build_caches=False,
        components={name: DatasetComponent(cache_dir=path) for name, path in paths.items()},
        train_weights={name: 0.0 for name in paths},
    )
    with trainer.use_device_mesh():
        evaluator = TaggedEvaluator(
            EvalBatch=trainer.EvalBatch,
            tagged_eval_sets=data.tagged_eval_sets(model_config.max_Pos.resize(experiment.SEQ_LEN)),
            loss_fn=eval_loss,
            tokenizer=tokenizer,
            axis_mapping=trainer.compute_axis_mapping,
        )
        for request in pending:
            checkpoint = checkpoint_metadata(request)
            if checkpoint != spec["checkpoints"][request["run_name"]]:
                raise ValueError(f"Frozen checkpoint changed: {request['run_name']}")
            with use_cpu_device():
                model = eqx.filter_eval_shape(
                    model_config.build, hax.Axis("vocab", len(tokenizer)), key=jax.random.PRNGKey(0)
                )
                model = load_checkpoint(model, checkpoint["path"], subpath="model")
            model = hax.shard_with_axis_mapping(model, trainer.parameter_axis_mapping)
            # Parent-tag averages are numpy.float32; immutable JSON receipts need Python scalars.
            metrics = {key: float(value) for key, value in eval_model(evaluator, model, prefix="eval").items()}
            scores = component_scores(metrics)
            control_error = abs(metrics[experiment.PRIMARY_METRIC] - spec["primary_bpb"][request["run_name"]])
            if control_error > PALOMA_TOLERANCE:
                raise ValueError(f"Restored PALOMA control failed for {request['run_name']}: {control_error}")
            result = {
                "spec_sha256": spec["spec_sha256"],
                "run_name": request["run_name"],
                "fingerprint": request["fingerprint"],
                "checkpoint": checkpoint,
                "component_bpb": scores,
                "macro_bpb": sum(scores.values()) / len(COMPONENTS),
                "paloma_control_error": control_error,
                "metrics": metrics,
            }
            if jax.process_index() == 0:
                persist_submission_plan(result, result_path(spec, request))
                logger.info("Completed %s: Uncheatable %.6f", request["run_name"], result["macro_bpb"])
            del model


async def submit(spec: dict) -> None:
    experiment.require_central1()
    validate_spec(spec)
    paths = await asyncio.to_thread(prepare_caches, spec)
    calls = []
    # Two batches reuse compilation across all checkpoints of the same model size.
    for target in (False, True):
        requests = [r for r in spec["training_plan"]["runs"] if (r["arm"] == "target") == target]
        pending = [r for r in requests if verified_result(spec, r) is None]
        if pending:
            worker = remote(
                evaluate_checkpoints,
                name=f"tpp10-uncheatable-{'target' if target else 'proxy'}",
                resources=TPU,
                env_vars={"MARIN_PREFIX": experiment.PREFIX},
            )
            calls.append(asyncio.to_thread(worker, spec, paths, pending))
    await asyncio.gather(*calls)


def collect(spec: dict, output: Path) -> None:
    values, rows = {}, []
    for request in spec["training_plan"]["runs"]:
        result = verified_result(spec, request)
        if result is None:
            raise ValueError(f"Evaluation not complete: {request['run_name']}")
        values[request["run_name"]] = result["macro_bpb"]
        rows.append(
            {
                "run_name": request["run_name"],
                "arm": request["arm"],
                "percent": request["percent"],
                "trainer_seed": request["trainer_seed"],
                "subset_seed": request["subset_seed"],
                "macro_bpb": result["macro_bpb"],
                **result["component_bpb"],
                "paloma_control_error": result["paloma_control_error"],
                "spec_sha256": spec["spec_sha256"],
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    with (output / "measurements.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = analysis.analyze(spec["training_plan"], values)
    result.update(metric=METRIC, spec_sha256=spec["spec_sha256"], aggregation=spec["aggregation"])
    (output / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    analysis.plot_result(
        result, output, metric_label="Uncheatable BPB", title="Same checkpoints, common Uncheatable evaluation"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build-from-plan", type=Path)
    action.add_argument("--submit", action="store_true")
    action.add_argument("--collect", type=Path)
    args = parser.parse_args()
    if args.build_from_plan:
        spec = build_spec(json.loads(args.build_from_plan.read_text()), args.spec)
        print(json.dumps({"spec_sha256": spec["spec_sha256"], "checkpoints": len(spec["checkpoints"])}))
        return
    spec = json.loads(args.spec.read_text())
    validate_spec(spec)
    if args.submit:
        asyncio.run(submit(spec))
    else:
        collect(spec, args.collect)


if __name__ == "__main__":
    main()
