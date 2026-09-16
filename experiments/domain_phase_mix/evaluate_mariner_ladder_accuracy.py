# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Evaluate frozen MARINER ladder HF checkpoints on the complete easy-overlap suite.

Run with ``uv run --all-packages --extra lm_eval python -m experiments.domain_phase_mix.evaluate_mariner_ladder_accuracy
--plan PLAN.json`` to validate, adding ``--submit`` only on an east5 Iris parent. The parent must have an
explicit 48-hour timeout; the current Fray remote API has no child timeout parameter. Every selected row is
released concurrently as one v5p-8 child in us-east5-a. Completed outputs are reused only after identity, coverage and
artifact hashes pass verification.

The JSON plan contains schema_version=1, region, zone, cache_uri, cache_manifest_sha256, output_root,
tpu_type, max_length, batch_size, seed=0, runtime_versions, lm_eval_revision, source_hashes, uv_lock_sha256,
and rows. Each row contains name, method, checkpoint_uri and checkpoint_files: a mapping from relative
filename to {size, generation, crc32c}. Additional row provenance is retained. The canonical hash of the
entire plan identifies its output directory; the plan must not contain its own hash.
"""

from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import fsspec
import jax
import jmp
from fray.types import ResourceConfig
from levanter.distributed import DistributedConfig
from levanter.eval_harness import (
    EvalHarnessMainConfig,
    LmEvalHarnessConfig,
    SampleLoggingConfig,
    ensure_lm_eval_available,
    run_eval_harness_main,
)
from levanter.models.qwen import Qwen3Config
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.py_utils import FailSafeJSONEncoder
from marin.evaluation.eval_dataset_cache import (
    MANIFEST_FILE,
    CacheManifest,
    load_eval_datasets_from_gcs,
)
from marin.evaluation.evaluation_config import convert_to_levanter_task_config
from marin.execution.remote import remote

from experiments.evals.olmo_base_easy_overlap import MMLU_SUBJECT_TO_CATEGORY, OLMO_BASE_EASY_OVERLAP_TASKS

logger = logging.getLogger(__name__)
REPO = Path(__file__).resolve().parents[2]
REGION = "us-east5"
ZONE = "us-east5-a"
PREFIX = "gs://marin-us-east5"
OUTPUT_ROOT = PREFIX + "/experiments/mariner_ladder_accuracy_20260914"
TPU_TYPE = "v5p-8"
EXPECTED_TASKS = frozenset(
    [f"mmlu_{subject}_5shot" for subject in MMLU_SUBJECT_TO_CATEGORY]
    + [cast(str, task.task_alias) for task in OLMO_BASE_EASY_OVERLAP_TASKS if task.name != "mmlu"]
)
RUNTIME_PACKAGES = ("lm-eval", "datasets", "transformers", "jax", "jaxlib")
ARTIFACTS = ("results.json.gz", "summary_metrics.json", "provenance.json")


def canonical_json(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def plan_sha256(plan: dict) -> str:
    return hashlib.sha256(canonical_json(plan)).hexdigest()


def code_pins() -> dict[str, str]:
    """Hash the evaluation implementation and its model, data and runtime boundaries."""
    paths = (
        "experiments/domain_phase_mix/evaluate_mariner_ladder_accuracy.py",
        "experiments/evals/olmo_base_easy_overlap.py",
        "lib/levanter/src/levanter/eval_harness.py",
        "lib/levanter/src/levanter/eval_harness_config.py",
        "lib/levanter/src/levanter/eval_harness_metrics.py",
        "lib/levanter/src/levanter/data/packing.py",
        "lib/levanter/src/levanter/data/loader.py",
        "lib/levanter/src/levanter/utils/jax_utils.py",
        "lib/levanter/src/levanter/utils/background_iterable.py",
        "lib/levanter/src/levanter/models/qwen.py",
        "lib/levanter/src/levanter/models/llama.py",
        "lib/levanter/src/levanter/compat/hf_checkpoints.py",
        "lib/levanter/src/levanter/tokenizers.py",
        "lib/levanter/src/levanter/trainer.py",
        "lib/levanter/src/levanter/distributed.py",
        "lib/marin/src/marin/evaluation/eval_dataset_cache.py",
        "lib/marin/src/marin/evaluation/evaluation_config.py",
        "lib/marin/src/marin/execution/remote.py",
        "lib/marin/src/marin/training/run_environment.py",
        "uv.lock",
    )
    return {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for path in paths}


def runtime_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in RUNTIME_PACKAGES}


def lm_eval_revision() -> str:
    metadata = importlib.metadata.distribution("lm-eval").read_text("direct_url.json")
    if metadata is None:
        raise ValueError("lm-eval lacks the required VCS installation provenance")
    return json.loads(metadata)["vcs_info"]["commit_id"]


def read_bytes(uri: str) -> bytes:
    with fsspec.open(uri, "rb") as handle:
        return handle.read()


def read_json(uri: str) -> dict:
    return json.loads(read_bytes(uri))


def validate_plan(plan: dict) -> None:
    """Check the frozen configuration and local code without initializing accelerators."""
    ensure_lm_eval_available()
    if plan["schema_version"] != 1 or plan["region"] != REGION or plan["zone"] != ZONE:
        raise ValueError("Expected schema 1 and explicit us-east5/us-east5-a placement")
    if plan["output_root"] != OUTPUT_ROOT or not plan["cache_uri"].startswith(PREFIX + "/"):
        raise ValueError("Evaluation outputs and dataset cache must use the approved east5 paths")
    if plan["tpu_type"] != TPU_TYPE or plan["max_length"] != 4096 or plan["seed"] != 0:
        raise ValueError("Expected v5p-8, max_length=4096 and the harness's fixed seed 0")
    if plan["batch_size"] <= 0 or plan["batch_size"] % 8:
        raise ValueError("batch_size must be a positive multiple of eight for the frozen evaluation protocol")
    if plan["source_hashes"] != code_pins():
        raise ValueError("Evaluation source differs from the frozen plan")
    if plan["uv_lock_sha256"] != code_pins()["uv.lock"]:
        raise ValueError("uv.lock differs from the frozen plan")
    if plan["runtime_versions"] != runtime_versions() or plan["lm_eval_revision"] != lm_eval_revision():
        raise ValueError("Installed evaluation packages differ from the frozen runtime")
    rows = plan["rows"]
    if not rows or len({row["name"] for row in rows}) != len(rows):
        raise ValueError("Plan must contain unique checkpoint row names")
    for row in rows:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", row["name"]) or not row["method"]:
            raise ValueError("Expected a safe row name and an explicit method")
        if not row["checkpoint_uri"].startswith(PREFIX + "/"):
            raise ValueError("Every checkpoint must be region-local in marin-us-east5")
        files = row["checkpoint_files"]
        if not {"config.json", "tokenizer.json"}.issubset(files):
            raise ValueError("Checkpoint pins must include model config and tokenizer.json")
        if not ({"model.safetensors", "model.safetensors.index.json"} & files.keys()):
            raise ValueError("Checkpoint pins must include HF model weights or the shard index")
        for filename, info in files.items():
            if filename.startswith("/") or ".." in Path(filename).parts:
                raise ValueError(f"Unsafe checkpoint filename: {filename}")
            if not info["generation"] or not info["crc32c"] or info["size"] <= 0:
                raise ValueError(f"Incomplete checkpoint object identity: {filename}")


def validate_inputs(plan: dict) -> CacheManifest:
    """Verify the regional cache manifest and every pinned checkpoint object without reading weights."""
    payload = read_bytes(plan["cache_uri"] + "/" + MANIFEST_FILE)
    if hashlib.sha256(payload).hexdigest() != plan["cache_manifest_sha256"]:
        raise ValueError("Regional evaluation cache manifest differs from the frozen plan")
    manifest = CacheManifest.from_dict(json.loads(payload))
    if not manifest.supports_full_offline_task_loading():
        raise ValueError("Prepare a complete regional HF cache including Hub/modules metadata before evaluation")
    if not {task.name for task in OLMO_BASE_EASY_OVERLAP_TASKS}.issubset(manifest.task_names):
        raise ValueError("Regional cache does not declare all eleven evaluation families")
    for row in plan["rows"]:
        root = row["checkpoint_uri"].rstrip("/")
        fs, _ = fsspec.core.url_to_fs(root)
        for filename, expected in row["checkpoint_files"].items():
            actual = fs.info(root + "/" + filename)
            if any(str(actual[key]) != str(expected[key]) for key in ("size", "generation", "crc32c")):
                raise ValueError(f"Checkpoint object changed: {root}/{filename}")
        if "model.safetensors.index.json" in row["checkpoint_files"]:
            shards = set(read_json(root + "/model.safetensors.index.json")["weight_map"].values())
            if not shards.issubset(row["checkpoint_files"]):
                raise ValueError(f"Checkpoint manifest omits weight shards: {root}")
    return manifest


def output_uri(plan: dict, row: dict) -> str:
    return f"{plan['output_root']}/{plan_sha256(plan)}/{row['name']}"


def harness_config(plan: dict) -> LmEvalHarnessConfig:
    # Match the immutable regional cache rather than upstream's renamed dataset repositories.
    dataset_paths = {"sciq": "sciq", "winogrande": "winogrande", "social_iqa": "social_i_qa"}
    tasks = [
        replace(task, dataset_path=dataset_paths[task.task]) if task.task in dataset_paths else task
        for task in convert_to_levanter_task_config(OLMO_BASE_EASY_OVERLAP_TASKS)
    ]
    return LmEvalHarnessConfig(
        task_spec=tasks,
        max_examples=None,
        max_length=plan["max_length"],
        log_samples=True,
        sample_logging=SampleLoggingConfig(log_all=True),
        apply_chat_template=False,
    )


def checkpoint_model_config(checkpoint_uri: str) -> Qwen3Config:
    """Restore the archived Qwen3 architecture and retain its own offline tokenizer reference."""
    converter = Qwen3Config(reference_checkpoint=checkpoint_uri, tokenizer=checkpoint_uri).hf_checkpoint_converter()
    hf_config = converter.hf_config_from_hf_checkpoint(checkpoint_uri)
    if hf_config.model_type != "qwen3":
        raise ValueError(f"Expected an archived Qwen3 checkpoint: {checkpoint_uri}")
    model = converter.config_from_hf_config(hf_config)
    # from_hf_config restores architecture fields, but resets the default checkpoint/tokenizer references.
    return replace(model, reference_checkpoint=checkpoint_uri, tokenizer=checkpoint_uri)


def task_document_counts(tasks: dict) -> dict[str, int]:
    """Count every leaf's complete evaluation split before loading the model."""
    counts = {}
    for task in tasks.values():
        if isinstance(task, dict):
            counts.update(task_document_counts(task))
        else:
            counts[task.config.task] = len(task.eval_docs)
    return counts


def validate_results(results: dict, expected_counts: dict[str, int]) -> None:
    """Require every leaf, accuracy metric, configuration and full per-document payload."""
    if set(expected_counts) != EXPECTED_TASKS or any(count <= 0 for count in expected_counts.values()):
        raise ValueError("Expected positive document counts for all 67 task leaves")
    for name, count in expected_counts.items():
        metrics = results["results"][name]
        accuracy = [value for key, value in metrics.items() if key.split(",")[0] in ("acc", "acc_norm")]
        if not accuracy or not all(math.isfinite(value) and 0 <= value <= 1 for value in accuracy):
            raise ValueError(f"Missing or invalid accuracy for {name}")
        counts = results["n-samples"][name]
        if counts["original"] != count or counts["effective"] != count:
            raise ValueError(f"Incomplete evaluation split for {name}: {counts}, expected {count}")
        samples = results["samples"][name]
        if len(samples) != count or len({sample["doc_id"] for sample in samples}) != count:
            raise ValueError(f"Incomplete or duplicated per-document samples for {name}")
        if len(metrics["outputs"]) < count:
            raise ValueError(f"Incomplete Levanter per-request outputs for {name}")
        expected_shots = 0 if name == "lambada_0shot" else 5
        if results["configs"][name]["num_fewshot"] != expected_shots:
            raise ValueError(f"Unexpected few-shot configuration for {name}")


def verified_result(plan: dict, row: dict) -> dict | None:
    """Resume only after reading and verifying every durable output artifact."""
    root = output_uri(plan, row)
    fs, _ = fsspec.core.url_to_fs(root)
    if not fs.exists(root + "/SUCCESS.json"):
        return None
    marker = read_json(root + "/SUCCESS.json")
    if marker["plan_sha256"] != plan_sha256(plan) or marker["row"] != row:
        raise ValueError(f"Completed evaluation identity mismatch: {root}")
    if set(marker["artifacts"]) != set(ARTIFACTS):
        raise ValueError(f"Completed evaluation artifact inventory is incomplete: {root}")
    data = {}
    for filename in ARTIFACTS:
        encoded = read_bytes(root + "/" + filename)
        identity = {"size": len(encoded), "sha256": hashlib.sha256(encoded).hexdigest()}
        if identity != marker["artifacts"][filename]:
            raise ValueError(f"Completed evaluation artifact changed: {root}/{filename}")
        data[filename] = json.loads(gzip.decompress(encoded) if filename.endswith(".gz") else encoded)
    provenance = data["provenance.json"]
    if provenance["plan_sha256"] != plan_sha256(plan) or provenance["row"] != row:
        raise ValueError(f"Completed evaluation provenance mismatch: {root}")
    validate_results(data["results.json.gz"], provenance["expected_task_samples"])
    return marker


def persist_results(plan: dict, row: dict, results: dict, counts: dict[str, int], topology: dict) -> None:
    root = output_uri(plan, row)
    encoded_results = json.dumps(results, cls=FailSafeJSONEncoder).encode()
    normalized = json.loads(encoded_results)
    validate_results(normalized, counts)
    summary = {key: value for key, value in normalized.items() if key not in ("results", "samples")}
    summary["results"] = {
        name: {key: value for key, value in metrics.items() if key != "outputs"}
        for name, metrics in normalized["results"].items()
    }
    provenance = {
        "plan_sha256": plan_sha256(plan),
        "plan": plan,
        "row": row,
        "expected_task_samples": counts,
        "runtime_versions": runtime_versions(),
        "lm_eval_revision": lm_eval_revision(),
        "topology": topology,
        "completed_at": datetime.now(UTC).isoformat(),
    }
    payloads = {
        "results.json.gz": gzip.compress(encoded_results, mtime=0),
        "summary_metrics.json": json.dumps(summary, cls=FailSafeJSONEncoder).encode(),
        "provenance.json": canonical_json(provenance),
    }
    identities = {}
    for filename, payload in payloads.items():
        uri = root + "/" + filename
        with fsspec.open(uri, "wb") as handle:
            handle.write(payload)
        expected = {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        actual = read_bytes(uri)
        if len(actual) != expected["size"] or hashlib.sha256(actual).hexdigest() != expected["sha256"]:
            raise ValueError(f"Durable output verification failed: {uri}")
        identities[filename] = expected
    marker = {"plan_sha256": plan_sha256(plan), "row": row, "artifacts": identities}
    with fsspec.open(root + "/SUCCESS.json", "wb") as handle:
        handle.write(canonical_json(marker))
    if read_json(root + "/SUCCESS.json") != marker:
        raise ValueError(f"Completion marker readback failed: {root}")
    logger.info("Completed %s: %s", row["name"], root)
    logger.info("All returned metrics: %s", json.dumps(summary["results"]))


def evaluate_row(plan: dict, row: dict) -> None:
    """Validate, stage the offline dataset cache, and evaluate one HF checkpoint."""
    validate_plan(plan)
    validate_inputs(plan)
    if verified_result(plan, row) is not None:
        logger.info("Verified completed evaluation: %s", row["name"])
        return
    if os.environ.get("HF_DATASETS_OFFLINE") != "1" or os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Workers must start with both HF_DATASETS_OFFLINE=1 and HF_HUB_OFFLINE=1")
    manifest = load_eval_datasets_from_gcs(plan["cache_uri"])
    if manifest is None or not manifest.supports_full_offline_task_loading():
        raise ValueError("Failed to stage the complete regional cache; repair it before resubmitting")
    config = harness_config(plan)
    counts = task_document_counts(config.to_task_dict())
    if set(counts) != EXPECTED_TASKS:
        raise ValueError(f"Offline cache task coverage differs: {sorted(set(counts) ^ EXPECTED_TASKS)}")

    # HF conversion can touch JAX; initialize distributed execution before opening the model.
    DistributedConfig().initialize()
    device_count = jax.device_count()
    if plan["batch_size"] < device_count or plan["batch_size"] % device_count:
        raise ValueError(f"Evaluation batch size {plan['batch_size']} is not divisible by {device_count} JAX devices")
    model = checkpoint_model_config(row["checkpoint_uri"])
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
        per_device_eval_parallelism=plan["batch_size"] // device_count,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_jaxprs=False,
        log_xla_hlo=False,
        shutdown_at_exit=False,
    )
    eval_config = EvalHarnessMainConfig(
        eval_harness=config,
        tokenizer=row["checkpoint_uri"],
        checkpoint_path=row["checkpoint_uri"],
        checkpoint_is_hf=True,
        trainer=trainer,
        model=model,
    )
    # The harness prints the complete sample payload. Keep it off the Iris log stream.
    with tempfile.TemporaryFile(mode="w+") as transcript, contextlib.redirect_stdout(transcript):
        results = run_eval_harness_main(eval_config)
    if jax.process_index() == 0:
        if results is None:
            raise ValueError("The leader returned no evaluation results")
        topology = {"devices": jax.device_count(), "processes": jax.process_count(), "backend": jax.default_backend()}
        persist_results(plan, row, results, counts, topology)


def submit(plan: dict) -> None:
    """Release all unfinished rows concurrently through the existing Fray remote runner."""
    if os.environ.get("MARIN_PREFIX") != PREFIX or not os.environ.get("IRIS_TASK_ID"):
        raise ValueError(
            "Run --submit from the explicitly placed east5 Iris parent with MARIN_PREFIX=gs://marin-us-east5"
        )
    pending = [row for row in plan["rows"] if verified_result(plan, row) is None]
    if not pending:
        logger.info("Every evaluation has verified durable outputs")
        return
    resources = ResourceConfig.with_tpu(TPU_TYPE, cpu=8, ram="64g", disk="32g", regions=(REGION,), zone=ZONE)
    with ThreadPoolExecutor(max_workers=len(pending)) as pool:
        futures = []
        for row in pending:
            worker = remote(
                evaluate_row,
                name=f"mariner-ladder-accuracy-{row['name']}-{plan_sha256(plan)[:12]}",
                resources=resources,
                pip_dependency_groups=["tpu", "lm_eval"],
                env_vars={"MARIN_PREFIX": PREFIX, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"},
            )
            futures.append(pool.submit(worker, plan, row))
        for future in futures:
            future.result()
    for row in plan["rows"]:
        if verified_result(plan, row) is None:
            raise ValueError(f"Child finished without verified durable results: {row['name']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    validate_inputs(plan)
    logger.info("Validated plan %s: %d checkpoints, 67 full task leaves", plan_sha256(plan), len(plan["rows"]))
    if args.submit:
        submit(plan)


if __name__ == "__main__":
    main()
