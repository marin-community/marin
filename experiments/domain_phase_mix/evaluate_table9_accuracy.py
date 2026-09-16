# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resumable Table-9 companion inference for frozen regional HF checkpoints.

Use --prepare to freeze a plan from an existing ladder checkpoint inventory.
--submit releases the chosen mode for every unfinished checkpoint concurrently.
Canaries use separate identities; a canary can never satisfy full coverage.
Generation is retained separately from grading so grader changes never repeat TPU work.
"""

import argparse
import copy
import gzip
import hashlib
import json
import logging
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import fsspec
import jax
import jmp
from fray.types import ResourceConfig, get_tpu_topology
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.distributed import DistributedConfig
from levanter.eval_harness import SampleLoggingConfig, _LmEvalHarnessWorker
from levanter.model_loading import load_hf_checkpoint
from levanter.tokenizers import load_tokenizer as load_marin_tokenizer
from levanter.tracker import NoopConfig
from levanter.trainer import TrainerConfig
from levanter.utils.tree_utils import inference_mode
from lm_eval.api.instance import Instance
from marin.evaluation.olmo_base_eval.accuracy import (
    CHOICE_BACKFILL_TASKS,
    GENERATION_BACKFILL_TASKS,
    choice_metrics,
    validate_task_samples,
)
from marin.evaluation.olmo_base_eval.bpb import encode_context_continuation
from marin.execution.remote import remote

from experiments.domain_phase_mix import evaluate_mariner_ladder_accuracy as existing

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
PREFIX = "gs://marin-us-east5"
REQUEST_ROOT = PREFIX + "/raw/eval-datasets/table9_accuracy/20260914"
OUTPUT_ROOT = PREFIX + "/experiments/table9_accuracy_20260914"
MODES = {"choices": CHOICE_BACKFILL_TASKS, "generation": GENERATION_BACKFILL_TASKS}
TPU_ZONES = {"v5p-8": "us-east5-a", "v6e-4": "us-east5-b", "v6e-8": "us-east5-b"}
MEMORY_PROBE_DECODE_TOKENS = 64


def digest(value: dict) -> str:
    return hashlib.sha256(existing.canonical_json(value)).hexdigest()


def source_pins() -> dict:
    paths = (
        "experiments/domain_phase_mix/evaluate_table9_accuracy.py",
        "lib/marin/src/marin/evaluation/olmo_base_eval/accuracy.py",
        "lib/marin/src/marin/evaluation/olmo_base_eval/bpb.py",
        "lib/levanter/src/levanter/model_loading.py",
        "lib/levanter/src/levanter/inference/engine.py",
        "lib/levanter/src/levanter/layers/attention.py",
        "lib/levanter/src/levanter/layers/kv_cache.py",
    )
    return existing.code_pins() | {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths}


def write_verified(uri: str, data: bytes) -> dict:
    with fsspec.open(uri, "wb") as handle:
        handle.write(data)
    if existing.read_bytes(uri) != data:
        raise ValueError(f"Artifact readback differs: {uri}")
    return {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}


def task_requests(plan: dict, name: str) -> list[dict]:
    spec = plan["request_manifest"]["tasks"][name]
    data = existing.read_bytes(plan["requests_uri"] + "/" + spec["file"])
    if hashlib.sha256(data).hexdigest() != spec["sha256"] or len(data) != spec["size"]:
        raise ValueError(f"Frozen requests changed: {name}")
    rows = json.loads(gzip.decompress(data))
    if len(rows) != spec["count"] or [r["doc_id"] for r in rows] != list(range(len(rows))):
        raise ValueError(f"Request document identity differs: {name}")
    return rows


def protocol(plan: dict) -> dict:
    return {k: v for k, v in plan.items() if k not in {"rows", "output_root"}}


def result_root(plan: dict, row: dict, name: str, limit: int) -> str:
    identity = {"protocol": protocol(plan), "checkpoint": row, "task": name, "limit": limit}
    return f"{plan['output_root']}/{row['name']}/{name}/{digest(identity)}"


def completed_task(plan: dict, row: dict, name: str, limit: int) -> dict | None:
    root = result_root(plan, row, name, limit)
    fs, _ = fsspec.core.url_to_fs(root)
    if not fs.exists(root + "/SUCCESS.json"):
        return None
    marker = existing.read_json(root + "/SUCCESS.json")
    payload = existing.read_bytes(root + "/samples.json.gz")
    if marker["artifact"] != {"sha256": hashlib.sha256(payload).hexdigest(), "size": len(payload)}:
        raise ValueError(f"Completed samples changed: {root}")
    samples = json.loads(gzip.decompress(payload))
    expected = (
        min(limit, plan["request_manifest"]["tasks"][name]["count"])
        if limit
        else plan["request_manifest"]["tasks"][name]["count"]
    )
    if marker["row"] != row or marker["protocol_sha256"] != digest(protocol(plan)) or marker["limit"] != limit:
        raise ValueError(f"Completion provenance differs: {root}")
    stage = "scored" if name in CHOICE_BACKFILL_TASKS else "generated_ungraded"
    if marker["task"] != name or marker["count"] != expected or marker["stage"] != stage:
        raise ValueError(f"Completion task/count/stage differs: {root}")
    if len(samples) != expected or [s["doc_id"] for s in samples] != list(range(expected)):
        raise ValueError(f"Incomplete samples: {root}")
    if name in CHOICE_BACKFILL_TASKS:
        validate_task_samples(name, samples, list(range(expected)), plan["request_manifest"]["tasks"][name]["metric"])
    elif any(not isinstance(s["generation"], str) for s in samples):
        raise ValueError(f"Missing generated text: {root}")
    return marker


def evaluation_resources(plan: dict) -> ResourceConfig:
    """Validate the fixed protocol and return its single-host TPU placement."""
    if plan["tpu_type"] not in TPU_ZONES or plan["zone"] != TPU_ZONES[plan["tpu_type"]]:
        raise ValueError("Unsupported single-host TPU type or mismatched east5 zone")
    if plan["region"] != "us-east5" or plan["max_length"] != 8192 or plan["batch_size"] != 8:
        raise ValueError("Expected east5, batch 8 and length 8192")
    return ResourceConfig.with_tpu(
        plan["tpu_type"], cpu=8, ram="64g", disk="32g", regions=(plan["region"],), zone=plan["zone"]
    )


def validate_plan(plan: dict) -> None:
    if plan["schema_version"] != 1:
        raise ValueError("Expected schema 1")
    evaluation_resources(plan)
    if plan["source_pins"] != source_pins() or plan["runtime_versions"] != existing.runtime_versions():
        raise ValueError("Frozen evaluation code/runtime differs")
    if plan["lm_eval_revision"] != existing.lm_eval_revision():
        raise ValueError("Frozen lm-eval revision differs")
    if set(plan["request_manifest"]["tasks"]) != set(CHOICE_BACKFILL_TASKS + GENERATION_BACKFILL_TASKS):
        raise ValueError("Request manifest must cover all 20 non-deferred backfill tasks")
    if not plan["rows"] or len({r["name"] for r in plan["rows"]}) != len(plan["rows"]):
        raise ValueError("Expected unique checkpoint rows")
    for uri in (plan["requests_uri"], plan["output_root"], *(r["checkpoint_uri"] for r in plan["rows"])):
        if not uri.startswith(PREFIX + "/"):
            raise ValueError("All inputs and outputs must be region-local")
    for row in plan["rows"]:
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", row["name"]):
            raise ValueError("Unsafe checkpoint name")
        files = row["checkpoint_files"]
        if not {"config.json", "tokenizer.json"} <= files.keys():
            raise ValueError("Missing checkpoint config/tokenizer identity")
        if not {"model.safetensors", "model.safetensors.index.json"} & files.keys():
            raise ValueError("Missing checkpoint weight identity")
        if "model.safetensors.index.json" in files:
            index = existing.read_json(row["checkpoint_uri"] + "/model.safetensors.index.json")
            if not set(index["weight_map"].values()) <= files.keys():
                raise ValueError("Checkpoint manifest omits weight shards")
        fs, _ = fsspec.core.url_to_fs(row["checkpoint_uri"])
        for filename, expected in row["checkpoint_files"].items():
            actual = fs.info(row["checkpoint_uri"] + "/" + filename)
            if any(str(actual[k]) != str(expected[k]) for k in ("generation", "crc32c", "size")):
                raise ValueError(f"Checkpoint object identity changed: {row['name']}/{filename}")


def verify_native_prompts(plan: dict) -> None:
    """Check exact context/gold parity region-locally before allocating TPUs."""
    if not os.environ.get("IRIS_TASK_ID") or os.environ.get("MARIN_PREFIX") != PREFIX:
        raise ValueError("Read the large native request set only from the east5 Iris parent")
    expected = {name: task_requests(plan, name) for name in plan["request_manifest"]["tasks"]}
    matched = {name: set() for name in expected}
    uri = plan["request_manifest"]["native_request_set"] + "/requests.jsonl"
    with fsspec.open(uri, "rt") as handle:
        for line in handle:
            native = json.loads(line)
            name = native["task"]
            if name not in expected:
                continue
            row = expected[name][native["doc_id"]]
            if (native["context"], native["continuation"]) != (row["context"], row["reference"]):
                raise ValueError(f"Native prompt or gold mismatch: {name}/{row['doc_id']}")
            matched[name].add(row["doc_id"])
    if any(len(matched[name]) != len(rows) for name, rows in expected.items()):
        raise ValueError("Native request inventory is incomplete")
    logger.info("Exact native prompt/gold parity: all %d documents", sum(map(len, expected.values())))


def scored_choices(requests: list[dict], scores: list[tuple[float, bool]], encode) -> list[dict]:
    """Normalize each continuation's likelihood using its exact joined tokenization."""
    if len(scores) != sum(len(r["choices"]) for r in requests):
        raise ValueError("Choice score count differs from the request inventory")
    out = []
    start = 0
    for r in requests:
        counts = [
            encode_context_continuation(encode, r["context"], c, bos_token_id=None).num_continuation_tokens
            for c in r["choices"]
        ]
        values = [float(p) for p, _ in scores[start : start + len(counts)]]
        start += len(counts)
        out.append(
            r
            | {
                "logprobs": values,
                "token_counts": counts,
                "metrics": choice_metrics(values, counts, r["choices"], r["gold_index"]),
            }
        )
    return out


def evaluate_task(harness, requests: list[dict], spec: dict) -> list[dict]:
    name = requests[0]["task"]
    if name in CHOICE_BACKFILL_TASKS:
        inputs = [
            Instance(
                request_type="loglikelihood", doc=r, arguments=(r["context"], c), idx=i, metadata=(name, r["doc_id"], 1)
            )
            for r in requests
            for i, c in enumerate(r["choices"])
        ]
        return scored_choices(requests, harness.loglikelihood(inputs), harness.tok_encode)
    inputs = [
        Instance(
            request_type="generate_until",
            doc=r,
            # The harness appends the EOS string to `until` in place; never hand it the plan's own dict.
            arguments=(r["context"], copy.deepcopy(spec["generation"])),
            idx=0,
            metadata=(name, r["doc_id"], 1),
        )
        for r in requests
    ]
    outputs = harness.generate_until(inputs)
    if len(outputs) != len(requests):
        raise ValueError(f"Generation count mismatch: {name}")
    return [r | {"generation": generated} for r, generated in zip(requests, outputs, strict=True)]


def memory_probe_requests(encode, mode: str, batch_size: int, max_length: int) -> list[Instance]:
    """Fill the real checkpoint's context window without using evaluation examples."""
    reserve = 1 if mode == "choices" else MEMORY_PROBE_DECODE_TOKENS
    context = " x" * (max_length - reserve)
    if len(encode(context)) != max_length - reserve:
        raise ValueError("Memory probe tokenizer did not produce the required context length")
    if mode == "choices":
        if len(encode(context + " x")) != max_length:
            raise ValueError("Memory probe continuation does not fill the context window")
        request_type = "loglikelihood"
        arguments = (context, " x")
    else:
        request_type = "generate_until"
        arguments = (context, {"max_gen_toks": reserve, "until": [], "temperature": 0.0})
    return [
        Instance(request_type=request_type, doc={}, arguments=arguments, idx=0, metadata=("memory_probe", i, 1))
        for i in range(batch_size)
    ]


def memory_probe_identity(plan: dict, row: dict, mode: str) -> dict:
    return {
        "row": row,
        "protocol_sha256": digest(protocol(plan)),
        "mode": mode,
        "batch_size": plan["batch_size"],
        "max_length": plan["max_length"],
        "backend": "tpu",
        "devices": get_tpu_topology(plan["tpu_type"]).chip_count,
    }


def completed_memory_probe(plan: dict, row: dict, mode: str) -> dict | None:
    uri = result_root(plan, row, "memory_" + mode, 0) + "/SUCCESS.json"
    fs, _ = fsspec.core.url_to_fs(uri)
    if not fs.exists(uri):
        return None
    marker = existing.read_json(uri)
    expected = memory_probe_identity(plan, row, mode)
    if any(marker[k] != value for k, value in expected.items()) or marker["count"] != plan["batch_size"]:
        raise ValueError(f"Memory probe provenance/count differs: {uri}")
    return marker


def run_memory_probe(harness, plan: dict, row: dict, mode: str) -> None:
    requests = memory_probe_requests(harness.tok_encode, mode, plan["batch_size"], plan["max_length"])
    logger.info("Memory probe %s/%s: %d full-window requests", row["name"], mode, len(requests))
    if mode == "choices":
        outputs = harness.loglikelihood(requests)
        if any(not math.isfinite(score) for score, _ in outputs):
            raise ValueError("Nonfinite memory-probe likelihood")
    else:
        outputs = harness.generate_until(requests)
        if any(not isinstance(output, str) for output in outputs):
            raise ValueError("Missing memory-probe generation")
    if len(outputs) != len(requests):
        raise ValueError("Incomplete memory probe")
    marker = memory_probe_identity(plan, row, mode) | {
        "count": len(outputs),
        "decode_token_budget": MEMORY_PROBE_DECODE_TOKENS if mode == "generation" else 0,
        "memory_stats": [device.memory_stats() for device in jax.local_devices()],
        "completed_at": datetime.now(UTC).isoformat(),
    }
    uri = result_root(plan, row, "memory_" + mode, 0) + "/SUCCESS.json"
    write_verified(uri, existing.canonical_json(marker))
    completed_memory_probe(plan, row, mode)
    harness.clear_sample_outputs()
    logger.info("Durable memory probe: %s", uri)


def checkpoint_tokenizers(checkpoint_uri: str, cache_dir: Path):
    """Load both evaluation interfaces from the checkpoint's tokenizer files."""
    # Keep staged files alive for Marin's local loader and later HF reconstruction.
    hf_tokenizer = load_tokenizer(checkpoint_uri, local_cache_dir=str(cache_dir))
    return hf_tokenizer, load_marin_tokenizer(hf_tokenizer.name_or_path)


def evaluate_row(plan: dict, row: dict, mode: str, limit: int) -> None:
    validate_plan(plan)
    pending = [name for name in MODES[mode] if completed_task(plan, row, name, limit) is None]
    needs_probe = completed_memory_probe(plan, row, mode) is None
    if not pending and not needs_probe:
        return
    DistributedConfig().initialize()
    if jax.default_backend() != "tpu" or jax.process_count() != 1:
        raise ValueError("Generation and companion scoring require a single-host slice")
    devices = jax.device_count()
    if devices != get_tpu_topology(plan["tpu_type"]).chip_count:
        raise ValueError("Actual TPU device count differs from the frozen plan")
    if plan["batch_size"] % devices:
        raise ValueError("Evaluation batch must divide evenly over TPU devices")
    trainer = TrainerConfig(
        tracker=NoopConfig(),
        mp=jmp.get_policy("p=bfloat16,c=bfloat16"),
        per_device_eval_parallelism=plan["batch_size"] // devices,
        distributed=DistributedConfig(initialize_jax_distributed=False),
        log_jaxprs=False,
        log_xla_hlo=False,
        shutdown_at_exit=False,
    )
    trainer.initialize()
    if trainer.EvalBatch.size != plan["batch_size"]:
        raise ValueError("Actual evaluation batch differs from the frozen plan")
    model_config = existing.checkpoint_model_config(row["checkpoint_uri"])
    hf_tokenizer, tokenizer = checkpoint_tokenizers(row["checkpoint_uri"], Path(".cache/table9-tokenizers"))
    with trainer.use_device_mesh():
        model = load_hf_checkpoint(
            model_config,
            row["checkpoint_uri"],
            axis_mapping=trainer.parameter_axis_mapping,
            tokenizer=hf_tokenizer,
            compute_dtype=trainer.mp.compute_dtype,
        )
        model = inference_mode(model, True)
        worker = _LmEvalHarnessWorker(
            trainer.EvalBatch,
            model.Pos.resize(plan["max_length"]),
            model,
            trainer.compute_axis_mapping,
            tokenizer,
            trainer.mp,
            max_packed_segments=64,
            generation_kwargs={"temperature": 0.0, "n": 1, "seed": 0},
            sample_logging_config=SampleLoggingConfig(log_all=True),
        )
        harness = worker.make_harness_lm()
        if needs_probe:
            run_memory_probe(harness, plan, row, mode)
        for name in pending:
            requests = task_requests(plan, name)
            if limit:
                requests = requests[:limit]
            spec = plan["request_manifest"]["tasks"][name]
            logger.info("Evaluating %s/%s: %d documents", row["name"], name, len(requests))
            samples = evaluate_task(harness, requests, spec)
            if name in CHOICE_BACKFILL_TASKS:
                validate_task_samples(name, samples, list(range(len(requests))), spec["metric"])
            elif any(not isinstance(s["generation"], str) for s in samples):
                raise ValueError(f"Missing generated text: {name}")
            payload = gzip.compress(json.dumps(samples, sort_keys=True, allow_nan=False).encode(), mtime=0)
            root = result_root(plan, row, name, limit)
            artifact = write_verified(root + "/samples.json.gz", payload)
            marker = {
                "row": row,
                "protocol_sha256": digest(protocol(plan)),
                "limit": limit,
                "task": name,
                "count": len(samples),
                "artifact": artifact,
                "stage": "scored" if mode == "choices" else "generated_ungraded",
                "completed_at": datetime.now(UTC).isoformat(),
            }
            write_verified(root + "/SUCCESS.json", existing.canonical_json(marker))
            completed_task(plan, row, name, limit)
            harness.clear_sample_outputs()
            logger.info("Durable %s: %s", marker["stage"], root)
        worker.stop()


def submit(plan: dict, mode: str, limit: int) -> None:
    if not os.environ.get("IRIS_TASK_ID") or os.environ.get("MARIN_PREFIX") != PREFIX:
        raise ValueError("Submission must run on an explicitly placed east5 Iris parent")
    validate_plan(plan)
    verify_native_prompts(plan)
    if not limit:
        for row in plan["rows"]:
            if completed_memory_probe(plan, row, mode) is None:
                raise ValueError(f"Full release requires the memory canary: {row['name']}/{mode}")
            for name in MODES[mode]:
                if completed_task(plan, row, name, 2) is None:
                    raise ValueError(f"Full release requires the two-document canary: {row['name']}/{name}")
    pending = [
        r
        for r in plan["rows"]
        if completed_memory_probe(plan, r, mode) is None
        or any(completed_task(plan, r, n, limit) is None for n in MODES[mode])
    ]
    if not pending:
        return
    for row in pending:
        _, tokenizer = checkpoint_tokenizers(row["checkpoint_uri"], Path(".cache/table9-tokenizers"))
        memory_probe_requests(
            partial(tokenizer.encode, add_special_tokens=False), mode, plan["batch_size"], plan["max_length"]
        )
        logger.info("Tokenizer and memory-probe tokenization preflight passed: %s", row["name"])
    resources = evaluation_resources(plan)
    with ThreadPoolExecutor(max_workers=len(pending)) as pool:
        futures = []
        for row in pending:
            child = remote(
                evaluate_row,
                name=f"table9-accuracy-{mode}-{row['name']}-{digest(protocol(plan))[:10]}-n{limit}",
                resources=resources,
                pip_dependency_groups=["tpu", "lm_eval"],
                env_vars={"MARIN_PREFIX": PREFIX, "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"},
            )
            futures.append(pool.submit(child, plan, row, mode, limit))
        for future in futures:
            future.result()


def prepare_plan(args) -> None:
    base = json.loads(args.checkpoint_plan.read_text())
    payload = (args.requests / "manifest.json").read_bytes()
    manifest = json.loads(payload)
    requests_uri = REQUEST_ROOT + "/" + hashlib.sha256(payload).hexdigest()
    for spec in manifest["tasks"].values():
        write_verified(requests_uri + "/" + spec["file"], (args.requests / spec["file"]).read_bytes())
    write_verified(requests_uri + "/manifest.json", payload)
    plan = {
        "schema_version": 1,
        "region": "us-east5",
        "zone": TPU_ZONES[args.tpu_type],
        "tpu_type": args.tpu_type,
        "max_length": 8192,
        "batch_size": 8,
        "requests_uri": requests_uri,
        "request_manifest": manifest,
        "output_root": OUTPUT_ROOT,
        "rows": base["rows"],
        "source_pins": source_pins(),
        "runtime_versions": existing.runtime_versions(),
        "lm_eval_revision": existing.lm_eval_revision(),
    }
    validate_plan(plan)
    args.plan.write_text(json.dumps(plan, indent=2) + "\n")
    logger.info("Frozen plan %s", digest(plan))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--checkpoint-plan", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--tpu-type", choices=TPU_ZONES)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--verify-prompts", action="store_true")
    parser.add_argument("--mode", choices=MODES, default="choices")
    parser.add_argument("--canary-documents", type=int, default=0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.canary_documents < 0:
        parser.error("Canary size must be nonnegative")
    if args.prepare:
        if not args.tpu_type or not args.checkpoint_plan or not args.requests:
            parser.error("--prepare requires --tpu-type, --checkpoint-plan and --requests")
        prepare_plan(args)
        return
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    if args.verify_prompts:
        verify_native_prompts(plan)
    if args.submit:
        submit(plan, args.mode, args.canary_documents)


if __name__ == "__main__":
    main()
