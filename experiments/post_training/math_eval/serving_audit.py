# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bind a completed native serving task to its exact offline-scored response table."""

import hashlib
import json
import math
from datetime import UTC, datetime

import pyarrow as pa
import pyarrow.parquet as pq
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.harness import build_records
from experiments.post_training.math_eval.pool import MODEL_TEMPLATES
from experiments.post_training.math_eval.rate import MODEL_PROFILES
from experiments.post_training.math_eval.rendering import (
    INCREMENTAL_RENDERING,
    render_serving_response,
    renderer_provenance,
)
from experiments.post_training.math_eval.scoring import SEMANTIC_DEPENDENCIES
from experiments.post_training.math_eval.serving import (
    SERVING_MODELS,
    load_rating_inputs,
    serving_configuration,
    serving_specification,
    verify_wheel_receipt,
)
from experiments.post_training.math_eval.serving_records import completion_request

DUMP_IDENTITY_KEYS = (
    "step",
    "dump_namespace",
    "rows",
    "unique_uids",
    "ordered_prompt_sha256",
    "ordered_response_sha256",
    "ordered_result_sha256",
)


def validate_serving_generation(config, generation, native_tasks, native_job):
    """Use controller task evidence and emitted runtime configuration, not caller labels."""
    model, _engine = serving_configuration(config)
    expected = serving_specification(config)
    if (
        generation.get("schema") != "math_eval_serving_generation_v1"
        or audit.canonical_sha(expected) != generation.get("specification_sha256")
        or audit.canonical_sha(generation.get("specification")) != generation.get("specification_sha256")
    ):
        raise ValueError("Serving generation differs from the frozen specification")
    tasks = native_tasks.get("tasks", [])
    if len(tasks) != 1:
        raise ValueError("Serving audit requires exactly one native task")
    task = tasks[0]
    controller = native_job["job"]
    gpu = controller.get("resources", {}).get("device", {}).get("gpu", {})
    allocation = generation.get("controller_allocation", {})
    allocated_gpu = allocation.get("resources", {}).get("device", {})
    if (
        controller.get("job_id", "") + "/0" != task["task_id"]
        or controller.get("state") != "JOB_STATE_SUCCEEDED"
        or controller.get("exit_code") != 0
        or controller.get("cluster") != "cw-us-east-02a"
        or task.get("cluster") != "cw-us-east-02a"
        or allocation.get("controller_scope") not in {"local", "cw-us-east-02a"}
        or controller.get("task_count") != 1
        or controller.get("completed_count") != 1
        or gpu.get("variant") != "H100"
        or gpu.get("count") != config.allocated_gpus
        or allocated_gpu.get("kind") != "gpu"
        or allocated_gpu.get("variant") != gpu.get("variant")
        or allocated_gpu.get("count") != gpu.get("count")
    ):
        raise ValueError("Controller allocation does not prove the expected GPU resources and region")
    attempts = task.get("attempts", [])
    if (
        task.get("task_id") != generation.get("task_id")
        or task.get("state") != "TASK_STATE_SUCCEEDED"
        or task.get("exit_code") != 0
        or task.get("current_attempt_id") != 0
        or len(attempts) != 1
        or attempts[0].get("attempt_id") != 0
        or attempts[0].get("state") != "TASK_STATE_SUCCEEDED"
        or attempts[0].get("exit_code") != 0
        or generation.get("attempt_id") != 0
        or generation.get("worker_region_hint") not in {None, "cw-us-east-02a"}
    ):
        raise ValueError("Serving task did not finish once successfully in the permitted region")
    start = int(task["started_at"]["epoch_ms"])
    finish = int(task["finished_at"]["epoch_ms"])
    if (
        start <= 0
        or finish <= start
        or int(attempts[0]["started_at"]["epoch_ms"]) != start
        or int(attempts[0]["finished_at"]["epoch_ms"]) != finish
        or allocation.get("started_at_ms") != start
        or (
            attempts[0].get("attempt_uid")
            and any(
                value != attempts[0]["attempt_uid"]
                for value in (generation.get("attempt_uid"), allocation.get("attempt_uid"))
            )
        )
    ):
        raise ValueError("Serving allocation lacks a matching native attempt identity and terminal interval")
    timing = generation.get("generation_and_native_scoring")
    if timing is not None and (
        not isinstance(timing, dict)
        or not start <= timing.get("started_at_ms", 0) < timing.get("finished_at_ms", 0) <= finish
        or not isinstance(timing.get("monotonic_seconds"), (int, float))
        or not math.isfinite(timing["monotonic_seconds"])
        or not 0 < timing["monotonic_seconds"] <= (finish - start) / 1000
        or abs(timing["monotonic_seconds"] - (timing["finished_at_ms"] - timing["started_at_ms"]) / 1000) > 1
    ):
        raise ValueError("Producer generation interval is not consistent with the native task lifetime")
    if generation.get("score_dependency_versions") != (SEMANTIC_DEPENDENCIES | {"reasoning-gym": "0.1.25"}):
        raise ValueError("Native scorer dependency versions differ from the frozen verifier")
    profile = MODEL_PROFILES[config.model]
    for key, expected_value in {
        "model_identity": SERVING_MODELS[config.model]["identity"],
        "model_config_sha256": SERVING_MODELS[config.model]["config_sha256"],
        "tokenizer_sha256": profile["tokenizer_sha256"],
        "prompt_template_id": profile["prompt_template_id"],
        "engine_global_seed": 17,
        "request_sampling_seed": None,
        "expected_ids_sha256": audit.canonical_sha(sorted(config.expected_ids)),
        "rows": len(config.expected_ids) * config.samples,
    }.items():
        if key not in generation or generation[key] != expected_value:
            raise ValueError(f"Serving generation has incompatible {key}")
    runtime = generation["runtime"]
    if "renderer" in generation and generation["renderer"] != renderer_provenance():
        raise ValueError("Serving renderer differs from the pinned replay implementation")
    verify_wheel_receipt("MARIN_VLLM_WHEEL_VERIFIED=" + json.dumps(runtime), runtime["api_version"])
    command = generation.get("native_command", [])
    flags = {
        "--seed": "17",
        "--generation-config": "vllm",
        "--max-model-len": str(model.max_model_len),
        "--tensor-parallel-size": str(config.tensor_parallel_size),
        "--served-model-name": model.model_id,
        "--dtype": "bfloat16",
    }
    if config.model == "snowball":
        flags.update(
            {
                "--data-parallel-size": "4",
                "--data-parallel-size-local": "4",
                "--data-parallel-backend": "mp",
                "--distributed-executor-backend": "mp",
                "--all2all-backend": "allgather_reducescatter",
                "--gpu-memory-utilization": "0.9",
                "--kv-cache-dtype": "auto",
                "--model-loader-extra-config": '{"concurrency":4,"distributed":false,"memory_limit":8589934592}',
                "--max-num-seqs": str(_engine.max_num_seqs),
            }
        )
        if command.count("--enable-expert-parallel") != 1:
            raise ValueError("Native Snowball command does not prove expert parallelism")
    if command.count("serve") != 1 or command[command.index("serve") + 1] != model.weights:
        raise ValueError("Native serving command uses a different model artifact")
    for flag, value in flags.items():
        if (
            command.count(flag) != 1
            or command.index(flag) + 1 >= len(command)
            or command[command.index(flag) + 1] != value
        ):
            raise ValueError(f"Native engine command does not prove {flag}")
    return {
        "producer_source_commit": config.source_commit,
        "contract_response_rendering": INCREMENTAL_RENDERING,
        "renderer": renderer_provenance(),
        "producer_renderer": generation.get("renderer", {"method": "decode_skip_special_tokens"}),
        "model_label": config.model,
        "checkpoint": generation["model_identity"],
        "engine_global_seed": 17,
        "request_sampling_seed": None,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_prompt_tokens": profile["max_prompt_tokens"],
        "max_response_tokens": profile["max_response_tokens"],
        "samples": config.samples,
        "tokenizer_sha256": profile["tokenizer_sha256"],
        "prompt_template_id": profile["prompt_template_id"],
        "run_id": generation["task_id"],
        "attempt_id": 0,
        "request_fingerprint": generation["specification_sha256"],
        "step": 0,
        "generated_at_utc": datetime.fromtimestamp(finish / 1000, UTC).isoformat(),
        "dump_uri": config.output_uri + "/dumped_evals/global_step_0_evals",
        "runtime": runtime,
        "execution_cluster": controller["cluster"],
        "gpu_variant": gpu["variant"],
        "gpu_count": gpu["count"],
        "generation_and_native_scoring": timing,
        "task_gpu_hours": (finish - start) * gpu["count"] / 3_600_000,
    }


def audit_serving_outputs(config, *, native_tasks, native_job, output_uri):
    """Read back raw artifacts, validate all requests and score frozen membership."""
    root = StoragePath(config.output_uri)
    generation = json.loads((root / "generation.json").read_bytes())
    protocol = validate_serving_generation(config, generation, native_tasks, native_job)
    items = load_rating_inputs(config)
    tokenizer_bytes = (StoragePath(SERVING_MODELS[config.model]["weights"]) / "tokenizer.json").read_bytes()
    if hashlib.sha256(tokenizer_bytes).hexdigest() != protocol["tokenizer_sha256"]:
        raise ValueError("Serving audit tokenizer changed")
    decoder = Tokenizer.from_str(tokenizer_bytes.decode())
    requests = [
        completion_request(item, decoder, model=config.model, samples=config.samples, api_model=f"rating-{config.model}")
        for item in items
    ]
    if (
        audit.canonical_sha([audit.canonical_sha(request) for request in requests])
        != generation["ordered_request_sha256"]
    ):
        raise ValueError("Serving request protocol differs from frozen question tokens")
    raw_bytes = (root / "dumped_evals/global_step_0_evals/rows.jsonl").read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != generation["raw_rows_sha256"]:
        raise ValueError("Serving response artifact changed after generation")
    raw_rows = [json.loads(line) for line in raw_bytes.splitlines()]
    if len(raw_rows) != generation["rows"]:
        raise ValueError("Serving response row count changed")
    for ordinal, row in enumerate(raw_rows):
        request = requests[ordinal // config.samples]
        if row["row_ordinal"] != ordinal or row["generation_request_sha256"] != audit.canonical_sha(request):
            raise ValueError("Serving response has a different request identity or order")
        text = render_serving_response(decoder, row["prompt_token_ids"], row["response_ids"])
        if hashlib.sha256(text.encode()).hexdigest() != row["engine_output_text_sha256"]:
            raise ValueError("Serving response differs from native verifier text")
        if "engine_output_text" in row and row["engine_output_text"] != text:
            raise ValueError("Preserved API text differs from the pinned incremental renderer")
    pool = StoragePath(config.pool_uri)
    receipt = build_records(
        config.output_uri,
        0,
        manifest=pq.read_table(pa.BufferReader((pool / "manifest.parquet").read_bytes())).to_pylist(),
        selection=json.loads((pool / "selection.json").read_bytes()),
        overlay=json.loads(StoragePath(config.overlay_uri).read_bytes()),
        expected_ids=list(config.expected_ids),
        template=MODEL_TEMPLATES[config.model],
        decoder=decoder,
        model=config.model,
        tokenizer_sha256=protocol["tokenizer_sha256"],
        samples=config.samples,
        wandb_metrics=generation["metrics"],
        output_uri=output_uri,
        metric_reference="serving_score_receipt",
    )
    result = {
        "schema": "math_eval_serving_audit_v1",
        "inference_evidence_pass": True,
        "clean_end_to_end": True,
        "generation_sha256": audit.canonical_sha(generation),
        "native_tasks_sha256": audit.canonical_sha(native_tasks),
        "native_job_sha256": audit.canonical_sha(native_job),
        "protocol": protocol,
        "eval_dump": receipt["audit"],
        "expected_ids_sha256": receipt["expected_ids_sha256"],
        "records_sha256": receipt["records_sha256"],
    }
    (StoragePath(output_uri) / "generation-audit.json").write_text(json.dumps(result, sort_keys=True))
    return result


def protocol_from_serving_audit(receipt, generation_audit):
    """Bind the rating reducer to the same already-audited response parquet."""
    if (
        not generation_audit.get("inference_evidence_pass")
        or not generation_audit.get("clean_end_to_end")
        or receipt.get("metric_reference") != "serving_score_receipt"
        or any(receipt["audit"][key] != generation_audit["eval_dump"][key] for key in DUMP_IDENTITY_KEYS)
        or any(receipt[key] != generation_audit[key] for key in ("expected_ids_sha256", "records_sha256"))
        or any(receipt[key] != generation_audit["protocol"][key] for key in ("tokenizer_sha256", "prompt_template_id"))
    ):
        raise ValueError("Serving generation audit does not certify these rating records")
    return generation_audit["protocol"]
