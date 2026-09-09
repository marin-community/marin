# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate one audited checkpoint with nine drained calibration panels."""

import hashlib
import importlib.metadata
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import requests
from iris.client.client import iris_ctx
from iris.cluster.client.job_info import get_job_info
from marin.inference.serve import local_inference
from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.calibration_protocol import (
    bounded_bytes,
    calibration_panels,
    calibration_protocol_receipt,
    calibration_request,
    calibration_rows,
    checkpoint_serving_configuration,
    load_calibration_inputs,
)
from experiments.post_training.math_eval.checkpoint_tokenizer import stage_tokenizer
from experiments.post_training.math_eval.export_binding import EAST_PREFIX, hash_export_files
from experiments.post_training.math_eval.pool import canonical_json
from experiments.post_training.math_eval.pool_audit import verify_verifier_sources
from experiments.post_training.math_eval.rate import MODEL_PROFILES
from experiments.post_training.math_eval.rendering import renderer_provenance
from experiments.post_training.math_eval.scoring import SEMANTIC_DEPENDENCIES
from experiments.post_training.math_eval.serving import MAX_FAILURE_RECEIPT_BYTES, verify_wheel_receipt
from experiments.post_training.math_eval.serving_records import serving_metrics


def run_panel(items, decoder, protocol, session, *, api_model, output_uri, provenance, seen_response_ids):
    """Return only after all HTTP requests and native scoring have drained."""
    output = StoragePath(output_uri)
    if output.exists():
        raise ValueError("Refusing to overwrite a calibration panel")
    requests_by_question = [calibration_request(item, decoder, protocol=protocol, api_model=api_model) for item in items]
    session.check_alive()
    started_ms, started_ns = time.time_ns() // 1_000_000, time.monotonic_ns()

    def generate(index):
        session.check_alive()
        response = requests.post(
            session.model.endpoint.base_url + "/completions", json=requests_by_question[index], timeout=600
        )
        response.raise_for_status()
        return response.json()

    rows = []
    # The context manager joins every worker before any next panel can begin.
    with ThreadPoolExecutor(max_workers=8) as executor:
        for index, response in enumerate(executor.map(generate, range(len(items)))):
            try:
                response_id = response.get("id")
                if response_id in seen_response_ids:
                    raise ValueError("Calibration endpoint reused a response identity")
                scored = calibration_rows(
                    items[index], requests_by_question[index], response, decoder, protocol=protocol, question_index=index
                )
                seen_response_ids.add(response_id)
                rows.extend(scored)
            except ValueError as error:
                failure = {
                    "schema": "math_eval_calibration_failure_v1",
                    "valid": False,
                    "protocol": calibration_protocol_receipt(protocol),
                    "item": items[index],
                    "request": requests_by_question[index],
                    "response": response,
                    "provenance": provenance,
                    "error": str(error),
                }
                payload = canonical_json(failure).encode()
                if len(payload) > MAX_FAILURE_RECEIPT_BYTES:
                    raise RuntimeError("Calibration failure receipt exceeds its8MiB bound") from error
                target = output / "validation-failure.json"
                if target.exists():
                    raise RuntimeError("Refusing to overwrite calibration failure evidence") from error
                target.write_bytes(payload)
                raise
    session.check_alive()
    finished_ms = time.time_ns() // 1_000_000
    elapsed = (time.monotonic_ns() - started_ns) / 1_000_000_000
    if len(rows) != len(items) * protocol.samples:
        raise ValueError("Calibration panel sample coverage is incomplete")
    return rows, {
        "started_at_ms": started_ms,
        "finished_at_ms": finished_ms,
        "monotonic_seconds": elapsed,
        "requests_returned": len(items),
        "http_executor_drained": True,
        "ordered_request_sha256": audit.canonical_sha(
            [audit.canonical_sha(request) for request in requests_by_question]
        ),
    }


def _controller_allocation():
    job = get_job_info()
    if job is None or job.attempt_id != 0:
        raise ValueError("Calibration requires one first-attempt Iris task")
    description = iris_ctx().client.describe_task(job.task_id)
    status, resources = description.status, description.resources
    device = resources.device
    if (
        status.execution_cluster_id not in {"local", "cw-us-east-02a"}
        or str(status.task_id) != str(job.task_id)
        or status.current_attempt_number != 0
        or len(status.attempts) != 1
        or device is None
        or device.kind != "gpu"
        or device.variant != "H100"
        or device.count != 1
    ):
        raise ValueError("Calibration allocation differs from one reviewed H100")
    return job, {
        "controller_scope": status.execution_cluster_id,
        "resources": asdict(resources),
        "attempt_uid": status.attempts[0].attempt_uid,
        "started_at_ms": None if status.attempts[0].started_at is None else status.attempts[0].started_at.epoch_ms(),
    }


def run_checkpoint_calibration(*, binding_uri, binding_sha256, output_uri, source_commit):
    """Keep model startup and all panels within one externally bounded H100 task.

    This produces evidence only. A separate controller/byte/token/harness audit
    must qualify the attempt; the nine panels never count as nine allocations.
    """
    if any(not uri.startswith(EAST_PREFIX) for uri in (binding_uri, output_uri)):
        raise ValueError("Calibration artifacts must remain east")
    if not re.fullmatch(r"[a-f0-9]{40}", source_commit):
        raise ValueError("Calibration source commit must be explicit")
    output = StoragePath(output_uri)
    if output.exists():
        raise ValueError("Refusing to overwrite a calibration attempt")
    job, allocation = _controller_allocation()
    verify_verifier_sources()
    versions = {name: importlib.metadata.version(name) for name in (*SEMANTIC_DEPENDENCIES, "reasoning-gym")}
    if versions != SEMANTIC_DEPENDENCIES | {"reasoning-gym": "0.1.25"}:
        raise ValueError("Calibration native scorer dependencies changed")
    binding = json.loads(bounded_bytes(binding_uri))
    model, engine = checkpoint_serving_configuration(binding, expected_binding_sha256=binding_sha256)
    actual_inventory = hash_export_files(model.weights)
    if actual_inventory != binding["content"]:
        raise ValueError("Actual checkpoint bytes differ from the audited export")
    model_config = json.loads(bounded_bytes(model.weights + "/config.json"))
    if (
        model_config.get("architectures") != ["Qwen3ForCausalLM"]
        or model_config.get("max_position_embeddings", 0) < 2048
    ):
        raise ValueError("Calibration export does not support the frozen Qwen model/context")
    staged_tokenizer = stage_tokenizer(binding["tokenizer_source"])
    if staged_tokenizer != model.tokenizer:
        raise ValueError("Calibration native tokenizer path differs from its staged byte inventory")
    tokenizer_bytes = bounded_bytes(staged_tokenizer + "/tokenizer.json")
    if hashlib.sha256(tokenizer_bytes).hexdigest() != MODEL_PROFILES["qwen"]["tokenizer_sha256"]:
        raise ValueError("Calibration tokenizer differs from frozen Qwen")
    decoder = Tokenizer.from_str(tokenizer_bytes.decode())
    _manifest, _selection, _overlay, batteries = load_calibration_inputs()
    panels = calibration_panels()
    rendering = renderer_provenance()
    specification = {
        "source_commit": source_commit,
        "binding_sha256": binding_sha256,
        "model": asdict(model),
        "engine": asdict(engine) | {"extra_metric_families": sorted(engine.extra_metric_families)},
        "panels": [calibration_protocol_receipt(panel) for panel in panels],
        "battery_ids_sha256": {
            split: audit.canonical_sha(sorted(row["prompt_sha256"] for row in items))
            for split, items in batteries.items()
        },
    }
    panel_receipts, response_ids = [], set()
    with local_inference(model, engine, num_chips=1) as session:
        endpoint = session.model.endpoint.base_url
        version_reply = requests.get(endpoint.removesuffix("/v1") + "/version", timeout=30)
        version_reply.raise_for_status()
        runtime = verify_wheel_receipt(session._served.environment.logs(), version_reply.json()["version"])
        native_command = list(session._served.environment._command)
        provenance = {
            "specification": specification,
            "runtime": runtime,
            "native_command": native_command,
            "controller_allocation": allocation,
            "renderer": rendering,
        }
        for ordinal, panel in enumerate(panels):
            panel_uri = str(output / panel.identity)
            rows, timing = run_panel(
                batteries[panel.split],
                decoder,
                panel,
                session,
                api_model=model.model_id,
                output_uri=panel_uri,
                provenance=provenance,
                seen_response_ids=response_ids,
            )
            if panel_receipts and timing["started_at_ms"] < panel_receipts[-1]["timing"]["finished_at_ms"]:
                raise ValueError("Calibration protocol panels overlapped")
            metrics = serving_metrics(rows, samples=panel.samples)
            dump = StoragePath(panel_uri) / "dumped_evals" / "global_step_0_evals"
            dump.mkdirs(exist_ok=False)
            raw = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
            (dump / "rows.jsonl").write_bytes(raw)
            (dump / "aggregated_results.jsonl").write_text(json.dumps(metrics, sort_keys=True))
            receipt = {
                "schema": "math_eval_calibration_panel_v1",
                "ordinal": ordinal,
                "protocol": calibration_protocol_receipt(panel),
                "timing": timing,
                "rows": len(rows),
                "raw_rows_sha256": hashlib.sha256(raw).hexdigest(),
                "metrics": metrics,
                "output_uri": panel_uri,
                "binding_sha256": binding_sha256,
                "specification_sha256": audit.canonical_sha(specification),
                "task_id": str(job.task_id),
                "attempt_uid": allocation["attempt_uid"],
                "shared_server_state": True,
                "independent_engine_restart": False,
            }
            (StoragePath(panel_uri) / "generation.json").write_text(canonical_json(receipt))
            panel_receipts.append(receipt)
            print(
                "CALIBRATION_PANEL_DRAINED "
                + json.dumps(
                    {
                        "ordinal": ordinal,
                        "panel": panel.identity,
                        "rows": len(rows),
                        "receipt_sha256": audit.canonical_sha(receipt),
                    }
                ),
                flush=True,
            )
        session.check_alive()
    result = {
        "schema": "math_eval_checkpoint_calibration_generation_v1",
        "specification": specification,
        "specification_sha256": audit.canonical_sha(specification),
        "binding": binding,
        "runtime": runtime,
        "renderer": rendering,
        "native_command": native_command,
        "score_dependency_versions": versions,
        "task_id": str(job.task_id),
        "attempt_id": job.attempt_id,
        "attempt_uid": allocation["attempt_uid"],
        "controller_allocation": allocation,
        "worker_region_hint": job.worker_region,
        "bundle_id": job.bundle_id,
        "panels": panel_receipts,
        "responses": sum(panel["rows"] for panel in panel_receipts),
        "requires_independent_terminal_and_harness_audit": True,
        "allocation_scope": "one task shared by nine drained panels; charge allocation once",
    }
    if result["responses"] != 29406:
        raise ValueError("Calibration does not cover all prescribed responses")
    (output / "generation.json").write_text(canonical_json(result))
    print(
        "CHECKPOINT_CALIBRATION_GENERATION_COMPLETE "
        + json.dumps({"rows": result["responses"], "receipt_sha256": audit.canonical_sha(result)}),
        flush=True,
    )
    return result
