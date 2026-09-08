# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Qualify checkpoint calibration panels without counting their shared task twice."""

import hashlib
import importlib.metadata
import json
import math
from contextlib import contextmanager
from dataclasses import asdict

from rigging.filesystem.storage_path import StoragePath
from tokenizers import Tokenizer

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.calibration_protocol import (
    BATTERIES,
    bounded_bytes,
    calibration_panels,
    calibration_protocol_receipt,
    calibration_request,
    calibration_rows,
    checkpoint_serving_configuration,
    load_calibration_inputs,
)
from experiments.post_training.math_eval.harness import build_records
from experiments.post_training.math_eval.pool import MODEL_TEMPLATES
from experiments.post_training.math_eval.rate import MODEL_PROFILES
from experiments.post_training.math_eval.rendering import renderer_provenance
from experiments.post_training.math_eval.scoring import SEMANTIC_DEPENDENCIES
from experiments.post_training.math_eval.semantic_worker import SemanticWorker
from experiments.post_training.math_eval.serving import verify_wheel_receipt
from experiments.post_training.math_eval.serving_audit import validate_native_serving_task

CALIBRATION_MAX_EVAL_BYTES = 512 * 1024**2


@contextmanager
def calibration_evaluation_bound():
    """Scope a finite larger object bound to this serial CPU calibration audit.

    K8 heldout has 16,072 rows, each with up to 1,024 prompt and response
    tokens plus forensic text. 512 MiB allows about 33 KiB per row; it is
    a finite implementation budget, not a guarantee for arbitrary output.
    The 1 MiB line limit and all token/scorer/membership checks are unchanged.
    """
    previous = audit.MAX_EVAL_BYTES
    try:
        audit.MAX_EVAL_BYTES = CALIBRATION_MAX_EVAL_BYTES
        yield
    finally:
        audit.MAX_EVAL_BYTES = previous


def validate_calibration_generation(generation, native_tasks, native_job, *, binding_sha256, source_commit, output_uri):
    """Require the independently frozen checkpoint, source, protocols and native allocation."""
    model, engine = checkpoint_serving_configuration(generation["binding"], expected_binding_sha256=binding_sha256)
    expected = {
        "source_commit": source_commit,
        "binding_sha256": binding_sha256,
        "model": asdict(model),
        "engine": asdict(engine) | {"extra_metric_families": sorted(engine.extra_metric_families)},
        "panels": [calibration_protocol_receipt(panel) for panel in calibration_panels()],
        "battery_ids_sha256": {split: frozen["ids_sha256"] for split, frozen in BATTERIES.items()},
    }
    # Dataclass tuples become arrays in the actual JSON wire representation.
    expected = json.loads(json.dumps(expected))
    if (
        generation.get("schema") != "math_eval_checkpoint_calibration_generation_v1"
        or generation.get("specification") != expected
        or generation.get("specification_sha256") != audit.canonical_sha(expected)
        or generation.get("responses") != 29406
        or generation.get("score_dependency_versions") != SEMANTIC_DEPENDENCIES | {"reasoning-gym": "0.1.25"}
        or generation.get("renderer") != renderer_provenance()
    ):
        raise ValueError("Calibration source, checkpoint, renderer, or frozen protocol differs")
    proof = validate_native_serving_task(generation, native_tasks, native_job, allocated_gpus=1)
    runtime = generation["runtime"]
    verify_wheel_receipt("MARIN_VLLM_WHEEL_VERIFIED=" + json.dumps(runtime), runtime["api_version"])
    command = generation.get("native_command", [])
    required = {
        "--seed": "17",
        "--generation-config": "vllm",
        "--max-model-len": "2048",
        "--tensor-parallel-size": "1",
        "--served-model-name": model.model_id,
        "--dtype": "bfloat16",
        "--max-num-seqs": str(engine.max_num_seqs),
    }
    if command.count("serve") != 1 or command[command.index("serve") + 1] != model.weights:
        raise ValueError("Calibration native command uses another checkpoint")
    for flag, value in required.items():
        if (
            command.count(flag) != 1
            or command.index(flag) + 1 >= len(command)
            or command[command.index(flag) + 1] != value
        ):
            raise ValueError(f"Calibration native command differs at {flag}")
    panels = generation.get("panels", [])
    if len(panels) != 9:
        raise ValueError("Calibration lacks its nine separate panels")
    previous_end = proof["start"]
    for ordinal, (panel, protocol) in enumerate(zip(panels, calibration_panels(), strict=True)):
        expected_fields = {
            "schema": "math_eval_calibration_panel_v1",
            "ordinal": ordinal,
            "protocol": calibration_protocol_receipt(protocol),
            "rows": BATTERIES[protocol.split]["rows"] * protocol.samples,
            "output_uri": output_uri.rstrip("/") + "/" + protocol.identity,
            "binding_sha256": binding_sha256,
            "specification_sha256": generation["specification_sha256"],
            "task_id": generation["task_id"],
            "attempt_uid": generation["attempt_uid"],
            "shared_server_state": True,
            "independent_engine_restart": False,
        }
        if any(panel.get(key) != value for key, value in expected_fields.items()):
            raise ValueError("Calibration panel identity, order or membership changed")
        timing = panel.get("timing", {})
        start, finish, elapsed = (timing.get(key) for key in ("started_at_ms", "finished_at_ms", "monotonic_seconds"))
        if (
            type(start) is not int
            or type(finish) is not int
            or not previous_end <= start < finish <= proof["finish"]
            or not isinstance(elapsed, (float, int))
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or abs(elapsed - (finish - start) / 1000) > 1
            or timing.get("http_executor_drained") is not True
            or timing.get("requests_returned") != BATTERIES[protocol.split]["rows"]
        ):
            raise ValueError("Calibration panels overlap, remain undrained, or lack valid timing")
        previous_end = finish
    return {
        "schema": "math_eval_checkpoint_calibration_native_audit_v1",
        "generation_sha256": audit.canonical_sha(generation),
        "native_tasks_sha256": audit.canonical_sha(native_tasks),
        "native_job_sha256": audit.canonical_sha(native_job),
        "binding_sha256": binding_sha256,
        "source_commit": source_commit,
        "task_gpu_hours": (proof["finish"] - proof["start"]) / 3_600_000,
        "gpu_count": 1,
        "execution_cluster": "cw-us-east-02a",
        "allocation_scope": "one task for all nine panels; do not sum its cost per panel",
        "native_evidence_pass": True,
        "requires_byte_token_and_harness_audit": True,
    }


def validate_panel_rows(raw, panel, requests, decoder, *, protocol, items, seen_response_ids):
    """Replay every original native token, scorer, request and response identity."""

    if hashlib.sha256(raw).hexdigest() != panel["raw_rows_sha256"]:
        raise ValueError("Calibration row artifact bytes changed")
    if (
        audit.canonical_sha([audit.canonical_sha(request) for request in requests])
        != panel["timing"]["ordered_request_sha256"]
    ):
        raise ValueError("Calibration request sequence changed")
    lines = raw.splitlines()
    if any(len(line) + 1 > audit.MAX_EVAL_LINE_BYTES for line in lines):
        raise ValueError("Calibration row exceeds the frozen line byte bound")
    rows = [json.loads(line) for line in lines]
    if len(rows) != len(items) * protocol.samples or len(rows) != panel["rows"]:
        raise ValueError("Calibration row coverage differs")
    for index, (item, request) in enumerate(zip(items, requests, strict=True)):
        group = rows[index * protocol.samples : (index + 1) * protocol.samples]
        response_id = group[0]["generation_response_id"]
        if response_id in seen_response_ids or any(row["generation_response_id"] != response_id for row in group):
            raise ValueError("Calibration native response identity overlaps or changed within a question")
        response = {
            "id": response_id,
            "model": request["model"],
            "choices": [
                {
                    "index": offset,
                    "text": row["engine_output_text"],
                    "token_ids": row["response_ids"],
                    "prompt_token_ids": row["prompt_token_ids"],
                    "finish_reason": row["stop_reason"],
                    "stop_reason": row["engine_stop_reason"],
                }
                for offset, row in enumerate(group)
            ],
            "usage": {
                "prompt_tokens": len(request["prompt"]),
                "completion_tokens": sum(len(row["response_ids"]) for row in group),
                "total_tokens": len(request["prompt"]) + sum(len(row["response_ids"]) for row in group),
            },
        }
        rescored = calibration_rows(item, request, response, decoder, protocol=protocol, question_index=index)
        if rescored != group or any(
            type(row["native_contract_correct"]) is not bool or type(row["row_ordinal"]) is not int for row in group
        ):
            raise ValueError("Calibration native record fields differ from exact request/token/scorer replay")
        seen_response_ids.add(response_id)
    return rows


def audit_checkpoint_calibration(output_uri, *, audit_uri, native_tasks, native_job, binding_sha256, source_commit):
    """Read back each panel and score it with the unchanged bounded semantic worker.

    Native correctness and completed correctness must match exactly. Advisory
    semantic timeouts remain unresolved; no implicit zero or cross-panel pooling.
    """

    if StoragePath(audit_uri).exists():
        raise ValueError("Refusing to overwrite checkpoint calibration audit evidence")
    versions = {name: importlib.metadata.version(name) for name in (*SEMANTIC_DEPENDENCIES, "reasoning-gym")}
    if versions != SEMANTIC_DEPENDENCIES | {"reasoning-gym": "0.1.25"}:
        raise ValueError("Calibration audit dependencies differ from the frozen scorers")
    generation = json.loads(bounded_bytes(output_uri + "/generation.json"))
    result = validate_calibration_generation(
        generation,
        native_tasks,
        native_job,
        binding_sha256=binding_sha256,
        source_commit=source_commit,
        output_uri=output_uri,
    )
    tokenizer_bytes = bounded_bytes(generation["binding"]["model_uri"] + "/tokenizer.json")
    tokenizer_sha = hashlib.sha256(tokenizer_bytes).hexdigest()
    if tokenizer_sha != MODEL_PROFILES["qwen"]["tokenizer_sha256"]:
        raise ValueError("Calibration audit tokenizer changed")
    decoder = Tokenizer.from_str(tokenizer_bytes.decode())
    manifest, selection, overlay, batteries = load_calibration_inputs()
    seen_response_ids, panel_audits = set(), []
    for panel, protocol in zip(generation["panels"], calibration_panels(), strict=True):
        panel_uri = panel["output_uri"]
        if json.loads(bounded_bytes(panel_uri + "/generation.json")) != panel:
            raise ValueError("Calibration panel receipt differs from overall generation")
        items = batteries[protocol.split]
        requests = [
            calibration_request(
                item, decoder, protocol=protocol, api_model=generation["specification"]["model"]["api_model"]
            )
            for item in items
        ]
        raw = bounded_bytes(panel_uri + "/dumped_evals/global_step_0_evals/rows.jsonl", limit=CALIBRATION_MAX_EVAL_BYTES)
        validate_panel_rows(
            raw, panel, requests, decoder, protocol=protocol, items=items, seen_response_ids=seen_response_ids
        )
        target = audit_uri.rstrip("/") + "/" + protocol.identity
        # A distinct receipt collection per panel binds every semantic result to
        # that panel's records. Worker startup has its own deadline.
        with (
            calibration_evaluation_bound(),
            SemanticWorker(
                source_sha256="b23dcab6a38211631a5a3b51938842ca34b958db6df01de2a5117f763c66b6da",
                startup_timeout=60,
                row_timeout=30,
                cleanup_timeout=1,
            ) as worker,
        ):
            receipt = build_records(
                panel_uri,
                0,
                manifest=manifest,
                selection=selection,
                overlay=overlay,
                expected_ids=[item["prompt_sha256"] for item in items],
                template=MODEL_TEMPLATES["qwen"],
                decoder=decoder,
                model="qwen",
                tokenizer_sha256=tokenizer_sha,
                samples=protocol.samples,
                wandb_metrics=panel["metrics"],
                output_uri=target,
                metric_reference="serving_score_receipt",
                semantic_worker=worker,
            )
        panel_audits.append(
            {
                "protocol": calibration_protocol_receipt(protocol),
                "generation_sha256": audit.canonical_sha(panel),
                "receipt_sha256": audit.canonical_sha(receipt),
                "records_sha256": receipt["records_sha256"],
                "expected_ids_sha256": receipt["expected_ids_sha256"],
                "rows": receipt["records"],
                "semantic_execution_sha256": receipt["semantic_execution_sha256"],
                "output_uri": target,
                "contract_metric_parity_verified": receipt["contract_metric_parity_verified"],
            }
        )
    if sum(row["rows"] for row in panel_audits) != 29406 or len(seen_response_ids) != 9316:
        raise ValueError("Calibration does not cover all unique native requests and response draws")
    result.update(
        panels=panel_audits,
        rows=29406,
        unique_native_requests=len(seen_response_ids),
        byte_token_and_harness_audit_pass=True,
        byte_bounds={
            "maximum_object_bytes": CALIBRATION_MAX_EVAL_BYTES,
            "maximum_line_bytes": audit.MAX_EVAL_LINE_BYTES,
        },
        requires_byte_token_and_harness_audit=False,
        semantic_execution={
            "source_sha256": "b23dcab6a38211631a5a3b51938842ca34b958db6df01de2a5117f763c66b6da",
            "startup_timeout_seconds": 60,
            "row_timeout_seconds": 30,
            "cleanup_timeout_seconds": 1,
            "actual_dependencies": versions,
            "scope": "successful rows preserve exact scorer output; wall-time expiry is unresolved",
        },
    )
    (StoragePath(audit_uri) / "calibration-audit.json").write_text(json.dumps(result, sort_keys=True))
    return result
