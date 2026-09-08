# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from dataclasses import asdict

import pytest

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval.calibration_audit import (
    calibration_evaluation_bound,
    validate_calibration_generation,
    validate_panel_rows,
)
from experiments.post_training.math_eval.calibration_protocol import (
    BATTERIES,
    CalibrationProtocol,
    calibration_panels,
    calibration_protocol_receipt,
    calibration_request,
    checkpoint_serving_configuration,
)
from experiments.post_training.math_eval.calibration_serving import run_panel
from experiments.post_training.math_eval.rendering import renderer_provenance
from tests.rl.test_math_eval_calibration_serving import endpoint_fixture
from tests.rl.test_math_eval_serving import generation_fixture


def fixture():
    _, generation, tasks, native_job = generation_fixture()
    files = {name: {"bytes": 1, "sha256": "a" * 64} for name in ("config.json", "tokenizer.json", "model.safetensors")}
    binding = {
        "schema": "math_eval_checkpoint_content_v1",
        "global_step": 96,
        "training_seed": 17,
        "model_uri": "s3://marin-us-east-02a/marin/checkpoint/hf",
        "content": {"files": files, "total_bytes": 3, "files_sha256": audit.canonical_sha(files)},
    }
    binding["binding_sha256"] = audit.canonical_sha(binding)
    model, engine = checkpoint_serving_configuration(binding, expected_binding_sha256=binding["binding_sha256"])
    specification = {
        "source_commit": "b" * 40,
        "binding_sha256": binding["binding_sha256"],
        "model": asdict(model),
        "engine": asdict(engine) | {"extra_metric_families": sorted(engine.extra_metric_families)},
        "panels": [calibration_protocol_receipt(p) for p in calibration_panels()],
        "battery_ids_sha256": {split: v["ids_sha256"] for split, v in BATTERIES.items()},
    }
    generation.update(
        schema="math_eval_checkpoint_calibration_generation_v1",
        binding=binding,
        specification=json.loads(json.dumps(specification)),
        specification_sha256=audit.canonical_sha(specification),
        renderer=renderer_provenance(),
        responses=29406,
    )
    generation["native_command"] = [
        "vllm",
        "serve",
        model.weights,
        "--seed",
        "17",
        "--generation-config",
        "vllm",
        "--max-model-len",
        "2048",
        "--tensor-parallel-size",
        "1",
        "--served-model-name",
        model.model_id,
        "--dtype",
        "bfloat16",
        "--max-num-seqs",
        "64",
    ]
    generation["panels"] = []
    for i, p in enumerate(calibration_panels()):
        generation["panels"].append(
            {
                "schema": "math_eval_calibration_panel_v1",
                "ordinal": i,
                "protocol": calibration_protocol_receipt(p),
                "rows": BATTERIES[p.split]["rows"] * p.samples,
                "output_uri": "s3://output/" + p.identity,
                "binding_sha256": binding["binding_sha256"],
                "specification_sha256": generation["specification_sha256"],
                "task_id": generation["task_id"],
                "attempt_uid": generation["attempt_uid"],
                "shared_server_state": True,
                "independent_engine_restart": False,
                "timing": {
                    "started_at_ms": 2000 + i * 10000,
                    "finished_at_ms": 3000 + i * 10000,
                    "monotonic_seconds": 1.0,
                    "http_executor_drained": True,
                    "requests_returned": BATTERIES[p.split]["rows"],
                },
            }
        )
    return (
        generation,
        tasks,
        native_job,
        {"binding_sha256": binding["binding_sha256"], "source_commit": "b" * 40, "output_uri": "s3://output"},
    )


def test_calibration_task_cost_is_charged_once_for_nine_panels():
    generation, tasks, job, expected = fixture()
    result = validate_calibration_generation(generation, tasks, job, **expected)
    assert result["task_gpu_hours"] == 1 and result["native_evidence_pass"]
    assert result["requires_byte_token_and_harness_audit"]


@pytest.mark.parametrize(
    "poison",
    [
        "order",
        "overlap",
        "undrained",
        "count",
        "temperature",
        "checkpoint",
        "source",
        "request_seed",
        "region",
        "gpus",
        "retry",
        "attempt",
        "native_cap",
        "claimed_restart",
    ],
)
def test_calibration_cannot_certify_poisoned_panel_or_native_provenance(poison):
    generation, tasks, job, expected = fixture()
    if poison == "order":
        generation["panels"].reverse()
    elif poison == "overlap":
        generation["panels"][1]["timing"]["started_at_ms"] = 2500
    elif poison == "undrained":
        generation["panels"][0]["timing"]["http_executor_drained"] = False
    elif poison == "count":
        generation["panels"][0]["rows"] -= 1
    elif poison == "temperature":
        generation["panels"][6]["protocol"]["temperature"] = 1
    elif poison == "checkpoint":
        expected["binding_sha256"] = "c" * 64
    elif poison == "source":
        expected["source_commit"] = "c" * 40
    elif poison == "request_seed":
        generation["panels"][0]["protocol"]["request_sampling_seed"] = 17
    elif poison == "region":
        tasks["tasks"][0]["cluster"] = "cw-rno2a"
    elif poison == "gpus":
        job["job"]["resources"]["device"]["gpu"]["count"] = 4
    elif poison == "retry":
        tasks["tasks"][0]["current_attempt_id"] = 1
    elif poison == "attempt":
        generation["panels"][2]["attempt_uid"] = "another"
    elif poison == "native_cap":
        generation["native_command"][generation["native_command"].index("--max-model-len") + 1] = "3072"
    else:
        generation["panels"][0]["independent_engine_restart"] = True
    with pytest.raises(ValueError):
        validate_calibration_generation(generation, tasks, job, **expected)


@pytest.mark.parametrize("poison", [None, "reward", "correct", "text", "request", "ordinal", "repeat_id"])
def test_calibration_raw_replay_binds_every_row_field(tmp_path, monkeypatch, poison):
    decoder, items, session, _ = endpoint_fixture(monkeypatch)
    protocol = CalibrationProtocol("heldout_stochastic", 4)
    rows, timing = run_panel(
        items,
        decoder,
        protocol,
        session,
        api_model="checkpoint",
        output_uri=str(tmp_path / "panel"),
        provenance={},
        seen_response_ids=set(),
    )
    seen = set()
    if poison == "reward":
        rows[0]["score"] = -1
    elif poison == "correct":
        rows[0]["native_contract_correct"] = False
    elif poison == "text":
        rows[0]["engine_output_text"] = "Answer: 6"
    elif poison == "request":
        rows[0]["generation_request_sha256"] = "0" * 64
    elif poison == "ordinal":
        rows[0]["row_ordinal"] = 1
    elif poison == "repeat_id":
        seen.add(rows[0]["generation_response_id"])
    raw = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    panel = {"raw_rows_sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows), "timing": timing}
    requests = [calibration_request(item, decoder, protocol=protocol, api_model="checkpoint") for item in items]
    if poison is None:
        assert (
            validate_panel_rows(raw, panel, requests, decoder, protocol=protocol, items=items, seen_response_ids=seen)
            == rows
        )
        assert len(seen) == 3
    else:
        with pytest.raises(ValueError):
            validate_panel_rows(raw, panel, requests, decoder, protocol=protocol, items=items, seen_response_ids=seen)


def test_calibration_object_bound_is_finite_scoped_and_restored_on_failure():
    before = audit.MAX_EVAL_BYTES
    line_before = audit.MAX_EVAL_LINE_BYTES
    with pytest.raises(RuntimeError):
        with calibration_evaluation_bound():
            assert audit.MAX_EVAL_BYTES == 512 * 1024**2
            assert audit.MAX_EVAL_LINE_BYTES == line_before == 1024**2
            raise RuntimeError("fixture audit failed")
    assert audit.MAX_EVAL_BYTES == before
