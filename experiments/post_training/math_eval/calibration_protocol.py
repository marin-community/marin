# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Separate checkpoint calibration protocols; preserve difficulty-rating defaults."""

import hashlib
import json
from dataclasses import asdict, dataclass

import pyarrow as pa
import pyarrow.parquet as pq
from marin.external_dependencies import VLLM_GPU_RELEASE
from marin.inference.config import ServedModelConfig, VllmEngineConfig, VllmLauncherType, VllmSource

from experiments.post_training import async_rl_audit as audit
from experiments.post_training.math_eval import launcher
from experiments.post_training.math_eval.checkpoint_tokenizer import tokenizer_stage_path
from experiments.post_training.math_eval.export_binding import EAST_PREFIX, validate_inventory
from experiments.post_training.math_eval.serving_records import _completion_rows, completion_request

BATTERIES = {
    "dev": {
        "rows": 256,
        "ids_sha256": "990473158d27b223cb585c174479b1e7104cc3ac3996ca7ce4820ae396ec4b69",
        "parquet_sha256": launcher.DEV_SHA256,
    },
    "heldout": {
        "rows": 2009,
        "ids_sha256": "5e39c446ac096fc63979f858ffcb4f165570b5106fc3a1783f8735c71853f4f3",
        "parquet_sha256": "e331365b3089c6a951fc4b7825a54957cf7514369e7d4e28f061a8d69354ea30",
    },
}


@dataclass(frozen=True)
class CalibrationProtocol:
    panel: str
    samples: int
    repetition: int = 1

    def __post_init__(self):
        if self.panel not in {"dev_greedy", "heldout_greedy", "heldout_stochastic"}:
            raise ValueError("Unknown calibration panel")
        if type(self.samples) is not int or type(self.repetition) is not int:
            raise ValueError("Calibration counts must be integers")
        if self.panel == "dev_greedy":
            valid = self.samples == 1 and 1 <= self.repetition <= 5
        elif self.panel == "heldout_greedy":
            valid = self.samples == 1 and self.repetition == 1
        else:
            valid = self.samples in {1, 4, 8} and self.repetition == 1
        if not valid:
            raise ValueError("Calibration panel changed its prescribed K or repeat count")

    @property
    def split(self):
        return "dev" if self.panel == "dev_greedy" else "heldout"

    @property
    def temperature(self):
        return 0.6 if self.panel == "heldout_stochastic" else 0.0

    @property
    def top_p(self):
        return 0.95 if self.panel == "heldout_stochastic" else 1.0

    @property
    def identity(self):
        return f"{self.panel}-k{self.samples}-repeat{self.repetition}"


def calibration_request(item, decoder, *, protocol, api_model):
    request = completion_request(item, decoder, model="qwen", samples=protocol.samples, api_model=api_model)
    return request | {"temperature": protocol.temperature, "top_p": protocol.top_p, "max_tokens": 1024}


def calibration_rows(item, request, response, decoder, *, protocol, question_index):
    expected = calibration_request(item, decoder, protocol=protocol, api_model=request["model"])
    return _completion_rows(
        item, request, response, decoder, model="qwen", question_index=question_index, expected=expected
    )


def calibration_panels():
    return (
        *(CalibrationProtocol("dev_greedy", 1, index) for index in range(1, 6)),
        CalibrationProtocol("heldout_greedy", 1),
        *(CalibrationProtocol("heldout_stochastic", samples) for samples in (1, 4, 8)),
    )


def validate_calibration_battery(content, manifest, selection, overlay, *, protocol):
    frozen = BATTERIES[protocol.split]
    if hashlib.sha256(content).hexdigest() != frozen["parquet_sha256"]:
        raise ValueError("Frozen calibration battery bytes changed")
    rows = pq.read_table(pa.BufferReader(content)).to_pylist()
    ids = [row["extra_info"]["prompt_sha256"] for row in rows]
    if len(ids) != frozen["rows"] or audit.canonical_sha(sorted(ids)) != frozen["ids_sha256"]:
        raise ValueError("Frozen calibration battery membership changed")
    launcher.validate_view(rows, manifest, selection, overlay, split=protocol.split, expected_ids=ids)
    return rows


def checkpoint_serving_configuration(binding, *, expected_binding_sha256, concurrency=8):
    """Build the endpoint identity after an independently qualified content binding."""
    unsigned = {key: value for key, value in binding.items() if key != "binding_sha256"}
    if (
        binding["binding_sha256"] != expected_binding_sha256
        or audit.canonical_sha(unsigned) != expected_binding_sha256
        or binding["schema"] != "math_eval_checkpoint_content_v1"
        or binding["global_step"] != 96
        or binding["training_seed"] not in {17, 29}
        or not binding["model_uri"].startswith(EAST_PREFIX)
        or type(concurrency) is not int
        or not 1 <= concurrency <= 32
    ):
        raise ValueError("Calibration checkpoint identity or concurrency changed")
    validate_inventory(binding["content"])
    if VLLM_GPU_RELEASE.source_commit != "f0d7cc7f587482e0ab771e3c9715e726eb914e60":
        raise ValueError("Qualify the new calibration renderer before serving")
    model = ServedModelConfig(
        weights=binding["model_uri"],
        tokenizer=tokenizer_stage_path(binding["tokenizer_source"]),
        api_model="checkpoint-" + expected_binding_sha256[:16],
        dtype="bfloat16",
        max_model_len=2048,
        tensor_parallel_size=1,
    )
    engine = VllmEngineConfig(
        launcher=VllmLauncherType.CUDA,
        source=VllmSource.MARIN_FORK,
        startup_timeout_seconds=600,
        max_num_seqs=concurrency * 8,
        extra_args=("--seed", "17", "--generation-config", "vllm", "--tokenizer", model.tokenizer),
    )
    return model, engine


def calibration_protocol_receipt(protocol):
    return asdict(protocol) | {
        "temperature": protocol.temperature,
        "top_p": protocol.top_p,
        "max_prompt_tokens": 1024,
        "max_response_tokens": 1024,
        "context_tokens": 2048,
        "engine_global_seed": 17,
        "request_sampling_seed": None,
        "scope": (
            "nine sequential drained panels share one server; no independent engine-restart claim; "
            "separate greedy and stochastic estimands, no cross-panel K scaling or covariance pooling"
        ),
    }


def bounded_bytes(uri, *, limit=32 * 1024**2):
    """Refuse oversized metadata before materializing it in memory."""
    filesystem, path = audit.fs_path(str(uri))
    if filesystem.info(path)["size"] > limit:
        raise ValueError("Calibration artifact exceeds its byte bound")
    with filesystem.open(path, "rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ValueError("Calibration artifact grew past its byte bound")
    return content


def load_calibration_inputs():
    """Read and verify the fixed batteries, original pool and acceptance overlay."""
    pool = launcher.POOL_URI
    manifest = pq.read_table(pa.BufferReader(bounded_bytes(pool + "/manifest.parquet"))).to_pylist()
    selection = json.loads(bounded_bytes(pool + "/selection.json"))
    overlay = json.loads(bounded_bytes(pool + "/" + launcher.BATTERY_PATH + "/audit-overlay.json"))
    lookup = {row["prompt_sha256"]: row for row in manifest}
    batteries = {}
    for split in ("dev", "heldout"):
        protocol = next(p for p in calibration_panels() if p.split == split)
        raw = bounded_bytes(pool + "/" + launcher.BATTERY_PATH + "/qwen/" + split + ".parquet")
        rows = validate_calibration_battery(raw, manifest, selection, overlay, protocol=protocol)
        batteries[split] = [lookup[row["extra_info"]["prompt_sha256"]] for row in rows]
    return manifest, selection, overlay, batteries
