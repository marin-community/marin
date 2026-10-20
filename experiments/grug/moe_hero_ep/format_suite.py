# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Evaluate several pinned Hero panels after one checkpoint restore in an existing GPU job."""

import argparse
import hashlib
import json
import logging
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

import draccus
import jax
import jax.numpy as jnp
from jax.experimental.multihost_utils import sync_global_devices
from levanter.cutlass_kernel_cache import cutlass_kernel_cache, install
from levanter.distributed import DistributedConfig
from levanter.grug.sharding import compact_grug_mesh
from levanter.tokenizers import load_tokenizer
from pydantic import Field, model_validator
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from rigging.timing import log_time

from experiments.grug.moe_hero_ep.format_eval import EOT_ID, Panel, Serialization, render_panel, score_content_panel
from experiments.grug.moe_hero_ep.format_measure import NativePtRequest, measure_pt_model, read_pt_panel
from experiments.grug.moe_hero_ep.format_sample import (
    NATIVE_BACKEND,
    NativePanelRequest,
    native_evaluation_model,
    native_logits,
    sample_panel,
)
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint, Record, digest
from experiments.grug.moe_hero_ep.weights import cast_weights, restore_weights

logger = logging.getLogger(__name__)


class PanelPurpose(StrEnum):
    FORMAT = "format"
    CONTENT = "content"


class SuitePanel(Record):
    panel: Panel
    serialization: Serialization
    purpose: PanelPurpose

    @property
    def result_key(self) -> str:
        return f"panels/{self.panel.panel_id}/{self.serialization}"


class NativeSuiteRequest(Record):
    checkpoint: Checkpoint
    model: dict
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    tokenizer: str = Field(min_length=1)
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    target_cluster: str = Field(min_length=1)
    expert_axis_size: int = Field(ge=1, le=1)
    replica_axis_size: int = Field(ge=1)
    panels: tuple[SuitePanel, ...] = Field(min_length=1)
    pt_repetitions: int = Field(ge=1, le=10)
    pt_panel_path: str | None = None
    pt_panel_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def common_identity(self) -> Self:
        if (self.pt_panel_path is None) != (self.pt_panel_sha256 is None):
            raise ValueError("PT measurement needs both its panel path and exact raw SHA256")
        if len({entry.result_key for entry in self.panels}) != len(self.panels):
            raise ValueError("A panel/serialization pair may occur only once; content also retains format grading")
        for entry in self.panels:
            panel = entry.panel
            if (
                panel.backend != NATIVE_BACKEND
                or panel.source_revision != self.source_revision
                or (panel.tokenizer, panel.tokenizer_revision) != (self.tokenizer, self.tokenizer_revision)
            ):
                raise ValueError("Every suite panel must share the pinned native backend, source and tokenizer")
            if entry.purpose == PanelPurpose.CONTENT and any(not case.expected for case in panel.cases):
                raise ValueError("Content grading requires an explicitly pinned reference for every case")
        return self

    @property
    def suite_id(self) -> str:
        return digest(self.model_dump(mode="json"))


def _write_json(path: StoragePath, value: dict) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    conditional_object(str(path)).write(raw, expected_version=None)
    return hashlib.sha256(raw).hexdigest()


def run_suite(request: NativeSuiteRequest, destination: str) -> dict:
    """Validate every panel before one restore; retain raw samples and final suite identity."""
    request = NativeSuiteRequest.model_validate(request.model_dump())
    logger.info(
        "Validate native suite %s: %d panels, PT repetitions=%d",
        request.suite_id,
        len(request.panels),
        request.pt_repetitions,
    )
    config = draccus.decode(GrugModelConfig, request.model)
    if config != native_evaluation_model(config):
        raise ValueError("Use native_evaluation_model to select dropless evaluation settings")
    root = StoragePath(destination)
    if root.exists():
        raise FileExistsError("Choose a new immutable suite destination")
    tokenizer = load_tokenizer(f"{request.tokenizer}@{request.tokenizer_revision}")
    if len(tokenizer) > config.vocab_size:
        raise ValueError("Pinned tokenizer vocabulary exceeds the actual model vocabulary")
    for entry in request.panels:
        if entry.panel.context_length > config.max_seq_len:
            raise ValueError("A suite panel exceeds the pinned model context")
        rendered = render_panel(entry.panel, tokenizer, entry.serialization)
        stop = tokenizer.eos_token_id if entry.serialization == Serialization.LEGACY else EOT_ID
        if stop is None or not 0 <= stop < config.vocab_size:
            raise ValueError("A suite stop token is outside the pinned model vocabulary")
        texts = [prompt.text for prompt in rendered.prompts]
        texts.extend(case.expected for case in entry.panel.cases if case.expected is not None)
        if any(
            token < 0 or token >= config.vocab_size
            for text in texts
            for token in tokenizer.encode(text, add_special_tokens=False)
        ):
            raise ValueError("A rendered suite prompt/reference exceeds the pinned model vocabulary")
    pt_request = None
    pt_panel = None
    if request.pt_panel_path is not None:
        assert request.pt_panel_sha256 is not None
        pt_request = NativePtRequest(
            panel_path=request.pt_panel_path,
            panel_sha256=request.pt_panel_sha256,
            checkpoint=request.checkpoint,
            model=draccus.encode(config),
            tokenizer=request.tokenizer,
            tokenizer_revision=request.tokenizer_revision,
            source_revision=request.source_revision,
            expert_axis_size=request.expert_axis_size,
            replica_axis_size=request.replica_axis_size,
        )
        pt_panel = read_pt_panel(pt_request)
        if pt_panel.sequence_length > config.max_seq_len or any(
            max(row.tokens) >= config.vocab_size for row in pt_panel.rows
        ):
            raise ValueError("PT panel exceeds the pinned model context or vocabulary")
    if jax.process_index() == 0:
        _write_json(root / "request.json", request.model_dump(mode="json"))
    sync_global_devices("hero-format-suite-reserved")
    mesh = compact_grug_mesh(expert_axis_size=request.expert_axis_size, replica_axis_size=request.replica_axis_size)
    actual_model = draccus.encode(config)
    panel_outputs = []
    pt_output = None
    with jax.set_mesh(mesh):
        logger.info(
            "Restore checkpoint step=%d from %s on mesh %s", request.checkpoint.step, request.checkpoint.uri, mesh
        )
        with log_time("Suite checkpoint restore and BF16 weight conversion"):
            model = cast_weights(
                restore_weights(request.checkpoint.uri, request.checkpoint.metadata_digest, config, mesh), jnp.bfloat16
            )
            jax.block_until_ready(model)
        if pt_panel is not None:
            assert pt_request is not None
            repeat_measurements = []
            for index in range(request.pt_repetitions):
                logger.info("Start PT repetition %d/%d: %d rows", index + 1, request.pt_repetitions, len(pt_panel.rows))
                with log_time(f"PT repetition {index + 1}/{request.pt_repetitions}"):
                    measurement = measure_pt_model(model, pt_panel, mesh)
                repeat_measurements.append(measurement)
                logger.info(
                    "Completed PT repetition %d/%d: loss=%.8f", index + 1, request.pt_repetitions, measurement["loss"]
                )
            measured = repeat_measurements[0]
            losses = [float(repeat["loss"]) for repeat in repeat_measurements]
            pt_report = {
                "request": pt_request.model_dump(),
                "panel_digest": digest(pt_panel.model_dump()),
                "model_digest": digest(actual_model),
                "backend": NATIVE_BACKEND,
                "compute_dtype": "bfloat16",
                "routing_control_dtype": "float32",
                **measured,
                "repeat_measurements": repeat_measurements,
                "variability": {
                    "repetitions": request.pt_repetitions,
                    "primary_loss": losses[0],
                    "losses": losses,
                    "loss_min": min(losses),
                    "loss_max": max(losses),
                    "loss_range": max(losses) - min(losses),
                },
            }
            pt_report["report_digest"] = digest(pt_report)
            pt_output = {
                "panel_sha256": request.pt_panel_sha256,
                "panel_digest": pt_report["panel_digest"],
                "report_digest": pt_report["report_digest"],
                "result_key": "pt/report.json",
            }
            if jax.process_index() == 0:
                pt_output["file_sha256"] = _write_json(root / "pt/report.json", pt_report)
                logger.info(
                    "Saved PT measurements: primary loss=%.8f, observed loss range=%.8f",
                    measured["loss"],
                    max(losses) - min(losses),
                )
        logits = native_logits(model, mesh)
        for index, entry in enumerate(request.panels):
            logger.info(
                "Start panel %d/%d: %s, purpose=%s, serialization=%s, cases=%d, context=%d",
                index + 1,
                len(request.panels),
                entry.panel.panel_id,
                entry.purpose,
                entry.serialization,
                len(entry.panel.cases),
                entry.panel.context_length,
            )
            panel_request = NativePanelRequest(
                panel=entry.panel,
                checkpoint=request.checkpoint,
                model=actual_model,
                serialization=entry.serialization,
                target_cluster=request.target_cluster,
                expert_axis_size=request.expert_axis_size,
                replica_axis_size=request.replica_axis_size,
            )
            with log_time(f"Panel {index + 1}/{len(request.panels)} generation"):
                samples, report = sample_panel(panel_request, tokenizer, logits, batch_size=jax.device_count())
            content = (
                score_content_panel(entry.panel, samples, tokenizer, entry.serialization)
                if entry.purpose == PanelPurpose.CONTENT
                else None
            )
            files = {}
            if jax.process_index() == 0:
                for name, value in (
                    ("panel.json", entry.panel.model_dump(mode="json")),
                    ("request.json", panel_request.model_dump(mode="json")),
                    ("samples.json", samples.model_dump(mode="json")),
                    ("report.json", report),
                ):
                    files[name] = _write_json(root / entry.result_key / name, value)
                if content is not None:
                    files["content.json"] = _write_json(root / entry.result_key / "content.json", content)
            logger.info("Completed panel %d/%d: sample=%s", index + 1, len(request.panels), samples.request.sample_id)
            panel_outputs.append(
                {
                    "panel_id": entry.panel.panel_id,
                    "serialization": entry.serialization.value,
                    "purpose": entry.purpose.value,
                    "content_panel_id": entry.panel.panel_id if entry.purpose == PanelPurpose.CONTENT else None,
                    "result_key": entry.result_key,
                    "sample_id": samples.request.sample_id,
                    "format_report_digest": report["report_digest"],
                    "content_report_digest": None if content is None else digest(content),
                    "file_sha256": files,
                }
            )
    manifest = {
        "suite_id": request.suite_id,
        "request": request.model_dump(mode="json"),
        "checkpoint": request.checkpoint.model_dump(),
        "model": actual_model,
        "model_digest": digest(actual_model),
        "source_revision": request.source_revision,
        "backend": NATIVE_BACKEND,
        "compute_dtype": "bfloat16",
        "routing_control_dtype": "float32",
        "devices": jax.device_count(),
        "processes": jax.process_count(),
        "device_platforms": sorted({device.platform for device in jax.devices()}),
        "mesh": dict(mesh.shape),
        "panels": panel_outputs,
        "pt": pt_output,
        "implementation_sha256": {
            name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in (
                "format_suite.py",
                "format_sample.py",
                "format_measure.py",
                "format_eval.py",
                "model.py",
                "weights.py",
            )
        },
        "completed_at": datetime.now(UTC).isoformat(),
    }
    manifest["manifest_digest"] = digest(manifest)
    if jax.process_index() == 0:
        _write_json(root / "manifest.json", manifest)
        logger.info("Native suite complete: manifest=%s at %s", manifest["manifest_digest"], destination)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("destination")
    args = parser.parse_args()
    request = NativeSuiteRequest.model_validate_json(args.request.read_bytes())
    DistributedConfig().initialize()
    configure_coreweave_s3()
    configure_logging(logging.INFO if jax.process_index() == 0 else logging.WARNING)
    if jax.default_backend() != "gpu" or any(device.platform != "gpu" for device in jax.devices()):
        raise ValueError("Run the native suite inside the explicitly allocated distributed GPU job")
    logger.info("Install the standard CuTeDSL kernel cache")
    with log_time("Kernel cache setup"):
        install(cutlass_kernel_cache())
    run_suite(request, args.destination)


if __name__ == "__main__":
    main()
