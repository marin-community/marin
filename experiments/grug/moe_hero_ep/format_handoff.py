# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Assemble candidate evaluations from completed, pinned native-suite measurements."""

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Self

import draccus
import numpy as np
from pydantic import Field, model_validator
from rigging.filesystem.conditional_object import conditional_object
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath

from experiments.grug.moe_hero_ep.averaging import _DONORS, read_prepared_provenance
from experiments.grug.moe_hero_ep.format_eval import (
    Candidate,
    CandidateEvaluation,
    Panel,
    Serialization,
    grade_panel,
    validated_format_metrics,
)
from experiments.grug.moe_hero_ep.format_measure import (
    NativePtRequest,
    PtMeasurementRow,
    PtRoutingBatch,
    read_pt_panel,
)
from experiments.grug.moe_hero_ep.format_sample import NATIVE_BACKEND, NativePanelRequest, native_evaluation_model
from experiments.grug.moe_hero_ep.format_suite import NativeSuiteRequest, PanelPurpose
from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint, Record, SampleResult, digest


class CandidateHandoffRequest(Record):
    candidate: Candidate
    parent: Candidate | None
    suite_root: str
    suite_manifest_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    checkpoint: Checkpoint
    model_config_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    tokenizer: str
    tokenizer_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    format_panel_id: str
    content_panel_id: str
    pt_panel_path: str
    pt_panel_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    long_context_panel_id: str | None = None
    context_extension_manifest: str | None = None
    context_extension_manifest_digest: str | None = None

    @model_validator(mode="after")
    def candidate_lineage(self) -> Self:
        parents = {
            Candidate.A: {None},
            Candidate.B: {Candidate.A},
            Candidate.C: {Candidate.A, Candidate.B},
            Candidate.D: {Candidate.C},
            Candidate.E: {Candidate.D},
        }
        if self.parent not in parents[self.candidate]:
            raise ValueError("Candidate letter and parent are inconsistent")
        if self.candidate == Candidate.E:
            if not self.context_extension_manifest or not self.context_extension_manifest_digest:
                raise ValueError("Candidate E requires its exact context-extension manifest pin")
        elif self.context_extension_manifest is not None or self.context_extension_manifest_digest is not None:
            raise ValueError("Only candidate E uses a context-extension manifest")
        return self


def _json(path: StoragePath, sha256: str | None = None) -> dict:
    raw = path.read_bytes()
    if sha256 is not None and hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError(f"Completed suite output bytes changed: {path}")
    return json.loads(raw)


def _bound_digest(record: dict, key: str) -> str:
    actual = digest({name: value for name, value in record.items() if name != key})
    if actual != record.get(key):
        raise ValueError(f"Bound measurement {key} changed")
    return actual


def _lineage(request: CandidateHandoffRequest, model: GrugModelConfig) -> dict:
    metadata = _json(StoragePath(request.checkpoint.uri) / "metadata.json")
    if digest(metadata) != request.checkpoint.metadata_digest or metadata.get("is_temporary") is not False:
        raise ValueError("Candidate checkpoint metadata changed or is not permanent")
    if metadata["step"] != request.checkpoint.step:
        raise ValueError("Candidate checkpoint step differs from its completion metadata")
    if request.candidate == Candidate.A:
        if "averaging_digest" in metadata or "hero_sft" in metadata or model.yarn is not None:
            raise ValueError("Candidate A must be a permanent PT checkpoint")
        return {"checkpoint_metadata": metadata}
    if request.candidate in (Candidate.B, Candidate.C):
        prepared = read_prepared_provenance(request.checkpoint.uri, request.checkpoint.metadata_digest)
        if prepared["candidate"] != request.candidate.value:
            raise ValueError("Candidate letter differs from its preparation provenance")
        if request.candidate == Candidate.C:
            if prepared["donors"] != [list(row) for row in _DONORS]:
                raise ValueError("Candidate C lacks the exact donor initialization")
            parent = prepared["selected_parent"]
            if request.parent == Candidate.B:
                if read_prepared_provenance(parent["path"], parent["metadata_digest"])["candidate"] != "B":
                    raise ValueError("Candidate C's actual parent is not B")
            elif "averaging_digest" in prepared["source_metadata"][0]:
                raise ValueError("Candidate C's actual averaged parent was declared as A")
        source_model = draccus.decode(GrugModelConfig, prepared["model"])
        provenance = {"checkpoint_metadata": metadata, "preparation": prepared}
    else:
        sft = metadata.get("hero_sft", {})
        if sft.get("candidate") != "D" or metadata.get("merge_base") != "BF16-rounded candidate C":
            raise ValueError("Candidate D/E must use an ordinary merged format-SFT export")
        base = sft["base"]
        prepared = read_prepared_provenance(base["checkpoint"], base["metadata_digest"])
        if prepared["candidate"] != "C":
            raise ValueError("Format-SFT export does not descend from candidate C")
        source_model = draccus.decode(GrugModelConfig, prepared["model"])
        provenance = {"checkpoint_metadata": metadata, "base_preparation": prepared}
        if request.candidate == Candidate.E:
            assert request.context_extension_manifest is not None
            extension = _json(StoragePath(request.context_extension_manifest))
            if _bound_digest(extension, "manifest_digest") != request.context_extension_manifest_digest:
                raise ValueError("Context-extension manifest differs from its declared pin")
            if (
                extension["candidate"] != "E"
                or extension["parent_candidate"] != "D"
                or extension["source_checkpoint"] != request.checkpoint.uri
                or extension["source_metadata_digest"] != request.checkpoint.metadata_digest
                or extension["source_model_digest"] != digest(draccus.encode(source_model))
            ):
                raise ValueError("Candidate E's manifest does not bind these D arrays and source forward configuration")
            source_model = draccus.decode(GrugModelConfig, extension["model"])
            if source_model.max_seq_len != 65536 or source_model.yarn is None:
                raise ValueError("Candidate E lacks the 65K YaRN forward configuration")
            provenance["context_extension"] = extension
    if digest(draccus.encode(native_evaluation_model(source_model))) != request.model_config_digest:
        raise ValueError("Measured forward model differs from the checkpoint's provenanced candidate configuration")
    return provenance


def _pt_measurements(report: dict, suite: NativeSuiteRequest, request: CandidateHandoffRequest) -> tuple[float, dict]:
    measured_request = NativePtRequest.model_validate(report["request"])
    if (
        measured_request.checkpoint != suite.checkpoint
        or measured_request.model != suite.model
        or measured_request.tokenizer != suite.tokenizer
        or measured_request.tokenizer_revision != suite.tokenizer_revision
        or measured_request.source_revision != suite.source_revision
        or measured_request.panel_sha256 != request.pt_panel_sha256
        or measured_request.expert_axis_size != suite.expert_axis_size
        or measured_request.replica_axis_size != suite.replica_axis_size
    ):
        raise ValueError("PT measurement belongs to another checkpoint, forward model, tokenizer or panel")
    panel = read_pt_panel(measured_request.model_copy(update={"panel_path": request.pt_panel_path}))
    if report["panel_digest"] != digest(panel.model_dump()):
        raise ValueError("Measured PT panel differs from its exact pinned token arrays")
    model = draccus.decode(GrugModelConfig, suite.model)
    rows = [PtMeasurementRow.model_validate(row) for row in report["rows"]]
    if [row.row_index for row in rows] != [row.row_index for row in panel.rows]:
        raise ValueError("PT measurement omitted, duplicated or reordered pinned rows")
    for row, source in zip(rows, panel.rows, strict=True):
        expected_weight = float(np.asarray(source.loss_weight, dtype=np.float32).sum(dtype=np.float64))
        if (
            row.arrays_sha256 != source.arrays_sha256
            or row.valid_tokens != sum(segment >= 0 for segment in source.segment_ids)
            or not math.isfinite(row.loss_sum)
            or row.loss_sum < 0
            or not row.loss_weight > 0
            or not math.isclose(row.loss_weight, expected_weight, rel_tol=1e-9)
            or not math.isclose(row.loss, row.loss_sum / row.loss_weight, rel_tol=1e-9, abs_tol=1e-9)
        ):
            raise ValueError("PT measured row disagrees with its actual pinned tokens, weights or finite loss")
    loss_sum, weight, valid = (
        sum(row.loss_sum for row in rows),
        sum(row.loss_weight for row in rows),
        sum(row.valid_tokens for row in rows),
    )
    if not all(
        math.isfinite(float(report[key]))
        for key in ("loss_sum", "loss_weight", "loss", "routing_entropy_mean", "router_z_loss")
    ):
        raise ValueError("Nonfinite PT aggregate measurement")
    if (
        not math.isclose(report["loss_sum"], loss_sum, rel_tol=1e-9, abs_tol=1e-9)
        or not math.isclose(report["loss_weight"], weight, rel_tol=1e-9)
        or not math.isclose(report["loss"], loss_sum / weight, rel_tol=1e-9, abs_tol=1e-9)
        or report["valid_tokens"] != valid
    ):
        raise ValueError("PT aggregate loss differs from the measured weighted row sums")
    batches = [PtRoutingBatch.model_validate(batch) for batch in report["routing_batches"]]
    if [index for batch in batches for index in batch.row_indices] != [row.row_index for row in rows]:
        raise ValueError("PT routing batches do not cover the exact measured rows")
    row_tokens = {row.row_index: row.valid_tokens for row in rows}
    counts = []
    for batch in batches:
        array = np.asarray(batch.routing_counts_per_layer)
        if (
            batch.valid_tokens != sum(row_tokens[index] for index in batch.row_indices)
            or batch.filler_rows < 0
            or array.shape != (model.num_layers, model.num_experts)
            or array.dtype.kind not in "iu"
            or np.any(array < 0)
            or not np.all(array.sum(axis=-1) == batch.valid_tokens * model.num_experts_per_token)
            or not math.isfinite(batch.routing_entropy_mean)
            or not math.isfinite(batch.router_z_loss)
        ):
            raise ValueError("PT expert assignment counts include missing, dropped or padding tokens")
        counts.append(array)
    aggregate = np.sum(counts, axis=0)
    if not np.array_equal(aggregate, report["routing_counts_per_layer"]):
        raise ValueError("PT aggregate expert counts differ from the actual batch counts")
    loads = np.concatenate([array / array.mean(axis=-1, keepdims=True) for array in (*counts, aggregate)])
    entropy = sum(batch.routing_entropy_mean * batch.valid_tokens for batch in batches) / valid
    z_loss = sum(batch.router_z_loss * batch.valid_tokens for batch in batches) / valid
    if not math.isclose(entropy, report["routing_entropy_mean"], rel_tol=1e-9, abs_tol=1e-9) or not math.isclose(
        z_loss, report["router_z_loss"], rel_tol=1e-9, abs_tol=1e-9
    ):
        raise ValueError("PT routing scalar aggregates differ from valid-token-weighted measurements")
    return loss_sum / weight, {
        "max_load_factor": float(loads.max()),
        "max_balance_violation": float(np.abs(loads - 1).max()),
        "routing_entropy_mean": entropy,
        "router_z_loss": z_loss,
    }


def assemble_candidate(request: CandidateHandoffRequest) -> tuple[CandidateEvaluation, dict]:
    """Validate completed suite bytes and derive a candidate without assigning manual metrics."""
    root = StoragePath(request.suite_root)
    manifest = _json(root / "manifest.json")
    if _bound_digest(manifest, "manifest_digest") != request.suite_manifest_digest:
        raise ValueError("Completed suite differs from the declared manifest pin")
    suite = NativeSuiteRequest.model_validate(manifest["request"])
    model = draccus.decode(GrugModelConfig, suite.model)
    actual_model = draccus.encode(model)
    if (
        manifest["suite_id"] != suite.suite_id
        or suite.checkpoint != request.checkpoint
        or manifest["checkpoint"] != suite.checkpoint.model_dump()
        or (suite.tokenizer, suite.tokenizer_revision) != (request.tokenizer, request.tokenizer_revision)
        or digest(actual_model) != request.model_config_digest
        or manifest["model"] != actual_model
        or manifest["model_digest"] != request.model_config_digest
        or suite.pt_panel_sha256 != request.pt_panel_sha256
        or manifest["source_revision"] != suite.source_revision
        or manifest["backend"] != NATIVE_BACKEND
        or manifest["compute_dtype"] != "bfloat16"
        or manifest["routing_control_dtype"] != "float32"
    ):
        raise ValueError("Suite checkpoint, model, tokenizer, PT panel or precision differs from the handoff pin")
    if _json(root / "request.json") != suite.model_dump(mode="json"):
        raise ValueError("Persisted suite request differs from its completion manifest")
    if native_evaluation_model(model) != model:
        raise ValueError("Suite forward model is not the declared native evaluation backend")
    suite = suite.model_copy(update={"model": actual_model})
    lineage = _lineage(request, model)
    reports, contents = {}, {}
    if len(manifest["panels"]) != len(suite.panels):
        raise ValueError("Completed suite panel outputs are incomplete")
    for entry, output in zip(suite.panels, manifest["panels"], strict=True):
        if (
            output["panel_id"] != entry.panel.panel_id
            or output["serialization"] != entry.serialization.value
            or output["purpose"] != entry.purpose.value
            or output["result_key"] != entry.result_key
        ):
            raise ValueError("Suite output changed its pinned panel, purpose or serialization")
        required = {"panel.json", "request.json", "samples.json", "report.json"}
        if entry.purpose == PanelPurpose.CONTENT:
            required.add("content.json")
        if output["file_sha256"].keys() != required:
            raise ValueError("Suite output is missing required measured artifacts")
        values = {name: _json(root / entry.result_key / name, output["file_sha256"][name]) for name in required}
        if Panel.model_validate(values["panel.json"]) != entry.panel:
            raise ValueError("Completed panel bytes differ from the pinned suite panel")
        panel_request = NativePanelRequest.model_validate(values["request.json"])
        samples = SampleResult.model_validate(values["samples.json"])
        if (
            panel_request.panel != entry.panel
            or panel_request.checkpoint != suite.checkpoint
            or panel_request.model != suite.model
            or panel_request.serialization != entry.serialization
            or panel_request.target_cluster != suite.target_cluster
            or samples.request.checkpoint != suite.checkpoint
            or samples.request.model != suite.model
            or samples.request.source_revision != suite.source_revision
            or output["sample_id"] != samples.request.sample_id
        ):
            raise ValueError("Completed samples belong to another checkpoint or model execution")
        report = values["report.json"]
        if digest(grade_panel(entry.panel, samples, entry.serialization)) != digest(report):
            raise ValueError("Formatting report differs from the retained raw generations")
        if report["report_digest"] != output["format_report_digest"]:
            raise ValueError("Suite format report digest changed")
        if entry.serialization == Serialization.MARIN:
            validated_format_metrics(report)
            reports[entry.panel.panel_id] = report
        if entry.purpose == PanelPurpose.CONTENT:
            content = values["content.json"]
            if (
                digest(content) != output["content_report_digest"]
                or output["content_panel_id"] != entry.panel.panel_id
                or content["panel_id"] != entry.panel.panel_id
                or content["sample_id"] != samples.request.sample_id
                or content["model_config_digest"] != request.model_config_digest
                or content["structural_report_digest"] != report["report_digest"]
                or set(content["outcomes"]) != {case.id for case in entry.panel.cases}
                or any(type(value) is not bool for value in content["outcomes"].values())
                or content["successes"] != sum(content["outcomes"].values())
                or content["total"] != len(entry.panel.cases)
            ):
                raise ValueError("Content measurement belongs to another panel, checkpoint or raw generation")
            contents[(entry.panel.panel_id, entry.serialization)] = content["outcomes"]
    pt = manifest["pt"]
    if not pt or pt["result_key"] != "pt/report.json" or pt["panel_sha256"] != request.pt_panel_sha256:
        raise ValueError("Suite lacks the exact PT loss and routing measurement")
    measured = _json(root / pt["result_key"], pt["file_sha256"])
    if (
        _bound_digest(measured, "report_digest") != pt["report_digest"]
        or measured["panel_digest"] != pt["panel_digest"]
        or measured["model_digest"] != request.model_config_digest
        or measured["backend"] != NATIVE_BACKEND
        or measured["compute_dtype"] != "bfloat16"
        or measured["routing_control_dtype"] != "float32"
    ):
        raise ValueError("PT report identity or digest changed")
    loss, routing = _pt_measurements(measured, suite, request)
    if request.format_panel_id not in reports:
        raise ValueError("Suite lacks the selected native Marin formatting panel")
    if any((request.content_panel_id, serialization) not in contents for serialization in Serialization):
        raise ValueError("Suite must measure the same selected content panel in both serializations")
    if request.long_context_panel_id is not None and (
        request.long_context_panel_id not in reports or reports[request.long_context_panel_id]["context_length"] != 65536
    ):
        raise ValueError("Suite lacks the selected actual65K Marin panel")
    candidate = CandidateEvaluation(
        candidate=request.candidate,
        parent=request.parent,
        checkpoint=request.checkpoint.uri,
        checkpoint_metadata_digest=request.checkpoint.metadata_digest,
        model_config_digest=request.model_config_digest,
        panel_id=request.format_panel_id,
        content_panel_id=request.content_panel_id,
        tokenizer_revision=request.tokenizer_revision,
        backend=NATIVE_BACKEND,
        evaluation_dtype="bfloat16",
        heldout_loss_panel=request.pt_panel_sha256,
        heldout_loss=loss,
        routing=routing,
        content={serialization: contents[(request.content_panel_id, serialization)] for serialization in Serialization},
        format_report=reports[request.format_panel_id],
        long_context_report=reports.get(request.long_context_panel_id),
        serving_parity_report=None,
        serving_parity_passed=False,
    )
    provenance = {
        "request": request.model_dump(mode="json"),
        "suite_manifest": manifest,
        "pt_report": measured,
        "lineage": lineage,
        "evaluation_id": candidate.evaluation_id,
        "routing_definition": (
            "Maximum over measured batches and aggregate layers of expert count / per-layer mean; "
            "balance violation is max absolute deviation of that ratio from 1. "
            "Entropy and z-loss are valid-token-weighted batch means."
        ),
        "serving_parity": "Pending; this CLI accepts no manual boolean or path as serving proof.",
    }
    provenance["assembly_digest"] = digest(provenance)
    return candidate, provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("destination")
    args = parser.parse_args()
    configure_coreweave_s3()
    request = CandidateHandoffRequest.model_validate_json(args.request.read_bytes())
    root = StoragePath(args.destination)
    if root.exists():
        raise FileExistsError("Choose a new immutable candidate handoff destination")
    candidate, provenance = assemble_candidate(request)
    for name, value in (("candidate.json", candidate.model_dump(mode="json")), ("assembly.json", provenance)):
        conditional_object(str(root / name)).write(
            (json.dumps(value, indent=2, sort_keys=True) + "\n").encode(), expected_version=None
        )


if __name__ == "__main__":
    main()
