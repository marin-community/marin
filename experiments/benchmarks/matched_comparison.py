# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the two saved reports produced by the matched checkpoint fixture."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from levanter.inference.benchmark import TokenWorkload

_EXECUTION_CONFIG_FIELDS = {"inference_attention_implementation"}
_HF_METADATA_FIELDS = {"_name_or_path", "transformers_version", "torch_dtype", "dtype"}


def compare_reports(manifest: dict, workload: TokenWorkload, reports: dict[str, dict]) -> dict:
    """Validate paired runtime evidence and summarize agreement before speed ratios."""
    native, vllm = reports["native"]["provenance"], reports["vllm"]["provenance"]
    if native["backend"] != "levanter" or vllm["backend"] != "vllm":
        raise ValueError("Expected native and vLLM reports")
    if not native["checkpoint"] or not (
        native["checkpoint"]["identity"] == vllm["checkpoint"] == manifest["checkpoint"]
    ):
        raise ValueError("Checkpoint identities differ")
    if not (native["dtype"] == vllm["dtype"] == manifest["dtype"]):
        raise ValueError("Model dtypes differ")
    if vllm["effective_dtype"] != manifest["dtype"]:
        raise ValueError("Effective vLLM dtype differs")
    native_hf, vllm_hf = native["checkpoint"]["hf_config"], vllm["hf_config"]
    hf_differences = {
        field: {"native": native_hf.get(field), "vllm": vllm_hf.get(field)}
        for field in native_hf.keys() | vllm_hf.keys()
        if native_hf.get(field) != vllm_hf.get(field)
    }
    if hf_differences.keys() - _HF_METADATA_FIELDS:
        raise ValueError(f"Loaded Hugging Face model configurations differ: {hf_differences}")
    if len(reports["native"]["warmup"]) != len(reports["vllm"]["warmup"]) or len(reports["native"]["samples"]) != len(
        reports["vllm"]["samples"]
    ):
        raise ValueError("Warmup or measured batch counts differ")
    differences = {}
    for backend, provenance in (("native", native), ("vllm", vllm)):
        config = provenance["model_config"]
        changed = {
            field: {"fixture": manifest["model_config"].get(field), "actual": config.get(field)}
            for field in config.keys() | manifest["model_config"].keys()
            if config.get(field) != manifest["model_config"].get(field)
        }
        if changed.keys() - _EXECUTION_CONFIG_FIELDS:
            raise ValueError(f"{backend} model architecture differs: {changed}")
        differences[backend] = changed
    device_count = len(native["devices"])
    if not (device_count == len(vllm["devices"]) and native["device_kind"] == vllm["devices"]):
        raise ValueError("Accelerator kinds or counts differ")
    if native["hardware_label"] != vllm["hardware_label"]:
        raise ValueError("Hardware allocation/topology labels differ")
    mesh = native["mesh"]
    args = vllm["engine_args"]
    expected_mesh = {"replica_dcn": 1, "data": 1, "context": 1, "expert": device_count, "model": 1}
    if mesh != expected_mesh or not (
        args["tensor_parallel_size"] == 1
        and args["data_parallel_size"] == args["data_parallel_size_local"] == device_count
        and args["enable_expert_parallel"] == (device_count > 1)
    ):
        raise ValueError("Requires native EP=N/data1/TP1 versus local vLLM DP=N/EP/TP1")
    if args["enable_prefix_caching"]:
        raise ValueError("Prefix caching changes the measured workload")
    deterministic = {}
    throughput = {}
    for backend, report in reports.items():
        if report["schema_version"] != 2:
            raise ValueError("Rerun both runtimes to capture the separate validation batch")
        if report["timing_boundary"] != "offline_batch_host_submission_to_host_tokens":
            raise ValueError("Timing boundaries differ")
        actual_workload = TokenWorkload(**report["workload"])
        if actual_workload != workload or report["workload_sha256"] != workload.sha256:
            raise ValueError(f"{backend} token workload differs")
        tokens = report["validation_tokens"]
        if len(tokens) != len(workload.prompts) or any(len(row) != workload.output_tokens for row in tokens):
            raise ValueError(f"{backend} validation generation is incomplete")
        digest = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()
        if digest != report["validation_output_sha256"]:
            raise ValueError(f"{backend} validation token digest is corrupt")
        all_batches = [report["first_batch_including_compile"], *report["warmup"], *report["samples"]]
        deterministic[backend] = all(row["output_sha256"] == digest for row in all_batches)
        rate = statistics.median(row["output_tokens_per_second"] for row in report["samples"])
        if not math.isfinite(rate) or rate <= 0 or rate != report["median_output_tokens_per_second"]:
            raise ValueError(f"{backend} throughput summary is invalid")
        throughput[backend] = rate
    first_difference = next(
        (
            {"request_index": request, "output_token_index": position, "native_token": a, "vllm_token": b}
            for request, (left, right) in enumerate(
                zip(reports["native"]["validation_tokens"], reports["vllm"]["validation_tokens"], strict=True)
            )
            for position, (a, b) in enumerate(zip(left, right, strict=True))
            if a != b
        ),
        None,
    )
    comparable_outputs = first_difference is None and all(deterministic.values())
    result = {
        "checkpoint": manifest["checkpoint"],
        "workload_sha256": workload.sha256,
        "dtype": manifest["dtype"],
        "hardware_label": native["hardware_label"],
        "device_kinds": native["device_kind"],
        "parallelism": {"native_mesh": mesh, "vllm_dp": device_count, "vllm_tp": 1, "vllm_ep": device_count},
        "execution_config_differences": differences,
        "hf_metadata_differences": hf_differences,
        "effective_execution": {name: report["provenance"]["effective_execution"] for name, report in reports.items()},
        "validation_output_hashes": {name: report["validation_output_sha256"] for name, report in reports.items()},
        "validation_tokens_agree": first_difference is None,
        "first_divergent_token": first_difference,
        "all_batch_hashes_agree_with_validation": deterministic,
        "median_output_tokens_per_second": throughput,
        "native_over_vllm_throughput": throughput["native"] / throughput["vllm"] if comparable_outputs else None,
        "ratio_withheld_reason": None if comparable_outputs else "Outputs differ across backends or batches",
    }
    return result


def compare_fixture(root: Path) -> dict:
    """Write a paired report, refusing incompatible inputs and withholding invalid ratios."""
    manifest = json.loads((root / "manifest.json").read_text())
    workload = TokenWorkload(**json.loads((root / "workload.json").read_text()))
    reports = {name: json.loads((root / f"{name}-result.json").read_text()) for name in ("native", "vllm")}
    result = compare_reports(manifest, workload, reports)
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result
