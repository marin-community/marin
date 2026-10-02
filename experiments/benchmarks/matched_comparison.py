# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare the two saved reports produced by the matched checkpoint fixture."""

import hashlib
import json
import math
import statistics
from pathlib import Path

from levanter.inference.benchmark import BATCH_TIMING_BOUNDARY, TokenWorkload
from levanter.models.snowball import GRUG_MOE_CANONICAL_CONFIG_FIELDS

MANIFEST_FILENAME = "manifest.json"
WORKLOAD_FILENAME = "workload.json"

_EXECUTION_CONFIG_FIELDS = {"inference_attention_implementation"}
_HF_METADATA_FIELDS = {"_name_or_path", "transformers_version", "torch_dtype", "dtype"}


def normalized_grug_hf_config(config: dict) -> dict:
    """Resolve the promoted Grug config's aliases and recipe defaults, checking conflicts."""
    result = {key: value for key, value in config.items() if key not in _HF_METADATA_FIELDS}
    if result.get("model_type") != "grug_moe":
        return result
    aliases = {alias: name for name, alias in GRUG_MOE_CANONICAL_CONFIG_FIELDS if name != alias}
    aliases.update(
        {
            "attention_head_dim": "head_dim",
            "intermediate_size": "moe_intermediate_size",
            "num_local_experts": "num_experts",
        }
    )
    for alias, canonical in aliases.items():
        if alias in result:
            if result[alias] != result[canonical]:
                raise ValueError(
                    f"Conflicting Grug config alias {alias}: {result[alias]} != {canonical}: {result[canonical]}"
                )
            del result[alias]
    # Schema-v1 is the fixed June recipe; schema-v2 exports its architecture fields.
    # Defaults and derived fields follow vllm/transformers_utils/configs/grugmoe.py
    # at the promoted 01911be34fac source, not arbitrary missing-field equivalence.
    defaults = {"disable_pko": True, "disable_long_rope": True, "qk_mult_long_scale": 1.0, "use_cache": True}
    if result["grugmoe_artifact_schema_version"] == 1:
        defaults.update(
            {
                "global_every": 4,
                "num_shared_experts": 1,
                "rope_fused": False,
                "sconv": False,
                "sconv_kernel": 4,
                "sconv_sites": ["k", "attn", "mlp"],
                "local_kv_heads": None,
                "global_kv_heads": None,
                "latent_dim": None,
            }
        )
    for name, value in defaults.items():
        result.setdefault(name, value)
    layers = result["num_hidden_layers"]
    schedule = [
        "full_attention" if (i + 1) % result["global_every"] == 0 or i == layers - 1 else "sliding_attention"
        for i in range(layers)
    ]
    derived = {
        "layer_types": schedule,
        "rope": {"theta": result["rope_theta"]},
        "rope_parameters": {"rope_type": "default", "rope_theta": result["rope_theta"]},
    }
    for name, value in derived.items():
        if result.get(name) is not None and result[name] != value:
            raise ValueError(
                f"Grug config {name} differs from its architecture-derived value: {result[name]} != {value}"
            )
        result[name] = value
    return result


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
    if normalized_grug_hf_config(native_hf) != normalized_grug_hf_config(vllm_hf):
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
        if report["timing_boundary"] != BATCH_TIMING_BOUNDARY:
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
        "raw_hf_config_differences": hf_differences,
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
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    workload = TokenWorkload(**json.loads((root / WORKLOAD_FILENAME).read_text()))
    reports = {name: json.loads((root / f"{name}-result.json").read_text()) for name in ("native", "vllm")}
    result = compare_reports(manifest, workload, reports)
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result
