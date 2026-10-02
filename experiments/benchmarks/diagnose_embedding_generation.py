# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare untimed ordinary generation with an embedding-only unary precision intervention."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
from pathlib import Path

import draccus
import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.benchmark import TokenWorkload, source_provenance, summarize_batch
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig
from levanter.main.inference_benchmark import measure_levanter_batch
from levanter.models.snowball import SnowballConfig

from experiments.benchmarks.diagnose_native_prefix import (
    DIAGNOSTIC_MAX_LAYERS,
    DIAGNOSTIC_MAX_SEQUENCES,
    DIAGNOSTIC_MAX_VOCAB,
    DIAGNOSTIC_MAX_WIDTH,
    DIAGNOSTIC_PAGE_SIZE,
    embedding_gate_weight_digests,
)
from experiments.benchmarks.matched_comparison import (
    MANIFEST_FILENAME,
    WORKLOAD_FILENAME,
    compare_reports,
    first_token_difference,
)
from experiments.benchmarks.snowball_trace import DiagnosticEmbeddingGate


def generation_interventions(model, workload: TokenWorkload, engine_config: InferenceEngineConfig) -> dict:
    """Generate from the same weights and fresh caches without returning activation traces."""
    results = {}
    for name in ("baseline", "embedding_fp32_silu_sigmoid"):
        variant = model
        if name != "baseline":
            gate = DiagnosticEmbeddingGate(model.transformer.embed_gated_norm, "float32", "float32")
            variant = eqx.tree_at(lambda m: m.transformer.embed_gated_norm, model, gate)
        engine = InferenceEngine.from_model_with_config(variant, None, engine_config)
        measurement = measure_levanter_batch(engine, workload)
        summary = summarize_batch(workload, measurement)
        results[name] = {"tokens": measurement.tokens, "output_sha256": summary.output_sha256}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.fixture
    manifest = json.loads((root / MANIFEST_FILENAME).read_text())
    workload = TokenWorkload(**json.loads((root / WORKLOAD_FILENAME).read_text()))
    reports = {name: json.loads((root / f"{name}-result.json").read_text()) for name in ("native", "vllm")}
    # Validate architecture, weights, workload, and topology before interpreting any intervention.
    comparison = compare_reports(manifest, workload, reports)
    native = reports["native"]["provenance"]
    config = draccus.decode(SnowballConfig, {k: v for k, v in native["model_config"].items() if k != "type"})
    if (
        native["model_config"]["type"] != "snowball"
        or config.num_layers > DIAGNOSTIC_MAX_LAYERS
        or config.hidden_dim > DIAGNOSTIC_MAX_WIDTH
        or config.vocab_size > DIAGNOSTIC_MAX_VOCAB
        or len(workload.prompts) > DIAGNOSTIC_MAX_SEQUENCES
        or max(map(len, workload.prompts)) + workload.output_tokens > DIAGNOSTIC_PAGE_SIZE
    ):
        raise ValueError("This diagnostic is bounded to small Snowball fixtures")
    checkpoint = root / "checkpoint"
    if hashlib.sha256((checkpoint / "model.safetensors").read_bytes()).hexdigest() != manifest["checkpoint"]:
        raise ValueError("Checkpoint file differs from the paired fixture")
    if jax.process_count() != 1 or [d.device_kind for d in jax.devices()] != native["device_kind"]:
        raise ValueError("Rerun on the same single-host accelerator topology as the saved native report")
    if jax.config.jax_default_matmul_precision != native["effective_execution"]["matmul_precision"]:
        raise ValueError("Matmul precision differs from the saved native report")
    mesh = compact_grug_mesh(model_axis_size=native["mesh"]["model"], expert_axis_size=native["mesh"]["expert"])
    if dict(mesh.shape) != native["mesh"]:
        raise ValueError("Native mesh differs from the saved report")
    engine_config = InferenceEngineConfig(
        **{**native["engine_config"], "compute_dtype": jnp.dtype(native["engine_config"]["compute_dtype"])}
    )
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping({"kv_head": "model", "heads": "model"}):
        converter = HFCheckpointConverter.from_hf(str(checkpoint))
        model = converter.load_pretrained(config.model_type, config=config, dtype=jnp.dtype(manifest["dtype"]))
        dtype = manifest["dtype"]
        model = jmp.get_policy(f"params={dtype},compute={dtype},output={dtype}").cast_to_compute(model)
        results = generation_interventions(model, workload, engine_config)
        digests = embedding_gate_weight_digests(model)
    baseline_difference = first_token_difference(results["baseline"]["tokens"], reports["native"]["validation_tokens"])
    if baseline_difference is not None:
        raise ValueError(f"Ordinary generation no longer reproduces the saved native baseline: {baseline_difference}")
    for result in results.values():
        result["first_difference_from_vllm"] = first_token_difference(
            result["tokens"], reports["vllm"]["validation_tokens"]
        )
        result["matches_vllm"] = result["first_difference_from_vllm"] is None
    report = {
        "boundary": "untimed_ordinary_engine_generation_without_activation_trace",
        "checkpoint": manifest["checkpoint"],
        "norm_weights": manifest.get("norm_weights"),
        "workload_sha256": workload.sha256,
        "parallelism": comparison["parallelism"],
        "effective_execution": comparison["effective_execution"],
        "engine_config": native["engine_config"],
        "source": dataclasses.asdict(source_provenance()),
        "versions": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "marin-levanter")},
        "saved_report_sha256": {
            name: hashlib.sha256((root / f"{name}-result.json").read_bytes()).hexdigest() for name in ("native", "vllm")
        },
        "embedding_gate_weight_sha256": digests,
        "intervention": {
            "site": "transformer.embed_gated_norm",
            "silu_dtype": "float32",
            "sigmoid_dtype": "float32",
            "cast_after_each_unary": dtype,
            "remaining_math": "unchanged",
        },
        "saved_vllm_tokens": reports["vllm"]["validation_tokens"],
        "baseline_reproduces_saved_native": True,
        "results": results,
        "throughput_comparison": None,
        "throughput_withheld_reason": "Diagnostic changes embedding gate arithmetic",
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
