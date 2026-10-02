# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark fixed token batches through the native Levanter inference engine."""

import argparse
import dataclasses
import importlib.metadata
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import jmp

from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.benchmark import BatchMeasurement, TokenWorkload, measure_batches
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.lm_model import LmConfig
from levanter.models.snowball import SnowballConfig

logger = logging.getLogger(__name__)


def measure_levanter_batch(engine: InferenceEngine, workload: TokenWorkload) -> BatchMeasurement:
    """Measure a batch admitted in one prefill, including host scheduling and extraction."""
    if len(workload.prompts) > min(engine.config.max_seqs, engine.config.max_seqs_in_prefill):
        raise ValueError("Latency measurement requires every request to fit in the first prefill")
    assert engine.config.max_prefill_size is not None
    if sum(map(len, workload.prompts)) > engine.config.max_prefill_size:
        raise ValueError("Latency measurement requires all prompt tokens to fit in the first prefill")
    requests = [
        Request(
            prompt_tokens=prompt,
            request_id=i,
            decode_params=dataclasses.replace(
                SeqDecodingParams.default(), max_num_tokens=jnp.asarray(len(prompt) + workload.output_tokens)
            ),
            n_generations=1,
        )
        for i, prompt in enumerate(workload.prompts)
    ]
    first_token = []
    start = time.perf_counter()

    def after_prefill(iteration):
        if iteration == 0:
            first_token.extend([time.perf_counter() - start] * len(requests))

    result = engine.generate(requests, step_callback=after_prefill)
    return BatchMeasurement(time.perf_counter() - start, first_token, result.tokens)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True, help="Levanter model JSON including type")
    parser.add_argument("--workload", type=Path, required=True, help="JSON prompts (token IDs) and output_tokens")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    parser.add_argument("--model-axis-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measured-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hardware-label", required=True, help="Provisioned accelerator and topology, e.g. v5p-8")
    args = parser.parse_args()
    config = draccus.decode(LmConfig, json.loads(args.model_config.read_text()))
    if not isinstance(config, SnowballConfig):
        raise ValueError("This synthetic benchmark currently supports SnowballConfig only; Hero needs native decode")
    workload = TokenWorkload(**json.loads(args.workload.read_text()))
    if any(token < 0 or token >= config.vocab_size for prompt in workload.prompts for token in prompt):
        raise ValueError("Prompt token outside the model vocabulary")
    max_seq_len = max(map(len, workload.prompts)) + workload.output_tokens
    if max_seq_len > config.max_seq_len:
        raise ValueError("Workload exceeds model context length")
    dtype = jnp.dtype(args.dtype)
    if jax.process_count() != 1:
        raise ValueError("This driver currently supports a single host; use all local devices through the model axis")
    mesh = compact_grug_mesh(model_axis_size=args.model_axis_size)
    setup_start = time.perf_counter()
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping({"kv_head": "model", "heads": "model"}):
        model = hax.named_jit(config.build)(hax.Axis("vocab", config.vocab_size), key=jax.random.PRNGKey(args.seed))
        model = jmp.get_policy(f"params={args.dtype},compute={args.dtype},output={args.dtype}").cast_to_compute(model)
        jax.block_until_ready(model)
        batch = len(workload.prompts)
        engine_config = InferenceEngineConfig(
            max_seq_len=max_seq_len,
            page_size=args.page_size,
            max_pages=batch * ((max_seq_len + args.page_size - 1) // args.page_size),
            max_seqs=batch,
            max_seqs_in_prefill=batch,
            max_queued_tokens=batch,
            max_prefill_size=sum(map(len, workload.prompts)),
            max_rounds=args.max_rounds,
            max_stop_seqs=0,
            max_stop_tokens=0,
            compute_dtype=dtype,
        )
        engine = InferenceEngine.from_model_with_config(model, None, engine_config)
        jax.block_until_ready(engine.gen_state)
        setup_elapsed = time.perf_counter() - setup_start
        result = measure_batches(
            workload,
            lambda tokens: measure_levanter_batch(engine, tokens),
            warmup_batches=args.warmup_batches,
            measured_batches=args.measured_batches,
        )
    result["provenance"] = {
        "backend": "levanter",
        "evidence_kind": "synthetic_random_weights",
        "checkpoint": None,
        "model_config": draccus.encode(config),
        "seed": args.seed,
        "dtype": args.dtype,
        "hardware_label": args.hardware_label,
        "devices": [str(device) for device in jax.devices()],
        "device_kind": [device.device_kind for device in jax.devices()],
        "mesh": dict(mesh.shape),
        "engine_config": {**dataclasses.asdict(engine_config), "compute_dtype": str(dtype)},
        "versions": {name: importlib.metadata.version(name) for name in ["jax", "jaxlib", "marin-levanter"]},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "environment": {name: os.environ.get(name) for name in ["XLA_FLAGS", "LIBTPU_INIT_ARGS", "RAGGED_DOT_IMPL"]},
    }
    result["model_and_cache_setup"] = setup_elapsed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Wrote benchmark result to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
