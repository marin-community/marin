# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark fixed token batches through the native Levanter inference engine."""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import os
import time
from pathlib import Path

import draccus
import haliax as hax
import jax
import jax.numpy as jnp
import jmp

from levanter.compat.hf_checkpoints import HFCheckpointConverter, RepoRef
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.benchmark import BatchMeasurement, TokenWorkload, measure_batches, source_provenance
from levanter.inference.engine import InferenceEngine, InferenceEngineConfig, Request
from levanter.inference.jit_scheduler import SeqDecodingParams
from levanter.models.lm_model import LmConfig
from levanter.models.hero import HeroConfig
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
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model-config", type=Path, help="Synthetic Levanter model JSON including type")
    source.add_argument(
        "--checkpoint", help="Snowball or Hero HF export: local directory, object-storage URL, or Hub repo"
    )
    parser.add_argument("--revision", help="Pinned Hub revision for --checkpoint")
    parser.add_argument(
        "--checkpoint-identity", help="Immutable export identity or digest, shared with vLLM provenance"
    )
    parser.add_argument("--workload", type=Path, required=True, help="JSON prompts (token IDs) and output_tokens")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], required=True)
    parser.add_argument("--model-axis-size", type=int, default=1)
    parser.add_argument("--expert-axis-size", type=int, default=1)
    parser.add_argument("--page-size", type=int, default=128)
    parser.add_argument("--max-rounds", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--measured-batches", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hardware-label", required=True, help="Provisioned accelerator and topology, e.g. v5p-8")
    parser.add_argument("--source-revision", help="Source commit supplied by the launcher for bundles without .git")
    parser.add_argument(
        "--source-dirty", choices=["true", "false"], help="Launcher checkout status; omitted means unknown"
    )
    args = parser.parse_args()
    source = source_provenance(
        args.source_revision, None if args.source_dirty is None else args.source_dirty == "true"
    )
    converter = None
    checkpoint_provenance = None
    if args.checkpoint:
        if not args.checkpoint_identity:
            parser.error("--checkpoint requires --checkpoint-identity for matched weight provenance")
        reference = RepoRef(args.checkpoint, args.revision)
        converter = HFCheckpointConverter.from_hf(reference)
        hf_config = converter.hf_config_from_hf_checkpoint(reference)
        config = converter.config_from_hf_config(hf_config)
        checkpoint_provenance = {
            "source": args.checkpoint,
            "revision": args.revision,
            "identity": args.checkpoint_identity,
            "hf_config": hf_config.to_dict(),
            "tokenizer_vocab_sha256": hashlib.sha256(
                json.dumps(converter.tokenizer.get_vocab(), sort_keys=True).encode()
            ).hexdigest(),
        }
    else:
        if args.revision or args.checkpoint_identity:
            parser.error("--revision and --checkpoint-identity require --checkpoint")
        config = draccus.decode(LmConfig, json.loads(args.model_config.read_text()))
    if not isinstance(config, (SnowballConfig, HeroConfig)):
        raise ValueError("This benchmark supports SnowballConfig and HeroConfig")
    workload = TokenWorkload(**json.loads(args.workload.read_text()))
    if any(token < 0 or token >= config.vocab_size for prompt in workload.prompts for token in prompt):
        raise ValueError("Prompt token outside the model vocabulary")
    max_seq_len = max(map(len, workload.prompts)) + workload.output_tokens
    if max_seq_len > config.max_seq_len:
        raise ValueError("Workload exceeds model context length")
    dtype = jnp.dtype(args.dtype)
    if jax.process_count() != 1:
        raise ValueError("This driver currently supports a single host; use all local devices through the model axis")
    mesh = compact_grug_mesh(model_axis_size=args.model_axis_size, expert_axis_size=args.expert_axis_size)
    setup_start = time.perf_counter()
    with hax.partitioning.set_mesh(mesh), hax.axis_mapping({"kv_head": "model", "heads": "model"}):
        if converter is None:
            model = hax.named_jit(config.build)(
                hax.Axis("vocab", config.vocab_size), key=jax.random.PRNGKey(args.seed)
            )
        else:
            model = converter.load_pretrained(config.model_type, config=config, dtype=dtype)
        model = jmp.get_policy(f"params={args.dtype},compute={args.dtype},output={args.dtype}").cast_to_compute(model)
        jax.block_until_ready(model)
        batch = len(workload.prompts)
        token_shards = jax.device_count() // args.model_axis_size
        decode_capacity = ((batch + token_shards - 1) // token_shards) * token_shards
        prefill_tokens = sum(map(len, workload.prompts))
        prefill_capacity = ((prefill_tokens + token_shards - 1) // token_shards) * token_shards
        engine_config = InferenceEngineConfig(
            max_seq_len=max_seq_len,
            page_size=args.page_size,
            max_pages=batch * ((max_seq_len + args.page_size - 1) // args.page_size),
            max_seqs=batch,
            max_seqs_in_prefill=batch,
            max_queued_tokens=decode_capacity,
            max_tokens_per_round=decode_capacity,
            max_prefill_size=prefill_capacity,
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
    result = dataclasses.asdict(result)
    result["provenance"] = {
        "backend": "levanter",
        "evidence_kind": "synthetic_random_weights" if converter is None else "checkpoint",
        "checkpoint": checkpoint_provenance,
        "model_config": draccus.encode(config),
        "seed": args.seed,
        "dtype": args.dtype,
        "hardware_label": args.hardware_label,
        "devices": [str(device) for device in jax.devices()],
        "device_kind": [device.device_kind for device in jax.devices()],
        "mesh": dict(mesh.shape),
        "engine_config": {**dataclasses.asdict(engine_config), "compute_dtype": str(dtype)},
        "versions": {name: importlib.metadata.version(name) for name in ["jax", "jaxlib", "marin-levanter"]},
        "git_commit": source.revision,
        "git_dirty": source.dirty,
        "source_origin": source.origin,
        "source_tree_hash": source.tree_hash,
        "environment": {name: os.environ.get(name) for name in ["XLA_FLAGS", "LIBTPU_INIT_ARGS", "RAGGED_DOT_IMPL"]},
    }
    result["model_and_cache_setup"] = setup_elapsed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    logger.info("Wrote benchmark result to %s", args.output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
