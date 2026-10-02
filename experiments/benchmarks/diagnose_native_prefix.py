# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Untimed Snowball shared-prefix logits with baseline and FP32-router computation."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jmp
import numpy as np
from levanter.compat.hf_checkpoints import HFCheckpointConverter
from levanter.grug.sharding import compact_grug_mesh
from levanter.inference.page_table import PageBatchInfo, PageTableSpec
from levanter.models.snowball import SnowballConfig

from experiments.benchmarks.snowball_trace import decode_with_trace

DIAGNOSTIC_PAGE_SIZE = 128


class PrefixBatch(NamedTuple):
    tokens: hax.NamedArray
    metadata: PageBatchInfo
    positions: hax.NamedArray


def _prefix_batch(sequences: list[list[int]], start: int, count: int):
    batch = len(sequences)
    page_size = DIAGNOSTIC_PAGE_SIZE
    valid = batch * count
    capacity = ((valid + jax.device_count() - 1) // jax.device_count()) * jax.device_count()
    tokens = np.zeros(capacity, dtype=np.int32)
    positions = np.zeros(capacity, dtype=np.int32)
    destinations = np.full(capacity, -1, dtype=np.int32)
    for request, sequence in enumerate(sequences):
        rows = slice(request * count, (request + 1) * count)
        tokens[rows] = sequence[start : start + count]
        positions[rows] = np.arange(start, start + count)
        destinations[rows] = request * page_size + positions[rows]
    info = PageBatchInfo(
        slot_ids=hax.named(jnp.arange(batch, dtype=jnp.int32), "seq"),
        page_indices=hax.named(jnp.arange(batch, dtype=jnp.int32)[:, None], ("seq", "page")),
        seq_lens=hax.named(jnp.full(batch, start + count, dtype=jnp.int32), "seq"),
        cu_q_lens=hax.named(jnp.arange(batch + 1, dtype=jnp.int32) * count, "seq"),
        num_seqs=jnp.array(batch, dtype=jnp.int32),
        new_token_dests=hax.named(jnp.asarray(destinations), "position"),
        page_size=page_size,
    )
    return PrefixBatch(hax.named(jnp.asarray(tokens), "position"), info, hax.named(jnp.asarray(positions), "position"))


def prefix_logits(model, sequences: list[list[int]], prefill_length: int) -> np.ndarray:
    """Return final paged logits after a prefill and optional teacher-forced decode steps."""
    batch, length = len(sequences), len(sequences[0])
    page_size = DIAGNOSTIC_PAGE_SIZE
    cache = model.initial_cache(PageTableSpec(num_pages=batch, page_size=page_size), dtype=jnp.bfloat16)
    run = hax.named_jit(lambda m, ids, state, info, positions: m.decode(ids, state, info, positions))
    phases = [(0, prefill_length), *((i, 1) for i in range(prefill_length, length))]
    final = None
    for start, count in phases:
        tokens, info, positions = _prefix_batch(sequences, start, count)
        logits, cache = run(model, tokens, cache, info, positions)
        final = np.asarray(logits.array)[np.arange(batch) * count + count - 1].astype(np.float32)
    assert final is not None
    return final


def prefix_trace(model, sequences: list[list[int]], baseline_logits: np.ndarray) -> dict:
    """Capture all layer stages from one paged prefill and disclose tracing's numerical effect."""
    count = len(sequences[0])
    tokens, info, positions = _prefix_batch(sequences, 0, count)
    cache = model.initial_cache(
        PageTableSpec(num_pages=len(sequences), page_size=DIAGNOSTIC_PAGE_SIZE), dtype=jnp.bfloat16
    )
    logits, _, stages = hax.named_jit(decode_with_trace)(model, tokens, cache, info, positions)
    arrays = {
        name: np.asarray(value).astype(np.int32 if name == "expert_ids" else np.float32)
        for name, value in stages.items()
    }
    final_indices = np.arange(len(sequences)) * count + count - 1
    actual_logits = np.asarray(logits.array).astype(np.float32)
    head = np.asarray(model.transformer.output_proj).astype(np.float64)
    return {
        "boundary": "untimed_paged_layer_scan_with_auxiliary_jit_outputs",
        "head_weight_sha256": hashlib.sha256(np.ascontiguousarray(head.T, dtype=np.float32).tobytes()).hexdigest(),
        "stages": {name: value.tolist() for name, value in arrays.items()},
        "logits": actual_logits.tolist(),
        "projection_fp64": (arrays["post_final_gate"].astype(np.float64) @ head).tolist(),
        "final_token_indices": final_indices.tolist(),
        "baseline_logit_max_absolute_error": float(np.max(np.abs(actual_logits[final_indices] - baseline_logits))),
        "baseline_top_tokens_agree": bool(
            np.array_equal(actual_logits[final_indices].argmax(-1), baseline_logits.argmax(-1))
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True, help="JSON sequences and original prefill_length")
    parser.add_argument("--expert-axis-size", type=int, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.prefixes.read_text())
    sequences = inputs["sequences"]
    lengths = {len(row) for row in sequences}
    if (
        len(lengths) != 1
        or not 0 < inputs["prefill_length"] <= len(sequences[0]) <= DIAGNOSTIC_PAGE_SIZE
        or len(sequences) > 8
    ):
        raise ValueError(
            f"Diagnostic requires at most eight equal-length prefixes of at most {DIAGNOSTIC_PAGE_SIZE} tokens"
        )
    if args.expert_axis_size != jax.device_count():
        raise ValueError("Diagnostic requires EP over every visible device, TP1/data1")
    converter = HFCheckpointConverter.from_hf(str(args.fixture / "checkpoint"))
    config = converter.config_from_hf_config(converter.hf_config_from_hf_checkpoint())
    if (
        not isinstance(config, SnowballConfig)
        or config.vocab_size > 4096
        or config.num_layers > 2
        or config.hidden_dim > 512
    ):
        raise ValueError("This diagnostic supports Snowball with at most two layers, width512, and vocabulary4096")
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=args.expert_axis_size)):
        model = converter.load_pretrained(config.model_type, config=config, dtype=jnp.bfloat16)
        model = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16").cast_to_compute(model)
        fp32_router = eqx.tree_at(
            lambda m: tuple(block.mlp.router for block in m.transformer.blocks),
            model,
            tuple(block.mlp.router.astype(jnp.float32) for block in model.transformer.blocks),
        )
        results = {}
        baseline_precision = jax.config.jax_default_matmul_precision
        for router_name, variant in (("bf16_router", model), ("fp32_router", fp32_router)):
            for precision_name, precision in (
                ("baseline_precision", baseline_precision),
                ("highest_precision", "highest"),
            ):
                with jax.default_matmul_precision(precision):
                    for mode, prefill in (
                        ("single_prefill", len(sequences[0])),
                        ("incremental", inputs["prefill_length"]),
                    ):
                        logits = prefix_logits(variant, sequences, prefill)
                        logprobs = np.asarray(jax.nn.log_softmax(jnp.asarray(logits), axis=-1))
                        top_ids = np.argsort(-logprobs, axis=-1)[:, :20]
                        results[f"{router_name}_{precision_name}_{mode}"] = {
                            "router_dtype": str(variant.transformer.blocks[0].mlp.router.dtype),
                            "matmul_precision": precision,
                            "logits": logits.tolist(),
                            "top_token_ids": top_ids.tolist(),
                            "top_logprobs": np.take_along_axis(logprobs, top_ids, axis=-1).tolist(),
                        }
        stage_trace = prefix_trace(
            model, sequences, np.asarray(results["bf16_router_baseline_precision_single_prefill"]["logits"])
        )
    result = {
        "stage_trace": stage_trace,
        "checkpoint": json.loads((args.fixture / "manifest.json").read_text())["checkpoint"],
        "prefixes": inputs,
        "expert_axis_size": args.expert_axis_size,
        "boundary": "untimed_teacher_forced_paged_logits",
        "results": results,
    }
    output = args.fixture / "native-prefix-diagnostic.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
