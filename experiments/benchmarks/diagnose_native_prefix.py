# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Untimed Snowball shared-prefix logits with baseline and FP32-router computation."""

import argparse
import json
from pathlib import Path

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


def prefix_logits(model, sequences: list[list[int]], prefill_length: int) -> np.ndarray:
    """Return final paged logits after a prefill and optional teacher-forced decode steps."""
    batch, length = len(sequences), len(sequences[0])
    page_size = 128
    cache = model.initial_cache(PageTableSpec(num_pages=batch, page_size=page_size), dtype=jnp.bfloat16)
    run = hax.named_jit(lambda m, ids, state, info, positions: m.decode(ids, state, info, positions))
    phases = [(0, prefill_length), *((i, 1) for i in range(prefill_length, length))]
    final = None
    for start, count in phases:
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
        logits, cache = run(
            model, hax.named(jnp.asarray(tokens), "position"), cache, info, hax.named(jnp.asarray(positions), "position")
        )
        final = np.asarray(logits.array)[np.arange(batch) * count + count - 1].astype(np.float32)
    assert final is not None
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--prefixes", type=Path, required=True, help="JSON sequences and original prefill_length")
    parser.add_argument("--expert-axis-size", type=int, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.prefixes.read_text())
    sequences = inputs["sequences"]
    lengths = {len(row) for row in sequences}
    if len(lengths) != 1 or not 0 < inputs["prefill_length"] <= len(sequences[0]) <= 128 or len(sequences) > 8:
        raise ValueError("Diagnostic requires at most eight equal-length prefixes of at most128 tokens")
    if args.expert_axis_size != jax.device_count():
        raise ValueError("Diagnostic requires EP over every visible device, TP1/data1")
    converter = HFCheckpointConverter.from_hf(str(args.fixture / "checkpoint"))
    config = converter.config_from_hf_config(converter.hf_config_from_hf_checkpoint())
    if not isinstance(config, SnowballConfig) or config.vocab_size > 4096:
        raise ValueError("This bounded diagnostic supports small Snowball fixtures only")
    with jax.set_mesh(compact_grug_mesh(expert_axis_size=args.expert_axis_size)):
        model = converter.load_pretrained(config.model_type, config=config, dtype=jnp.bfloat16)
        model = jmp.get_policy("params=bfloat16,compute=bfloat16,output=bfloat16").cast_to_compute(model)
        fp32_router = eqx.tree_at(
            lambda m: tuple(block.mlp.router for block in m.transformer.blocks),
            model,
            tuple(block.mlp.router.astype(jnp.float32) for block in model.transformer.blocks),
        )
        results = {}
        for name, variant in (("baseline", model), ("fp32_router_highest", fp32_router)):
            with jax.default_matmul_precision(
                "highest" if name == "fp32_router_highest" else jax.config.jax_default_matmul_precision
            ):
                for mode, prefill in (("single_prefill", len(sequences[0])), ("incremental", inputs["prefill_length"])):
                    logits = prefix_logits(variant, sequences, prefill)
                    logprobs = np.asarray(jax.nn.log_softmax(jnp.asarray(logits), axis=-1))
                    top_ids = np.argsort(-logprobs, axis=-1)[:, :20]
                    results[f"{name}_{mode}"] = {
                        "logits": logits.tolist(),
                        "top_token_ids": top_ids.tolist(),
                        "top_logprobs": np.take_along_axis(logprobs, top_ids, axis=-1).tolist(),
                    }
    result = {
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
