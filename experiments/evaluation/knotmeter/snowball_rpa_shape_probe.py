# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Verify the long-context Snowball attention shape without model weights."""

import argparse
import json
import os
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from jax.sharding import AxisType
from jax.sharding import PartitionSpec as P
from levanter.grug.attention import ragged_paged_attention


def _attention_with_default_fp32_tiles(q, pages, *metadata):
    result = ragged_paged_attention(
        q.astype(jnp.float32),
        pages.astype(jnp.float32),
        *metadata,
        sm_scale=128**-0.5,
        implementation="tpu",
    )
    return result.astype(q.dtype)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiles", choices=("bounded", "default"), default="bounded")
    args = parser.parse_args()
    context = 65536
    page_size = 128
    dtype = jnp.bfloat16
    pages_count = context // page_size
    records = {}
    mesh = jax.make_mesh((len(jax.devices()),), ("expert",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        pages = jnp.ones((pages_count, page_size, 10, 128), dtype, out_sharding=P())
        page_indices = jnp.arange(pages_count, dtype=jnp.int32, out_sharding=P())[None, :]
        cases = [("prefill", context, 51870, 51870, None), ("decode", 4, 1, 51871, None)]
        if args.tiles == "bounded":
            cases.append(("decode_window", 4, 1, 51871, 2048))
        for label, query_tokens, valid_tokens, kv_length, window in cases:
            if args.tiles == "default":
                attention = jax.jit(_attention_with_default_fp32_tiles)
            else:
                attention = jax.jit(
                    partial(ragged_paged_attention, sm_scale=128**-0.5, sliding_window=window, implementation="tpu_fp32")
                )
            inputs = (
                jnp.zeros((query_tokens, 5, 4, 128), dtype, out_sharding=P()),
                pages,
                jnp.array([kv_length], jnp.int32, out_sharding=P()),
                page_indices,
                jnp.array([0, valid_tokens], jnp.int32, out_sharding=P()),
                jnp.array(1, jnp.int32, out_sharding=P()),
            )
            started = time.monotonic()
            compiled = attention.lower(*inputs).compile()
            compile_seconds = time.monotonic() - started
            output = jax.block_until_ready(compiled(*inputs))
            assert bool(jnp.all(output[:valid_tokens] == 1))
            assert bool(jnp.all(output[valid_tokens:] == 0))
            times = []
            for _ in range(3):
                started = time.monotonic()
                jax.block_until_ready(compiled(*inputs))
                times.append(time.monotonic() - started)
            memory = compiled.memory_analysis()
            records[label] = {
                "query_buffer_tokens": query_tokens,
                "valid_query_tokens": valid_tokens,
                "kv_tokens": kv_length,
                "sliding_window": window,
                "compile_seconds": compile_seconds,
                "seconds": times,
                "temp_bytes": memory.temp_size_in_bytes if memory is not None else None,
            }
    result = {"devices": [device.device_kind for device in jax.devices()], "tiles": args.tiles, "records": records}
    (Path(os.environ["IRIS_OUTPUT_DIR"]) / "rpa-shape.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
