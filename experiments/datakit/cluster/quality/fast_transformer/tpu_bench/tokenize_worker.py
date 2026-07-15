# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fork-pool tokenization worker for the fast scoring pipeline.

Tokenization is CPU-heavy Python-and-Rust glue: the Rust ``encode_batch`` releases the GIL,
but the surrounding Python caps thread scaling to ~4x, so this pipeline tokenizes in a *fork*
process pool (true multi-core, no GIL) instead. A child runs no jax device ops -- it only
tokenizes -- so it never initializes the TPU runtime (the parent owns the chips), which is
safe only because the pool is forked BEFORE the parent initializes JAX.

Each child packs a block of window texts to ``[W, max_tokens]`` int32 and hands it back through
a ``shared_memory`` segment (the parent unlinks it), so a shard's ~750 MB of packed ids never
travels as a pickle over the pool's pipes.
"""

import os
from multiprocessing import resource_tracker, shared_memory

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np

from experiments.datakit.cluster.quality.fast_transformer.tpu_bench.common import (
    PAD_ID,
    load_remap_meta,
    load_shared_tokenizer,
    pack_windows,
    remap_to_array,
)

_STATE: dict = {}


def child_init(model_dir: str) -> None:
    """Pool initializer: bind the shared tokenizer + dense remap once per child process.

    Forked after the parent warmed both, so ``load_shared_tokenizer``/``remap_to_array`` hit
    the inherited caches (copy-on-write) rather than re-staging or re-downloading.
    """
    remap, tokenizer_name, max_tokens = load_remap_meta(model_dir)
    _STATE["tokenizer"] = load_shared_tokenizer(tokenizer_name)
    _STATE["lut"] = remap_to_array(remap)
    _STATE["max_tokens"] = max_tokens


def child_warm(_i: int = 0) -> int:
    """Force the tokenizer to bind in this child (called once per proc at pool startup)."""
    pack_windows(["warm up the tokenizer"], _STATE["tokenizer"], _STATE["lut"], _STATE["max_tokens"])
    return _i


def child_pack(win_texts: list[str]) -> tuple[str, tuple[int, int], int]:
    """Tokenize + pack a block to shared memory -> (shm_name, shape, n_real_tokens).

    The parent reads the segment by name and is responsible for ``unlink``-ing it.
    """
    packed = pack_windows(win_texts, _STATE["tokenizer"], _STATE["lut"], _STATE["max_tokens"])
    n_tokens = int((packed != PAD_ID).sum())
    shm = shared_memory.SharedMemory(create=True, size=packed.nbytes)
    np.ndarray(packed.shape, dtype=packed.dtype, buffer=shm.buf)[:] = packed
    # The parent unlinks the segment, so drop this child's resource_tracker registration --
    # otherwise the tracker double-frees at child exit and spams "leaked shared_memory" warnings.
    resource_tracker.unregister(shm._name, "shared_memory")
    shm.close()
    return shm.name, packed.shape, n_tokens
