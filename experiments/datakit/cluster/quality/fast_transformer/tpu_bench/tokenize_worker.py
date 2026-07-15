# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Child-process tokenization worker for the fast scoring pipeline.

Deliberately imports NO jax: the fast stage spawns a process pool to tokenize across the
v6e host's many cores, and a spawned child must not import the TPU runtime (only the
parent process owns the 4 chips). Each child loads the tokenizer + vocab lookup once via
``child_init`` and then packs window-text batches to ``[N, max_tokens]`` int32.
"""

import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

from experiments.datakit.cluster.quality.fast_transformer.tpu_bench.common import (
    load_remap_meta,
    pack_windows,
    remap_to_array,
)

_STATE: dict = {}


def child_init(model_dir: str) -> None:
    """Pool initializer: load tokenizer + dense remap once per child process."""
    remap, tokenizer_name, max_tokens = load_remap_meta(model_dir)
    _STATE["lut"] = remap_to_array(remap, len(remap) + 2)
    _STATE["tokenizer_name"] = tokenizer_name
    _STATE["max_tokens"] = max_tokens


def child_warm(_i: int = 0) -> int:
    """Force the tokenizer to load in this child (called once per proc at pool startup)."""
    pack_windows(["warm up the tokenizer"], _STATE["tokenizer_name"], _STATE["lut"], _STATE["max_tokens"])
    return _i


def child_tokenize(texts: list[str]):
    """Tokenize + pack a batch of window texts -> ``[len(texts), max_tokens]`` int32."""
    return pack_windows(texts, _STATE["tokenizer_name"], _STATE["lut"], _STATE["max_tokens"])
