# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Step wrappers for slicing Levanter caches.

This module provides step definitions that wrap the library functions in
marin.tokenize.slice_cache.
"""

from levanter.data.text import LmDatasetSourceConfigBase

from marin.execution import deferred, output, step
from marin.tokenize.slice_cache import SliceCacheConfig
from marin.tokenize.slice_cache import _slice_cache_in_ray as _slice_cache_lib

# Mark library function as deferred
slice_cache_lib = deferred(_slice_cache_lib)


@step(name="{name}")
def slice_cache(
    name: str,
    input_config: LmDatasetSourceConfigBase,
    num_tokens: int,
    seed: int = 42,
    tokenizer_spec: str = "stanford-crfm/marin-tokenizer",
) -> SliceCacheConfig:
    """
    Create a step that slices a Levanter cache to produce a subsample.

    This step reads a Levanter cache and produces a subsample of that cache by
    sampling documents randomly from the cache (without replacement) until the
    desired number of tokens is reached.

    Args:
        name: Name for this slice cache step (used in output path)
        input_config: The input cache configuration (LmDatasetSourceConfigBase)
        num_tokens: The number of tokens to include in the sliced cache
        seed: The random seed for shuffling the dataset (default: 42)
        tokenizer_spec: The tokenizer specification (default: "stanford-crfm/marin-tokenizer")

    Returns:
        SliceCacheConfig to be executed by the step wrapper
    """
    return slice_cache_lib(
        SliceCacheConfig(
            input_config=input_config,
            num_tokens=num_tokens,
            seed=seed,
            tokenizer=tokenizer_spec,
            cache_path=output(),
        )
    )
