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

"""Script to count tokens from GCS caches and output values for domains.py.

This script queries the token counts from existing GCS caches and prints
Python code that can be copied into domains.py to hardcode the weights.

Usage:
    python -m experiments.domain_phase_mix.count_tokens

The script assumes caches exist at gs://marin-us-central1/tokenized/...
"""

import os

import numpy as np


def get_token_count_from_cache(cache_path: str) -> int:
    """Get the total token count from a tokenized cache.

    Args:
        cache_path: Path to the cache directory (GCS or local).

    Returns:
        Total number of tokens in the cache.
    """
    from levanter.store.cache import TreeCache

    exemplar = {"input_ids": np.array([0], dtype=np.int32)}
    train_path = os.path.join(cache_path, "train")

    cache = TreeCache.load(train_path, exemplar=exemplar)
    token_count = cache.store.tree["input_ids"].data_size
    return token_count


# Cache paths on GCS (relative to MARIN_PREFIX)
CACHE_PATHS = {
    # Nemotron HQ splits (using llama3_tokenizer)
    "nemotron_cc/hq_actual": "tokenized/nemotron_cc/hq_actual-5af4cc",
    "nemotron_cc/medium_high": "tokenized/nemotron_cc/medium_high-d21701",
    "nemotron_cc/medium": "tokenized/nemotron_cc/medium-d86506",
    # Full Nemotron (additional splits)
    "nemotron_cc/hq_synth": "tokenized/nemotron_cc/hq_synth-3525e2",
    "nemotron_cc/medium_low": "tokenized/nemotron_cc/medium_low-0fdb07",
    "nemotron_cc/low_actual": "tokenized/nemotron_cc/low_actual-cb3f2c",
    # Dolmino splits
    "dolmino/dclm": "tokenized/dolmino/dclm-6c18eb",
    "dolmino/flan": "tokenized/dolmino/flan-d71ec1",
    "dolmino/pes2o": "tokenized/dolmino/pes2o-d22243",
    "dolmino/stackexchange": "tokenized/dolmino/stackexchange-271a84",
    "dolmino/wiki": "tokenized/dolmino/wiki-c31b74",
    # SFT datasets (using marin_tokenizer)
    "tulu_3_sft_mixture": "tokenized/tulu_3_sft_mixture_marin_tokenizer-c0f545",
    "openthoughts_114k_math": "tokenized/openthoughts_114k_math_marin_tokenizer-2ec574",
    "verifiable_math_problems": "tokenized/verifiable_math_problems_marin_tokenizer-a665df",
}


def main():
    prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")

    print("=" * 80)
    print("Token counts from GCS caches")
    print(f"Prefix: {prefix}")
    print("=" * 80)
    print()

    results = {}
    errors = []

    for name, relative_path in CACHE_PATHS.items():
        full_path = os.path.join(prefix, relative_path)
        try:
            token_count = get_token_count_from_cache(full_path)
            results[name] = token_count
            print(f"{name}: {token_count:,} tokens ({token_count / 1e9:.2f}B)")
        except Exception as e:
            errors.append((name, str(e)))
            print(f"{name}: ERROR - {e}")

    print()
    print("=" * 80)
    print("Python code for domains.py")
    print("=" * 80)
    print()

    # Group by domain
    nemotron = [
        "nemotron_cc/hq_actual",
        "nemotron_cc/hq_synth",
        "nemotron_cc/medium_high",
        "nemotron_cc/medium",
        "nemotron_cc/medium_low",
        "nemotron_cc/low_actual",
    ]
    dolmino = ["dolmino/dclm", "dolmino/flan", "dolmino/pes2o", "dolmino/stackexchange", "dolmino/wiki"]
    sft = ["tulu_3_sft_mixture", "openthoughts_114k_math", "verifiable_math_problems"]

    print("# Nemotron token counts")
    print("NEMOTRON_TOKENS = {")
    for name in nemotron:
        if name in results:
            print(f'    "{name}": {results[name]},  # {results[name] / 1e9:.2f}B tokens')
    print("}")
    print()

    print("# Dolmino token counts")
    print("DOLMINO_TOKENS = {")
    for name in dolmino:
        short_name = name.split("/")[1]
        if name in results:
            print(f'    "{short_name}": {results[name]},  # {results[name] / 1e9:.2f}B tokens')
    print("}")
    print()

    print("# SFT token counts")
    print("SFT_TOKENS = {")
    for name in sft:
        if name in results:
            print(f'    "{name}": {results[name]},  # {results[name] / 1e9:.2f}B tokens')
    print("}")
    print()

    # Print totals
    print("# Domain totals:")
    if all(n in results for n in nemotron):
        total = sum(results[n] for n in nemotron)
        print(f"# Nemotron: {total:,} tokens ({total / 1e9:.2f}B)")
    if all(n in results for n in dolmino):
        total = sum(results[n] for n in dolmino)
        print(f"# Dolmino: {total:,} tokens ({total / 1e9:.2f}B)")
    if all(n in results for n in sft):
        total = sum(results[n] for n in sft)
        print(f"# SFT: {total:,} tokens ({total / 1e9:.2f}B)")

    if errors:
        print()
        print("=" * 80)
        print("ERRORS:")
        print("=" * 80)
        for name, error in errors:
            print(f"  {name}: {error}")


if __name__ == "__main__":
    main()
