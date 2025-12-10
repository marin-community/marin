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

from functools import cache
from random import Random

from dupekit import hash_xxh3_128

# NOTE: adapted from https://stackoverflow.com/a/52534632, the purpose is not to
# be fast, but rather to have a baseline for the rust implementation. We could not
# use use datasketch because of numpy dependency issues.


def minhash(s: set[str], vector_length: int = 128, prime=4294967311) -> list[int]:
    """
    Given a set `s`, pass each member of the set through all permutation functions,
    and set the `ith` position of `vec` to the `ith` permutation function's output if
    that output is smaller than `vec[i]`.

    Args:
        s: set of strings to compute the minhash for
        vector_length: length of the minhash vector

        prime: a large prime number for the hash functions

    Returns:
        A list of ints representing the minhash of the set `s`.
    """
    # initialize a minhash of length N with positive infinity values
    vec = [float("inf") for i in range(vector_length)]

    for val_str in s:
        val = hash_xxh3_128(val_str.encode())

        # loop over each "permutation function"
        for perm_idx, perm_vals in enumerate(_minhash_perms(vector_length)):
            a, b = perm_vals

            # pass `val` through the `ith` permutation function
            output = (a * val + b) % prime

            # conditionally update the `ith` value of vec
            if vec[perm_idx] > output:
                vec[perm_idx] = output

    # the returned vector represents the minimum hash of the set s
    return vec  # type: ignore[bad-return]


@cache
def _minhash_perms(vector_length: int, max_val: int = (2**32) - 1, seed: int = 42) -> list[tuple[int, int]]:
    """
    Generate permutation coefficients for MinHash.

    Args:
        vector_length: Number of permutation functions to generate
        max_val: Maximum value for random coefficients
        seed: Random seed for reproducibility (default: 42)

    Returns:
        List of (a, b) coefficient tuples for permutation functions
    """
    rng = Random(seed)
    return [(rng.randint(0, max_val), rng.randint(0, max_val)) for _ in range(vector_length)]
