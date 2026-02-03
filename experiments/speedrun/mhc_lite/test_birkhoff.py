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

"""Unit tests for Birkhoff-von Neumann doubly stochastic matrix construction."""

import itertools
import math

import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from haliax import Axis

from experiments.speedrun.mhc_lite.main import _birkhoff_von_neumann, _generate_permutation_matrices


class TestPermutationMatrices:
    """Tests for permutation matrix generation."""

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_correct_count(self, n: int):
        """Should generate exactly n! permutation matrices."""
        perm_matrices = _generate_permutation_matrices(n)
        assert perm_matrices.shape[0] == math.factorial(n)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_correct_shape(self, n: int):
        """Each permutation matrix should be n x n."""
        perm_matrices = _generate_permutation_matrices(n)
        assert perm_matrices.shape == (math.factorial(n), n, n)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_binary_entries(self, n: int):
        """Permutation matrices should only contain 0s and 1s."""
        perm_matrices = _generate_permutation_matrices(n)
        assert jnp.all((perm_matrices == 0) | (perm_matrices == 1))

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_row_sums(self, n: int):
        """Each row should sum to 1."""
        perm_matrices = _generate_permutation_matrices(n)
        row_sums = perm_matrices.sum(axis=2)  # sum over columns
        assert jnp.allclose(row_sums, 1.0)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_column_sums(self, n: int):
        """Each column should sum to 1."""
        perm_matrices = _generate_permutation_matrices(n)
        col_sums = perm_matrices.sum(axis=1)  # sum over rows
        assert jnp.allclose(col_sums, 1.0)

    def test_identity_is_first(self):
        """Identity permutation (0,1,2,...) should be first."""
        n = 4
        perm_matrices = _generate_permutation_matrices(n)
        identity = jnp.eye(n)
        assert jnp.allclose(perm_matrices[0], identity)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_all_unique(self, n: int):
        """All permutation matrices should be unique."""
        perm_matrices = _generate_permutation_matrices(n)
        n_fact = math.factorial(n)
        # Flatten each matrix and check uniqueness
        flat = perm_matrices.reshape(n_fact, -1)
        for i in range(n_fact):
            for j in range(i + 1, n_fact):
                assert not jnp.allclose(flat[i], flat[j]), f"Matrices {i} and {j} are identical"


class TestBirkhoffVonNeumann:
    """Tests for Birkhoff-von Neumann doubly stochastic construction."""

    @pytest.fixture
    def setup_n4(self):
        """Setup for n=4 streams (24 permutations)."""
        n = 4
        n_fact = math.factorial(n)
        perm_matrices = _generate_permutation_matrices(n)
        NumPerms = Axis("num_perms", n_fact)
        StreamOut = Axis("stream_out", n)
        StreamIn = Axis("stream_in", n)
        return perm_matrices, NumPerms, StreamOut, StreamIn, n

    def test_uniform_weights_gives_uniform_matrix(self, setup_n4):
        """Uniform weights should give uniform doubly stochastic matrix."""
        perm_matrices, NumPerms, StreamOut, StreamIn, n = setup_n4
        # Uniform weights (all zeros -> equal softmax)
        weights = hax.zeros((NumPerms,))
        result = _birkhoff_von_neumann(weights, perm_matrices, NumPerms, StreamOut, StreamIn)
        # Result is scaled by n, so divide to get actual DS matrix
        ds_matrix = result.array / n
        # Uniform DS matrix has all entries = 1/n
        expected = jnp.ones((n, n)) / n
        assert jnp.allclose(ds_matrix, expected, atol=1e-6)

    def test_single_permutation_weight(self, setup_n4):
        """Strongly weighting one permutation should approximate that permutation."""
        perm_matrices, NumPerms, StreamOut, StreamIn, n = setup_n4
        n_fact = math.factorial(n)
        # Put all weight on permutation index 5
        target_idx = 5
        weights_arr = jnp.full((n_fact,), -100.0)
        weights_arr = weights_arr.at[target_idx].set(100.0)
        weights = hax.named(weights_arr, (NumPerms,))
        result = _birkhoff_von_neumann(weights, perm_matrices, NumPerms, StreamOut, StreamIn)
        ds_matrix = result.array / n
        expected = perm_matrices[target_idx]
        assert jnp.allclose(ds_matrix, expected, atol=1e-6)

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 123])
    def test_row_sums_to_one(self, setup_n4, seed: int):
        """Rows of doubly stochastic matrix should sum to 1."""
        perm_matrices, NumPerms, StreamOut, StreamIn, n = setup_n4
        key = jrandom.PRNGKey(seed)
        weights_arr = jrandom.normal(key, (math.factorial(n),))
        weights = hax.named(weights_arr, (NumPerms,))
        result = _birkhoff_von_neumann(weights, perm_matrices, NumPerms, StreamOut, StreamIn)
        ds_matrix = result.array / n
        row_sums = ds_matrix.sum(axis=1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-6), f"Row sums: {row_sums}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 123])
    def test_column_sums_to_one(self, setup_n4, seed: int):
        """Columns of doubly stochastic matrix should sum to 1."""
        perm_matrices, NumPerms, StreamOut, StreamIn, n = setup_n4
        key = jrandom.PRNGKey(seed)
        weights_arr = jrandom.normal(key, (math.factorial(n),))
        weights = hax.named(weights_arr, (NumPerms,))
        result = _birkhoff_von_neumann(weights, perm_matrices, NumPerms, StreamOut, StreamIn)
        ds_matrix = result.array / n
        col_sums = ds_matrix.sum(axis=0)
        assert jnp.allclose(col_sums, 1.0, atol=1e-6), f"Column sums: {col_sums}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 123])
    def test_non_negative(self, setup_n4, seed: int):
        """All entries should be non-negative."""
        perm_matrices, NumPerms, StreamOut, StreamIn, n = setup_n4
        key = jrandom.PRNGKey(seed)
        weights_arr = jrandom.normal(key, (math.factorial(n),))
        weights = hax.named(weights_arr, (NumPerms,))
        result = _birkhoff_von_neumann(weights, perm_matrices, NumPerms, StreamOut, StreamIn)
        ds_matrix = result.array / n
        assert jnp.all(ds_matrix >= -1e-10), f"Min value: {ds_matrix.min()}"

    @pytest.mark.parametrize("seed", [0, 1, 2, 42, 123])
    def test_max_one(self, setup_n4, seed: int):
        """All entries should be at most 1."""
        perm_matrices, NumPerms, StreamOut, StreamIn, n = setup_n4
        key = jrandom.PRNGKey(seed)
        weights_arr = jrandom.normal(key, (math.factorial(n),))
        weights = hax.named(weights_arr, (NumPerms,))
        result = _birkhoff_von_neumann(weights, perm_matrices, NumPerms, StreamOut, StreamIn)
        ds_matrix = result.array / n
        assert jnp.all(ds_matrix <= 1.0 + 1e-10), f"Max value: {ds_matrix.max()}"

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_different_sizes(self, n: int):
        """Should work for different stream counts."""
        n_fact = math.factorial(n)
        perm_matrices = _generate_permutation_matrices(n)
        NumPerms = Axis("num_perms", n_fact)
        StreamOut = Axis("stream_out", n)
        StreamIn = Axis("stream_in", n)
        key = jrandom.PRNGKey(0)
        weights_arr = jrandom.normal(key, (n_fact,))
        weights = hax.named(weights_arr, (NumPerms,))
        result = _birkhoff_von_neumann(weights, perm_matrices, NumPerms, StreamOut, StreamIn)
        ds_matrix = result.array / n
        # Check doubly stochastic properties
        assert jnp.allclose(ds_matrix.sum(axis=0), 1.0, atol=1e-6)
        assert jnp.allclose(ds_matrix.sum(axis=1), 1.0, atol=1e-6)
        assert jnp.all(ds_matrix >= -1e-10)
        assert jnp.all(ds_matrix <= 1.0 + 1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
