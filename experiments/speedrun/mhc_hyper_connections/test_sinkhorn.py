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

"""Tests for the Sinkhorn normalization in mHC."""

import haliax as hax
import jax.numpy as jnp
import jax.random as jrandom
import pytest
from haliax import Axis

from experiments.speedrun.mhc_hyper_connections.main import _sinkhorn_log


def _make_logits(key, n: int):
    """Create test logits as a NamedArray."""
    StreamOut = Axis("stream_out", n)
    StreamIn = Axis("stream_in", n)
    raw = jrandom.normal(key, (n, n))
    return hax.named(raw, (StreamOut, StreamIn)), StreamOut, StreamIn


class TestSinkhornDoublyStochastic:
    """Test that Sinkhorn produces approximately doubly stochastic matrices."""

    def test_sinkhorn_rows_sum_to_one(self):
        """After Sinkhorn, row sums (output mixing weights) should be exact."""
        key = jrandom.PRNGKey(42)
        logits, StreamOut, StreamIn = _make_logits(key, 4)
        result = _sinkhorn_log(logits, StreamOut, StreamIn, num_iters=20, tau=0.05)
        row_sums = hax.sum(result, axis=StreamIn)
        # Row sums are exact (we end with row normalization)
        assert jnp.allclose(row_sums.array, 1.0, atol=1e-5), f"Row sums: {row_sums.array}"

    def test_sinkhorn_cols_sum_to_one(self):
        """After Sinkhorn, column sums should be approximately 1."""
        key = jrandom.PRNGKey(42)
        logits, StreamOut, StreamIn = _make_logits(key, 4)
        result = _sinkhorn_log(logits, StreamOut, StreamIn, num_iters=20, tau=0.05)
        col_sums = hax.sum(result, axis=StreamOut)
        # Column sums can have more variance - within 15%
        assert jnp.allclose(col_sums.array, 1.0, atol=0.15), f"Column sums: {col_sums.array}"

    def test_sinkhorn_doubly_stochastic_various_sizes(self):
        """Test doubly stochastic property for various matrix sizes.

        Uses enough iterations to achieve good convergence for each size.
        The mHC paper uses n=4 streams with 20 iterations.
        """
        key = jrandom.PRNGKey(123)
        # Scale iterations with matrix size for reasonable convergence
        for n, num_iters in [(2, 20), (3, 20), (4, 20), (8, 50), (16, 100)]:
            key, subkey = jrandom.split(key)
            logits, StreamOut, StreamIn = _make_logits(subkey, n)
            result = _sinkhorn_log(logits, StreamOut, StreamIn, num_iters=num_iters, tau=0.05)

            row_sums = hax.sum(result, axis=StreamIn).array
            col_sums = hax.sum(result, axis=StreamOut).array

            # Row sums are exact (we end with row normalization), column sums approximate
            assert jnp.allclose(row_sums, 1.0, atol=1e-5), f"n={n}: Row sums not exact: {row_sums}"
            assert jnp.allclose(col_sums, 1.0, atol=0.25), f"n={n}: Col sums not ~1: {col_sums}"

    def test_sinkhorn_non_negative(self):
        """Sinkhorn output should be non-negative."""
        key = jrandom.PRNGKey(999)
        logits, StreamOut, StreamIn = _make_logits(key, 4)
        result = _sinkhorn_log(logits, StreamOut, StreamIn, num_iters=20, tau=0.05)
        assert jnp.all(result.array >= 0), f"Negative values found: {result.array}"

    def test_sinkhorn_convergence_with_iterations(self):
        """More iterations should improve column sum convergence (rows are already exact)."""
        key = jrandom.PRNGKey(42)
        logits, StreamOut, StreamIn = _make_logits(key, 4)

        # Row sums are exact after the loop (we end with row normalization)
        # So test column sum convergence instead
        result_5 = _sinkhorn_log(logits, StreamOut, StreamIn, num_iters=5, tau=0.05)
        col_error_5 = jnp.abs(hax.sum(result_5, axis=StreamOut).array - 1.0).max()

        result_50 = _sinkhorn_log(logits, StreamOut, StreamIn, num_iters=50, tau=0.05)
        col_error_50 = jnp.abs(hax.sum(result_50, axis=StreamOut).array - 1.0).max()

        assert col_error_50 <= col_error_5, f"More iterations should reduce col error: {col_error_5} vs {col_error_50}"

    def test_sinkhorn_high_iterations_nearly_exact(self):
        """With many iterations, both row and column sums should be nearly exact."""
        key = jrandom.PRNGKey(42)
        logits, StreamOut, StreamIn = _make_logits(key, 4)
        result = _sinkhorn_log(logits, StreamOut, StreamIn, num_iters=100, tau=0.05)

        row_sums = hax.sum(result, axis=StreamIn).array
        col_sums = hax.sum(result, axis=StreamOut).array

        # Row sums are always exact, column sums converge with iterations
        assert jnp.allclose(row_sums, 1.0, atol=1e-5), f"Row sums: {row_sums}"
        assert jnp.allclose(col_sums, 1.0, atol=0.01), f"Col sums: {col_sums}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
