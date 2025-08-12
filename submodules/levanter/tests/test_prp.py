import jax.numpy as jnp
import jax.random as jrandom
import numpy
import numpy as np
import pytest

from levanter.data._prp import FeistelPermutation, LcgPermutation


def test_permutation_creates_valid_instance():
    length = 100
    prng_key = jrandom.PRNGKey(0)
    permutation = LcgPermutation(length, prng_key)
    assert permutation.length == length
    assert 0 < permutation.a < length
    assert 0 <= permutation.b < length


@pytest.mark.parametrize("PermutationClass", [LcgPermutation, FeistelPermutation])
def test_permutation_with_single_index_returns_correct_value(PermutationClass):
    length = 10
    prng_key = jrandom.PRNGKey(0)
    permutation = PermutationClass(length, prng_key)
    index = 5
    result = permutation(index)
    assert isinstance(result, int)
    assert result != index  # In most cases, result should not equal the input for a permutation


@pytest.mark.parametrize("PermutationClass", [LcgPermutation, FeistelPermutation])
def test_permutation_with_array_returns_correct_values(PermutationClass):
    length = 10
    prng_key = jrandom.PRNGKey(0)
    permutation = PermutationClass(length, prng_key)
    indices = jnp.arange(length)
    results = permutation(indices)
    assert isinstance(results, numpy.ndarray)
    assert len(results) == length
    assert jnp.sum(results == indices) <= 2


@pytest.mark.parametrize("PermutationClass", [LcgPermutation, FeistelPermutation])
def test_permutation_is_bijective_over_full_range(PermutationClass):
    length = 10
    prng_key = jrandom.PRNGKey(0)
    permutation = PermutationClass(length, prng_key)
    indices = jnp.arange(length)
    permuted = permutation(indices)
    # Check if all elements are unique, which is a necessary condition for a bijective function
    assert len(jnp.unique(permuted)) == length


@pytest.mark.parametrize("PermutationClass", [LcgPermutation, FeistelPermutation])
def test_permutation_handles_edge_case_length_one(PermutationClass):
    length = 1
    prng_key = jrandom.PRNGKey(0)
    permutation = PermutationClass(length, prng_key)
    result = permutation(0)
    assert result == 0  # With length 1, the only valid output is the input


@pytest.mark.parametrize("PermutationClass", [LcgPermutation, FeistelPermutation])
def test_permutation_rejects_invalid_indices(PermutationClass):
    length = 10
    prng_key = jrandom.PRNGKey(0)
    permutation = PermutationClass(length, prng_key)
    with pytest.raises(IndexError):
        permutation(-1)  # Test negative index
    with pytest.raises(IndexError):
        permutation(length)  # Test index equal to length


@pytest.mark.parametrize("PermutationClass", [LcgPermutation, FeistelPermutation])
def test_permutation_is_deterministic(PermutationClass):
    length = 4
    prng_key = jrandom.PRNGKey(0)
    permutation = PermutationClass(length, prng_key)
    indices = np.arange(length, dtype=np.uint64)
    results = permutation(indices)
    prng_key = jrandom.PRNGKey(0)
    permutation = PermutationClass(length, prng_key)
    results2 = permutation(indices)
    assert jnp.all(results == results2)


@pytest.mark.parametrize("PermutationClass", [LcgPermutation, FeistelPermutation])
def test_permutation_is_deterministic1(PermutationClass):
    length = 4
    prng_key = jrandom.PRNGKey(1)
    permutation = PermutationClass(length, prng_key)
    indices = jnp.arange(length)
    results = permutation(indices)
    prng_key = jrandom.PRNGKey(1)
    permutation = PermutationClass(length, prng_key)
    results2 = permutation(indices)
    assert jnp.all(results == results2)


@pytest.mark.parametrize("PermutationClass", [LcgPermutation, FeistelPermutation])
def test_permutation_handles_large_length_no_overflow(PermutationClass):
    large_length = 2**34
    prng_key = jrandom.PRNGKey(0)
    permutation = PermutationClass(large_length, prng_key)
    index = 2**32  # A large index within the range
    result = permutation(index)
    assert isinstance(result, int)
    assert 0 <= result < large_length


@pytest.mark.parametrize("PermutationClass", [LcgPermutation, FeistelPermutation])
@pytest.mark.parametrize("dtype", [np.uint16, np.uint32, np.uint64])
def test_handles_reasonable_dtypes(PermutationClass, dtype):
    length = 31000
    prng_key = jrandom.PRNGKey(0)
    permutation = PermutationClass(length, prng_key)
    index = np.arange(length, dtype=dtype)
    result = permutation(index)
    assert isinstance(result, np.ndarray)
    assert len(result) == length
    # check it's a permutation
    sorted = np.sort(result)
    assert np.all(sorted == np.arange(length, dtype=dtype))
