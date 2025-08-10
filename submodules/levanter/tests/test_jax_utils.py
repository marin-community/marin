import jax
import numpy as np
import pytest

from levanter.utils.jax_utils import best_effort_sharding, create_fsdp_mesh
from test_utils import skip_if_not_enough_devices


def _assert_can_put_with_sharding(array, sharding):
    try:
        jax.device_put(array, sharding)
    except ValueError:
        # assert False, f"Could not put array with shape {array.shape} with sharding {sharding}"
        raise AssertionError(f"Could not put array with shape {array.shape} with sharding {sharding}")


@skip_if_not_enough_devices(8)
def test_best_effort_sharding():
    if len(jax.devices()) % 8 != 0:
        pytest.skip("Not enough devices")
    # 1D array, 8 devices
    array = np.arange(8)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(2, 4)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(4, 2)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(8, 1)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(1, 8)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(2, 2, 2)
    sharding = best_effort_sharding(array.shape)
    _assert_can_put_with_sharding(array, sharding)

    devices = jax.devices()[:6]

    array = array.reshape(2, 2, 2)
    sharding = best_effort_sharding(array.shape, devices=devices)
    _assert_can_put_with_sharding(array, sharding)


@pytest.mark.parametrize("fsdp_size", [1, 2, 4, 8])
def test_best_effort_sharding_with_mesh(fsdp_size):
    if fsdp_size > len(jax.devices()):
        pytest.skip("Not enough devices")
    elif len(jax.devices()) % fsdp_size != 0:
        pytest.skip("Number of devices is not a multiple of fsdp_size")

    mesh = create_fsdp_mesh(len(jax.devices()) // fsdp_size, fsdp_size, 1)

    array = np.arange(8)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(2, 4)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(4, 2)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(8, 1)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(1, 8)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)

    array = array.reshape(2, 2, 2)
    sharding = best_effort_sharding(array.shape, mesh=mesh)
    _assert_can_put_with_sharding(array, sharding)


def test_tree_broadcast_to_simple():
    from levanter.utils.jax_utils import tree_broadcast_to

    # Test with simple nested dicts
    prefix = {"a": 1, "b": 2}
    target = {"a": {"x": 10, "y": 20}, "b": {"z": 30}}
    result = tree_broadcast_to(prefix, target)

    assert result == {"a": {"x": 1, "y": 1}, "b": {"z": 2}}

    # Test with lists
    prefix = [1, 2]
    target = [[10, 20], [30, 40]]
    result = tree_broadcast_to(prefix, target)

    assert result == [[1, 1], [2, 2]]

    # Test with tuples
    prefix = (1, 2)
    target = ((10, 20), (30, 40))
    result = tree_broadcast_to(prefix, target)

    assert result == ((1, 1), (2, 2))


def test_tree_broadcast_to_mixed_types():
    from levanter.utils.jax_utils import tree_broadcast_to

    # Test with mixed types
    prefix = {"a": 1, "b": 2}
    target = {"a": [10, 20], "b": {"x": 30, "y": 40}}
    result = tree_broadcast_to(prefix, target)

    assert result == {"a": [1, 1], "b": {"x": 2, "y": 2}}


def test_tree_broadcast_to_with_equinox():
    import equinox as eqx
    import jax.numpy as jnp

    from levanter.utils.jax_utils import tree_broadcast_to

    class SimpleModule(eqx.Module):
        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, key):
            self.weight = jnp.ones((2, 2))
            self.bias = jnp.ones(2)

    # Create a module and a target tree
    key = jax.random.PRNGKey(0)
    module = SimpleModule(key)
    target = {"module": module, "other": {"x": 1, "y": 2}}

    # Test broadcasting a scalar
    prefix = 42
    result = tree_broadcast_to(prefix, target)

    assert result["module"].weight == 42
    assert result["module"].bias == 42
    assert result["other"]["x"] == 42
    assert result["other"]["y"] == 42

    # Test broadcasting a dict
    prefix = {"module": 1, "other": 2}
    result = tree_broadcast_to(prefix, target)

    assert result["module"].weight == 1
    assert result["module"].bias == 1
    assert result["other"]["x"] == 2
    assert result["other"]["y"] == 2


def test_tree_broadcast_to_edge_cases():
    from levanter.utils.jax_utils import tree_broadcast_to

    # Test with empty trees
    prefix = None
    target = {"a": {}, "b": {}}
    result = tree_broadcast_to(prefix, target, is_leaf=lambda x: x is None)
    assert result == {"a": {}, "b": {}}

    # Test with None values
    prefix = {"a": None, "b": 2}
    target = {"a": {"x": 10}, "b": {"y": 20}}
    result = tree_broadcast_to(prefix, target, is_leaf=lambda x: x is None)
    assert result == {"a": {"x": None}, "b": {"y": 2}}

    # Test with is_leaf function
    def is_leaf(x):
        return isinstance(x, list)

    prefix = [1, 2]
    target = {"a": [10, 20], "b": [30, 40]}
    result = tree_broadcast_to(prefix, target, is_leaf=is_leaf)
    assert result == {"a": [1, 2], "b": [1, 2]}
