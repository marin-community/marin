# Copyright 2025 The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import dslice

import haliax as hax


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


def test_named_ref_basic_get_set(key):
    X = hax.Axis("x", 4)
    Y = hax.Axis("y", 3)
    array = hax.random.uniform(key, (X, Y))
    ref = hax.new_ref(array)

    assert ref.shape == array.shape
    assert ref.axes == array.axes

    slice_val = ref[{"x": slice(1, 3)}]
    assert slice_val.axes == (X.resize(2), Y)

    new_block = hax.ones_like(slice_val) * 2.0
    ref[{"x": slice(1, 3)}] = new_block
    updated = ref.value()
    assert jnp.allclose(updated[{"x": slice(1, 3)}].array, jnp.asarray(new_block.array))


def test_named_ref_scalar_update():
    X = hax.Axis("x", 5)
    ref = hax.new_ref(hax.zeros(X))
    ref[{"x": 2}] = 3.14
    assert pytest.approx(ref.value()[{"x": 2}].array.item()) == 3.14


def test_slice_ref_composition():
    Layer = hax.Axis("layer", 6)
    Head = hax.Axis("head", 8)
    cache = hax.zeros((Layer, Head))
    cache_ref = hax.new_ref(cache)

    layers_1_4 = cache_ref.slice({"layer": slice(1, 4)})
    assert layers_1_4.axes == (Layer.resize(3), Head)

    layers_1_4[{"layer": 0}] = hax.arange(Head).astype(jnp.float32)
    value = cache_ref.value()
    expected_row = jnp.arange(Head.size, dtype=jnp.float32)
    assert jnp.allclose(value[{"layer": 1}].array, expected_row)

    single_layer = layers_1_4.slice({"layer": 0})
    assert single_layer.axes == (Head,)
    single_layer[{"head": 0}] = 7.0
    updated = cache_ref.value()
    assert updated[{"layer": 1, "head": 0}].array == pytest.approx(7.0)


def test_slice_ref_with_dslice():
    Layer = hax.Axis("layer", 10)
    Head = hax.Axis("head", 4)
    cache_ref = hax.new_ref(hax.zeros((Layer, Head)))

    sub = cache_ref.slice({"layer": hax.ds(3, 3)})
    assert sub.axes == (Layer.resize(3), Head)

    sub[{"layer": 1, "head": slice(None)}] = hax.ones(Head)

    value = cache_ref.value()
    assert jnp.allclose(value[{"layer": 4}].array, jnp.ones((Head.size,), dtype=jnp.float32))
    assert isinstance(sub._prefix[0], type(dslice(0, 1)))


def test_dslice_combination_branches():
    X = hax.Axis("x", 10)
    base = hax.new_ref(hax.arange(X))

    ds_ref = base.slice({"x": hax.ds(2, 5)})

    ds_plus_slice = ds_ref.slice({"x": slice(1, 3)})
    prefix = ds_plus_slice._prefix[0]
    assert isinstance(prefix, type(dslice(0, 1)))
    assert prefix.start == 3 and prefix.size == 2

    ds_plus_ds = ds_ref.slice({"x": hax.ds(1, 2)})
    prefix = ds_plus_ds._prefix[0]
    assert prefix.start == 3 and prefix.size == 2

    single = ds_ref.slice({"x": 1})
    assert single._prefix[0] == 3

    neg = ds_ref.slice({"x": -1})
    assert neg._prefix[0] == 6

    list_ref = ds_ref.slice({"x": [0, 2]})
    assert list_ref._prefix[0] == [2, 4]

    array_ref = ds_ref.slice({"x": jnp.array([0, 1], dtype=jnp.int32)})
    assert jnp.array_equal(array_ref._prefix[0], jnp.array([2, 3]))

    try:
        array_reslice = array_ref.slice({"x": jnp.array([0], dtype=jnp.int32)})
        assert array_reslice._prefix[0] == jnp.array([2], dtype=jnp.int32)
    except NotImplementedError:
        # Advanced indexing on advanced indexing is not supported yet.
        pass

    Sel = hax.Axis("sel", 2)
    named_idx = hax.arange(Sel).astype(jnp.int32)
    named_ref = ds_ref.slice({"x": named_idx})
    assert isinstance(named_ref._prefix[0], hax.NamedArray)
    assert jnp.array_equal(named_ref._prefix[0].array, jnp.array([2, 3]))

    try:
        named_reslice = named_ref.slice({"x": hax.NamedArray(jnp.array([1], dtype=jnp.int32), (Sel.resize(1),))})
        assert isinstance(named_reslice._prefix[0], hax.NamedArray)
        assert jnp.array_equal(named_reslice._prefix[0].array, jnp.array([3]))
    except NotImplementedError:
        # Advanced indexing on advanced indexing is not supported yet.
        pass

    strided = base.slice({"x": slice(1, 9, 2)})
    strided_sub = strided.slice({"x": slice(1, 3)})
    strided_prefix = strided_sub._prefix[0]
    assert (strided_prefix.start, strided_prefix.size, strided_prefix.stride) == (3, 2, 2)

    strided_list = strided.slice({"x": [0, 1]})
    assert strided_list._prefix[0] == [1, 3]

    strided_array = strided.slice({"x": jnp.array([0, 1], dtype=jnp.int32)})
    assert jnp.array_equal(strided_array._prefix[0], jnp.array([1, 3]))

    Sel2 = hax.Axis("sel2", 2)
    named_idx2 = hax.arange(Sel2).astype(jnp.int32)
    strided_named = strided.slice({"x": named_idx2})
    assert isinstance(strided_named._prefix[0], hax.NamedArray)
    assert jnp.array_equal(strided_named._prefix[0].array, jnp.array([1, 3]))


def test_slice_value_and_unsliced():
    X = hax.Axis("x", 6)
    Y = hax.Axis("y", 3)
    base = hax.new_ref(hax.arange((X, Y)))

    slice_ref = base.slice({"x": slice(2, 5)})
    slice_value = slice_ref.value()
    expected = base.value()[{"x": slice(2, 5)}]
    assert jnp.allclose(slice_value.array, expected.array)
    assert slice_value.axes == expected.axes

    slice_ref[{"x": 0}] = hax.zeros(Y).astype(slice_ref.dtype)
    assert jnp.allclose(base.value()[{"x": 2}].array, hax.zeros(Y).array)

    unsliced = slice_ref.unsliced()
    assert unsliced is not slice_ref
    assert unsliced.axes == base.axes
    assert jnp.allclose(unsliced.value().array, base.value().array)
    assert unsliced._prefix == tuple(slice(None) for _ in base.axes)
    assert unsliced._ref is base._ref


def test_freeze_returns_named_array():
    X = hax.Axis("x", 3)
    ref = hax.new_ref(hax.arange(X))
    frozen = hax.freeze(ref)
    assert isinstance(frozen, hax.NamedArray)
    assert frozen.axes == ref.axes
    assert jnp.allclose(frozen.array, ref.value().array)


def test_swap_returns_previous_value():
    X = hax.Axis("x", 4)
    ref = hax.new_ref(hax.zeros(X))
    prev = hax.swap(ref, {"x": slice(1, 3)}, hax.ones(X.resize(2)))
    assert isinstance(prev, hax.NamedArray)
    assert prev.axes == (X.resize(2),)
    assert jnp.allclose(prev.array, 0.0)
    assert jnp.allclose(ref.value()[{"x": slice(1, 3)}].array, jnp.ones((2,), dtype=jnp.float32))


def test_named_ref_jit_plumbing():
    X = hax.Axis("x", 5)
    ref = hax.new_ref(hax.zeros(X))

    @jax.jit
    def write_and_read(ref):
        ref[{"x": 1}] = 4.2
        return ref[{"x": 1}]

    out = write_and_read(ref)
    assert isinstance(out, hax.NamedArray)
    assert out.axes == ()
    assert pytest.approx(out.array.item()) == 4.2
    assert jnp.allclose(ref.value()[{"x": 1}].array, jnp.asarray(4.2))


def test_named_ref_is_pytree_leaf():
    X = hax.Axis("x", 3)
    ref = hax.new_ref(hax.zeros(X))

    leaves = jax.tree_util.tree_leaves(ref)
    assert len(leaves) == 1
    assert leaves[0] is ref._ref

    structure = jax.tree_util.tree_structure(ref)
    rebuilt = jax.tree_util.tree_unflatten(structure, leaves)

    assert isinstance(rebuilt, hax.NamedRef)
    assert rebuilt.axes == ref.axes
    assert rebuilt._prefix == ref._prefix


def test_with_scan():
    X = hax.Axis("x", 5)
    ref = hax.new_ref(hax.zeros(X))

    @jax.jit
    def foo(ref, xs):
        def scan_fn(_, x):
            ref_slice = ref.slice({"x": x})
            ref_slice[...] = (x * x).astype(ref_slice.dtype)
            return None, x * 2

        return hax.scan(scan_fn, X)(None, xs)[1]

    out = foo(ref, jnp.arange(X.size))

    assert jnp.all(ref.value().array == jnp.arange(X.size) ** 2)

    assert jnp.all(out == jnp.arange(X.size) * 2)


def test_grad_scan():
    X = hax.Axis("x", 4)

    def f(x):
        x_ref = hax.new_ref(hax.zeros(X))

        def scan_fn(_, i):
            slice = x_ref.slice({"x": i})
            slice[...] = jnp.sin(x * i)
            return None, None

        hax.scan(scan_fn, X)(None, jnp.arange(X.size))
        return x_ref[...].sum().scalar()

    df = jax.grad(f)(1.0)
    assert pytest.approx(df) == jnp.sum(jnp.cos(jnp.arange(X.size)) * jnp.arange(X.size))
