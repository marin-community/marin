# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

<<<<<<< HEAD
import tempfile

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file

from levanter.compat.fsspec_safetensor import load_tensor_dict


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", ["float32", "int8", "uint64"])
async def test_various_dtypes(tmp_path, dtype):
    data = {
        "x": (
            np.random.randint(0, 100, (4, 5)).astype(dtype) if "int" in dtype else np.random.randn(4, 5).astype(dtype)
        ),
    }

    local_path = tmp_path / f"test_{dtype}.safetensors"
    save_file(data, local_path)

    ref = load_file(str(local_path))
    uri = f"file://{local_path}"
    virtual = await load_tensor_dict(uri)

    for key in ref:
        np.testing.assert_array_equal(await virtual[key].read(), ref[key])


@pytest.mark.asyncio
async def test_bfloat16_support(tmp_path):
    data = {"bf": np.random.randn(4, 4).astype(np.dtype("bfloat16"))}
    path = tmp_path / "bf16.safetensors"
    save_file(data, path)
    ref = load_file(str(path))
    virtual = await load_tensor_dict(f"file://{path}")

    np.testing.assert_array_equal(await virtual["bf"].read(), ref["bf"])


@pytest.mark.asyncio
async def test_memory_filesystem():
    import fsspec

    # Create sample file in memory
    data = {"mem": np.random.randn(2, 3).astype(np.float32)}

    with tempfile.NamedTemporaryFile() as f:
        save_file(data, f.name)
        serialized = f.read()

    fs = fsspec.filesystem("memory")
    with fs.open("/test.safetensors", "wb") as f:
        f.write(serialized)

    # Use file URI
    virtual = await load_tensor_dict("memory://test.safetensors")
    ref = data["mem"]
    result = await virtual["mem"].read()
    np.testing.assert_array_equal(result, ref)


@pytest.mark.asyncio
async def test_various_keys_in_one_file(tmp_path):
    data = {
        "x": np.random.randn(4, 5).astype(np.float32),
        "y": np.random.randn(5, 8).astype(np.float32),
        "z": np.random.randn(4, 5).astype(np.int32),
    }

    path = tmp_path / "test.safetensors"
    save_file(data, path)

    ref = load_file(str(path))
    virtual = await load_tensor_dict(f"file://{path}")
    for key in ref:
        np.testing.assert_array_equal(await virtual[key].read(), ref[key])
        np.testing.assert_array_equal(await virtual[key].read(), data[key])


@pytest.mark.asyncio
async def test_virtual_slicing(tmp_path):
    data = {"slice": np.arange(100, dtype=np.int32).reshape(10, 10)}

    path = tmp_path / "slice.safetensors"
    save_file(data, path)
    ref = load_file(str(path))
    virtual = await load_tensor_dict(f"file://{path}")

    # Read a slice
    ts_arr = virtual["slice"]
    sliced = await ts_arr[2:5, 4:7].read()
    expected = ref["slice"][2:5, 4:7]

    np.testing.assert_array_equal(sliced, expected)


# try using gcs
@pytest.mark.asyncio
async def test_gcs(tmp_path):
    import fsspec

    data = {
        "x": np.random.randn(4, 5).astype(np.float32),
        "y": np.random.randn(5, 8).astype(np.float32),
        "z": np.random.randn(4, 5).astype(np.int32),
    }

    local_path = str(tmp_path / "various.safetensors")
    save_file(data, local_path)

    test_data = "gs://levanter-data/test/various.safetensors"

    fs = fsspec.filesystem("gcs")
    try:
        if not fs.exists(test_data):
            fs.put(local_path, test_data)
    except Exception:
        pytest.skip("No test data found")

    virtual = await load_tensor_dict(test_data)
    ref = load_file(local_path)

    for key in ref:
        np.testing.assert_array_equal(await virtual[key].read(), ref[key])


@pytest.mark.asyncio
async def test_strided_reads(tmp_path):
    data = {"mat": np.arange(100, dtype=np.float32).reshape(10, 10)}

    path = tmp_path / "weird_strides.safetensors"
    save_file(data, path)
    ref = load_file(str(path))
    virtual = await load_tensor_dict(f"file://{path}")
    ts_arr = virtual["mat"]

    # Normal read for sanity
    np.testing.assert_array_equal(await ts_arr.read(), ref["mat"])

    # Transpose
    expected = ref["mat"].T
    actual = await ts_arr.transpose([1, 0]).read()
    np.testing.assert_array_equal(actual, expected)

    # Step slicing (every other column)
    expected = ref["mat"][:, ::2]
    actual = await ts_arr[:, ::2].read()
    np.testing.assert_array_equal(actual, expected)

    # Reversed rows
    expected = ref["mat"][::-1]
    actual = await ts_arr[::-1].read()
    np.testing.assert_array_equal(actual, expected)

    # Slice + transpose + step
    expected = ref["mat"][2:8, ::-2].T
    actual = await ts_arr[2:8, ::-2].transpose([1, 0]).read()
    np.testing.assert_array_equal(actual, expected)
=======
import numpy as np
import pytest
import jax.numpy as jnp
from fsspec.asyn import AsyncFileSystem
from safetensors.numpy import load_file, save_file
from levanter.compat.fsspec_safetensor import (
    read_safetensors_fsspec,
)
from levanter.compat.hf_checkpoints import HFCheckpointConverter


class _InMemoryAsyncFS(AsyncFileSystem):
    def __init__(self, payload: bytes):
        super().__init__()
        self._payload = payload

    async def _cat_file(self, path: str, start: int | None = None, end: int | None = None, **_) -> bytes:
        start = 0 if start is None else start
        end = len(self._payload) if end is None else end
        return self._payload[start:end]


@pytest.mark.asyncio
async def test_read_safetensors_fsspec_roundtrip(tmp_path):
    data = {
        "foo": np.arange(12, dtype=np.float32).reshape(3, 4),
        "bar": (np.random.randn(2, 3) * 5).astype(np.float32),
        "baz": np.arange(6, dtype=np.int32),
    }
    path = tmp_path / "roundtrip.safetensors"
    save_file(data, path)

    arrays = await read_safetensors_fsspec(
        f"file://{path}",
        dtype_override=jnp.float16,
        sharding_fn=lambda _: None,
    )

    assert set(arrays.keys()) == set(data.keys())
    np.testing.assert_array_equal(np.asarray(arrays["baz"]), data["baz"])
    assert arrays["foo"].dtype == jnp.float16
    np.testing.assert_allclose(np.asarray(arrays["foo"]), data["foo"].astype(np.float16))


def test_load_from_remote_file_url(tmp_path, monkeypatch):
    data = {
        "foo": np.random.randn(4, 4).astype(np.float32),
        "bar": np.random.randn(3, 2).astype(np.float32),
    }
    path = tmp_path / "model.safetensors"
    save_file(data, path)

    expected = load_file(str(path))

    # This monkeypatching offends me but fine

    monkeypatch.setattr("levanter.compat.hf_checkpoints.best_effort_sharding", (lambda shape, mesh: None))

    def _jit_stub(fn, *args, **kwargs):
        def _wrapped(x):
            return fn(x)

        return _wrapped

    monkeypatch.setattr("levanter.compat.hf_checkpoints.jax.jit", _jit_stub)
    monkeypatch.setattr("levanter.compat.hf_checkpoints.jax.lax.with_sharding_constraint", lambda x, _: x)

    converter = HFCheckpointConverter.__new__(HFCheckpointConverter)
    converter.__dict__.update(
        {
            "LevConfigClass": None,
            "reference_checkpoint": None,
            "HfConfigClass": None,
            "tokenizer": None,
            "feature_extractor": None,
            "config_overrides": None,
            "trust_remote_code": False,
            "ignore_prefix": None,
        }
    )

    remote_state = converter._load_from_remote(f"file://{tmp_path}", dtype=None)

    assert set(remote_state.keys()) == set(expected.keys())
    for key in expected:
        np.testing.assert_array_equal(np.array(remote_state[key]), expected[key])


@pytest.mark.asyncio
async def test_dtype_override(tmp_path):
    data = {
        "floaty": np.arange(start=0, stop=1, step=0.1, dtype=np.float32),
        "ints": np.arange(6, dtype=np.int32).reshape(2, 3),
    }
    path = tmp_path / "dtype.safetensors"
    save_file(data, path)

    tensors = await read_safetensors_fsspec(
        f"file://{path}",
        dtype_override=jnp.bfloat16,
        sharding_fn=lambda _: None,
    )

    assert tensors["floaty"].dtype == jnp.bfloat16
    np.testing.assert_allclose(
        np.asarray(tensors["floaty"], dtype=np.float32),
        data["floaty"].astype(np.float32),
        rtol=1e-3,
        atol=1e-3,
    )
    assert tensors["ints"].dtype == jnp.int32
    np.testing.assert_array_equal(np.asarray(tensors["ints"]), data["ints"])
>>>>>>> origin/main
