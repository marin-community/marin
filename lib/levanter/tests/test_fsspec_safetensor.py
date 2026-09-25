# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import struct
import threading

import numpy as np
import pytest
import jax.numpy as jnp
from fsspec.asyn import AsyncFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from safetensors.numpy import load_file, save, save_file
from levanter.compat.fsspec_safetensor import (
    StagingByteBudget,
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


def test_load_sharded_remote_reads_overlap(monkeypatch):
    first = np.arange(8, dtype=np.float32)
    second = np.arange(8, dtype=np.float32) + 10
    shard_bytes = {"first.safetensors": save({"first": first}), "second.safetensors": save({"second": second})}
    rendezvous = threading.Barrier(2, timeout=5)

    class ObservedFS(MemoryFileSystem):
        def cat_file(self, path, start=None, end=None, **kwargs):
            if path.endswith(".safetensors") and start is not None:
                name = path.rsplit("/", 1)[-1]
                data_start = 8 + struct.unpack("<Q", shard_bytes[name][:8])[0]
                if start >= data_start:
                    rendezvous.wait()
            return super().cat_file(path, start=start, end=end, **kwargs)

    fs = ObservedFS()
    for name, payload in shard_bytes.items():
        fs.pipe(f"/model/{name}", payload)
    fs.pipe(
        "/model/model.safetensors.index.json",
        json.dumps({"weight_map": {"first": "first.safetensors", "second": "second.safetensors"}}).encode(),
    )

    monkeypatch.setattr("levanter.compat.hf_checkpoints.url_to_fs", lambda _: (fs, "/model"))
    monkeypatch.setattr("levanter.compat.hf_checkpoints.best_effort_sharding", lambda shape, mesh: None)

    converter = HFCheckpointConverter.__new__(HFCheckpointConverter)
    state = converter._load_from_remote("memory://model")

    np.testing.assert_array_equal(np.asarray(state["first"]), first)
    np.testing.assert_array_equal(np.asarray(state["second"]), second)


@pytest.mark.asyncio
async def test_safetensors_reads_respect_shared_staging_budget(monkeypatch):
    payload = save({"first": np.arange(32, dtype=np.float32), "second": np.arange(32, dtype=np.float32)})
    data_start = 8 + struct.unpack("<Q", payload[:8])[0]
    monkeypatch.setattr("levanter.compat.fsspec_safetensor.DEFAULT_CHUNK_SIZE_BYTES", 128)
    monkeypatch.setattr("levanter.compat.fsspec_safetensor.MAX_CONCURRENT_CHUNKS", 2)

    class ObservedFS(AsyncFileSystem):
        def __init__(self):
            super().__init__()
            self.active = 0
            self.peak = 0

        async def _cat_file(self, path, start=None, end=None, **kwargs):
            if start is not None and start >= data_start:
                self.active += 1
                self.peak = max(self.peak, self.active)
                await asyncio.sleep(0)
                self.active -= 1
            return payload[start:end]

        async def _size(self, path):
            return len(payload)

    fs = ObservedFS()
    arrays = await read_safetensors_fsspec("memory://model", fs=fs, staging_budget=StagingByteBudget(256))

    assert fs.peak == 1
    np.testing.assert_array_equal(np.asarray(arrays["first"]), np.arange(32, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(arrays["second"]), np.arange(32, dtype=np.float32))

    fs_without_pressure = ObservedFS()
    await read_safetensors_fsspec("memory://model", fs=fs_without_pressure, staging_budget=StagingByteBudget(512))
    assert fs_without_pressure.peak == 2


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
