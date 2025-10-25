# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import glob
import os
from functools import reduce
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar
from contextlib import contextmanager

import draccus
import equinox as eqx
import jax
from jax.sharding import Mesh
import numpy as np
import pytest
from chex import assert_trees_all_close
from equinox import nn as nn
from jax._src.random import PRNGKey
from transformers import AutoConfig, BatchEncoding

import haliax as hax
from haliax.partitioning import ResourceAxis, set_mesh

from levanter.checkpoint import _get_fs_and_plain_path
from levanter.data._preprocessor import BatchProcessor
from levanter.data.sharded_datasource import ShardedDataSource
from levanter.layers.attention import AttentionMask


T = TypeVar("T")


def skip_if_not_enough_devices(count: int):
    return pytest.mark.skipif(len(jax.devices()) < count, reason=f"Not enough devices ({len(jax.devices())})")


class MLP(eqx.Module):
    """slightly less annoying MLP, used for testing purposes"""

    layers: List[nn.Linear]
    activation: Callable = eqx.field(static=True)
    final_activation: Callable = eqx.field(static=True)
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    width_size: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)

    def __init__(
        self,
        in_size: int,
        out_size: int,
        width_size: int,
        depth: int,
        activation: Callable = jax.nn.relu,
        final_activation: Callable = lambda x: x,
        *,
        key: "jax.random.PRNGKey",
        **kwargs,
    ):
        """**Arguments**:

        - `in_size`: The size of the input layer.
        - `out_size`: The size of the output layer.
        - `width_size`: The size of each hidden layer.
        - `depth`: The number of hidden layers.
        - `activation`: The activation function after each hidden layer. Defaults to
            ReLU.
        - `final_activation`: The activation function after the output layer. Defaults
            to the identity.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """

        super().__init__(**kwargs)
        keys = jax.random.split(key, depth + 1)
        layers = []
        if depth == 0:
            layers.append(nn.Linear(in_size, out_size, key=keys[0]))
        else:
            layers.append(nn.Linear(in_size, width_size, key=keys[0]))
            for i in range(depth - 1):
                layers.append(nn.Linear(width_size, width_size, key=keys[i + 1]))
            layers.append(nn.Linear(width_size, out_size, key=keys[-1]))
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.width_size = width_size
        self.depth = depth
        self.activation = activation  # type: ignore
        self.final_activation = final_activation  # type: ignore

    def __call__(self, x, *, key: Optional["jax.random.PRNGKey"] = None):
        """**Arguments:**

        - `x`: A JAX array with shape `(in_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        A JAX array with shape `(out_size,)`.
        """
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        return x


def assert_trees_not_close(a, b):
    try:
        assert_trees_all_close(jax.tree_util.tree_leaves(arrays_only(a)), jax.tree_util.tree_leaves(arrays_only(b)))
    except AssertionError:
        pass
    else:
        raise AssertionError("Trees are equal")


def arrays_only(x):
    return eqx.filter(x, eqx.is_array_like)


def has_torch():
    try:
        import torch  # noqa F401

        return True
    except ImportError:
        return False


def has_soundlibs():
    try:
        import librosa  # noqa F401
        import soundfile  # noqa F401
        import torchcodec  # noqa F401

        return True
    except ImportError:
        return False


def skip_if_no_torch(f):
    return pytest.mark.skipif(not has_torch(), reason="torch not installed")(f)


def skip_if_no_soundlibs(f):
    return pytest.mark.skipif(not has_soundlibs(), reason="soundfile/librosa not installed")(f)


def skip_if_module_missing(module: str):
    def try_import_module(module):
        try:
            __import__(module)
        except ImportError:
            return False
        else:
            return True

    return pytest.mark.skipif(not try_import_module(module), reason=f"{module} not installed")


def skip_if_checkpoint_not_accessible(path: str):
    def try_load_path(path):
        try:
            fs, path_to_open = _get_fs_and_plain_path(path)
            fs.open(path_to_open, "rb")
        except Exception:
            return False
        else:
            return True

    return pytest.mark.skipif(not try_load_path(path), reason="Checkpoint not accessible")


def skip_if_hf_model_not_accessible(model_id: str):
    def try_load_hf(model_id):
        try:
            AutoConfig.from_pretrained(model_id)
        except Exception:
            return False
        else:
            return True

    return pytest.mark.skipif(not try_load_hf(model_id), reason="HuggingFace model not accessible")


def skip_in_ci(fn_or_msg):
    if isinstance(fn_or_msg, str):

        def decorator(fn):
            return pytest.mark.skipif("CI" in os.environ, reason=fn_or_msg)(fn)

        return decorator

    return pytest.mark.skipif("CI" in os.environ, reason="skipped in CI")(fn_or_msg)


class IdentityProcessor(BatchProcessor[BatchEncoding, BatchEncoding]):
    def __call__(self, batch: Sequence[BatchEncoding]) -> BatchEncoding:
        stacked = reduce(_stack_batch_encodings, batch)
        return stacked

    @property
    def output_exemplar(self):
        return BatchEncoding({})

    @property
    def num_cpus(self) -> int:
        return 0

    @property
    def metadata(self) -> Dict[str, Any]:
        return {}


class ShardsDataSource(ShardedDataSource[T]):
    def __init__(self, docs: List[List[T]]):
        self.docs = docs

    @property
    def shard_names(self) -> Sequence[str]:
        return [str(i) for i in range(len(self.docs))]

    def open_shard_at_row(self, shard_name: str, row: int):
        return self.docs[int(shard_name)][row:]


class SingleShardDocumentSource(ShardedDataSource[T]):
    def __init__(self, docs: List[T]):
        self.docs = docs

    @property
    def shard_names(self) -> Sequence[str]:
        return ["0"]

    def open_shard_at_row(self, shard_name: str, row: int):
        return self.docs[row:]


def parameterize_with_configs(pattern, config_path=None):
    test_path = os.path.dirname(os.path.abspath(__file__))
    if config_path is None:
        config_path = os.path.join(test_path, "..", "config")

    configs = glob.glob(os.path.join(config_path, pattern))
    return pytest.mark.parametrize("config_file", configs, ids=lambda x: f"{os.path.basename(x)}")


def check_load_config(config_class, config_file):
    try:
        draccus.parse(config_class, config_file, args=[])
    except Exception as e:
        raise Exception(f"failed to parse {config_file}") from e


def check_model_works_with_seqlen(model_type, config, input_len):
    key = PRNGKey(0)
    Vocab = hax.Axis("vocab", 128)
    model = model_type.init(Vocab, config, key=key)
    input_ids = hax.arange(config.Pos.resize(input_len), dtype=jax.numpy.int32)
    causal_mask = AttentionMask.causal()
    a1 = model(input_ids, key=key, attn_mask=causal_mask)
    assert a1.axis_size("position") == input_len


def _stack_batch_encodings(a: BatchEncoding, b: BatchEncoding) -> BatchEncoding:
    """Stacks two batch encodings together, assuming that the keys are the same."""

    def _ensure_batched(x):
        if len(x) == 0:
            return list(x)
        elif isinstance(x[0], Sequence) or isinstance(x[0], np.ndarray):
            return list(x)
        else:
            return [x]

    return BatchEncoding({k: _ensure_batched(a[k]) + _ensure_batched(b[k]) for k in a.keys()})


def create_test_mesh(tensor_parallelism: int = 1, skip_ok: bool = False) -> Mesh:
    """Create a Mesh with separate data and tensor parallel axes."""

    if tensor_parallelism < 1:
        raise ValueError("tensor_parallelism must be at least 1")

    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices available for creating a mesh")

    if len(devices) % tensor_parallelism != 0:
        if skip_ok:
            pytest.skip(
                f"Cannot create mesh with tensor_parallelism={tensor_parallelism} using {len(devices)} devices"
            )
        raise ValueError(
            f"Cannot create mesh with tensor_parallelism={tensor_parallelism} using {len(devices)} devices"
        )

    mesh_devices = np.array(devices).reshape(-1, tensor_parallelism)
    return Mesh(mesh_devices, (ResourceAxis.DATA, ResourceAxis.MODEL))


@contextmanager
def use_test_mesh(tensor_parallelism: int = 1, *, mesh: Optional[Mesh] = None, skip_ok: bool = False):
    """Context manager that activates a data/model Mesh and sets haliax's global mesh."""
    if mesh is None:
        mesh = create_test_mesh(tensor_parallelism, skip_ok=skip_ok)
    with set_mesh(mesh):
        yield mesh
