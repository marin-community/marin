"""Tests for bridges module."""

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from grugfuzz import hf_state_dict_to_jax, jax_state_dict_to_torch, jax_to_torch, torch_to_jax


class TestTorchToJax:
    def test_1d_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0])
        j = torch_to_jax(t)
        assert isinstance(j, jnp.ndarray)
        np.testing.assert_array_equal(np.array(j), t.numpy())

    def test_2d_tensor(self):
        t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        j = torch_to_jax(t)
        assert j.shape == (2, 2)
        np.testing.assert_array_equal(np.array(j), t.numpy())

    def test_preserves_dtype_float32(self):
        t = torch.tensor([1.0, 2.0], dtype=torch.float32)
        j = torch_to_jax(t)
        assert j.dtype == jnp.float32

    def test_preserves_dtype_int(self):
        t = torch.tensor([1, 2, 3], dtype=torch.int64)
        j = torch_to_jax(t)
        assert j.dtype == jnp.int32
        np.testing.assert_array_equal(np.array(j), t.numpy())

    def test_requires_grad_tensor(self):
        t = torch.tensor([1.0, 2.0], requires_grad=True)
        j = torch_to_jax(t)
        np.testing.assert_array_equal(np.array(j), t.detach().numpy())


class TestJaxToTorch:
    def test_1d_array(self):
        j = jnp.array([1.0, 2.0, 3.0])
        t = jax_to_torch(j)
        assert isinstance(t, torch.Tensor)
        np.testing.assert_array_equal(t.numpy(), np.array(j))

    def test_2d_array(self):
        j = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        t = jax_to_torch(j)
        assert t.shape == torch.Size([2, 2])

    def test_device_cpu(self):
        j = jnp.array([1.0, 2.0])
        t = jax_to_torch(j, device="cpu")
        assert t.device == torch.device("cpu")

    def test_int_input_casts_to_float32(self):
        j = jnp.array([1, 2, 3], dtype=jnp.int32)
        t = jax_to_torch(j)
        assert t.dtype == torch.int32
        np.testing.assert_array_equal(t.numpy(), np.array(j, dtype=np.int32))


class TestHfStateDictToJax:
    def test_converts_all_tensors(self):
        state_dict = {
            "layer.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "layer.bias": torch.tensor([0.1, 0.2]),
        }
        jax_dict = hf_state_dict_to_jax(state_dict)

        assert set(jax_dict.keys()) == set(state_dict.keys())
        for k in state_dict:
            assert isinstance(jax_dict[k], jnp.ndarray)
            np.testing.assert_array_equal(np.array(jax_dict[k]), state_dict[k].numpy())

    def test_preserves_shapes(self):
        state_dict = {
            "embed": torch.randn(1000, 128),
            "attn.q": torch.randn(128, 128),
        }
        jax_dict = hf_state_dict_to_jax(state_dict)

        assert jax_dict["embed"].shape == (1000, 128)
        assert jax_dict["attn.q"].shape == (128, 128)


class TestJaxStateDictToTorch:
    def test_converts_all_arrays(self):
        jax_dict = {
            "layer.weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "layer.bias": jnp.array([0.1, 0.2]),
        }
        torch_dict = jax_state_dict_to_torch(jax_dict)

        assert set(torch_dict.keys()) == set(jax_dict.keys())
        for k in jax_dict:
            assert isinstance(torch_dict[k], torch.Tensor)
            np.testing.assert_array_equal(torch_dict[k].numpy(), np.array(jax_dict[k]))


class TestRoundTrip:
    def test_torch_jax_torch(self):
        original = torch.randn(3, 4, 5)
        jax_arr = torch_to_jax(original)
        back = jax_to_torch(jax_arr)
        np.testing.assert_allclose(back.numpy(), original.numpy(), rtol=1e-6)

    def test_jax_torch_jax(self):
        original = jnp.array(np.random.randn(3, 4, 5).astype(np.float32))
        torch_tensor = jax_to_torch(original)
        back = torch_to_jax(torch_tensor)
        np.testing.assert_allclose(np.array(back), np.array(original), rtol=1e-6)
