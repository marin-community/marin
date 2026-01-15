# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
import uuid

import equinox as eqx
import fsspec
import jax
import jax.numpy as jnp
import jmp
import pytest
import safetensors
from chex import assert_trees_all_close, assert_trees_all_equal
from jax.random import PRNGKey

from haliax import Axis
from haliax.state_dict import ModuleWithStateDictSerialization, to_torch_compatible_state_dict

from levanter.compat.hf_checkpoints import (
    HF_STREAMING_ENV_VAR,
    SAFE_TENSORS_INDEX_NAME,
    SAFE_TENSORS_MODEL,
    ModelWithHfSerializationMixin,
    _convert_to_hf_url,
    _convert_to_jnp,
    _is_hf_model_id,
    _should_use_hf_streaming,
)
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from test_utils import skip_if_no_torch
from tests.test_utils import use_test_mesh


@skip_if_no_torch
def test_conversion_to_jnp_bfloat16():
    import torch

    x = torch.arange(10, dtype=torch.bfloat16) / 3.14
    with pytest.raises(TypeError):
        x.cpu().numpy()

    x_jnp = _convert_to_jnp(x, None)
    assert x_jnp.dtype == jnp.bfloat16
    assert x_jnp.shape == x.shape
    assert_trees_all_close(x_jnp, jnp.arange(10, dtype=jnp.bfloat16) / 3.14)


def test_save_sharded_checkpoints():
    nano_config = Gpt2Config(hidden_dim=64, num_heads=2, num_layers=2, resid_pdrop=0.0, use_flash_attention=False)
    converter = nano_config.hf_checkpoint_converter()

    mp = jmp.get_policy("f32")

    with tempfile.TemporaryDirectory() as tmpdir:
        with use_test_mesh():
            nano_model = Gpt2LMHeadModel.init(converter.Vocab, nano_config, key=PRNGKey(3))
            nano_model = mp.cast_to_param(nano_model)

            converter.save_pretrained(nano_model, tmpdir, max_shard_size=1024)

        # make sure we saved a few different files
        import glob

        assert len(glob.glob(tmpdir + "/*.safetensors")) > 1

        with use_test_mesh():
            loaded_model = converter.load_pretrained(
                Gpt2LMHeadModel, ref=tmpdir, config=nano_model.config, dtype=mp.param_dtype
            )

        assert loaded_model.config == nano_model.config
        assert loaded_model.Vocab == nano_model.Vocab

        assert_trees_all_equal(
            nano_model,
            loaded_model,
        )


# A simple wrapper to include diverse dtypes in a model
class BasicModelWrapper(ModuleWithStateDictSerialization, ModelWithHfSerializationMixin):
    model: Gpt2LMHeadModel
    an_int_param: jax.Array
    a_bool_buffer: jax.Array
    a_float_param: jax.Array

    @property
    def config(self):
        return self.model.config

    @property
    def Vocab(self):
        return self.model.Vocab

    @classmethod
    def init(cls, Vocab: Axis, config: Gpt2Config, *, key: PRNGKey) -> "BasicModelWrapper":
        model = Gpt2LMHeadModel.init(Vocab, config, key=key)
        an_int_param = jnp.array([-1, 0, 1, 2, 3], dtype=jnp.int32)
        a_bool_buffer = jnp.array([True, False, True, False], dtype=jnp.bool_)
        a_float_param = jnp.array([10.0, 20.0, 30.0], dtype=jnp.float32)

        return BasicModelWrapper(
            model=model,
            an_int_param=an_int_param,
            a_bool_buffer=a_bool_buffer,
            a_float_param=a_float_param,
        )

    def _state_dict_key_map(self):
        # This tells the serialization logic how to map attribute names to state dict keys
        # We want the wrapper's parameters to be at the top level.
        # The Gpt2LMHeadModel's parameters will be nested under "model" if we follow its own mapping.
        return {
            "model": "model",
            "an_int_param": "an_int_param",
            "a_bool_buffer": "a_bool_buffer",
            "a_float_param": "a_float_param",
        }


def test_save_pretrained_with_custom_dtype():
    gpt2_config = Gpt2Config(num_layers=1, num_heads=1, hidden_dim=32, use_flash_attention=False)
    converter = gpt2_config.hf_checkpoint_converter()

    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(tmpdir, exist_ok=True)

        with use_test_mesh():
            # Wrap the model
            wrapped_model = BasicModelWrapper.init(converter.Vocab, gpt2_config, key=PRNGKey(0))

            # Ensure initial dtypes are as expected
            assert wrapped_model.model.transformer.blocks.stacked.attn.c_attn.weight.array.dtype == jnp.float32
            assert wrapped_model.a_float_param.dtype == jnp.float32
            assert wrapped_model.an_int_param.dtype == jnp.int32
            assert wrapped_model.a_bool_buffer.dtype == jnp.bool_

            converter.save_pretrained(
                wrapped_model,
                tmpdir,
                save_tokenizer=False,
                save_reference_code=False,
                max_shard_size=int(10e9),
                dtype=jnp.bfloat16,
            )

        saved_file = os.path.join(tmpdir, SAFE_TENSORS_MODEL)
        assert os.path.exists(saved_file)

        tensors = safetensors.safe_open(saved_file, framework="jax", device="cpu")

        # Check dtypes in the saved file
        # Gpt_model float params should be bfloat16
        print(tensors.keys())
        assert tensors.get_tensor("model.wte.weight").dtype == jnp.bfloat16
        # Wrapper's own float param should be bfloat16
        assert tensors.get_tensor("a_float_param").dtype == jnp.bfloat16
        # Int and Bool params should remain unchanged
        assert tensors.get_tensor("an_int_param").dtype == jnp.int32
        assert tensors.get_tensor("a_bool_buffer").dtype == jnp.bool_

        # This part is tricky because load_pretrained is for LmWithHfSerializationMixin and expects a certain structure.
        # We saved a TestModelWrapper's state_dict.
        # For now, verifying the saved file's dtypes is the primary goal for this unit test.
        # A full load test would require TestModelWrapper to be an LmWithHfSerializationMixin, which is overkill.


def test_save_pretrained_default_dtype():
    gpt2_config = Gpt2Config(num_layers=1, num_heads=1, hidden_dim=32, use_flash_attention=False)
    converter = gpt2_config.hf_checkpoint_converter()

    mp_policy = jmp.get_policy("float32")
    expected_float_dtype = mp_policy.param_dtype  # usually float32

    # Cast float params to expected_float_dtype to be sure
    def cast_floats(x):
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(expected_float_dtype)
        return x

    with tempfile.TemporaryDirectory() as tmpdir:
        with use_test_mesh():
            wrapped_model = BasicModelWrapper.init(converter.Vocab, gpt2_config, key=PRNGKey(0))
            wrapped_model = jax.tree_util.tree_map(cast_floats, wrapped_model, is_leaf=eqx.is_array)

            assert (
                wrapped_model.model.transformer.blocks.stacked.attn.c_attn.weight.array.dtype == expected_float_dtype
            )
            assert wrapped_model.a_float_param.dtype == expected_float_dtype
            assert wrapped_model.an_int_param.dtype == jnp.int32
            assert wrapped_model.a_bool_buffer.dtype == jnp.bool_

        # Save the wrapped_model without passing dtype
        # Similar to the above test, using _save_pretrained_local for direct testing.

        with use_test_mesh():
            converter.save_pretrained(
                wrapped_model,
                tmpdir,
                save_tokenizer=False,
                save_reference_code=False,
                max_shard_size=int(10e9),
            )  # No dtype override

        saved_file = os.path.join(tmpdir, SAFE_TENSORS_MODEL)
        assert os.path.exists(saved_file)

        tensors = safetensors.safe_open(saved_file, framework="jax", device="cpu")

        # Check dtypes in the saved file - all should be original
        assert tensors.get_tensor("model.wte.weight").dtype == expected_float_dtype
        assert tensors.get_tensor("a_float_param").dtype == expected_float_dtype
        assert tensors.get_tensor("an_int_param").dtype == jnp.int32
        assert tensors.get_tensor("a_bool_buffer").dtype == jnp.bool_


def test_save_pretrained_to_memory_fs():
    fs = fsspec.filesystem("memory")
    path = f"memory://levanter/hf-save/{uuid.uuid4().hex}"

    gpt2_config = Gpt2Config(num_layers=4, num_heads=1, hidden_dim=32, use_flash_attention=False)
    converter = gpt2_config.hf_checkpoint_converter()

    try:
        fs.rm(path, recursive=True)
    except FileNotFoundError:
        pass

    with use_test_mesh():
        model = Gpt2LMHeadModel.init(converter.Vocab, gpt2_config, key=PRNGKey(4))

        converter.save_pretrained(
            model,
            path,
            max_shard_size=128,
            save_tokenizer=False,
            save_reference_code=False,
            save_feature_extractor=False,
        )

        stored_files = {fs._strip_protocol(file) for file in fs.find(path)}
        base_path = fs._strip_protocol(path)
        safetensor_files = {file for file in stored_files if file.endswith(".safetensors")}

        assert len(safetensor_files) > 1
        assert f"{base_path}/config.json" in stored_files
        assert f"{base_path}/{SAFE_TENSORS_INDEX_NAME}" in stored_files

        for file in safetensor_files:
            with fs.open(file, "rb") as fh:
                assert fh.read(1) != b""

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "model")
        fs.get(f"{base_path}/", local_path, recursive=True)

        with use_test_mesh():
            reloaded_model = converter.load_pretrained(Gpt2LMHeadModel, ref=local_path, config=gpt2_config)

        original_state = to_torch_compatible_state_dict(model)
        reloaded_state = to_torch_compatible_state_dict(reloaded_model)
        assert original_state.keys() == reloaded_state.keys()
        for key, original_value in original_state.items():
            reloaded_value = reloaded_state[key]
            assert_trees_all_close(original_value, reloaded_value)

    fs.rm(base_path, recursive=True)


# Tests for HF streaming functionality

def test_convert_to_hf_url():
    """Test that model IDs are correctly converted to hf:// URLs."""
    # Simple model ID
    assert _convert_to_hf_url("meta-llama/Llama-2-7b") == "hf://meta-llama/Llama-2-7b"

    # Model ID with revision
    assert _convert_to_hf_url("meta-llama/Llama-2-7b", "main") == "hf://meta-llama/Llama-2-7b@main"
    assert _convert_to_hf_url("gpt2", "v1.0") == "hf://gpt2@v1.0"

    # Empty revision should not append @
    assert _convert_to_hf_url("gpt2", None) == "hf://gpt2"


def test_is_hf_model_id():
    """Test that we correctly identify HF model IDs vs local paths and URLs."""
    # HF model IDs
    assert _is_hf_model_id("meta-llama/Llama-2-7b") is True
    assert _is_hf_model_id("gpt2") is True
    assert _is_hf_model_id("mistralai/Mistral-7B-v0.1") is True

    # URLs should not be HF model IDs
    assert _is_hf_model_id("hf://meta-llama/Llama-2-7b") is False
    assert _is_hf_model_id("gs://bucket/path/model") is False
    assert _is_hf_model_id("s3://bucket/path/model") is False

    # Local paths should not be HF model IDs
    assert _is_hf_model_id("/absolute/path/to/model") is False
    assert _is_hf_model_id("./relative/path") is False
    assert _is_hf_model_id("../parent/path") is False


def test_should_use_hf_streaming_env_var():
    """Test that the streaming mode environment variable is correctly detected."""
    original_value = os.environ.get(HF_STREAMING_ENV_VAR)

    try:
        # Test various enabled values
        os.environ[HF_STREAMING_ENV_VAR] = "1"
        assert _should_use_hf_streaming() is True

        os.environ[HF_STREAMING_ENV_VAR] = "true"
        assert _should_use_hf_streaming() is True

        os.environ[HF_STREAMING_ENV_VAR] = "TRUE"
        assert _should_use_hf_streaming() is True

        os.environ[HF_STREAMING_ENV_VAR] = "yes"
        assert _should_use_hf_streaming() is True

        # Test disabled values
        os.environ[HF_STREAMING_ENV_VAR] = "0"
        assert _should_use_hf_streaming() is False

        os.environ[HF_STREAMING_ENV_VAR] = "false"
        assert _should_use_hf_streaming() is False

        os.environ[HF_STREAMING_ENV_VAR] = ""
        assert _should_use_hf_streaming() is False

        # Test unset
        del os.environ[HF_STREAMING_ENV_VAR]
        assert _should_use_hf_streaming() is False

    finally:
        # Restore original value
        if original_value is not None:
            os.environ[HF_STREAMING_ENV_VAR] = original_value
        elif HF_STREAMING_ENV_VAR in os.environ:
            del os.environ[HF_STREAMING_ENV_VAR]


@pytest.mark.network
def test_load_model_with_hf_streaming():
    """Test that loading a model with HF streaming enabled works correctly.

    This test verifies that when MARIN_HF_USE_STREAMING=1 is set:
    1. The model loads successfully from HuggingFace Hub
    2. The model is loaded via the hf:// fsspec protocol
    3. The local HF cache is not populated (streamed directly)
    """
    from levanter.models.gpt2 import Gpt2Config

    # Use a very small model for testing
    model_id = "hf-internal-testing/tiny-random-gpt2"

    original_value = os.environ.get(HF_STREAMING_ENV_VAR)
    original_hf_cache = os.environ.get("HF_HOME")

    try:
        # Set up a temporary HF cache directory to verify streaming doesn't use it
        with tempfile.TemporaryDirectory() as tmp_hf_cache:
            os.environ["HF_HOME"] = tmp_hf_cache
            os.environ[HF_STREAMING_ENV_VAR] = "1"

            gpt2_config = Gpt2Config(
                num_layers=2,
                num_heads=2,
                hidden_dim=32,
                use_flash_attention=False,
            )
            converter = gpt2_config.hf_checkpoint_converter()

            with use_test_mesh():
                # This should trigger streaming mode
                state_dict = converter.load_state_dict(model_id)

            # Verify we got a state dict
            assert isinstance(state_dict, dict)
            assert len(state_dict) > 0

            # Check that no model files were downloaded to the cache
            # (the hf:// protocol should stream directly)
            hub_cache = os.path.join(tmp_hf_cache, "hub")
            if os.path.exists(hub_cache):
                # If hub directory exists, it should be empty or have no model files
                cached_files = []
                for root, dirs, files in os.walk(hub_cache):
                    cached_files.extend(files)
                model_files = [f for f in cached_files if f.endswith((".safetensors", ".bin"))]
                assert len(model_files) == 0, f"Expected no model files in cache, found: {model_files}"

    finally:
        # Restore original values
        if original_value is not None:
            os.environ[HF_STREAMING_ENV_VAR] = original_value
        elif HF_STREAMING_ENV_VAR in os.environ:
            del os.environ[HF_STREAMING_ENV_VAR]

        if original_hf_cache is not None:
            os.environ["HF_HOME"] = original_hf_cache
        elif "HF_HOME" in os.environ:
            del os.environ["HF_HOME"]
