import os
import tempfile

import equinox as eqx
import jax
import jax.numpy as jnp
import jmp
import pytest
import safetensors
from chex import assert_trees_all_close, assert_trees_all_equal
from jax.random import PRNGKey

from haliax import Axis
from haliax.state_dict import ModuleWithStateDictSerialization

from levanter.compat.hf_checkpoints import SAFE_TENSORS_MODEL, ModelWithHfSerializationMixin, _convert_to_jnp
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from test_utils import skip_if_no_torch


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

    nano_model = Gpt2LMHeadModel.init(converter.Vocab, nano_config, key=PRNGKey(3))

    mp = jmp.get_policy("f32")
    nano_model = mp.cast_to_param(nano_model)

    with tempfile.TemporaryDirectory() as tmpdir:
        converter.save_pretrained(nano_model, tmpdir, max_shard_size=1024)

        # make sure we saved a few different files
        import glob

        assert len(glob.glob(tmpdir + "/*.safetensors")) > 1

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
    # Wrap the model
    wrapped_model = BasicModelWrapper.init(converter.Vocab, gpt2_config, key=PRNGKey(0))

    # Ensure initial dtypes are as expected
    assert wrapped_model.model.transformer.blocks.stacked.attn.c_attn.weight.array.dtype == jnp.float32
    assert wrapped_model.a_float_param.dtype == jnp.float32
    assert wrapped_model.an_int_param.dtype == jnp.int32
    assert wrapped_model.a_bool_buffer.dtype == jnp.bool_

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the wrapped_model with dtype=jnp.bfloat16
        # Note: converter.save_pretrained expects a ModelWithHfSerializationMixin,
        # so we pass the gpt_model, but we want its state_dict to be part of the larger
        # state_dict from wrapped_model. This requires a bit of manual handling for the test.
        # The actual dtype conversion logic in _save_pretrained_local uses to_torch_compatible_state_dict,
        # which will pick up all params if called on wrapped_model.

        # Let's simulate how HFCheckpointConverter would get the state_dict from the *actual* model being saved (gpt_model)
        # but then for the test, we want to create a combined state dict.
        # This is a bit of a hack for the test setup because HFCheckpointConverter is designed for LmWithHfSerializationMixin.
        # The core logic we are testing (_save_pretrained_local's dtype conversion) works on a generic state_dict.

        # For a cleaner test of _save_pretrained_local's specific behavior, we could directly call it
        # with a manually constructed state_dict.
        # However, to test the full converter.save_pretrained path, we need a ModelWithHfSerializationMixin.

        # Let's adjust: we'll save the gpt_model, but then inspect a state_dict that *would have been*
        # created if TestModelWrapper were the one being serialized by to_torch_compatible_state_dict.
        # This is getting a bit convoluted.

        # Simpler approach: The dtype conversion happens in _save_pretrained_local on whatever state_dict it receives.
        # HFCheckpointConverter.save_pretrained calls to_torch_compatible_state_dict(model_to_save).
        # So, if we want to test the selective conversion, the model_to_save (gpt_model here)
        # itself should contain these diverse dtypes, or to_torch_compatible_state_dict should be
        # patched/mocked for this test to return a dict with diverse types.

        # Let's use the TestModelWrapper and directly call _save_pretrained_local,
        # as this directly tests the target logic with a controlled state_dict.
        # We'll need to save a dummy config.json as _save_pretrained_local expects it.
        os.makedirs(tmpdir, exist_ok=True)

        # This is what we're testing the internals of:
        converter._save_pretrained_local(
            wrapped_model,
            tmpdir,
            save_tokenizer=False,
            save_reference_code=False,
            max_shard_size=10e9,
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

        # (Optional) Attempt to load the model back using the converter
        # This part is tricky because load_pretrained is for LmWithHfSerializationMixin and expects a certain structure.
        # We saved a TestModelWrapper's state_dict.
        # For now, verifying the saved file's dtypes is the primary goal for this unit test.
        # A full load test would require TestModelWrapper to be an LmWithHfSerializationMixin, which is overkill.


def test_save_pretrained_default_dtype():
    gpt2_config = Gpt2Config(num_layers=1, num_heads=1, hidden_dim=32, use_flash_attention=False)
    converter = gpt2_config.hf_checkpoint_converter()

    wrapped_model = BasicModelWrapper.init(converter.Vocab, gpt2_config, key=PRNGKey(0))

    mp_policy = jmp.get_policy("float32")
    expected_float_dtype = mp_policy.param_dtype  # usually float32

    # Cast float params to expected_float_dtype to be sure
    def cast_floats(x):
        if eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(expected_float_dtype)
        return x

    wrapped_model = jax.tree_util.tree_map(cast_floats, wrapped_model, is_leaf=lambda x: eqx.is_array(x))

    assert wrapped_model.model.transformer.blocks.stacked.attn.c_attn.weight.array.dtype == expected_float_dtype
    assert wrapped_model.a_float_param.dtype == expected_float_dtype
    assert wrapped_model.an_int_param.dtype == jnp.int32
    assert wrapped_model.a_bool_buffer.dtype == jnp.bool_

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save the wrapped_model without passing dtype
        # Similar to the above test, using _save_pretrained_local for direct testing.

        converter._save_pretrained_local(
            wrapped_model, tmpdir, save_tokenizer=False, save_reference_code=False, max_shard_size=10e9, dtype=None
        )  # No dtype override

        saved_file = os.path.join(tmpdir, SAFE_TENSORS_MODEL)
        assert os.path.exists(saved_file)

        tensors = safetensors.safe_open(saved_file, framework="jax", device="cpu")

        # Check dtypes in the saved file - all should be original
        assert tensors.get_tensor("model.wte.weight").dtype == expected_float_dtype
        assert tensors.get_tensor("a_float_param").dtype == expected_float_dtype
        assert tensors.get_tensor("an_int_param").dtype == jnp.int32
        assert tensors.get_tensor("a_bool_buffer").dtype == jnp.bool_
