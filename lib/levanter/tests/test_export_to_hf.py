# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import glob
import json
import os
import tempfile
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import AxisType
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast
from transformers import GPT2Config as HfGpt2Config

import haliax

import levanter.main.export_lm_to_hf as export_lm_to_hf
from levanter.testing import tiny_corpus
from levanter.checkpoint import save_checkpoint
from levanter.compat.hf_checkpoints import HFCheckpointConverter, SAFE_TENSORS_INDEX_NAME
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.models.snowball import SnowballConfig
from levanter.utils.jax_utils import is_inexact_arrayish, local_cpu_mesh
from levanter.testing.helpers import has_torch
from haliax._src.state_dict import flatten_modules_for_export, to_state_dict


class TokenizerlessGpt2Config(Gpt2Config):
    def hf_checkpoint_converter(self, ref_checkpoint: str | None = None) -> HFCheckpointConverter["Gpt2Config"]:
        return HFCheckpointConverter(
            self.__class__,
            reference_checkpoint=None,
            HfConfigClass=HfGpt2Config,
            tokenizer=None,
            ignore_prefix="transformer",
        )


def test_export_lm_to_hf():
    model_config = Gpt2Config(
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
        use_flash_attention=True,
        hidden_dim=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        data_config = tiny_corpus.tiny_corpus_config(tmpdir)
        tok = data_config.the_tokenizer
        Vocab = haliax.Axis("vocab", len(tok))
        model = Gpt2LMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))
        # in our trainer, we only export the trainable params
        trainable, non_trainable = eqx.partition(model, is_inexact_arrayish)

        save_checkpoint({"model": trainable}, 0, f"{tmpdir}/ckpt")

        trainer = SimpleNamespace(
            device_mesh=contextlib.nullcontext(),
            parameter_axis_mapping={},
        )
        output_dir = f"{tmpdir}/output"
        config = export_lm_to_hf.ConvertLmConfig(
            trainer=trainer,
            checkpoint_path=f"{tmpdir}/ckpt",
            output_dir=output_dir,
            model=model_config,
            use_cpu=True,
        )
        export_lm_to_hf.main(config)

        # save_pretrained must persist a loadable HF config plus a weights shard.
        config_path = os.path.join(output_dir, "config.json")
        assert os.path.exists(config_path)
        weights = glob.glob(os.path.join(output_dir, "*.safetensors")) + glob.glob(os.path.join(output_dir, "*.bin"))
        assert weights, f"no weights file written under {output_dir}"

        with open(config_path) as f:
            hf_config = json.load(f)
        # the exported config must round-trip our model shape, not just be present
        assert hf_config["n_layer"] == 2
        assert hf_config["n_head"] == 2

        if has_torch():
            reloaded = AutoModelForCausalLM.from_pretrained(output_dir)
            assert reloaded.config.n_layer == 2
            assert reloaded.config.vocab_size == len(tok)


def test_export_lm_to_hf_custom_subpath_without_tokenizer():
    model_config = TokenizerlessGpt2Config(
        num_layers=1,
        num_heads=2,
        max_seq_len=16,
        use_flash_attention=False,
        hidden_dim=16,
    )
    vocab_size = 64

    with tempfile.TemporaryDirectory() as tmpdir:
        Vocab = haliax.Axis("vocab", vocab_size)
        model = Gpt2LMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))
        trainable, _ = eqx.partition(model, is_inexact_arrayish)
        save_checkpoint({"params": trainable}, 0, f"{tmpdir}/ckpt")

        trainer = SimpleNamespace(
            device_mesh=contextlib.nullcontext(),
            parameter_axis_mapping={},
        )
        output_dir = f"{tmpdir}/output"
        config = export_lm_to_hf.ConvertLmConfig(
            trainer=trainer,
            checkpoint_path=f"{tmpdir}/ckpt",
            checkpoint_subpath="params",
            output_dir=output_dir,
            model=model_config,
            save_tokenizer=False,
            override_vocab_size=vocab_size,
            max_shard_size=512,
            use_cpu=True,
        )
        export_lm_to_hf.main(config)

        with open(os.path.join(output_dir, "config.json")) as f:
            hf_config = json.load(f)
        assert hf_config["vocab_size"] == vocab_size
        assert hf_config["n_layer"] == 1
        assert hf_config["n_head"] == 2
        assert not os.path.exists(os.path.join(output_dir, "tokenizer.json"))
        assert os.path.exists(os.path.join(output_dir, SAFE_TENSORS_INDEX_NAME))
        assert len(glob.glob(os.path.join(output_dir, "*.safetensors"))) > 1


@pytest.mark.parametrize("model_kind", ["gpt2", "snowball"])
def test_export_dpo_policy_subtree_to_bfloat16(tmp_path, model_kind):
    if model_kind == "gpt2":
        model_config = TokenizerlessGpt2Config(
            num_layers=1, num_heads=2, max_seq_len=16, use_flash_attention=False, hidden_dim=16
        )
    else:
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({f"t{i}": i for i in range(64)})))
        tokenizer_path = tmp_path / "tokenizer"
        tokenizer.save_pretrained(tokenizer_path)
        model_config = SnowballConfig(
            vocab_size=64,
            hidden_dim=32,
            intermediate_dim=32,
            shared_expert_intermediate_dim=32,
            num_experts=8,
            num_experts_per_token=2,
            num_layers=1,
            num_heads=4,
            num_kv_heads=2,
            head_dim=8,
            max_seq_len=16,
            sliding_window=4,
            attention_implementation="reference",
            tokenizer=str(tokenizer_path),
        )
    mesh_axis_type = AxisType.Explicit if model_config.requires_explicit_mesh_axes else AxisType.Auto
    Vocab = haliax.Axis("vocab", 64)
    with local_cpu_mesh(mesh_axis_type):
        policy = model_config.build(Vocab, key=jax.random.PRNGKey(0))
        reference = model_config.build(Vocab, key=jax.random.PRNGKey(1))
    policy_params, _ = eqx.partition(policy, is_inexact_arrayish)
    reference_params, _ = eqx.partition(reference, is_inexact_arrayish)
    checkpoint_path = str(tmp_path / "checkpoint")
    save_checkpoint({"model": {"policy": policy_params, "reference": reference_params}}, 1, checkpoint_path)
    output_dir = str(tmp_path / "output")
    export_lm_to_hf.main(
        export_lm_to_hf.ConvertLmConfig(
            trainer=SimpleNamespace(device_mesh=contextlib.nullcontext(), parameter_axis_mapping={}),
            checkpoint_path=checkpoint_path,
            checkpoint_subpath="model/policy",
            output_dir=output_dir,
            model=model_config,
            save_tokenizer=False,
            override_vocab_size=Vocab.size,
            max_shard_size=512,
            export_dtype="bfloat16",
            use_cpu=True,
        )
    )
    with local_cpu_mesh(mesh_axis_type):
        exported = model_config.hf_checkpoint_converter().load_state_dict(output_dir)
    expected = to_state_dict(flatten_modules_for_export(policy_params))
    assert exported.keys() == expected.keys()
    for name, value in exported.items():
        assert value.dtype == jnp.bfloat16
        np.testing.assert_array_equal(value, expected[name].astype(jnp.bfloat16))


def test_export_pinned_tokenizer_revision(tmp_path, monkeypatch):
    old_tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "old": 1})))
    pinned_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "old": 1, "pinned": 2}))
    )
    snapshots = {("test/tokenizer", None): old_tokenizer, ("test/tokenizer", "snapshot"): pinned_tokenizer}

    def from_pretrained(model_name_or_path, *, revision=None, **kwargs):
        return snapshots[model_name_or_path, revision]

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", from_pretrained)
    model_config = TokenizerlessGpt2Config(
        num_layers=1, num_heads=2, max_seq_len=16, use_flash_attention=False, hidden_dim=16
    )
    model = Gpt2LMHeadModel.init(haliax.Axis("vocab", len(pinned_tokenizer)), model_config, key=jax.random.PRNGKey(0))
    params, _ = eqx.partition(model, is_inexact_arrayish)
    checkpoint_path = str(tmp_path / "checkpoint")
    save_checkpoint({"model": params}, 0, checkpoint_path)
    output_dir = str(tmp_path / "output")
    export_lm_to_hf.main(
        export_lm_to_hf.ConvertLmConfig(
            trainer=SimpleNamespace(device_mesh=contextlib.nullcontext(), parameter_axis_mapping={}),
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            model=model_config,
            tokenizer="test/tokenizer@snapshot",
            use_cpu=True,
        )
    )
    restored_tokenizer = PreTrainedTokenizerFast.from_pretrained(output_dir)
    assert restored_tokenizer.get_vocab() == pinned_tokenizer.get_vocab()
    with open(os.path.join(output_dir, "config.json")) as f:
        assert json.load(f)["vocab_size"] == len(pinned_tokenizer)
