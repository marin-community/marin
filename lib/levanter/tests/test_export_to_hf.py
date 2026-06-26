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
from transformers import AutoModelForCausalLM

import haliax

import levanter.main.export_lm_to_hf as export_lm_to_hf
import tiny_test_corpus
from levanter.checkpoint import save_checkpoint
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.utils.jax_utils import is_inexact_arrayish
from test_utils import has_torch


def test_export_lm_to_hf():
    model_config = Gpt2Config(
        num_layers=2,
        num_heads=2,
        max_seq_len=32,
        use_flash_attention=True,
        hidden_dim=32,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        data_config = tiny_test_corpus.tiny_corpus_config(tmpdir)
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
