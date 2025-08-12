import os
import tempfile

import equinox as eqx
import jax
import pytest
from transformers import AutoModelForCausalLM

import haliax

import levanter.main.export_lm_to_hf as export_lm_to_hf
import tiny_test_corpus
from levanter.checkpoint import save_checkpoint
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.utils.jax_utils import is_inexact_arrayish
from test_utils import has_torch


@pytest.mark.entry
def test_export_lm_to_hf():
    # just testing if train_lm has a pulse
    model_config = Gpt2Config(
        num_layers=2,
        num_heads=2,
        seq_len=32,
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

        try:
            config = export_lm_to_hf.ConvertLmConfig(
                checkpoint_path=f"{tmpdir}/ckpt",
                output_dir=f"{tmpdir}/output",
                model=model_config,
            )
            export_lm_to_hf.main(config)

            if has_torch():
                AutoModelForCausalLM.from_pretrained(f"{tmpdir}/output")

        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
