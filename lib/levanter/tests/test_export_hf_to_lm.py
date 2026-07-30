# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import os
import tempfile
from types import SimpleNamespace

import equinox as eqx
import jax
import numpy as np

import haliax

import levanter.main.export_hf_to_lm as export_hf_to_lm
import levanter.main.export_lm_to_hf as export_lm_to_hf
import tiny_test_corpus
from levanter.checkpoint import load_checkpoint, save_checkpoint
from levanter.compat.hf_checkpoints import load_tokenizer
from levanter.models.gpt2 import Gpt2Config, Gpt2LMHeadModel
from levanter.utils.jax_utils import is_inexact_arrayish
from test_utils import use_test_mesh

# Pad the model embedding past the tokenizer vocab (as Qwen does for TPU efficiency), so the
# emit_padded_tokenizer path has something to pad.
_VOCAB_PAD = 8


def test_import_hf_to_lm_saves_model_under_subpath_and_emits_padded_tokenizer():
    """levanter -> HF -> levanter round-trip through export_hf_to_lm.

    The model is saved under a ``model`` subtree (so it loads with ``subpath="model"``, matching a
    TrainerState's model field) and a tokenizer padded to the model vocab is emitted at the output root.
    """
    model_config = Gpt2Config(num_layers=2, num_heads=2, max_seq_len=32, hidden_dim=32, use_flash_attention=True)

    with tempfile.TemporaryDirectory() as tmpdir, use_test_mesh():
        tokenizer = tiny_test_corpus.tiny_corpus_config(tmpdir).the_tokenizer
        vocab_size = len(tokenizer) + _VOCAB_PAD
        Vocab = haliax.Axis("vocab", vocab_size)
        model = Gpt2LMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))

        # levanter -> HF: write config.json + safetensors + the (unpadded) tokenizer. No torch needed.
        levanter_ckpt = os.path.join(tmpdir, "levanter_ckpt")
        trainable, _ = eqx.partition(model, is_inexact_arrayish)
        save_checkpoint({"model": trainable}, 0, levanter_ckpt)
        hf_dir = os.path.join(tmpdir, "hf")
        export_lm_to_hf.main(
            export_lm_to_hf.ConvertLmConfig(
                trainer=SimpleNamespace(device_mesh=contextlib.nullcontext(), parameter_axis_mapping={}),
                checkpoint_path=levanter_ckpt,
                checkpoint_subpath="model",
                output_dir=hf_dir,
                model=model_config,
                override_vocab_size=vocab_size,  # the embedding is padded past the tokenizer
                use_cpu=True,
            )
        )

        # HF -> levanter: the model as a `model/` subtree + a padded tokenizer at the output root.
        native_dir = os.path.join(tmpdir, "native")
        export_hf_to_lm.main(
            export_hf_to_lm.ImportHfConfig(
                hf_checkpoint=hf_dir,
                output_path=native_dir,
                model=model_config,
                use_hf_model_config=False,
                dtype="float32",
                resize_vocab_to_match_tokenizer=False,
                subpath="model",
                emit_padded_tokenizer=True,
            )
        )

        # The checkpoint root holds the metadata; the model is a `model` subtree inside it (an OCDBT
        # key prefix, not an on-disk dir), reachable with subpath="model" below. The tokenizer is at the root.
        assert os.path.exists(os.path.join(native_dir, "metadata.json"))
        # The emitted tokenizer is padded up to the model vocab, so a downstream Vocab axis built from
        # len(tokenizer) matches the checkpoint embedding.
        padded_tokenizer = load_tokenizer(native_dir)
        assert len(padded_tokenizer) == vocab_size
        assert len(padded_tokenizer) == len(tokenizer) + _VOCAB_PAD

        # The model loads back from the `model` subtree with weights identical to the original.
        reloaded = load_checkpoint(model, native_dir, subpath="model")
        for original_leaf, reloaded_leaf in zip(
            jax.tree_util.tree_leaves(eqx.filter(model, is_inexact_arrayish)),
            jax.tree_util.tree_leaves(eqx.filter(reloaded, is_inexact_arrayish)),
            strict=True,
        ):
            np.testing.assert_allclose(np.asarray(original_leaf), np.asarray(reloaded_leaf), rtol=1e-5, atol=1e-5)
