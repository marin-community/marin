# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import tempfile

import jax

import haliax

import levanter.main.viz_logprobs as viz_logprobs
import tiny_test_corpus
from levanter.checkpoint import save_checkpoint
from levanter.distributed import DistributedConfig
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.tracker import NoopConfig


def test_viz_lm():
    model_config = LlamaConfig(
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        hidden_dim=32,
        max_seq_len=64,
    )

    with tempfile.TemporaryDirectory() as f:
        data_config, _ = tiny_test_corpus.construct_small_data_cache(f)
        tok = data_config.the_tokenizer
        Vocab = haliax.Axis("vocab", len(tok))
        model = LlamaLMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))

        save_checkpoint({"model": model}, 0, f"{f}/ckpt")

        viz_path = f"{f}/viz"
        config = viz_logprobs.VizLmConfig(
            data=data_config,
            model=model_config,
            trainer=viz_logprobs.TrainerConfig(
                per_device_eval_parallelism=len(jax.devices()),
                max_eval_batches=1,
                tracker=NoopConfig(),
                require_accelerator=False,
                distributed=DistributedConfig(initialize_jax_distributed=False),
            ),
            checkpoint_path=f"{f}/ckpt",
            num_docs=len(jax.devices()),
            path=viz_path,
        )
        viz_logprobs.main(config)

        # main() writes one HTML log-prob visualization per validation set; assert at
        # least one non-empty artifact landed so a silently-empty render is caught.
        html_files = glob.glob(f"{viz_path}*.html") + glob.glob(os.path.join(viz_path, "*.html"))
        assert html_files, f"no viz HTML written under {viz_path}"
        assert all(os.path.getsize(p) > 0 for p in html_files)
