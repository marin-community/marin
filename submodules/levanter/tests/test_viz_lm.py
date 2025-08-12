import os
import tempfile

import jax
import pytest

import haliax

import levanter.main.viz_logprobs as viz_logprobs
import tiny_test_corpus
from levanter.checkpoint import save_checkpoint
from levanter.distributed import RayConfig
from levanter.models.llama import LlamaConfig, LlamaLMHeadModel
from levanter.tracker import NoopConfig


@pytest.mark.entry
def test_viz_lm():
    # just testing if eval_lm has a pulse
    # save a checkpoint
    model_config = LlamaConfig(
        num_layers=2,
        num_heads=2,
        num_kv_heads=2,
        hidden_dim=32,
        seq_len=64,
    )

    with tempfile.TemporaryDirectory() as f:
        try:
            data_config, _ = tiny_test_corpus.construct_small_data_cache(f)
            tok = data_config.the_tokenizer
            Vocab = haliax.Axis("vocab", len(tok))
            model = LlamaLMHeadModel.init(Vocab, model_config, key=jax.random.PRNGKey(0))

            save_checkpoint({"model": model}, 0, f"{f}/ckpt")

            config = viz_logprobs.VizLmConfig(
                data=data_config,
                model=model_config,
                trainer=viz_logprobs.TrainerConfig(
                    per_device_eval_parallelism=len(jax.devices()),
                    max_eval_batches=1,
                    tracker=NoopConfig(),
                    require_accelerator=False,
                    ray=RayConfig(auto_start_cluster=False),
                ),
                checkpoint_path=f"{f}/ckpt",
                num_docs=len(jax.devices()),
                path=f"{f}/viz",
            )
            viz_logprobs.main(config)
        finally:
            try:
                os.unlink("wandb")
            except Exception:
                pass
