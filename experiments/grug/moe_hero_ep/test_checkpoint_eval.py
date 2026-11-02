# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices


@pytest.mark.timeout(240)
def test_checkpoint_eval_scores_saved_weights_at_each_length_and_pads_the_final_window():
    script = """
        import dataclasses
        import tempfile

        import jax
        import jax.numpy as jnp
        import jmp
        import numpy as np
        from fray.cluster import ResourceConfig
        from levanter.checkpoint import save_checkpoint
        from levanter.data.text.datasets import DatasetComponent, LmDataConfig
        from levanter.data.text.formats import TextLmDatasetFormat
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.store.cache import write_levanter_cache
        from levanter.trainer import TrainerConfig
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel

        from experiments.grug.moe_hero_ep import model
        from experiments.grug.moe_hero_ep.launch_checkpoint_eval import CheckpointEvalConfig, evaluate_checkpoint
        from experiments.grug.moe_hero_ep.train import _reshard_tree_to_mesh

        config = model.GrugModelConfig(
            vocab_size=64, hidden_dim=32, intermediate_dim=16, shared_expert_intermediate_dim=16,
            num_shared_experts=1, num_experts=4, num_experts_per_token=2, num_layers=2, num_heads=4,
            num_kv_heads=2, local_kv_heads=2, global_kv_heads=2, head_dim=8, max_seq_len=8,
            sliding_window=4, global_every=2, initializer_std=0.1, qk_mult=1.0,
            attention_implementation="reference", moe_implementation="fixed_all_to_all",
        )
        # 37 tokens: 4 windows of 8 and 2 windows of 16, each followed by a padded window of the final 5 tokens.
        stream = (np.arange(37) % 61 + 1).astype(np.int32)
        with tempfile.TemporaryDirectory() as directory:
            tokenizer = Tokenizer(WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 64)}}, unk_token="<unk>"))
            tokenizer.save(directory + "/tokenizer.json")
            write_levanter_cache([{"input_ids": stream}], directory + "/tiny/validation", metadata={})
            data = LmDataConfig(
                components={
                    "paloma/tiny": DatasetComponent(cache_dir=directory + "/tiny", format=TextLmDatasetFormat())
                },
                tokenizer=directory,
            )
            train_mesh = compact_grug_mesh(expert_axis_size=4, replica_axis_size=1)
            with jax.set_mesh(train_mesh):
                trained = model.Transformer.init(config, key=jax.random.key(3))
                state = {
                    "step": jnp.array(17, dtype=jnp.int32),
                    "params": trained,
                    "opt_state": {"momentum": jnp.ones((8, 8))},
                }
                save_checkpoint(state, step=17, checkpoint_path=directory + "/step-17", is_temporary=False)

            def scores(qk_mult):
                eval_config = CheckpointEvalConfig(
                    model=dataclasses.replace(config, qk_mult=qk_mult),
                    data=data,
                    trainer=TrainerConfig(mp=jmp.get_policy("float32")),
                    resources=ResourceConfig.with_cpu(),
                    checkpoint_path=directory + "/step-17",
                    seq_lens=(8, 16),
                    eval_tokens_per_batch=64,
                    dropless_moe_implementation="scatter",
                    compute_bpb=False,
                )
                return evaluate_checkpoint(eval_config)

            step, metrics = scores(1.0)
            _, rescaled = scores(1.5)

            # The reference: the saved weights scored directly, window by window.
            reference = {}
            eval_mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
            with jax.set_mesh(eval_mesh):
                weights = _reshard_tree_to_mesh(trained, train_mesh, eval_mesh)
                for seq_len in (8, 16):
                    # Right padding never reaches a real token under causal attention, so zero rows
                    # and zero weights reproduce the padded final window. Zero-weight rows also fill
                    # the batch to the four data shards.
                    row_count = 8 if seq_len == 8 else 4
                    rows = np.zeros((row_count * seq_len,), dtype=np.int32)
                    rows[: len(stream)] = stream
                    rows = rows.reshape(row_count, seq_len)
                    flat_weight = np.arange(row_count * seq_len) < len(stream) - 1
                    loss_weight = flat_weight.astype(np.float32).reshape(row_count, seq_len)
                    loss_weight[:, -1] = 0
                    per_token = weights.next_token_loss(
                        jnp.asarray(rows), jnp.asarray(loss_weight), mask=model.AttentionMask.causal(),
                        reduction="none", logsumexp_weight=None,
                    )
                    reference[seq_len] = float(jnp.sum(per_token * loss_weight) / loss_weight.sum())

        assert step == 17
        for seq_len in (8, 16):
            prefix = f"eval_dropless/seq_len_{seq_len}"
            assert metrics[prefix + "/paloma/tiny/tokens"] == 37 - -(-37 // seq_len)
            np.testing.assert_allclose(metrics[prefix + "/paloma/tiny/loss"], reference[seq_len], rtol=1e-4)
            assert abs(rescaled[prefix + "/loss"] - metrics[prefix + "/loss"]) > 1e-4
    """
    run_on_cpu_devices(script, device_count=4)
