# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import tempfile

import jax
import jax.numpy as jnp
import jmp
import numpy as np
import pytest
from fray.cluster import ResourceConfig
from levanter.checkpoint import save_checkpoint
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.grug.sharding import compact_grug_mesh
from levanter.store.cache import write_levanter_cache
from levanter.testing.devices import use_test_devices
from levanter.trainer import TrainerConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from experiments.grug.moe_hero_ep import model
from experiments.grug.moe_hero_ep.launch_checkpoint_eval import (
    CheckpointEvalConfig,
    EvalCheckpoint,
    eval_tokens_per_batch,
    evaluate_checkpoints,
)
from experiments.grug.moe_hero_ep.train import _reshard_tree_to_mesh


@pytest.mark.timeout(240)
def test_checkpoint_eval_scores_each_checkpoint_at_each_length_with_its_own_qk_mult_and_pads_the_final_window():
    with use_test_devices(4):
        config = model.GrugModelConfig(
            vocab_size=64,
            hidden_dim=32,
            intermediate_dim=16,
            shared_expert_intermediate_dim=16,
            num_shared_experts=1,
            num_experts=4,
            num_experts_per_token=2,
            num_layers=2,
            num_heads=4,
            num_kv_heads=2,
            local_kv_heads=2,
            global_kv_heads=2,
            head_dim=8,
            max_seq_len=8,
            sliding_window=4,
            global_every=2,
            initializer_std=0.1,
            qk_mult=1.0,
            attention_implementation="reference",
            moe_implementation="fixed_all_to_all",
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
                validation_remainder="pad",
            )
            with jax.set_mesh(compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)):
                trained = model.Transformer.init(config, key=jax.random.key(3))
                state = {
                    "step": jnp.array(17, dtype=jnp.int32),
                    "params": trained,
                    "opt_state": {"momentum": jnp.ones((8, 8))},
                }
                save_checkpoint(state, step=17, checkpoint_path=directory + "/step-17", is_temporary=False)
                # The same weights saved at a later step, scored with a larger qk_mult.
                save_checkpoint(
                    {**state, "step": jnp.array(23, dtype=jnp.int32)},
                    step=23,
                    checkpoint_path=directory + "/step-23",
                    is_temporary=False,
                )

            eval_config = CheckpointEvalConfig(
                model=config,
                data=data,
                trainer=TrainerConfig(mp=jmp.get_policy("float32")),
                resources=ResourceConfig.with_cpu(),
                checkpoints=(EvalCheckpoint(directory + "/step-17", 1.0), EvalCheckpoint(directory + "/step-23", 1.5)),
                seq_lens=(8, 16),
                eval_tokens_per_batch=64,
                dropless_moe_implementation="scatter",
                compute_bpb=False,
            )
            (step, metrics), (later, rescaled) = evaluate_checkpoints(eval_config)

            # The reference: the saved weights scored directly, window by window.
            reference = {}
            eval_mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
            with jax.set_mesh(eval_mesh):
                weights = _reshard_tree_to_mesh(trained, eval_mesh)
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
                        jnp.asarray(rows),
                        jnp.asarray(loss_weight),
                        mask=model.AttentionMask.causal(),
                        reduction="none",
                        logsumexp_weight=None,
                    )
                    reference[seq_len] = float(jnp.sum(per_token * loss_weight) / loss_weight.sum())

        assert (step, later) == (17, 23)
        for seq_len in (8, 16):
            prefix = f"eval_dropless/seq_len_{seq_len}"
            np.testing.assert_allclose(metrics[prefix + "/paloma/tiny/loss"], reference[seq_len], rtol=1e-4)
            assert abs(rescaled[prefix + "/loss"] - metrics[prefix + "/loss"]) > 1e-4


@pytest.mark.parametrize(("nodes", "sequences"), [(16, 16), (8, 8), (4, 4)])
def test_fewer_nodes_keep_the_tokens_per_gpu_and_one_row_per_context_group(nodes, sequences):
    tokens = eval_tokens_per_batch(nodes=nodes, context_axis_size=4, seq_lens=(4096, 65536, 262144))
    assert tokens // 262144 == sequences


@pytest.mark.parametrize(
    ("nodes", "context", "seq_len"),
    [
        (3, 4, 262144),  # does not divide the rack
        (16, 3, 4096),  # context does not divide the GPUs
        (2, 1, 262144),  # two sequences cannot fill eight batch shards
    ],
)
def test_a_shape_that_does_not_divide_is_refused(nodes, context, seq_len):
    with pytest.raises(ValueError, match=r"must divide|does not give whole sequences"):
        eval_tokens_per_batch(nodes=nodes, context_axis_size=context, seq_lens=(seq_len,))
