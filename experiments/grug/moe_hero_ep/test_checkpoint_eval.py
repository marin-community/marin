# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from levanter.testing.cpu_devices import run_on_cpu_devices


@pytest.mark.timeout(240)
def test_checkpoint_eval_scores_each_checkpoint_at_each_length_with_its_own_qk_mult():
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
        from experiments.grug.moe_hero_ep.launch_checkpoint_eval import (
            CheckpointEvalConfig, EvalCheckpoint, evaluate_checkpoints,
        )
        from experiments.grug.moe_hero_ep.train import _reshard_tree_to_mesh

        config = model.GrugModelConfig(
            vocab_size=64, hidden_dim=32, intermediate_dim=16, shared_expert_intermediate_dim=16,
            num_shared_experts=1, num_experts=4, num_experts_per_token=2, num_layers=2, num_heads=4,
            num_kv_heads=2, local_kv_heads=2, global_kv_heads=2, head_dim=8, max_seq_len=8,
            sliding_window=4, global_every=2, initializer_std=0.1, qk_mult=1.0,
            attention_implementation="reference", moe_implementation="fixed_all_to_all",
        )
        # 37 tokens: 4 windows of 8 and 2 windows of 16, each dropping the final 5 tokens.
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
            with jax.set_mesh(compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)):
                trained = model.Transformer.init(config, key=jax.random.key(3))
                state = {
                    "step": jnp.array(17, dtype=jnp.int32),
                    "params": trained,
                    "opt_state": {"momentum": jnp.ones((8, 8))},
                }
                save_checkpoint(state, step=17, checkpoint_path=directory + "/step-17", is_temporary=False)
                # The same weights saved at a later step, scored with a larger qk_mult.
                later_state = {**state, "step": jnp.array(23, dtype=jnp.int32)}
                save_checkpoint(later_state, step=23, checkpoint_path=directory + "/step-23", is_temporary=False)

            eval_config = CheckpointEvalConfig(
                model=config,
                data=data,
                trainer=TrainerConfig(mp=jmp.get_policy("float32")),
                resources=ResourceConfig.with_cpu(),
                checkpoints=(
                    EvalCheckpoint.single(directory + "/step-17", 1.0),
                    # A 50/50 merge of two saves of the same weights is those weights.
                    EvalCheckpoint(((directory + "/step-17", 0.5), (directory + "/step-23", 0.5)), 1.5, label="m"),
                ),
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
                    windows = stream[: len(stream) // seq_len * seq_len].reshape(-1, seq_len)
                    # Zero-weight rows fill the batch to the four data shards.
                    rows = np.zeros((4, seq_len), dtype=np.int32)
                    rows[: len(windows)] = windows
                    loss_weight = np.zeros((4, seq_len), dtype=np.float32)
                    loss_weight[: len(windows), : seq_len - 1] = 1
                    per_token = weights.next_token_loss(
                        jnp.asarray(rows), jnp.asarray(loss_weight), mask=model.AttentionMask.causal(),
                        reduction="none", logsumexp_weight=None,
                    )
                    reference[seq_len] = float(jnp.sum(per_token * loss_weight) / loss_weight.sum())

        assert (step, later) == (17, 23)
        for seq_len in (8, 16):
            prefix = f"eval_dropless/seq_len_{seq_len}"
            assert metrics[prefix + "/paloma/tiny/tokens"] == (37 // seq_len) * (seq_len - 1)
            np.testing.assert_allclose(metrics[prefix + "/paloma/tiny/loss"], reference[seq_len], rtol=1e-4)
            assert abs(rescaled[prefix + "/loss"] - metrics[prefix + "/loss"]) > 1e-4
    """
    run_on_cpu_devices(script, device_count=4)


@pytest.mark.timeout(240)
def test_hyperball_merge_keeps_each_sphere_norm_and_linear_merge_shrinks_it():
    script = """
        import tempfile

        import jax
        import jax.numpy as jnp
        import jmp
        import numpy as np
        from levanter.checkpoint import save_checkpoint
        from levanter.grug.sharding import compact_grug_mesh
        from levanter.utils.jax_utils import leaf_key_paths

        from experiments.grug.moe_hero_ep import model
        from experiments.grug.moe_hero_ep.launch_checkpoint_eval import (
            EvalCheckpoint, MergeSpace, hyperball_axes, merged_params,
        )
        from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig

        config = model.GrugModelConfig(
            vocab_size=64, hidden_dim=32, intermediate_dim=16, shared_expert_intermediate_dim=16,
            num_shared_experts=1, num_experts=4, num_experts_per_token=2, num_layers=2, num_heads=4,
            num_kv_heads=2, local_kv_heads=2, global_kv_heads=2, head_dim=8, max_seq_len=8,
            sliding_window=4, global_every=2, initializer_std=0.1, qk_mult=1.0,
            attention_implementation="reference", moe_implementation="fixed_all_to_all",
        )
        mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
        mp = jmp.get_policy("float32")
        with tempfile.TemporaryDirectory() as directory, jax.set_mesh(mesh):
            first = model.Transformer.init(config, key=jax.random.key(3))
            second = model.Transformer.init(config, key=jax.random.key(4))
            for step, params in ((17, first), (23, second)):
                state = {"step": jnp.array(step, dtype=jnp.int32), "params": params}
                save_checkpoint(state, step=step, checkpoint_path=f"{directory}/step-{step}", is_temporary=False)
            sources = ((directory + "/step-17", 0.5), (directory + "/step-23", 0.5))

            linear, step, metrics = merged_params(config, EvalCheckpoint(sources, 1.0), mesh=mesh, mp=mp)
            on_ball = EvalCheckpoint(sources, 1.0, space=MergeSpace.HYPERBALL)
            ball, _, _ = merged_params(config, on_ball, mesh=mesh, mp=mp)

            paths = jax.tree.leaves(leaf_key_paths(first))
            groups = jax.tree.leaves(GrugMoeMuonHConfig().create_mask(first))
            leaves = zip(jax.tree.leaves(first), jax.tree.leaves(second), jax.tree.leaves(linear),
                         jax.tree.leaves(ball), paths, groups, strict=True)
            spheres = 0
            for a, b, lin, hb, path, group in leaves:
                np.testing.assert_allclose(lin, (a + b) / 2, rtol=1e-6, atol=1e-7)
                axes = hyperball_axes(path, a.ndim, group)
                if axes is None:
                    np.testing.assert_allclose(hb, lin, rtol=1e-6)
                    continue
                spheres += 1
                norm = lambda x: np.sqrt(np.sum(np.square(np.asarray(x)), axis=axes, keepdims=True))
                np.testing.assert_allclose(norm(hb), (norm(a) + norm(b)) / 2, rtol=1e-5)
                # The same direction as the linear average, only rescaled.
                np.testing.assert_allclose(np.asarray(hb) / norm(hb), np.asarray(lin) / norm(lin), atol=1e-6)
        assert step == 23 and spheres > 0
        assert metrics["merge/norm_ratio_min"] < metrics["merge/norm_ratio_mean"] < 1.0
    """
    run_on_cpu_devices(script, device_count=1)
