# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import tempfile
from pathlib import Path

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

from experiments.grug.moe_hero_ep import launch_checkpoint_eval, model
from experiments.grug.moe_hero_ep.checkpoint_merge import (
    merge_plan_from_json,
    reproject_to_reference,
    running_form,
    weighted_sum,
)
from experiments.grug.moe_hero_ep.launch_checkpoint_eval import (
    CheckpointEvalConfig,
    EvalCheckpoint,
    eval_tokens_per_batch,
    evaluate_checkpoints,
    evaluate_merge_plan,
)
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
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


def _tiny_merge_config() -> model.GrugModelConfig:
    return model.GrugModelConfig(
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


def _tree_digest(root: Path) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in root.rglob("*") if path.is_file()}


@pytest.mark.timeout(480)
def test_merge_plan_scores_each_candidate_as_its_recipe_without_writing_to_the_checkpoints(tmp_path):
    """Every code path of a pass (running sum, own accumulator, bias mix, reprojection) against direct arithmetic."""
    config = _tiny_merge_config()
    stream = (np.arange(64) % 61 + 1).astype(np.int32)
    windows = {"T": [40], "U3": [40, 30, 20], "S10": [10]}
    plan = merge_plan_from_json(
        json.dumps(
            {
                "checkpoint_root": str(tmp_path / "checkpoints"),
                "qk_mult": 1.0,
                "score_raw_steps": [30],
                "windows": windows,
                "candidates": [
                    {"name": "raw", "mix": {"T": 1.0}},
                    {"name": "raw+pending", "mix": {"T": 1.0}, "bias": {"source": "pending", "mix": {"T": 1.0}}},
                    {"name": "raw+meanbias", "mix": {"T": 1.0}, "bias": {"source": "params", "mix": {"U3": 1.0}}},
                    {"name": "U3", "mix": {"U3": 1.0}},
                    {"name": "blend", "mix": {"T": 0.75, "U3": 0.25}},
                    {"name": "blend+reproject", "mix": {"T": 0.75, "U3": 0.25}, "reproject_to_newest": True},
                    {"name": "sparse", "mix": {"T": 0.5, "S10": 0.5}},
                ],
            }
        )
    )
    with use_test_devices(4):
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
            sources: dict[int, model.Transformer] = {}
            pending: dict[int, jax.Array] = {}
            with jax.set_mesh(compact_grug_mesh(expert_axis_size=2, replica_axis_size=1)):
                for seed, step in enumerate((10, 20, 30, 40)):
                    params = model.Transformer.init(config, key=jax.random.key(seed))
                    # Stored bias and pending bias differ per source, as they do on a real trajectory.
                    stored = jax.random.normal(jax.random.key(100 + seed), (2, 4))
                    params = model.apply_qb_betas(params, stored)
                    sources[step] = params
                    pending[step] = jax.random.normal(jax.random.key(200 + seed), (2, 4))
                    save_checkpoint(
                        {"step": jnp.array(step, dtype=jnp.int32), "params": params, "pending_qb_betas": pending[step]},
                        step=step,
                        checkpoint_path=f"{tmp_path}/checkpoints/step-{step}",
                        is_temporary=False,
                    )
            before = _tree_digest(tmp_path)
            eval_config = CheckpointEvalConfig(
                model=config,
                data=data,
                trainer=TrainerConfig(mp=jmp.get_policy("float32")),
                resources=ResourceConfig.with_cpu(),
                checkpoints=(),
                merge=plan,
                seq_lens=(8,),
                eval_tokens_per_batch=64,
                dropless_moe_implementation="scatter",
                compute_bpb=False,
            )
            scores, distances = evaluate_merge_plan(eval_config)
            assert _tree_digest(tmp_path) == before

            def mix(weights: dict[int, float]) -> model.Transformer:
                leaves = [jax.tree.map(lambda x, w=w: w * np.asarray(x), sources[s]) for s, w in weights.items()]
                return jax.tree.map(lambda *xs: jnp.asarray(sum(xs)), *leaves)

            def bias_of(step: int) -> jax.Array:
                return sources[step].stacked_blocks.stacked.mlp.router_bias

            third = 1.0 / 3.0
            u3 = {40: third, 30: third, 20: third}
            blend = {40: 0.75 + 0.25 * third, 30: 0.25 * third, 20: 0.25 * third}
            mean_bias = sum(bias_of(step) for step in u3) / 3
            expected = {
                "step-30": sources[30],
                "raw": sources[40],
                "raw+pending": model.apply_qb_betas(sources[40], pending[40]),
                "raw+meanbias": model.apply_qb_betas(sources[40], -mean_bias),
                "U3": mix(u3),
                "blend": mix(blend),
                "blend+reproject": _reproject_reference(mix(blend), sources[40]),
                "sparse": mix({40: 0.5, 10: 0.5}),
            }

            eval_mesh = compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)
            with jax.set_mesh(eval_mesh):
                rows = jnp.asarray(stream[:64].reshape(8, 8))
                loss_weight = np.ones((8, 8), dtype=np.float32)
                loss_weight[:, -1] = 0
                loss_weight = jnp.asarray(loss_weight)

                def direct(params: model.Transformer) -> float:
                    per_token = _reshard_tree_to_mesh(params, eval_mesh).next_token_loss(
                        rows, loss_weight, mask=model.AttentionMask.causal(), reduction="none", logsumexp_weight=None
                    )
                    return float(jnp.sum(per_token * loss_weight) / loss_weight.sum())

                reference = {name: direct(params) for name, params in expected.items()}

    prefix = "eval_merge/{}/seq_len_8/paloma/tiny/loss"
    assert set(scores) == set(reference)
    for name, value in reference.items():
        np.testing.assert_allclose(scores[name][prefix.format(name)], value, rtol=1e-4, err_msg=name)
    # The recipes are distinct models, so a candidate that silently scored the raw weights would fail above.
    assert len({round(value, 4) for value in reference.values()}) == len(reference)

    leaf = "token_embed"
    reference_distance = np.sum((np.asarray(sources[30].token_embed) - np.asarray(sources[40].token_embed)) ** 2)
    np.testing.assert_allclose(distances[30][f"{leaf}/dist2_newest"], reference_distance, rtol=1e-4)
    # The running sum holds all sources before this one, thus the mean of two at step 20.
    prior_mean = (np.asarray(sources[40].token_embed) + np.asarray(sources[30].token_embed)) / 2
    np.testing.assert_allclose(
        distances[20][f"{leaf}/dist2_prior_mean"],
        np.sum((np.asarray(sources[20].token_embed) - prior_mean) ** 2),
        rtol=1e-4,
    )


def _reproject_reference(params: model.Transformer, target: model.Transformer) -> model.Transformer:
    """Independent reference for the fixed-norm groups: whole-matrix when 2-D, per layer when stacked."""
    labels = GrugMoeMuonHConfig().create_mask(params)

    def rescale(label, leaf, ref):
        if label not in ("muonh", "adamh"):
            return leaf
        leaf_np, ref_np = np.asarray(leaf), np.asarray(ref)
        if leaf_np.ndim == 2:
            return jnp.asarray(leaf_np * np.linalg.norm(ref_np) / np.linalg.norm(leaf_np))
        out = [layer * np.linalg.norm(r) / np.linalg.norm(layer) for layer, r in zip(leaf_np, ref_np, strict=True)]
        return jnp.asarray(np.stack(out))

    return jax.tree.map(rescale, labels, params, target)


def test_reprojection_restores_each_fixed_norm_to_the_reference_and_leaves_adam_groups_alone():
    with use_test_devices(1):
        config = _tiny_merge_config()
        with jax.set_mesh(compact_grug_mesh(expert_axis_size=1, replica_axis_size=1)):
            reference = model.Transformer.init(config, key=jax.random.key(0))
            other = model.Transformer.init(config, key=jax.random.key(1))
            mean = weighted_sum([(0.5, reference), (0.5, other)])
            reprojected = reproject_to_reference(mean, reference)
            mlp = reprojected.stacked_blocks.stacked.mlp
            ref_mlp = reference.stacked_blocks.stacked.mlp
            # Expert banks hold one sphere per layer, across all of a layer's experts.
            np.testing.assert_allclose(
                np.linalg.norm(np.asarray(mlp.expert_mlp.w_up).reshape(2, -1), axis=1),
                np.linalg.norm(np.asarray(ref_mlp.expert_mlp.w_up).reshape(2, -1), axis=1),
                rtol=1e-5,
            )
            np.testing.assert_allclose(
                np.linalg.norm(np.asarray(reprojected.output_proj)),
                np.linalg.norm(np.asarray(reference.output_proj)),
                rtol=1e-5,
            )
            # Adam groups have no fixed norm, so they keep the plain mean.
            np.testing.assert_array_equal(np.asarray(reprojected.token_embed), np.asarray(mean.token_embed))
            np.testing.assert_array_equal(np.asarray(mlp.router), np.asarray(mean.stacked_blocks.stacked.mlp.router))


def test_stage1_plan_reads_each_source_once_newest_first_with_tsa_alpha_on_the_mean():
    plan_path = Path(launch_checkpoint_eval.__file__).parent / "hero_merge_plan_stage1.json"
    plan = merge_plan_from_json(plan_path.read_text())
    read_order = plan.source_steps()
    assert read_order[0] == 390251 and read_order[-2:] == (384000, 378000) and len(read_order) == 13
    candidates = {candidate.name: candidate for candidate in plan.candidates}
    # Terminal shrinkage averaging: alpha weights the mean, so a25 stays closest to the newest checkpoint.
    a25 = running_form(candidates["B11a25"].weights, read_order)
    assert a25.last_index == 10
    np.testing.assert_allclose([a25.newest, a25.running], [0.75, 0.25 / 11], rtol=1e-9)
    u4 = running_form(candidates["U4"].weights, read_order)
    np.testing.assert_allclose([u4.newest, u4.running, u4.last_index], [0.0, 0.25, 3], atol=1e-12)
    # The LR-adjusted mix skips the window, so it needs its own accumulator.
    assert running_form(candidates["W3"].weights, read_order) is None
