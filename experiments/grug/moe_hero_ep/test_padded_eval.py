# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import functools

import jax
import jmp
import numpy as np
import pytest
from haliax.partitioning import set_mesh
from levanter.data.text.datasets import DatasetComponent, LmDataConfig
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.eval import eval_model
from levanter.grug.sharding import compact_grug_mesh
from levanter.store.cache import write_levanter_cache
from levanter.testing.devices import use_test_devices
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from experiments.grug.moe_hero_ep import model, train


@pytest.mark.multidevice
def test_padded_window_scores_a_corpus_shorter_than_the_eval_length(tmp_path):
    # Three tokens in a 16-token window on CP2 x EP2: one whole context shard holds only padding. Right
    # padding must not reach a real token, so the loss must match the same corpus in an 8-token window.
    tokenizer = Tokenizer(WordLevel({"<unk>": 0, **{str(i): i for i in range(1, 128)}}, unk_token="<unk>"))
    tokenizer.save(str(tmp_path / "tokenizer.json"))
    cache_path = str(tmp_path / "tiny")
    write_levanter_cache([{"input_ids": np.array([1, 2, 3], dtype=np.int32)}], cache_path + "/validation", metadata={})
    data = LmDataConfig(
        components={"paloma/tiny": DatasetComponent(cache_dir=cache_path, format=TextLmDatasetFormat())},
        tokenizer=str(tmp_path),
        validation_remainder="pad",
    )
    config = model.GrugModelConfig(
        vocab_size=128,
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
        max_seq_len=16,
        sliding_window=4,
        global_every=2,
        capacity_factor=4.0,
        initializer_std=0.1,
        sconv=True,
        attention_implementation="reference",
        moe_implementation="fixed_all_to_all",
    )
    with use_test_devices(4):
        train_mesh = compact_grug_mesh(replica_axis_size=1, expert_axis_size=2, context_axis_size=2)
        eval_mesh = train.expert_collapsed_eval_mesh(train_mesh)
        with jax.set_mesh(train_mesh):
            transformer = model.Transformer.init(config, key=jax.random.key(0))
        losses = {}
        with set_mesh(eval_mesh):
            params = train._move_tree_to_eval_mesh(transformer, train_mesh)
            for seq_len in (8, 16):
                evaluator = train.build_tagged_evaluator(
                    data_config=data,
                    max_seq_len=seq_len,
                    mesh=eval_mesh,
                    eval_cfg=train.GrugEvalConfig(eval_batch_size=2, compute_bpb=False),
                    mp=jmp.get_policy("float32"),
                    model_transform=functools.partial(train._to_dropless_local, implementation="scatter"),
                )
                losses[seq_len] = eval_model(evaluator, params)["loss"]

    assert np.isfinite(losses[16])
    assert losses[16] > 0
    np.testing.assert_allclose(losses[16], losses[8], rtol=1e-5)
