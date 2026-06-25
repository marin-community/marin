# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Golden test: the inline-protocol DCLM experiment must resolve to the same
executor config as the ``default_train`` version.

This is the anti-drift guarantee for the experiment-redesign Stage 1: readable
inline code and the executed config provably cannot diverge. We compare the
executor's *content version* (the hashed output path), which is what determines
cache identity and is immune to callable identity — comparing the configs with
``==`` would spuriously fail on freshly-built closures.
"""

from fray.cluster import ResourceConfig
from marin.execution.executor import compute_output_path, executor_context

from experiments.defaults import default_train
from experiments.pretraining_datasets.dclm import dclm_mixture_config_llama3
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tutorials.dclm_1b_1x_inline import build, llama_1_4b_dclm


def _old_dclm_step():
    training_config = SimpleTrainConfig(
        train_seq_len=2048,
        resources=ResourceConfig.with_tpu("v4-128"),
        train_batch_size=256,
        num_train_steps=int(28.8e9) // (256 * 2048),
        learning_rate=3e-3,
        weight_decay=0.033,
        min_lr_ratio=0.1,
        warmup=5000,
        z_loss_weight=1e-4,
    )
    return default_train(
        name="dclm_1b_1x_how_to",
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_1_4b_dclm,
        train_config=training_config,
        tags=["HOWTOS", "DCLM_1B_1X"],
    )


def test_inline_dclm_matches_default_train(monkeypatch):
    # W&B group is a runtime binding: default_train reads it from the env and folds
    # it into the config, the inline helper does not. Clear it so the comparison
    # is on the protocol axes only.
    monkeypatch.delenv("WANDB_GROUP", raising=False)

    with executor_context():
        old = _old_dclm_step()
        new = build()

    assert old.name == new.name, f"\n old: {old.name}\n new: {new.name}"
    old_path = compute_output_path(old.name, old.config)
    new_path = compute_output_path(new.name, new.config)
    assert old_path == new_path, f"\n old: {old_path}\n new: {new_path}"
