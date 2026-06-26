# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The inline-protocol DCLM experiment must resolve to the same training *decisions*
as the ``default_train`` recipe, so the readable inline code and the executed config
cannot drift.

The inline run is a lazy artifact and ``default_train`` is an executor step, so their
output and cache *paths* differ by construction (explicit ``name@version`` vs content
hash). We compare the path-independent decisions: model, optimizer, batch/steps/seq,
z-loss, eval cadence, and the training mixture weights.
"""

from dataclasses import asdict

from fray.cluster import ResourceConfig
from marin.execution.executor import Executor, executor_context
from marin.execution.lazy import materialized_config

from experiments.defaults import default_train
from experiments.pretraining_datasets.dclm import dclm_mixture_config_llama3
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tutorials.dclm_1b_1x_inline import build, llama_1_4b_dclm


def _decisions(train_lm_config) -> dict:
    """The path-independent experimental decisions of a resolved ``TrainLmConfig``."""
    tc = train_lm_config
    return {
        "model": asdict(tc.model),
        "optimizer": asdict(tc.optimizer),
        "train_batch_size": tc.trainer.train_batch_size,
        "num_train_steps": tc.trainer.num_train_steps,
        "train_seq_len": tc.train_seq_len,
        "z_loss_weight": tc.z_loss_weight,
        "steps_per_eval": tc.trainer.steps_per_eval,
        "eval_harness_steps": tc.eval_harness_steps,
        # The mixture policy: training components and their weights. Validation sets are
        # zero-weight and are named differently in the lazy vs executor catalogs.
        "train_weights": {name: weight for name, weight in tc.data.train_weights.items() if weight > 0},
    }


def _default_train_decisions() -> dict:
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
    step = default_train(
        name="dclm_1b_1x_how_to",
        tokenized=dclm_mixture_config_llama3,
        model_config=llama_1_4b_dclm,
        train_config=training_config,
        tags=["HOWTOS", "DCLM_1B_1X"],
    )
    executor = Executor(prefix="gs://marin-golden", executor_info_base_path="gs://marin-golden/experiments")
    executor.compute_version(step, is_pseudo_dep=False)
    return _decisions(executor.configs[step].train_config)


def test_inline_dclm_matches_default_train_decisions(monkeypatch):
    # W&B group is a runtime binding default_train folds in from the env; clear it so
    # the comparison is on the protocol axes only.
    monkeypatch.delenv("WANDB_GROUP", raising=False)

    with executor_context():
        old = _default_train_decisions()
    new = _decisions(materialized_config(build(), "gs://marin-golden").train_config)

    assert old == new, f"\n old: {old}\n new: {new}"
