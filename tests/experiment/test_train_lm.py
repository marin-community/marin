# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``train_lm`` must assemble the decisions it is handed and chain checkpoints.

These pin the assembler's contract independent of the executor: the resolved
``TrainLmConfig`` carries exactly the model/optimizer/budget/z-loss it was given,
``evals=None`` produces no harness while a suite wires one in, and ``init_from`` both
adds the parent as a dependency and seeds ``initialize_from_checkpoint_path``.
"""

from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.optim import AdamConfig
from marin.execution.lazy import lower, materialized_config
from marin.experiment.data import mixture, tokenized
from marin.experiment.train import train_lm

from experiments.recipes import core_tasks

_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
_PREFIX = "gs://marin-test"
_MODEL = LlamaConfig(max_seq_len=128, hidden_dim=64, intermediate_dim=128, num_heads=4, num_kv_heads=4, num_layers=2)
_OPTIMIZER = AdamConfig(learning_rate=1e-3, weight_decay=0.1, warmup=10, min_lr_ratio=0.1)
_RESOURCES = ResourceConfig.with_tpu("v4-8")


def _corpus():
    return tokenized("corpus", source="org/corpus", tokenizer=_TOKENIZER)


def _build(*, evals=None, init_from=None, deps):
    return train_lm(
        name="checkpoints/unit",
        model=_MODEL,
        optimizer=_OPTIMIZER,
        data=lambda ctx: mixture(ctx, {deps[0]: 1.0}),
        deps=deps,
        batch_size=8,
        seq_len=128,
        num_train_steps=50,
        z_loss_weight=1e-4,
        evals=evals,
        resources=_RESOURCES,
        init_from=init_from,
    )


def test_assembles_the_given_decisions():
    corpus = _corpus()
    pod = materialized_config(_build(deps=(corpus,)), _PREFIX)
    tc = pod.train_config

    assert tc.model is _MODEL
    assert tc.optimizer is _OPTIMIZER
    assert tc.train_seq_len == 128
    assert tc.z_loss_weight == 1e-4
    assert tc.trainer.train_batch_size == 8
    assert tc.trainer.num_train_steps == 50
    assert pod.output_path == f"{_PREFIX}/checkpoints/unit/v1"
    # The TPU rides as a run-arg, resolved into the pod config, never the fingerprint.
    assert pod.resources == _RESOURCES


def test_evals_none_means_no_harness():
    corpus = _corpus()
    tc = materialized_config(_build(deps=(corpus,)), _PREFIX).train_config
    assert tc.eval_harness is None
    assert tc.eval_harness_steps is None


def test_eval_suite_wires_a_harness():
    corpus = _corpus()
    tc = materialized_config(_build(deps=(corpus,), evals=core_tasks(every=2000)), _PREFIX).train_config
    assert tc.eval_harness is not None
    assert tc.eval_harness_steps == 2000


def test_init_from_chains_the_parent():
    parent = _build(deps=(_corpus(),))
    child_corpus = tokenized("corpus2", source="org/corpus2", tokenizer=_TOKENIZER)
    child = _build(deps=(child_corpus,), init_from=parent)

    # The parent is both a build dependency (so it materializes first) and the weights
    # this run initializes from.
    assert parent in child.recipe.deps
    tc = materialized_config(child, _PREFIX).train_config
    assert tc.initialize_from_checkpoint_path == f"{_PREFIX}/checkpoints/unit/v1/checkpoints"


def test_lowers_to_a_runnable_graph():
    corpus = _corpus()
    spec = lower(_build(deps=(corpus,)))
    assert spec.name == "checkpoints/unit"
    # The corpus dependency is lowered into the graph.
    assert any(dep.name == "corpus" for dep in spec.deps)
