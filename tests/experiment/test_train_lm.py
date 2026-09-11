# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""``train_lm`` must assemble the decisions it is handed and chain checkpoints.

These pin the assembler's contract independent of the executor: the resolved
``TrainLmConfig`` carries exactly the model/optimizer/budget/z-loss it was given,
``evals=None`` produces no harness while a suite wires one in, and ``init_from`` both
adds the parent as a dependency and seeds ``initialize_from_checkpoint_path``.

``mixture`` reads each dataset's tokenizer/format from its record at run time, so the
helpers below seed the minimal ``TokenizedCache`` record a built dataset would leave (a
real run materializes the datasets first, as build dependencies) before assembling.
"""

import json
import math
from pathlib import Path

import pytest
from fray.cluster import ResourceConfig
from levanter.models.llama import LlamaConfig
from levanter.optim.config import AdamConfig
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import ArtifactStep, materialized_config
from marin.experiment.data import tokenized
from marin.experiment.train import EvalSuite, train_lm
from marin.processing.tokenize.tokenize import TokenizedCache

from experiments.evals.task_configs import CORE_TASKS

_TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
_V = "2026.06.28"
_MODEL = LlamaConfig(max_seq_len=128, hidden_dim=64, intermediate_dim=128, num_heads=4, num_kv_heads=4, num_layers=2)
_OPTIMIZER = AdamConfig(learning_rate=1e-3, weight_decay=0.1, warmup=10, min_lr_ratio=0.1)
_RESOURCES = ResourceConfig.with_tpu("v4-8")
_TOKENIZED_CACHE = f"{TokenizedCache.__module__}.{TokenizedCache.__qualname__}"


def _seed_caches(train: ArtifactStep, prefix: str) -> None:
    """Write the minimal record a built ``TokenizedCache`` dep would leave, so ``mixture``
    can read each dataset's tokenizer/format offline."""
    for dep in train.deps:
        if dep.artifact_type is TokenizedCache:
            write_record(
                ArtifactRecord(
                    name=dep.name,
                    version=dep.version,
                    output_path=dep.path(prefix),
                    result_type=_TOKENIZED_CACHE,
                    config={"tokenizer": _TOKENIZER, "format": {"text_key": "text"}},
                )
            )


def _seed_train_tokens(train: ArtifactStep, prefix: str, total_tokens: int) -> None:
    """Write the ``train`` split's ``.stats.json`` a built cache dep would leave, so
    ``num_train_epochs`` can read its token total offline (as ``TokenizedCache.num_train_tokens``)."""
    for dep in train.deps:
        if dep.artifact_type is TokenizedCache:
            stats_dir = Path(dep.path(prefix)) / "train"
            stats_dir.mkdir(parents=True, exist_ok=True)
            (stats_dir / ".stats.json").write_text(json.dumps({"total_tokens": total_tokens, "total_elements": 1}))


def _assemble(train: ArtifactStep, prefix: str):
    """The pod config a run of ``train`` would receive, with its dataset records seeded."""
    _seed_caches(train, prefix)
    return materialized_config(train, prefix)


def _corpus():
    return tokenized("corpus", source="org/corpus", tokenizer=_TOKENIZER, version=_V)


def _build(*, evals=None, init_from=None, datasets):
    return train_lm(
        name="checkpoints/unit",
        model=_MODEL,
        optimizer=_OPTIMIZER,
        datasets=datasets,
        batch_size=8,
        seq_len=128,
        num_train_steps=50,
        z_loss_weight=1e-4,
        evals=evals,
        resources=_RESOURCES,
        init_from=init_from,
        version=_V,
    )


def test_assembles_the_given_decisions(tmp_path):
    prefix = str(tmp_path)
    corpus = _corpus()
    pod = _assemble(_build(datasets={corpus: 1.0}), prefix)
    tc = pod.train_config

    assert tc.model is _MODEL
    assert tc.optimizer is _OPTIMIZER
    assert tc.train_seq_len == 128
    assert tc.z_loss_weight == 1e-4
    assert tc.trainer.train_batch_size == 8
    assert tc.trainer.num_train_steps == 50
    assert pod.output_path == f"{prefix}/checkpoints/unit/{_V}"
    # The TPU rides as a run-arg, resolved into the pod config, never the fingerprint.
    assert pod.resources == _RESOURCES


def test_datasets_assemble_the_mixture_internally(tmp_path):
    corpus = _corpus()
    tc = _assemble(_build(datasets={corpus: 0.7}), str(tmp_path)).train_config
    # train_lm folds the {handle: weight} mapping into a mixture keyed by the handle name.
    assert tc.data.train_weights == {"corpus": 0.7}


def test_validation_handles_join_at_weight_zero(tmp_path):
    corpus = _corpus()
    held_out = tokenized("heldout", source="org/heldout", tokenizer=_TOKENIZER, version=_V)
    train = train_lm(
        name="checkpoints/unit",
        model=_MODEL,
        optimizer=_OPTIMIZER,
        datasets={corpus: 1.0},
        validation=(held_out,),
        batch_size=8,
        seq_len=128,
        num_train_steps=50,
        z_loss_weight=1e-4,
        evals=None,
        resources=_RESOURCES,
        version=_V,
    )
    tc = _assemble(train, str(tmp_path)).train_config
    assert tc.data.train_weights == {"corpus": 1.0, "heldout": 0.0}
    # Both datasets are build dependencies so they materialize before training.
    assert {dep.name for dep in train.deps} == {"corpus", "heldout"}


def test_evals_none_means_no_harness(tmp_path):
    corpus = _corpus()
    tc = _assemble(_build(datasets={corpus: 1.0}), str(tmp_path)).train_config
    assert tc.eval_harness is None
    assert tc.eval_harness_steps is None


def test_eval_suite_wires_a_harness(tmp_path):
    corpus = _corpus()
    tc = _assemble(_build(datasets={corpus: 1.0}, evals=EvalSuite(CORE_TASKS, every=2000)), str(tmp_path)).train_config
    assert tc.eval_harness is not None
    assert tc.eval_harness_steps == 2000


def test_init_from_chains_the_parent(tmp_path):
    parent = _build(datasets={_corpus(): 1.0})
    child_corpus = tokenized("corpus2", source="org/corpus2", tokenizer=_TOKENIZER, version=_V)
    child = _build(datasets={child_corpus: 1.0}, init_from=parent)

    # The parent is both a build dependency (so it materializes first) and the weights
    # this run initializes from.
    assert parent in child.deps
    tc = _assemble(child, str(tmp_path)).train_config
    assert tc.initialize_from_checkpoint_path == f"{tmp_path}/checkpoints/unit/{_V}/checkpoints"


def test_lowers_to_a_runnable_graph():
    corpus = _corpus()
    spec = _build(datasets={corpus: 1.0}).lower()
    assert spec.name == "checkpoints/unit"
    # The corpus dependency is lowered into the graph.
    assert any(dep.name == "corpus" for dep in spec.deps)


def _build_epochs(*, datasets, num_train_epochs):
    return train_lm(
        name="checkpoints/unit",
        model=_MODEL,
        optimizer=_OPTIMIZER,
        datasets=datasets,
        batch_size=8,
        seq_len=128,
        num_train_epochs=num_train_epochs,
        z_loss_weight=1e-4,
        evals=None,
        resources=_RESOURCES,
        version=_V,
    )


def test_num_train_epochs_resolves_steps_from_token_count(tmp_path):
    prefix = str(tmp_path)
    corpus = _corpus()
    train = _build_epochs(datasets={corpus: 1.0}, num_train_epochs=3)
    _seed_train_tokens(train, prefix, total_tokens=4096)

    # A pass over the tokens is 4096 / (seq_len 128 * batch 8) = 4 steps; 3 epochs -> 12 steps.
    # This is derived from the packed token total, not from any hand-set num_train_steps.
    tc = _assemble(train, prefix).train_config
    assert tc.trainer.num_train_steps == math.ceil(3 * 4096 / (128 * 8)) == 12


def test_num_train_epochs_rounds_up_a_partial_final_batch(tmp_path):
    prefix = str(tmp_path)
    corpus = _corpus()
    train = _build_epochs(datasets={corpus: 1.0}, num_train_epochs=1)
    # One token past four full batches must still be covered, so the count rounds up to 5.
    _seed_train_tokens(train, prefix, total_tokens=128 * 8 * 4 + 1)

    tc = _assemble(train, prefix).train_config
    assert tc.trainer.num_train_steps == 5


def test_num_train_epochs_is_part_of_the_run_identity(tmp_path):
    corpus = _corpus()
    one = _build_epochs(datasets={corpus: 1.0}, num_train_epochs=1)
    two = _build_epochs(datasets={corpus: 1.0}, num_train_epochs=2)
    # The token total is unknown at fingerprint time, but the epoch count still distinguishes runs
    # so a 1-epoch and a 2-epoch run do not collide on one checkpoint.
    assert one.fingerprint() != two.fingerprint()


@pytest.mark.parametrize("num_train_steps,num_train_epochs", [(50, 2), (None, None)])
def test_exactly_one_of_steps_or_epochs_is_required(num_train_steps, num_train_epochs):
    with pytest.raises(ValueError, match="Exactly one of num_train_steps or num_train_epochs"):
        train_lm(
            name="checkpoints/unit",
            model=_MODEL,
            optimizer=_OPTIMIZER,
            datasets={_corpus(): 1.0},
            batch_size=8,
            seq_len=128,
            num_train_steps=num_train_steps,
            num_train_epochs=num_train_epochs,
            z_loss_weight=1e-4,
            evals=None,
            resources=_RESOURCES,
            version=_V,
        )


def test_num_train_epochs_rejects_a_mixture():
    a = tokenized("a", source="org/a", tokenizer=_TOKENIZER, version=_V)
    b = tokenized("b", source="org/b", tokenizer=_TOKENIZER, version=_V)
    with pytest.raises(ValueError, match="single training dataset"):
        _build_epochs(datasets={a: 0.5, b: 0.5}, num_train_epochs=1)


def test_dev_checkpoints_are_per_user_while_datasets_stay_shared(tmp_path, monkeypatch):
    def build_as(user: str):
        monkeypatch.setattr("rigging.provenance._getuser", lambda: user)
        return train_lm(
            name="checkpoints/unit",
            model=_MODEL,
            optimizer=_OPTIMIZER,
            datasets={_corpus(): 1.0},
            batch_size=8,
            seq_len=128,
            num_train_steps=50,
            z_loss_weight=1e-4,
            evals=None,
            resources=_RESOURCES,
            version="dev",
        )

    alice = build_as("alice")
    bob = build_as("bob")

    # Two people running the same dev experiment write to distinct, per-user checkpoint paths
    # instead of clobbering each other.
    assert alice.path(str(tmp_path)) == f"{tmp_path}/users/alice/checkpoints/unit/dev"
    assert bob.path(str(tmp_path)) == f"{tmp_path}/users/bob/checkpoints/unit/dev"

    # The shared dataset keeps its un-namespaced name in the assembled data config, so its cache
    # is reused across users rather than re-tokenized per person.
    tc = _assemble(alice, str(tmp_path)).train_config
    assert tc.data.train_weights == {"corpus": 1.0}
