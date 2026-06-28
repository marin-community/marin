# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: an LR/WD sweep over a 30M DCLM model, in the lazy-artifact style.

A sweep on the DCLM mixture with held-out validation and the CORE eval harness wired in,
selecting on the held-out loss:

- the DCLM training components and the Paloma + Uncheatable validation sets are
  :class:`~marin.execution.artifact.Dataset` handles, tokenized once and shared by every
  trial;
- :func:`~marin.experiment.sweep.sweep` fans out one
  :func:`~marin.experiment.train.train_lm` trial per grid point;
- each trial dispatches its own training job, which records its validation loss next to its
  checkpoints;
- :func:`~marin.experiment.sweep.select` reads each trial's recorded ``eval/loss`` (the
  validation micro-average) and reduces the trials to the lowest-loss one.

Run it against a cluster (with ``MARIN_PREFIX`` pointing at a bucket co-regional with
``RESOURCES``)::

    python -m experiments.tutorials.train_tiny_sweep
"""

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.artifact import Checkpoint
from marin.execution.lazy import Lazy, lower, resolve
from marin.execution.step_runner import StepRunner
from marin.experiment.sweep import Selection, select, sweep
from marin.experiment.train import train_lm

from experiments.evals.uncheatable import uncheatable_validation
from experiments.llama import llama3_tokenizer, llama_30m
from experiments.paloma import paloma_validation
from experiments.pretraining_datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_datasets
from experiments.recipes import core_tasks

# A single-host TPU slice; each trial trains on its own slice. This is a run-arg, not
# part of a trial's identity — re-running on a different TPU is the same checkpoint.
RESOURCES = ResourceConfig.with_tpu("v4-8")

# The selection metric: levanter logs the validation micro-average loss under this key,
# so the run's final value ranks the trials on held-out data.
SELECTION_METRIC = "eval/loss"

# The corpus and held-out validation: tokenized once into shared caches, reused by every
# trial by name@version. The first trial to run tokenizes them (their own Fray jobs).
_train = dclm_datasets(tokenizer=llama3_tokenizer)
_validation = [*paloma_validation(tokenizer=llama3_tokenizer), *uncheatable_validation(tokenizer=llama3_tokenizer)]
_weighted = {_train[name]: DCLM_MIXTURE_WEIGHTS[name] for name in _train}


def trial(*, learning_rate: float, weight_decay: float, version: str = "v1") -> Lazy[Checkpoint]:
    """One sweep trial: a 30M llama trained on the DCLM mixture, ready to select over.

    The swept hyperparameters are literals in the optimizer, so each grid point gets a
    distinct ``name@version`` and fingerprint; the TPU is a run-arg, excluded from
    identity. :func:`select` reads the checkpoint's recorded ``train_lm`` metrics and
    ranks it by ``SELECTION_METRIC``.
    """
    return train_lm(
        name=f"checkpoints/tutorial-dclm-30m-sweep/lr{learning_rate}-wd{weight_decay}",
        version=version,
        model=llama_30m,
        optimizer=AdamConfig(learning_rate=learning_rate, weight_decay=weight_decay),
        datasets=_weighted,
        validation=_validation,
        batch_size=128,
        seq_len=llama_30m.max_seq_len,
        num_train_steps=10000,
        z_loss_weight=None,
        evals=core_tasks(every=10000),
        resources=RESOURCES,
        tags=["llama", "30m", "dclm", "tutorial", "sweep", "test20251117"],
    )


def best_trial(*, version: str = "v1") -> Lazy[Selection]:
    """The full LR/WD sweep, reduced to its lowest validation-loss trial."""
    trials = sweep(
        trial,
        learning_rate=[3e-4, 6e-4, 1e-3],
        weight_decay=[0.0, 0.1, 0.2],
    )
    return select("sweeps/tutorial-dclm-30m-sweep", version, trials, metric=SELECTION_METRIC, mode="min")


if __name__ == "__main__":
    # Lower the sweep to a StepSpec graph and run it: the shared caches tokenize, each
    # trial trains (its own TPU job), then `select` records the winner.
    best = best_trial()
    StepRunner().run([lower(best)])

    selection = resolve(best)
    print(f"best trial: {selection.winner} ({SELECTION_METRIC}={selection.score:.4f})")
