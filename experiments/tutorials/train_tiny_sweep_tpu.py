# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: an LR/WD sweep over a tiny model on TPU, in the lazy-artifact style.

An LR/WD grid search authored as lazy artifacts, end to end:

- the corpus is a :class:`~marin.execution.lazy.Dataset` handle, tokenized once and
  shared by every trial;
- :func:`~marin.experiment.sweep.sweep` fans out one trial per grid point — each is a
  :func:`~marin.experiment.train.train_lm` :class:`~marin.execution.lazy.Checkpoint`,
  wrapped with :func:`~marin.experiment.sweep.annotate` so the sweep knows where to
  read its score. The swept hyperparameters are literals, so each trial gets a distinct
  ``name@version``;
- each trial dispatches its own TPU training job, which records its metrics next to its
  checkpoints (no metrics payload to thread back through the graph);
- :func:`~marin.experiment.sweep.select` reads each trial's recorded loss from its
  output and reduces the trials to the lowest-loss one.

There is no executor, no import-time step graph, and no lock coordination: the
``StepRunner`` schedules the trials and the selection from the lowered graph.

Run it against a cluster (with ``MARIN_PREFIX`` pointing at a bucket co-regional
with ``TRAIN_RESOURCES``)::

    python -m experiments.tutorials.train_tiny_sweep_tpu
"""

from fray.cluster import ResourceConfig
from levanter.optim import AdamConfig
from marin.execution.artifact import Artifact as ArtifactIO
from marin.execution.lazy import Checkpoint, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, tokenized
from marin.experiment.sweep import AnnotatedCheckpoint, annotate, select, sweep
from marin.experiment.train import train_lm

from experiments.llama import llama3_tokenizer, llama_30m

# A single-host TPU slice; each trial trains on its own slice. This is a run-arg, not
# part of a trial's identity — re-running on a different TPU is the same checkpoint.
TRAIN_RESOURCES = ResourceConfig.with_tpu("v4-8")

# The selection metric: levanter logs the train loss every step under this key, and the
# run's final value ranks the trials. Selecting on a held-out "eval/loss" works
# identically (`select(metric="eval/loss")`); this tutorial uses train loss so each trial
# stays self-contained, with no separate validation set to configure.
SELECTION_METRIC = "train/loss"

# The corpus: SlimPajama-6B, llama3-tokenized once into a shared cache. The first trial
# to run tokenizes it (its own Fray job); the rest reuse the cache by name@version.
slimpajama = tokenized(
    "slimpajama-6b",
    source="DKYoon/SlimPajama-6B",
    tokenizer=llama3_tokenizer,
    resources=ResourceConfig(ram="64g", disk="64g"),
)


def trial(*, learning_rate: float, weight_decay: float, version: str = "v1") -> AnnotatedCheckpoint:
    """One sweep trial: a tiny llama trained on the shared corpus, ready to select over.

    The swept hyperparameters are literals in the optimizer, so each grid point gets a
    distinct ``name@version`` and fingerprint; the TPU is a run-arg, excluded from
    identity, so re-running on different hardware does not fork the trial.
    :func:`~marin.experiment.sweep.annotate` pairs the checkpoint with the default
    reader for ``train_lm`` metrics, so :func:`select` can rank it by ``SELECTION_METRIC``.
    """
    checkpoint = train_lm(
        name=f"checkpoints/tiny-sweep/lr{learning_rate}-wd{weight_decay}",
        version=version,
        model=llama_30m,
        optimizer=AdamConfig(learning_rate=learning_rate, weight_decay=weight_decay),
        data=lambda ctx: mixture(ctx, {slimpajama: 1.0}),
        deps=(slimpajama,),
        batch_size=128,
        seq_len=llama_30m.max_seq_len,
        num_train_steps=10000,
        z_loss_weight=None,
        evals=None,
        resources=TRAIN_RESOURCES,
        tags=["llama", "30m", "slimpajama-6b", "tutorial", "sweep"],
    )
    return annotate(checkpoint)


def best_trial(*, version: str = "v1") -> Checkpoint:
    """The full LR/WD sweep, reduced to its lowest-loss trial."""
    trials = sweep(
        trial,
        learning_rate=[3e-4, 6e-4, 1e-3],
        weight_decay=[0.0, 0.1, 0.2],
    )
    return select("sweeps/tiny-sweep", version, trials, metric=SELECTION_METRIC, mode="min")


if __name__ == "__main__":
    # Lower the sweep to a StepSpec graph and run it: each trial trains (its own TPU
    # job, tokenizing the shared corpus first), then `select` records the winner.
    best = best_trial()
    StepRunner().run([lower(best)])

    selection = ArtifactIO.from_path(best.path())
    print(f"best trial: {selection['winner']} ({SELECTION_METRIC}={selection['score']:.4f})")
