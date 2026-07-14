# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: an LR/WD sweep over a 30M DCLM model, in the lazy-artifact style.

A sweep on the DCLM mixture with held-out validation and the CORE eval harness wired in,
selecting on the held-out loss:

- the DCLM training components and the Paloma + Uncheatable validation sets are tokenized
  :class:`~marin.processing.tokenize.tokenize.TokenizedCache` handles, tokenized once and
  shared by every trial;
- :func:`~marin.experiment.sweep.sweep` fans out one
  :func:`~marin.experiment.train.train_lm` trial per grid point;
- each trial dispatches its own training job, which records its validation loss next to its
  checkpoints;
- selection is ordinary Python: resolve each trial's
  :class:`~marin.training.training.LevanterCheckpoint`, read its recorded validation loss via
  :meth:`~marin.training.training.LevanterCheckpoint.training_metrics`, and keep the lowest.

Run it against a cluster (with ``MARIN_PREFIX`` pointing at a bucket co-regional with
``RESOURCES``)::

    python -m experiments.tutorials.train_tiny_sweep
"""

from fray.cluster import ResourceConfig
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep, run
from marin.experiment.sweep import sweep
from marin.experiment.train import EvalSuite, train_lm
from marin.training.training import LevanterCheckpoint

from experiments.datasets.dclm import DCLM_MIXTURE_WEIGHTS, dclm_datasets
from experiments.datasets.paloma import paloma_datasets
from experiments.datasets.uncheatable import uncheatable_datasets
from experiments.evals.task_configs import CORE_TASKS
from experiments.llama import llama3_tokenizer, llama_30m

# A single-host TPU slice; each trial trains on its own slice. This is a runtime arg, not
# part of a trial's identity — re-running on a different TPU is the same checkpoint.
RESOURCES = ResourceConfig.with_tpu("v4-8")

# The corpus and held-out validation: tokenized once into shared caches, reused by every
# trial by name@version. The first trial to run tokenizes them (their own Fray jobs).
_train = dclm_datasets(tokenizer=llama3_tokenizer)
_validation = [
    *paloma_datasets(tokenizer=llama3_tokenizer).values(),
    *uncheatable_datasets(tokenizer=llama3_tokenizer).values(),
]
_weighted = {_train[name]: DCLM_MIXTURE_WEIGHTS[name] for name in _train}


def trial(*, learning_rate: float, weight_decay: float, version: str = "dev") -> ArtifactStep[LevanterCheckpoint]:
    """One sweep trial: a 30M llama trained on the DCLM mixture, ready to select over.

    The swept hyperparameters are literals in the optimizer, so each grid point gets a
    distinct ``name@version`` and fingerprint; the TPU is a runtime arg, excluded from
    identity. The default ``dev`` version rebuilds each run while iterating on the sweep; pin
    a calendar version for trials you want cached.
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
        evals=EvalSuite(CORE_TASKS, every=10000),
        resources=RESOURCES,
        tags=["llama", "30m", "dclm", "tutorial", "sweep", "test20251117"],
    )


if __name__ == "__main__":
    trials = sweep(
        trial,
        learning_rate=[3e-4, 6e-4, 1e-3],
        weight_decay=[0.0, 0.1, 0.2],
    )

    # Build every trial and get back its resolved checkpoint: the shared caches tokenize once,
    # then each trial trains on its own TPU job (in parallel), recording its validation loss next
    # to its checkpoints. run() returns the loaded, typed LevanterCheckpoint per trial.
    checkpoints = run(*trials)

    # Selection is ordinary code over the resolved checkpoints: read each one's recorded held-out
    # loss through its typed artifact and keep the lowest.
    scored = [(t, ckpt.training_metrics().eval_loss) for t, ckpt in zip(trials, checkpoints, strict=True)]
    best, best_loss = min(scored, key=lambda pair: pair[1])
    print(f"best trial: {best.name} (eval_loss={best_loss:.4f})")
