# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tutorial: an LR/WD sweep over a tiny model on TPU, in the lazy-artifact style.

An LR/WD grid search authored as lazy artifacts, end to end:

- the corpus is a :class:`~marin.execution.lazy.Dataset` handle, tokenized once and
  shared by every trial;
- :func:`~marin.experiment.sweep.sweep` fans out one
  :class:`~marin.execution.lazy.Checkpoint` trial per grid point — the swept
  hyperparameters are literals, so each trial gets a distinct ``name@version``;
- each trial dispatches its own TPU training job (the launcher step runs inline),
  then reads the run's final loss back and returns it as the trial's metrics;
- :func:`~marin.experiment.sweep.select` reduces the trials to the lowest-loss one.

There is no executor, no import-time step graph, and no lock coordination: the
``StepRunner`` schedules the trials and the selection from the lowered graph.

Run it against a cluster (with ``MARIN_PREFIX`` pointing at a bucket co-regional
with ``TRAIN_RESOURCES``)::

    python -m experiments.tutorials.train_tiny_sweep_tpu
"""

from dataclasses import dataclass

from fray.cluster import ResourceConfig
from levanter.data.text import LmDataConfig
from marin.execution.artifact import Artifact as ArtifactIO
from marin.execution.lazy import Artifact, Checkpoint, Recipe, RunContext, lower
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.experiment.data import mixture, tokenized
from marin.experiment.sweep import select, sweep
from marin.scaling_laws.eval_metrics_reader import read_eval_records

from experiments.defaults import _run_training_on_worker, prepare_lm_train
from experiments.llama import llama3_tokenizer, llama_30m
from experiments.simple_train_config import SimpleTrainConfig

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


@dataclass(frozen=True)
class TrialConfig:
    """The concrete inputs one sweep trial trains on."""

    label: str
    out: str
    data: LmDataConfig
    learning_rate: float
    weight_decay: float
    resources: ResourceConfig


def _train_and_eval(config: TrialConfig) -> dict:
    """Train one trial on a TPU and return its metrics.

    Builds the trainer config from the resolved data and hyperparameters, dispatches
    training as its own Fray job on ``config.resources`` (this launcher step runs
    inline), then reads the run's final loss from the metrics the training wrote
    alongside its checkpoints. The returned mapping is the trial's artifact payload,
    which :func:`~marin.experiment.sweep.select` reduces over.
    """
    train_config = SimpleTrainConfig(
        resources=config.resources,
        train_batch_size=128,
        num_train_steps=10000,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    job_name, raw_config = prepare_lm_train(
        name=config.label,
        tokenized=config.data,
        model_config=llama_30m,
        train_config=train_config,
        tags=["llama", "30m", "slimpajama-6b", "tutorial", "sweep"],
        use_default_validation=False,
    )
    remote(_run_training_on_worker, resources=config.resources, name=job_name)(
        job_name, raw_config, config.out, config.resources
    )

    summary = read_eval_records([config.out])[-1]["summary"]
    return {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        SELECTION_METRIC: summary[SELECTION_METRIC],
    }


def trial(*, learning_rate: float, weight_decay: float, version: str = "v1") -> Checkpoint:
    """One sweep trial as a :class:`~marin.execution.lazy.Checkpoint` handle.

    The swept hyperparameters are literals in the config, so each grid point gets a
    distinct ``name@version`` and fingerprint; the TPU is a run-arg, excluded from
    identity, so re-running on different hardware does not fork the trial.
    """
    name = f"checkpoints/tiny-sweep/lr{learning_rate}-wd{weight_decay}"

    def build_config(ctx: RunContext) -> TrialConfig:
        return TrialConfig(
            label=f"tiny-sweep-lr{learning_rate}-wd{weight_decay}",
            out=ctx.out,
            data=mixture(ctx, {slimpajama: 1.0}),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            resources=ctx.run_arg("train_resources"),
        )

    return Checkpoint(
        name=name,
        version=version,
        recipe=Recipe(
            fn=_train_and_eval,
            build_config=build_config,
            deps=(slimpajama,),
            run_args={"train_resources": TRAIN_RESOURCES},
        ),
    )


def best_trial(*, version: str = "v1") -> Artifact:
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
