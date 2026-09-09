# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit a GP on a named HF swarm and persist its state as a Marin artifact."""

import argparse
from pathlib import Path

import numpy as np
from marin.execution.artifact import Artifact, write_artifact

from experiments.datakit.mixprior.hf import HF_DATASET, Data, load_data, load_hf
from experiments.datakit.mixprior.model import GP, Features, fit
from experiments.datakit.mixprior.objective import Objective, fit_objective

PARAMETERS_FILENAME = "model.npz"


class GPArtifact(Artifact):
    """A fitted GP with its training-data identity and objective definition."""

    hf_dataset: str
    hf_revision: str
    swarm_id: str
    components: list[str]
    observation_ids: list[str]
    objective_metrics: list[str]
    target_metrics: list[str]
    hinge_epsilon: float


def save_model(path: Path, model: GP, data: Data, objective: Objective, revision: str) -> GPArtifact:
    """Save prediction state and the observations excluded from future search."""
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    metrics = [data.labels[i] for i in objective.columns]
    artifact = GPArtifact(
        path=str(path),
        hf_dataset=HF_DATASET,
        hf_revision=revision,
        swarm_id=data.name,
        components=data.components,
        observation_ids=data.observation_ids,
        objective_metrics=metrics,
        target_metrics=[name for name, target in zip(metrics, objective.target_mask, strict=True) if target],
        hinge_epsilon=objective.epsilon,
    )
    np.savez(
        path / PARAMETERS_FILENAME,
        available=model.features.available,
        budgets=model.features.budgets,
        membership=model.features.membership,
        train_x=model.train_x,
        coefficients=model.coefficients,
        lengthscale=model.lengthscale,
        cholesky=model.cholesky,
        alpha=model.alpha,
        center=model.center,
        scale=model.scale,
        observed_weights=data.weights,
    )
    write_artifact(artifact.result_payload(), str(path))
    return artifact


def load_model(path: Path) -> tuple[GPArtifact, GP, np.ndarray]:
    """Restore a saved GP and observed weights without fetching data or fitting."""
    path = path.resolve()
    artifact = GPArtifact.raw_load(str(path))
    with np.load(path / PARAMETERS_FILENAME, allow_pickle=False) as arrays:
        model = GP(
            features=Features(arrays["available"], arrays["budgets"], arrays["membership"]),
            train_x=arrays["train_x"],
            coefficients=arrays["coefficients"],
            lengthscale=float(arrays["lengthscale"]),
            cholesky=arrays["cholesky"],
            alpha=arrays["alpha"],
            center=float(arrays["center"]),
            scale=float(arrays["scale"]),
        )
        observed = arrays["observed_weights"]
    return artifact, model, observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Hugging Face dataset commit")
    parser.add_argument("--swarm", required=True)
    parser.add_argument("--data", type=Path, help="Use a local HF snapshot instead of downloading")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the fitted GP artifact")
    args = parser.parse_args()
    data = load_data(args.data, args.swarm) if args.data else load_hf(args.revision, args.swarm)
    objective = fit_objective(data)
    model = fit(data, *objective(data.outcomes))
    save_model(args.output, model, data, objective, args.revision)
    print(f"Saved GP trained on {len(data.weights)} observations to {args.output}")


if __name__ == "__main__":
    main()
