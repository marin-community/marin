# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Fit metric acquisition on a named HF swarm and persist a Marin artifact."""

import argparse
from dataclasses import fields
from pathlib import Path

import jax
import numpy as np
from marin.execution.artifact import Artifact, write_artifact

from experiments.datakit.mixprior.acquisition import AdditiveMetricGP, KernelConfig
from experiments.datakit.mixprior.calibration import CALIBRATION_FOLDS, fit_calibrated, new_design_indices
from experiments.datakit.mixprior.hf import HF_DATASET, Data, load_data, load_hf
from experiments.datakit.mixprior.objective import Objective, fit_objective
from experiments.datakit.mixprior.observations import group_observations

PARAMETERS_FILENAME = "model.npz"


class MetricArtifact(Artifact):
    """Fitted metric acquisition, data identity, and objective definition."""

    hf_dataset: str
    hf_revision: str
    swarm_id: str
    components: list[str]
    observation_ids: list[str]
    objective_metrics: list[str]
    target_metrics: list[str]
    hinge_epsilon: float
    grouped_design_count: int
    available_tokens: list[float]
    phase_budgets: list[float]
    calibration_revision: str
    calibration_observation_ids: list[str]
    calibration_folds: int


def save_model(
    path: Path,
    model: AdditiveMetricGP,
    data: Data,
    objective: Objective,
    revision: str,
    calibration_revision: str,
    calibration_observation_ids: list[str],
) -> MetricArtifact:
    """Save the fitted metric model and the observations excluded from future search."""
    path = path.resolve()
    path.mkdir(parents=True, exist_ok=True)
    metrics = [data.labels[i] for i in objective.columns]
    artifact = MetricArtifact(
        path=str(path),
        hf_dataset=HF_DATASET,
        hf_revision=revision,
        swarm_id=data.name,
        components=data.components,
        observation_ids=data.observation_ids,
        objective_metrics=metrics,
        target_metrics=[name for name, target in zip(metrics, objective.target_mask, strict=True) if target],
        hinge_epsilon=objective.epsilon,
        grouped_design_count=len(model.train_roots),
        available_tokens=data.available_tokens.tolist(),
        phase_budgets=data.phase_budgets.tolist(),
        calibration_revision=calibration_revision,
        calibration_observation_ids=calibration_observation_ids,
        calibration_folds=CALIBRATION_FOLDS,
    )
    model = jax.device_get(model)
    arrays = {
        field.name: getattr(model, field.name) for field in fields(AdditiveMetricGP) if field.name != "kernel_config"
    }
    arrays.update({f"kernel_{field.name}": getattr(model.kernel_config, field.name) for field in fields(KernelConfig)})
    np.savez(path / PARAMETERS_FILENAME, **arrays, observed_weights=data.weights)
    write_artifact(artifact.result_payload(), str(path))
    return artifact


def load_model(path: Path, device: jax.Device | None = None) -> tuple[MetricArtifact, AdditiveMetricGP, np.ndarray]:
    """Restore acquisition state and observed weights without fetching data or fitting."""
    if not jax.config.x64_enabled:
        raise ValueError("Metric loading requires jax_enable_x64=True")
    path = path.resolve()
    artifact = MetricArtifact.raw_load(str(path))
    with np.load(path / PARAMETERS_FILENAME, allow_pickle=False) as arrays:
        observed = arrays["observed_weights"]
        parameters = {name: arrays[name] for name in arrays.files if name != "observed_weights"}
    parameters = jax.device_put(parameters, device)
    config = KernelConfig(**{field.name: parameters.pop(f"kernel_{field.name}") for field in fields(KernelConfig)})
    model = AdditiveMetricGP(kernel_config=config, **parameters)
    return artifact, model, observed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="Hugging Face dataset commit")
    parser.add_argument("--swarm", required=True)
    parser.add_argument(
        "--calibration-revision", required=True, help="Earlier HF commit defining the new calibration designs"
    )
    parser.add_argument("--calibration-data", type=Path, help="Local snapshot of the earlier revision")
    parser.add_argument("--data", type=Path, help="Use a local HF snapshot instead of downloading")
    parser.add_argument("--output", type=Path, required=True, help="Directory for the fitted acquisition artifact")
    parser.add_argument("--device", choices=("cpu", "gpu"), help="Default: JAX's preferred device")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    device = jax.devices(args.device)[0]
    data = load_data(args.data, args.swarm) if args.data else load_hf(args.revision, args.swarm)
    objective = fit_objective(data)
    grouped = group_observations(data)
    previous = (
        load_data(args.calibration_data, args.swarm)
        if args.calibration_data
        else load_hf(args.calibration_revision, args.swarm)
    )
    indices = new_design_indices(grouped.data, previous)
    model = fit_calibrated(grouped.data, objective, grouped.counts, indices, device)
    calibration_ids = [data.observation_ids[i] for i in new_design_indices(data, previous)]
    save_model(args.output, model, data, objective, args.revision, args.calibration_revision, calibration_ids)
    print(
        f"Saved acquisition model trained on {len(grouped.data.weights)} distinct designs "
        f"using {device} to {args.output}"
    )


if __name__ == "__main__":
    main()
