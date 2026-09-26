# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.datakit.mixprior.campaign import Campaign
from experiments.datakit.mixprior.data import (
    Swarm,
    SwarmObservations,
    SwarmProvenance,
    load_observations,
    sha256,
    write_record,
)
from experiments.datakit.mixprior.objective import HINGE_TASKS, UNCAPPED_TARGET_TASKS, VarianceNormalizedObjective


@pytest.fixture(scope="module")
def swarm_observations(tmp_path_factory: pytest.TempPathFactory) -> SwarmObservations:
    swarm_id = "target-fixture"
    root = tmp_path_factory.mktemp("target")
    observations = root / "observations.parquet"
    buckets = root / "buckets.parquet"
    labels = list(HINGE_TASKS)
    designs = (
        ({"c00q0": 0.5, "c00q1": 0.5}, {"c00q0": 0.5, "c00q1": 0.5}),
        ({"c00q0": 0.8, "c00q1": 0.2}, {"c00q0": 0.2, "c00q1": 0.8}),
    )
    rows = []
    for design_index, (phase0, phase1) in enumerate(designs):
        for seed_index, noise in enumerate((-0.1, 0.1)):
            run_name = f"design-{design_index}-seed-{seed_index}"
            rows.append(
                {
                    "observation_id": f"{swarm_id}:{run_name}",
                    "swarm_id": swarm_id,
                    "run_name": run_name,
                    "group": "marin_proportional" if design_index == 0 else "bayesian_optimization",
                    "phase0_weights": phase0,
                    "phase1_weights": phase1,
                    "grouped_bpb": {
                        label: 1.0 + 0.01 * index + 0.2 * design_index + noise for index, label in enumerate(labels)
                    },
                }
            )
    pd.DataFrame(rows).to_parquet(observations, index=False)
    write_record(
        buckets,
        {
            "cells": [
                {
                    "cell": "c00q0",
                    "domain": "domain-0",
                    "quality": 0,
                    "available_tokens": 10.0,
                },
                {
                    "cell": "c00q1",
                    "domain": "domain-0",
                    "quality": 1,
                    "available_tokens": 10.0,
                },
            ]
        },
    )
    return load_observations(observations, buckets, swarm_id)


@pytest.fixture(scope="module")
def objective(swarm_observations: SwarmObservations) -> VarianceNormalizedObjective:
    return VarianceNormalizedObjective.fit(
        swarm_observations.labels,
        swarm_observations.outcomes,
        np.asarray(swarm_observations.groups) == "marin_proportional",
        epsilon=0.0,
        metrics=tuple(swarm_observations.labels),
        hinge_tasks=HINGE_TASKS,
        uncapped_tasks=UNCAPPED_TARGET_TASKS,
        observation_sd=np.full(len(swarm_observations.labels), 0.1),
    )


@pytest.fixture
def tiny_campaign() -> Campaign:
    label = "logprob_humaneval_10shot"
    objective = _objective([label])
    target = _swarm(
        "target",
        weights=np.asarray(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.0, 1.0]],
            ]
        ),
        outcomes=np.asarray([[-1.0], [1.0]]),
    )
    source = _swarm(
        "source",
        weights=np.asarray(
            [
                [[0.9, 0.1], [0.9, 0.1]],
                [[0.1, 0.9], [0.1, 0.9]],
            ]
        ),
        outcomes=np.asarray([[-0.8], [0.8]]),
    )
    return Campaign(
        target=target,
        sources=(source,),
        objective=objective,
        objective_metadata={"kind": "test"},
    )


@pytest.fixture
def campaign_bundle(tmp_path: Path, swarm_observations: SwarmObservations) -> tuple[Path, Path]:
    root = tmp_path / "campaign"
    root.mkdir()
    swarm_id = "rav-test"
    swarm_dir = root / "swarms" / swarm_id
    swarm_dir.mkdir(parents=True)
    observations = swarm_dir / "observations.parquet"
    buckets = swarm_dir / "buckets.parquet"
    content_path = swarm_dir / "content.parquet"
    frame = pd.DataFrame(
        {
            "observation_id": [f"{swarm_id}:{run_name}" for run_name in swarm_observations.run_names],
            "swarm_id": swarm_id,
            "run_name": swarm_observations.run_names,
            "group": swarm_observations.groups,
            "phase0_weights": [
                dict(zip(swarm_observations.mixture_components, row[0], strict=True))
                for row in swarm_observations.weights
            ],
            "phase1_weights": [
                dict(zip(swarm_observations.mixture_components, row[1], strict=True))
                for row in swarm_observations.weights
            ],
            "grouped_bpb": [
                dict(zip(swarm_observations.labels, row, strict=True)) for row in swarm_observations.outcomes
            ],
        }
    )
    frame.to_parquet(observations, index=False)
    write_record(
        buckets,
        {
            "cells": [
                {
                    "cell": cell,
                    "domain": "domain-0",
                    "quality": index,
                    "available_tokens": float(swarm_observations.available_tokens[index]),
                }
                for index, cell in enumerate(swarm_observations.mixture_components)
            ]
        },
    )
    write_record(
        content_path,
        {
            "basis_id": "frozen-basis",
            "cells": swarm_observations.mixture_components,
            "matrix": np.eye(len(swarm_observations.mixture_components)).tolist(),
            "provenance": {"manifest": "content-manifest"},
        },
    )
    swarm = {
        "schema_version": 1,
        "swarm_id": swarm_id,
        "observations": {
            "path": observations.name,
            "sha256": sha256(observations),
            "schema": "mixture-observations-v1",
        },
        "buckets": {
            "path": buckets.name,
            "sha256": sha256(buckets),
        },
        "content": {
            "path": content_path.name,
            "sha256": sha256(content_path),
        },
        "phase_budgets": [1.5, 0.5],
        "store_uri": "hf://datasets/test/store",
        "store_artifact_uri": "hf://datasets/test/store-artifact.parquet",
        "training_recipe": "grug-test",
        "tokenizer": "test-tokenizer",
        "evaluation_suite": "test-evals",
        "model_active_parameters": 1,
        "model_total_parameters": 2,
        "physical_training_tokens": 2,
        "simulated_training_tokens": 2,
    }
    swarm_path = swarm_dir / "swarm.parquet"
    write_record(swarm_path, swarm)
    basis_dir = root / "basis"
    basis_dir.mkdir()
    lookup = basis_dir / "lookup.npy"
    np.save(lookup, np.arange(len(swarm_observations.mixture_components)))
    basis_path = basis_dir / "basis.parquet"
    write_record(
        basis_path,
        {
            "schema_version": 1,
            "basis_id": "frozen-basis",
            "lookup": {
                "path": lookup.name,
                "sha256": sha256(lookup),
            },
        },
    )
    registry_path = root / "swarm_registry.parquet"
    write_record(
        registry_path,
        {
            "schema_version": 1,
            "content_bases": [
                {
                    "basis_id": "frozen-basis",
                    "path": basis_path.relative_to(root).as_posix(),
                    "sha256": sha256(basis_path),
                }
            ],
            "swarms": [
                {
                    "swarm_id": swarm_id,
                    "path": swarm_path.relative_to(root).as_posix(),
                    "sha256": sha256(swarm_path),
                }
            ],
        },
    )
    manifest_path = root / "transfer_campaign.parquet"
    write_record(
        manifest_path,
        {
            "schema_version": 1,
            "registry": {
                "path": registry_path.name,
                "sha256": sha256(registry_path),
            },
            "target_swarm": swarm_id,
            "source_swarms": [],
            "objective_reference_swarm": swarm_id,
            "noise_reference_swarm": swarm_id,
            "response_tasks": [task for task in HINGE_TASKS if task != "include_mean"],
            "objective_epsilon": 0.0,
        },
    )
    return manifest_path, observations


def _objective(labels: list[str]) -> VarianceNormalizedObjective:
    return VarianceNormalizedObjective(
        labels=labels,
        reference_mean=np.zeros(len(labels)),
        reference_std=np.ones(len(labels)),
        task_correlation=np.eye(len(labels)),
        reference_count=2,
        epsilon=0.0,
        metrics=tuple(labels),
        hinge_tasks=tuple(labels),
        uncapped_tasks=tuple(labels),
        observation_sd=np.full(len(labels), 0.05),
    )


def _swarm(name: str, *, weights: np.ndarray, outcomes: np.ndarray) -> Swarm:
    cells = ["c00q0", "c00q1"]
    data = SwarmObservations(
        observation_ids=[f"{name}:{index}" for index in range(len(weights))],
        run_names=[f"{name}-{index}" for index in range(len(weights))],
        groups=["bayesian_optimization"] * len(weights),
        mixture_components=cells,
        component_metadata=[
            {
                "cell": cell,
                "domain": "domain-0",
                "quality": index,
                "available_tokens": 1.0,
            }
            for index, cell in enumerate(cells)
        ],
        available_tokens=np.ones(len(cells)),
        weights=weights,
        labels=["logprob_humaneval_10shot"],
        outcomes=outcomes,
    )
    return Swarm(
        swarm_id=name,
        data=data,
        phase_budgets=np.ones(2),
        content_basis_id="tiny-basis",
        content_matrix=np.eye(len(cells)),
        provenance=SwarmProvenance(
            store_uri=f"hf://datasets/test/{name}",
            store_artifact_uri=f"hf://datasets/test/{name}/artifact.parquet",
            training_recipe="test-recipe",
            tokenizer="test-tokenizer",
            evaluation_suite="test-evals",
            model_active_parameters=1,
            model_total_parameters=1,
            physical_training_tokens=1,
            simulated_training_tokens=2,
            content_provenance={"fixture": True},
        ),
    )
