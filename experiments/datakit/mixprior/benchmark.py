# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory

import jax
import numpy as np
from rigging.filesystem.storage_path import StoragePath
from scipy.stats import spearmanr

from experiments.datakit.mixprior.campaign import Campaign, load_campaign
from experiments.datakit.mixprior.data import Swarm, SwarmObservations, canonical_mixture_rows
from experiments.datakit.mixprior.huggingface import download_campaign
from experiments.datakit.mixprior.objective import ObjectiveObservations, ScalarObjective
from experiments.datakit.mixprior.quadratic_exposure import fit_quadratic_exposure_model
from experiments.datakit.mixprior.search import (
    POSTERIOR_MEAN,
    Acquisition,
    acquire_candidate,
    noisy_expected_improvement,
)
from experiments.datakit.mixprior.surrogate import JaxMixturePredictor, default_device

CAMPAIGN_URI = (
    "hf://datasets/marin-community/grug-moe-mix-swarm"
    "@283eacf18b66b7888b59fd8f889d6be134aee879/registry/v1/transfer_campaign.parquet"
)
CAMPAIGN_SHA256 = "5ed5fb024590dd4707b802caf8fe728be1b8d73375c100139f884b0728f0cca2"
RANK_PREFIXES = (2, 3, 5, 10, 20, 40)
REGRET_HORIZON = 20
REGRET_BATCH_SIZE = 5
SHORTLIST_SIZE = 5
VALIDATION_REPLAY = "validation"
CALIBRATION_REPLAY = "calibration"
REPLAY_TRANSITIONS = (VALIDATION_REPLAY, CALIBRATION_REPLAY)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AggregatedObjective:
    base: ScalarObjective
    target_swarm: str
    target_observations: dict[str, tuple[float, float]]

    def observations(self, swarm: Swarm) -> ObjectiveObservations:
        if swarm.swarm_id != self.target_swarm:
            return self.base.observations(swarm)
        rows = [self.target_observations[observation_id] for observation_id in swarm.data.observation_ids]
        return ObjectiveObservations(
            np.asarray([row[0] for row in rows]),
            np.asarray([row[1] for row in rows]),
        )


def subset_swarm(swarm: Swarm, indices: list[int]) -> Swarm:
    data = swarm.data
    return replace(
        swarm,
        data=SwarmObservations(
            observation_ids=[data.observation_ids[index] for index in indices],
            run_names=[data.run_names[index] for index in indices],
            groups=[data.groups[index] for index in indices],
            mixture_components=data.mixture_components,
            component_metadata=data.component_metadata,
            available_tokens=data.available_tokens,
            weights=data.weights[indices],
            labels=data.labels,
            outcomes=data.outcomes[indices],
        ),
    )


def transition(campaign: Campaign, name: str) -> tuple[Swarm, tuple[Swarm, ...]]:
    swarms = {swarm.swarm_id: swarm for swarm in (campaign.target, *campaign.sources)}
    legacy = swarms["legacy-swarm-d512"]
    first_store = swarms["harrier-store-b262968b-d768"]
    second_store = swarms["harrier-store-0381a974-d768"]
    if name == VALIDATION_REPLAY:
        return second_store, (legacy, first_store)
    if name == CALIBRATION_REPLAY:
        return first_store, (legacy,)
    raise ValueError(name)


def aggregate_replicates(campaign: Campaign, target: Swarm) -> tuple[Swarm, AggregatedObjective]:
    observations = campaign.objective.observations(target)
    groups: dict[bytes, list[int]] = {}
    for index, key in enumerate(canonical_mixture_rows(target.data.weights)):
        groups.setdefault(key.tobytes(), []).append(index)

    representatives = []
    aggregate_by_id = {}
    for indices in groups.values():
        representative = indices[0]
        precision = 1.0 / observations.variances[indices]
        value = float(np.sum(precision * observations.values[indices]) / np.sum(precision))
        variance = float(1.0 / np.sum(precision))
        observation_id = target.data.observation_ids[representative]
        representatives.append(representative)
        aggregate_by_id[observation_id] = (value, variance)

    return subset_swarm(target, representatives), AggregatedObjective(
        campaign.objective,
        target.swarm_id,
        aggregate_by_id,
    )


def rank_replay(
    campaign: Campaign,
    name: str,
    device: jax.Device,
) -> list[dict[str, float | int]]:
    target, sources = transition(campaign, name)
    actual = campaign.objective.observations(target).values
    rows = []
    for prefix in RANK_PREFIXES:
        observed = subset_swarm(target, list(range(prefix)))
        fitted = fit_quadratic_exposure_model(replace(campaign, target=observed, sources=sources), device)
        predicted = fitted.predict(target, target.data.weights[prefix:]).mean
        held_out = actual[prefix:]
        winner = int(np.argmax(predicted))
        shortlist_size = min(SHORTLIST_SIZE, len(predicted))
        shortlist = np.argpartition(predicted, -shortlist_size)[-shortlist_size:]
        row = {
            "prefix": prefix,
            "spearman": float(spearmanr(predicted, held_out).statistic),
            "winner_regret": float(np.max(held_out) - held_out[winner]),
            "shortlist_regret": float(np.max(held_out) - np.max(held_out[shortlist])),
        }
        rows.append(row)
        logger.info("%s prefix %d: %s", name, prefix, row)
    return rows


def anchor_indices(target: Swarm) -> list[int]:
    return sorted(
        range(len(target.data.weights)),
        key=lambda index: hashlib.sha256(canonical_mixture_rows(target.data.weights[index : index + 1])).digest(),
    )


def acquisition_batch(
    acquisition: Acquisition,
    model: JaxMixturePredictor,
    observed: Swarm,
    target: Swarm,
    remaining: list[int],
    count: int,
) -> list[int]:
    pending = []
    for _ in range(count):
        available = [index for index in remaining if index not in pending]
        pending_weights = target.data.weights[pending] if pending else None
        acquired = acquire_candidate(
            acquisition,
            model,
            observed,
            target.data.weights[available],
            pending_weights,
        )
        pending.append(available[acquired.pool_index])
    return pending


def regret_replay(
    campaign: Campaign,
    target: Swarm,
    sources: tuple[Swarm, ...],
    initial: list[int],
    acquisition: Acquisition,
    device: jax.Device,
) -> dict[str, object]:
    objective = campaign.objective.observations(target).values
    best_possible = float(objective.max())
    selected = list(initial)
    curve = []
    while len(selected) <= REGRET_HORIZON:
        curve.append(
            {
                "evaluations": len(selected),
                "simple_regret": best_possible - float(objective[selected].max()),
            }
        )
        if len(selected) == REGRET_HORIZON:
            break

        observed = subset_swarm(target, selected)
        fitted = fit_quadratic_exposure_model(replace(campaign, target=observed, sources=sources), device)
        remaining = [index for index in range(len(target.data.weights)) if index not in selected]
        count = min(REGRET_BATCH_SIZE, REGRET_HORIZON - len(selected))
        selected.extend(acquisition_batch(acquisition, fitted, observed, target, remaining, count))
        logger.info(
            "%s selected %d/%d, regret %.6f",
            acquisition.name,
            len(selected),
            REGRET_HORIZON,
            best_possible - float(objective[selected].max()),
        )
    return {"acquisition": acquisition.name, "initial": initial, "selected": selected, "curve": curve}


def run_rank(campaign: Campaign) -> dict[str, object]:
    device = default_device()
    results = {name: rank_replay(campaign, name, device) for name in REPLAY_TRANSITIONS}
    return {
        "results": results,
        "mean_spearman": {name: float(np.mean([row["spearman"] for row in rows])) for name, rows in results.items()},
    }


def run_regret(campaign: Campaign, block: int) -> dict[str, object]:
    results = {}
    for name in REPLAY_TRANSITIONS:
        target, sources = transition(campaign, name)
        target, objective = aggregate_replicates(campaign, target)
        replay_campaign = replace(campaign, target=target, sources=sources, objective=objective)
        anchors = anchor_indices(target)
        initial = [anchors[(block + offset) % len(anchors)] for offset in range(3)]
        results[name] = [
            regret_replay(replay_campaign, target, sources, initial, acquisition, default_device())
            for acquisition in (POSTERIOR_MEAN, noisy_expected_improvement(10_000 + block))
        ]
    return {"block": block, "results": results}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("rank", "regret"))
    parser.add_argument("--block", type=int, default=0)
    parser.add_argument("--output")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with TemporaryDirectory(prefix="mixprior-benchmark-") as temporary:
        manifest = download_campaign(CAMPAIGN_URI, CAMPAIGN_SHA256, Path(temporary) / "campaign")
        campaign = load_campaign(manifest)
        if args.mode == "rank":
            payload = run_rank(campaign)
        else:
            payload = run_regret(campaign, args.block)

    if args.output:
        StoragePath(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print("RESULT " + json.dumps(payload), flush=True)


if __name__ == "__main__":
    main()
