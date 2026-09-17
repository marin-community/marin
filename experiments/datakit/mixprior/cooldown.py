# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate cooldown candidates with the merged hero's earlier stages fixed."""

import argparse
import copy
import json
from pathlib import Path

import jax
import numpy as np

from experiments.datakit.mixprior.hf import write_candidates
from experiments.datakit.mixprior.search import acquisition_scores, search_cooldown
from experiments.datakit.mixprior.train import load_model
from experiments.grug.moe_hero_ep.harrier_mix_schedule import (
    COOLDOWN_TOKENS,
    MAX_COMPONENT_EPOCHS,
    MIXTURE_SWITCH_FRACTION,
    PRETRAIN_TOKENS,
    TOTAL_TOKENS,
)

MERGED_MIXTURE = Path(__file__).resolve().parents[2] / "grug/moe_hero_ep/harrier_mix_2026_08_18.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for candidates and complete schedules")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--pool-size", type=int, default=65_536)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument(
        "--radius", type=float, default=0.05, help="Maximum cooldown total variation from merged weights"
    )
    parser.add_argument("--minimum-distance", type=float, default=0.01)
    parser.add_argument("--device", choices=("cpu", "gpu"))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output directory already exists; choose a new directory")
    jax.config.update("jax_enable_x64", True)
    artifact, model, observed = load_model(args.model, jax.devices(args.device)[0])
    spec = json.loads(MERGED_MIXTURE.read_text())
    phases = spec["phases"]
    if [phase["name"] for phase in phases] != ["initial", "main", "cooldown"]:
        raise ValueError("Expected the merged initial/main/cooldown schedule")
    if artifact.swarm_id != phases[1]["swarm_id"] or artifact.swarm_id != phases[2]["swarm_id"]:
        raise ValueError("The fitted model must use the merged mixture's swarm")
    if set(artifact.components) != set(spec["available_tokens"]) or any(
        set(phase["weights"]) != set(artifact.components) for phase in phases
    ):
        raise ValueError("Model and merged schedule must name the same cells")
    weights = np.array([[phase["weights"][cell] for cell in artifact.components] for phase in phases])
    if not np.isfinite(weights).all() or np.any(weights < 0) or not np.allclose(weights.sum(-1), 1):
        raise ValueError("Merged schedule weights must be finite simplexes")
    available = np.array([spec["available_tokens"][cell] for cell in artifact.components])
    if not np.array_equal(available, artifact.available_tokens):
        raise ValueError("Model and merged schedule have different component token counts")
    initial_tokens = TOTAL_TOKENS * MIXTURE_SWITCH_FRACTION
    consumed = initial_tokens * weights[0] + (PRETRAIN_TOKENS - initial_tokens) * weights[1]
    selected = search_cooldown(
        model,
        weights[1:],
        available,
        consumed,
        COOLDOWN_TOKENS,
        observed,
        pool_size=args.pool_size,
        batch_size=args.batch_size,
        seed=args.seed,
        max_epochs=MAX_COMPONENT_EPOCHS,
        radius=args.radius,
        minimum_distance=args.minimum_distance,
    )
    scores = acquisition_scores(model, selected)
    anchor_score = float(acquisition_scores(model, weights[None, 1:])[0])
    args.output.mkdir(parents=True)
    write_candidates(
        args.output / "candidates.parquet", selected, artifact.swarm_id, artifact.components, artifact.hf_revision
    )
    records = []
    for index, (candidate, score) in enumerate(zip(selected, scores, strict=True), start=1):
        schedule = copy.deepcopy(spec)
        schedule["phases"][2] = {
            "name": "cooldown",
            "weights": dict(zip(artifact.components, candidate[1].tolist(), strict=True)),
            "source_file": "candidates.parquet",
            "candidate": index,
            "swarm_id": artifact.swarm_id,
            "hf_revision": artifact.hf_revision,
        }
        filename = f"schedule-{index:02d}.json"
        (args.output / filename).write_text(json.dumps(schedule, indent=2) + "\n")
        records.append(
            {
                "candidate": index,
                "schedule": filename,
                "predicted_score": float(score),
                "predicted_gain": float(score - anchor_score),
                "cooldown_tv": float(np.abs(candidate[1] - weights[2]).sum() / 2),
                "max_component_epochs": float(((consumed + COOLDOWN_TOKENS * candidate[1]) / available).max()),
            }
        )
    summary = {
        "hf_revision": artifact.hf_revision,
        "source_mixture": str(MERGED_MIXTURE),
        "source_schedule": spec,
        "seed": args.seed,
        "radius": args.radius,
        "minimum_distance": args.minimum_distance,
        "anchor_predicted_score": anchor_score,
        "fixed_prefix_tokens": PRETRAIN_TOKENS,
        "cooldown_tokens": COOLDOWN_TOKENS,
        "model_phase_budgets": artifact.phase_budgets,
        "candidates": records,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(selected)} cooldown candidates and complete schedules to {args.output}")


if __name__ == "__main__":
    main()
