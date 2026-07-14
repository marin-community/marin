# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import dataclasses
import json

import numpy as np

from experiments.probabilistic_dataflow.compiler import (
    Autoregressive,
    AvailabilityError,
    FactorizationError,
    ParallelQuery,
    Refine,
    compile_query,
)
from experiments.probabilistic_dataflow.scientific_metrics import field_rmse, score, spectral_error
from experiments.probabilistic_dataflow.synthetic import (
    advection_example,
    advection_problem,
    factorized_structure_problem,
    leaky_normalization_problem,
)
from experiments.probabilistic_dataflow.training import (
    build_mixed_synthetic_batch,
    record_order_equivariance_error,
    train_cross_domain_smoke,
    train_smoke,
)


def run_demo(*, training_steps: int) -> dict:
    advection = advection_problem()
    advection_values = advection_example(advection, seed=0).values
    parallel = compile_query(advection.query, ParallelQuery(advection.targets))
    refinement = compile_query(
        advection.query,
        Refine(ParallelQuery(advection.targets), steps=3, resample_fraction=0.25),
    )

    structure, _sequence, contacts, distances = factorized_structure_problem()
    autoregressive = compile_query(structure.query, Autoregressive((contacts, distances)))
    try:
        compile_query(structure.query, ParallelQuery((contacts, distances)))
    except FactorizationError as exc:
        factorization_diagnostic = str(exc)
    else:
        raise AssertionError("Dependent factors unexpectedly compiled as a parallel query")

    leaky = leaky_normalization_problem()
    try:
        compile_query(leaky.query)
    except AvailabilityError as exc:
        leakage_diagnostic = str(exc)
    else:
        raise AssertionError("Future-derived normalization unexpectedly passed availability analysis")

    packed, codec = build_mixed_synthetic_batch(examples_per_problem=2)
    result = train_smoke(steps=training_steps) if training_steps > 0 else None
    cross_domain_result = train_cross_domain_smoke(steps=training_steps) if training_steps > 0 else None
    initial = np.asarray(advection_values["initial"])
    truth = np.asarray(advection_values["future"])
    persistence = np.tile(initial, 3)
    return {
        "programs": ["synthetic_advection", "synthetic_contacts", "synthetic_structure"],
        "parallel_plan": _plan_summary(parallel),
        "refinement_plan": _plan_summary(refinement),
        "autoregressive_plan": _plan_summary(autoregressive),
        "factorization_diagnostic": factorization_diagnostic,
        "leakage_diagnostic": leakage_diagnostic,
        "packing": {
            "shape": list(packed.token_ids.shape),
            "supervised_tokens": int(packed.loss_weights.sum()),
            "sequences": len(packed.locations),
            "vocab_size": codec.vocab_size,
            "scientific_position_count": codec.scientific_position_count,
        },
        "persistence_scores": [
            dataclasses.asdict(metric) for metric in score(persistence, truth, metrics=(field_rmse, spectral_error))
        ],
        "record_order_equivariance_max_logit_error": record_order_equivariance_error(),
        "training": dataclasses.asdict(result) if result is not None else None,
        "cross_domain_training": dataclasses.asdict(cross_domain_result) if cross_domain_result is not None else None,
    }


def _plan_summary(plan) -> list[dict]:
    return [
        {
            "call": call.id,
            "operator": call.operator,
            "targets": list(call.target_ids),
            "context": list(call.context_ids),
            "dependencies": list(call.dependency_call_ids),
            "iteration": call.iteration,
            "approximations": list(call.approximation_notes),
        }
        for call in plan.calls
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the probabilistic scientific dataflow compiler spike")
    parser.add_argument("--training-steps", type=int, default=80)
    args = parser.parse_args()
    print(json.dumps(run_demo(training_steps=args.training_steps), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
