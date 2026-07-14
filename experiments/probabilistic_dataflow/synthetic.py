# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from experiments.probabilistic_dataflow.compiler import ConcreteExample
from experiments.probabilistic_dataflow.dsl import (
    Budget,
    Environment,
    Evidence,
    FieldType,
    MeshAxis,
    OrderedAxis,
    Program,
    Query,
    SetAxis,
    Source,
    UnorderedPairAxis,
    Value,
    learned_joint,
)

DEPLOYMENT = Environment("deployment", execution_time=0)
TRAINING = Environment("training", execution_time=1)
SYNTHETIC_SPLIT = frozenset({"synthetic-train"})
ALL_ENVIRONMENTS = frozenset({DEPLOYMENT.name, TRAINING.name})


@dataclass(frozen=True)
class SyntheticProblem:
    program: Program
    query: Query
    targets: tuple[Value, ...]


def scalar_forecast_problem() -> SyntheticProblem:
    measurement = FieldType("measurement", bins=16)

    with Program("scalar_forecast") as program:
        current = program.variable(
            "current",
            measurement,
            source=Source("synthetic.scalar_current", 0, ALL_ENVIRONMENTS, SYNTHETIC_SPLIT),
        )
        future = program.sample("future", measurement, learned_joint(current, name="scalar_transition"))

    evidence = Evidence().bind(current, environment=DEPLOYMENT.name)
    query = Query(program, evidence, (future,), DEPLOYMENT)
    return SyntheticProblem(program, query, (future,))


def advection_problem() -> SyntheticProblem:
    cell = MeshAxis("cell", 4, coordinates=((0.0,), (0.25,), (0.5,), (0.75,)))
    time = OrderedAxis("time", 3)
    state = FieldType("state", (cell,), bins=16)
    forcing_type = FieldType("forcing", (time, cell), bins=16)
    trajectory = FieldType("state_trajectory", (time, cell), bins=16)

    with Program("synthetic_advection") as program:
        initial = program.variable(
            "initial",
            state,
            source=Source("synthetic.initial", 0, ALL_ENVIRONMENTS, SYNTHETIC_SPLIT),
        )
        forcing = program.variable(
            "forcing",
            forcing_type,
            source=Source("synthetic.forcing", 0, ALL_ENVIRONMENTS, SYNTHETIC_SPLIT),
        )
        future = program.sample("future", trajectory, learned_joint(initial, forcing, name="advection_transition"))

    evidence = Evidence().bind(initial, environment=DEPLOYMENT.name).bind(forcing, environment=DEPLOYMENT.name)
    query = Query(
        program,
        evidence,
        (future,),
        DEPLOYMENT,
        Budget(model_calls=8, generated_tokens=64),
    )
    return SyntheticProblem(program, query, (future,))


def symmetric_pairs_problem() -> SyntheticProblem:
    residue = SetAxis("residue", 4)
    pair = UnorderedPairAxis.of(residue)
    sequence_type = FieldType("residue_class", (residue,), bins=8)
    contacts_type = FieldType("contact", (pair,), bins=2)

    with Program("synthetic_contacts") as program:
        sequence = program.variable(
            "sequence",
            sequence_type,
            source=Source("synthetic.sequence", 0, ALL_ENVIRONMENTS, SYNTHETIC_SPLIT),
        )
        contacts = program.sample("contacts", contacts_type, learned_joint(sequence, name="contact_map"))

    evidence = Evidence().bind(sequence, environment=DEPLOYMENT.name)
    query = Query(
        program,
        evidence,
        (contacts,),
        DEPLOYMENT,
        Budget(model_calls=8, generated_tokens=64),
    )
    return SyntheticProblem(program, query, (contacts,))


def factorized_structure_problem() -> tuple[SyntheticProblem, Value, Value, Value]:
    residue = SetAxis("residue", 4)
    pair = UnorderedPairAxis.of(residue)
    sequence_type = FieldType("residue_class", (residue,), bins=8)
    contacts_type = FieldType("contact", (pair,), bins=2)
    distance_type = FieldType("distance", (pair,), bins=8)

    with Program("synthetic_structure") as program:
        sequence = program.variable(
            "sequence",
            sequence_type,
            source=Source("synthetic.sequence", 0, ALL_ENVIRONMENTS, SYNTHETIC_SPLIT),
        )
        contacts = program.sample("contacts", contacts_type, learned_joint(sequence, name="contact_map"))
        distances = program.sample(
            "distances",
            distance_type,
            learned_joint(sequence, contacts, name="distance_given_contacts"),
        )

    evidence = Evidence().bind(sequence, environment=DEPLOYMENT.name)
    query = Query(
        program,
        evidence,
        (contacts, distances),
        DEPLOYMENT,
        Budget(model_calls=16, generated_tokens=128),
    )
    return SyntheticProblem(program, query, (contacts, distances)), sequence, contacts, distances


def leaky_normalization_problem() -> SyntheticProblem:
    cell = MeshAxis("cell", 4)
    state = FieldType("state", (cell,), bins=16)

    with Program("leaky_normalization") as program:
        initial = program.variable(
            "initial",
            state,
            source=Source("synthetic.initial", 0, ALL_ENVIRONMENTS, SYNTHETIC_SPLIT),
        )
        future_statistics = program.variable(
            "future_statistics",
            state,
            source=Source("synthetic.future_statistics", 1, ALL_ENVIRONMENTS, SYNTHETIC_SPLIT),
        )
        normalization = program.map("normalization", future_statistics, operation="mean_and_variance")
        forecast = program.sample(
            "forecast",
            state,
            learned_joint(initial, normalization, name="normalized_forecast"),
        )

    evidence = Evidence().bind(initial, environment=DEPLOYMENT.name).bind(normalization, environment=DEPLOYMENT.name)
    query = Query(program, evidence, (forecast,), DEPLOYMENT)
    return SyntheticProblem(program, query, (forecast,))


def scalar_forecast_example(problem: SyntheticProblem) -> ConcreteExample:
    return ConcreteExample(
        id="scalar-0",
        program_name=problem.program.name,
        values={"current": (3,), "future": (5,)},
    )


def advection_example(problem: SyntheticProblem, *, seed: int) -> ConcreteExample:
    rng = np.random.default_rng(seed)
    initial = rng.integers(0, 16, size=4, dtype=np.int32)
    forcing = rng.integers(0, 4, size=(3, 4), dtype=np.int32)
    future_steps = []
    current = initial
    for step_forcing in forcing:
        current = (np.roll(current, 1) + step_forcing) % 16
        future_steps.append(current)
    future = np.stack(future_steps)
    return ConcreteExample(
        id=f"advection-{seed}",
        program_name=problem.program.name,
        values={
            "initial": tuple(int(value) for value in initial),
            "forcing": tuple(int(value) for value in forcing.flat),
            "future": tuple(int(value) for value in future.flat),
        },
    )


def symmetric_pairs_example(problem: SyntheticProblem, *, seed: int) -> ConcreteExample:
    rng = np.random.default_rng(seed)
    sequence = rng.integers(0, 4, size=4, dtype=np.int32)
    pair_axis = problem.targets[0].value_type.axes[0]
    if not isinstance(pair_axis, UnorderedPairAxis):
        raise TypeError(f"Expected UnorderedPairAxis, got {type(pair_axis).__name__}")
    contacts = tuple(int(sequence[left] == sequence[right]) for left, right in pair_axis.canonical_pairs())
    return ConcreteExample(
        id=f"contacts-{seed}",
        program_name=problem.program.name,
        values={
            "sequence": tuple(int(value) for value in sequence),
            "contacts": contacts,
        },
    )


def factorized_structure_example(problem: SyntheticProblem, *, seed: int) -> ConcreteExample:
    rng = np.random.default_rng(seed)
    sequence = rng.integers(0, 4, size=4, dtype=np.int32)
    pair_axis = problem.program.value("contacts").value_type.axes[0]
    if not isinstance(pair_axis, UnorderedPairAxis):
        raise TypeError(f"Expected UnorderedPairAxis, got {type(pair_axis).__name__}")
    pairs = pair_axis.canonical_pairs()
    contacts = tuple(int(sequence[left] == sequence[right]) for left, right in pairs)
    distances = tuple(
        int((abs(int(sequence[left]) - int(sequence[right])) + 1 - contact) % 8)
        for (left, right), contact in zip(pairs, contacts, strict=True)
    )
    return ConcreteExample(
        id=f"structure-{seed}",
        program_name=problem.program.name,
        values={
            "sequence": tuple(int(value) for value in sequence),
            "contacts": contacts,
            "distances": distances,
        },
    )
