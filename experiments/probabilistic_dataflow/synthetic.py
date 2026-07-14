# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from experiments.probabilistic_dataflow.compiler import ConcreteExample
from experiments.probabilistic_dataflow.dsl import (
    AttentionPattern,
    Budget,
    DocumentSpec,
    FieldType,
    InferenceProgram,
    MeshAxis,
    OrderedAxis,
    PositionMode,
    SetAxis,
    Source,
    UnorderedPairAxis,
    Value,
)

SYNTHETIC_SPLIT = frozenset({"synthetic-train"})
SCIENTIFIC_FULL = DocumentSpec(attention=AttentionPattern.FULL, positions=PositionMode.SCIENTIFIC)
SMALL_BUDGET = Budget(model_calls=8, generated_tokens=64)
STRUCTURE_BUDGET = Budget(model_calls=16, generated_tokens=128)


def scalar_forecast_program() -> InferenceProgram:
    measurement = FieldType("measurement", bins=16)
    program = InferenceProgram("scalar_forecast", budget=SMALL_BUDGET)
    current = program.input_value(
        "current",
        measurement,
        source=Source(
            name="synthetic.scalar_current",
            split_keys=SYNTHETIC_SPLIT,
        ),
    )
    future = program.generate(
        "future",
        measurement,
        context=(current,),
        document=SCIENTIFIC_FULL,
        factor_name="scalar_transition",
    )
    return program.finish(future)


def advection_program() -> InferenceProgram:
    program, _initial, _forcing, future = _advection_program("synthetic_advection")
    return program.finish(future)


def refined_advection_program() -> InferenceProgram:
    program, initial, forcing, future = _advection_program("synthetic_advection")
    program.refine(
        future,
        context=(initial, forcing),
        document=SCIENTIFIC_FULL,
        resample_fraction=0.25,
    )
    program.refine(
        future,
        context=(initial, forcing),
        document=SCIENTIFIC_FULL,
        resample_fraction=0.25,
    )
    return program.finish(future)


def _advection_program(name: str) -> tuple[InferenceProgram, Value, Value, Value]:
    cell = MeshAxis("cell", 4, coordinates=((0.0,), (0.25,), (0.5,), (0.75,)))
    time = OrderedAxis("time", 3)
    state = FieldType("state", (cell,), bins=16)
    forcing_type = FieldType("forcing", (time, cell), bins=16)
    trajectory = FieldType("state_trajectory", (time, cell), bins=16)
    program = InferenceProgram(name, budget=SMALL_BUDGET)
    initial = program.input_value(
        "initial",
        state,
        source=Source(name="synthetic.initial", split_keys=SYNTHETIC_SPLIT),
    )
    forcing = program.input_value(
        "forcing",
        forcing_type,
        source=Source(name="synthetic.forcing", split_keys=SYNTHETIC_SPLIT),
    )
    future = program.generate(
        "future",
        trajectory,
        context=(initial, forcing),
        document=SCIENTIFIC_FULL,
        factor_name="advection_transition",
    )
    return program, initial, forcing, future


def contacts_program() -> InferenceProgram:
    residue = SetAxis("residue", 4)
    pair = UnorderedPairAxis.of(residue)
    sequence_type = FieldType("residue_class", (residue,), bins=8)
    contacts_type = FieldType("contact", (pair,), bins=2)
    program = InferenceProgram("synthetic_contacts", budget=SMALL_BUDGET)
    sequence = program.input_value(
        "sequence",
        sequence_type,
        source=Source(name="synthetic.sequence", split_keys=SYNTHETIC_SPLIT),
    )
    contacts = program.generate(
        "contacts",
        contacts_type,
        context=(sequence,),
        document=SCIENTIFIC_FULL,
        factor_name="contact_map",
    )
    return program.finish(contacts)


def structure_program() -> InferenceProgram:
    residue = SetAxis("residue", 4)
    pair = UnorderedPairAxis.of(residue)
    sequence_type = FieldType("residue_class", (residue,), bins=8)
    contacts_type = FieldType("contact", (pair,), bins=2)
    distance_type = FieldType("distance", (pair,), bins=8)
    program = InferenceProgram("synthetic_structure", budget=STRUCTURE_BUDGET)
    sequence = program.input_value(
        "sequence",
        sequence_type,
        source=Source(name="synthetic.sequence", split_keys=SYNTHETIC_SPLIT),
    )
    contacts = program.generate(
        "contacts",
        contacts_type,
        context=(sequence,),
        document=SCIENTIFIC_FULL,
        factor_name="contact_map",
    )
    distances = program.generate(
        "distances",
        distance_type,
        context=(sequence, contacts),
        document=SCIENTIFIC_FULL,
        factor_name="distance_given_contacts",
    )
    return program.finish(contacts, distances)


def scalar_forecast_example(program: InferenceProgram) -> ConcreteExample:
    return ConcreteExample(
        id="scalar-0",
        program_name=program.name,
        values={"current": (3,), "future": (5,)},
    )


def advection_example(program: InferenceProgram, *, seed: int) -> ConcreteExample:
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
        program_name=program.name,
        values={
            "initial": tuple(int(value) for value in initial),
            "forcing": tuple(int(value) for value in forcing.flat),
            "future": tuple(int(value) for value in future.flat),
        },
    )


def contacts_example(program: InferenceProgram, *, seed: int) -> ConcreteExample:
    rng = np.random.default_rng(seed)
    sequence = rng.integers(0, 4, size=4, dtype=np.int32)
    pair_axis = program.value("contacts").value_type.axes[0]
    if not isinstance(pair_axis, UnorderedPairAxis):
        raise TypeError(f"Expected UnorderedPairAxis, got {type(pair_axis).__name__}")
    contacts = tuple(int(sequence[left] == sequence[right]) for left, right in pair_axis.canonical_pairs())
    return ConcreteExample(
        id=f"contacts-{seed}",
        program_name=program.name,
        values={
            "sequence": tuple(int(value) for value in sequence),
            "contacts": contacts,
        },
    )


def structure_example(program: InferenceProgram, *, seed: int) -> ConcreteExample:
    rng = np.random.default_rng(seed)
    sequence = rng.integers(0, 4, size=4, dtype=np.int32)
    pair_axis = program.value("contacts").value_type.axes[0]
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
        program_name=program.name,
        values={
            "sequence": tuple(int(value) for value in sequence),
            "contacts": contacts,
            "distances": distances,
        },
    )
