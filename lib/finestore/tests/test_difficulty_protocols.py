# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read historical measurements alongside the registered dynamic Large contract."""

import json
from dataclasses import asdict
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from finestore.rl.difficulty import (
    LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID,
    PROTOCOL_ID,
    PROTOCOLS,
    DifficultyObservation,
    SamplingSettings,
    Solver,
)
from finestore.schema import arrow_schema


def observation(protocol_id: str) -> dict[str, Any]:
    protocol = PROTOCOLS[protocol_id]
    context, inputs, outputs, top_p, top_k, min_p, repetition, presence, frequency = protocol.sampling
    reference = {"uri": "memory://difficulty/evidence.json", "sha256": "a" * 64}
    release = {"repo": "fixture", "revision": "frozen", "url": "https://example.org/fixture/frozen"}
    settings: dict[str, Any] = {
        "context_tokens": context,
        "input_tokens": inputs,
        "output_tokens": outputs,
        "temperature": protocol.temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition,
        "presence_penalty": presence,
        "frequency_penalty": frequency,
        "thinking": "on",
        "deviations": [],
    }
    if protocol.dynamic_output is not None:
        settings["dynamic_output"] = asdict(protocol.dynamic_output)
    return {
        "measurement_id": protocol_id,
        "protocol_id": protocol_id,
        "created_at": "2026-11-27T12:00:00+00:00",
        "source_id": "fixture",
        "release": release,
        "converter_version": "frozen",
        "solver": "large",
        "status": "measured",
        "reason": None,
        "sample": {
            "source_id": "fixture",
            "release": release,
            "converter_version": "frozen",
            "converter_revision": "b" * 40,
            "population_count": 1,
            "task_specs_sha256": "c" * 64,
            "drop_manifest": reference,
            "seed": 0,
            "pre_oracle_filter": False,
            "resample_after_oracle_filter": False,
        },
        "sample_receipt": reference,
        "model": {
            "model": protocol.models[Solver.LARGE],
            "revision": "d" * 40,
            "tokenizer": protocol.models[Solver.LARGE],
            "tokenizer_revision": "d" * 40,
            "chat_template_sha256": "e" * 64,
            "serving_receipt": reference,
        },
        "settings": settings,
        "n": 1,
        "solved": 0,
        "wilson_95": {"low": 0.0, "high": 0.8},
        "infrastructure_attempts": 0,
        "pending_tasks": 0,
        "blocked_tasks": 0,
        "attempts_receipt": reference,
        "gpu_hours": 1.0,
        "input_tokens": 100,
        "output_tokens": 1000,
    }


def test_legacy_and_dynamic_measurements_survive_mixed_parquet_and_schema_evolution(tmp_path):
    old = observation(PROTOCOL_ID)
    old_model = DifficultyObservation.model_validate(old)
    dynamic = DifficultyObservation.model_validate(observation(LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID))
    old_schema = arrow_schema(DifficultyObservation)
    settings = old_schema.field("settings").type
    # Reconstruct the historical schema, which had no dynamic_output column.
    settings = pa.struct([field for field in settings if field.name != "dynamic_output"])
    old_schema = old_schema.set(old_schema.get_field_index("settings"), pa.field("settings", settings))
    old_path = tmp_path / "old.parquet"
    pq.write_table(pa.Table.from_pylist([old], schema=old_schema), old_path)
    restored = DifficultyObservation.model_validate(pq.read_table(old_path).to_pylist()[0])
    assert restored == old_model
    assert "dynamic_output" not in restored.model_dump(mode="json")["settings"]
    mixed = tmp_path / "mixed.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [m.model_dump(mode="json") for m in [restored, dynamic]], schema=arrow_schema(DifficultyObservation)
        ),
        mixed,
    )
    assert [DifficultyObservation.model_validate(row) for row in pq.read_table(mixed).to_pylist()] == [
        old_model,
        dynamic,
    ]
    assert DifficultyObservation.model_validate_json(dynamic.model_dump_json()) == dynamic
    assert dynamic.settings.dynamic_output.output_capacity(100) == 65372
    assert dynamic.settings.dynamic_output.output_capacity(49088) == 16384
    with pytest.raises(ValueError, match="minimum output"):
        dynamic.settings.dynamic_output.output_capacity(49089)


@pytest.mark.parametrize("mutation", ["old_identity", "fixed_split", "other_solver", "sampling_deviation"])
def test_dynamic_measurement_cannot_be_mislabeled_or_relax_the_frozen_settings(mutation):
    data = observation(LARGE_DYNAMIC_OUTPUT_PROTOCOL_ID)
    if mutation == "old_identity":
        data["protocol_id"] = PROTOCOL_ID
    elif mutation == "fixed_split":
        data["settings"] = observation(PROTOCOL_ID)["settings"]
    elif mutation == "other_solver":
        data["solver"] = "small"
    else:
        data["settings"]["top_p"] = 0.8
        data["settings"]["deviations"] = ["must not change the ruled sampling contract"]
    with pytest.raises(ValueError):
        DifficultyObservation.model_validate(data)


def test_historical_settings_json_keeps_its_original_fields():
    original = observation(PROTOCOL_ID)["settings"]
    restored = SamplingSettings.model_validate_json(json.dumps(original))
    assert json.loads(restored.model_dump_json()) == original
