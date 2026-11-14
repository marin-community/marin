# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete three-task measurements for installed Atlas API and browser tests."""

from finestore.rl.curation import SourceDeclaration
from finestore.rl.difficulty import PROTOCOL_ID, SOLVER_MODELS, SOLVER_THINKING, DifficultyMeasurement, Solver


def difficulty_measurement(declaration: SourceDeclaration, solver: Solver, solved: int = 1) -> DifficultyMeasurement:
    assert declaration.pipeline is not None
    assert declaration.actor_census_input is not None
    reference = {"uri": "s3://fixture/evidence.json", "sha256": "a" * 64}
    sample = {
        "source_id": declaration.id,
        "release": declaration.current_release.model_dump(mode="json"),
        "converter_version": declaration.pipeline.version,
        "converter_revision": declaration.actor_census_input.converter_revision,
        "population_count": 3,
        "task_specs_sha256": declaration.actor_census_input.task_specs_sha256,
        "drop_manifest": reference,
        "seed": 0,
        "tasks": [{"task_id": str(index), "task_sha256": str(index) * 64} for index in range(3)],
        "pre_oracle_filter": True,
        "resample_after_oracle_filter": True,
    }
    return DifficultyMeasurement.model_validate(
        {
            "measurement_id": f"{declaration.id}-{solver}-{solved}",
            "protocol_id": PROTOCOL_ID,
            "created_at": "2026-11-13T23:00:00+00:00",
            "source_id": declaration.id,
            "release": declaration.current_release.model_dump(mode="json"),
            "converter_version": declaration.pipeline.version,
            "solver": solver,
            "status": "measured",
            "reason": None,
            "sample": sample,
            "sample_receipt": reference,
            "model": {
                "model": SOLVER_MODELS[solver],
                "revision": "d" * 40,
                "tokenizer": f"fixture/{solver}",
                "tokenizer_revision": "d" * 40,
                "chat_template_sha256": "e" * 64,
                "serving_receipt": reference,
            },
            "settings": {
                "context_tokens": 65536,
                "input_tokens": 49152,
                "output_tokens": 16384,
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 20,
                "min_p": 0,
                "repetition_penalty": 1,
                "presence_penalty": 0,
                "frequency_penalty": 0,
                "thinking": SOLVER_THINKING[solver],
                "deviations": [],
            },
            "n": 3,
            "solved": solved,
            "wilson_95": (
                {"low": 0.06149194472039621, "high": 0.7923403991979522}
                if solved == 1
                else {"low": 0.4385029682449546, "high": 1}
            ),
            "infrastructure_attempts": 2,
            "pending_tasks": 0,
            "attempts_receipt": reference,
            "gpu_hours": 0.1,
            "input_tokens": 1000,
            "output_tokens": 100,
        }
    )
