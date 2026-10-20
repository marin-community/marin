# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical source records shared by selection and deployment regression tests."""

import json

from taskcompendium.runtime.resources import inline_resource


def closure_row(
    identifier,
    *,
    prefix='{"a":',
    expected='{"a":1}',
    tokens=(128000, 42, 128005),
    family="json_object",
    control="continue_required",
):
    config = {
        "extras": {
            "prompt_token_ids": list(tokens),
            "prefix": prefix,
            "expected": expected,
            "format": "json",
            "family": family,
            "control": control,
        },
        "config": {},
    }
    task = {
        "id": identifier,
        "source": {"dataset": "structure-closure", "revision": "frozen", "row": identifier, "importer_revision": "1"},
        "context": {"events": [{"type": "message", "role": "user", "content": "Continue the prefix."}]},
        "environment_requirements": {},
        "answer_type": "text",
        "answer_format": {"kind": "plain_text"},
        "grader": {"kind": "session"},
        "resources": {"verifier": [inline_resource("config.json", json.dumps(config).encode()).model_dump(mode="json")]},
    }
    # Include whitespace to ensure import does not serialize the native payload again.
    return {
        "lowered_task_spec": json.dumps(
            {
                "task": task,
                "runtime": {"task_machine": None, "verifier_machine": None},
                "session": {"task_session": "structure_closure", "max_turns": 1},
            },
            indent=2,
        ),
        "teacher_route": "structure-closure",
    }
