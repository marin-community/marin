# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attach source-bound benchmark tests to clean replay, never to training messages."""

import base64
import hashlib
import shlex


def fixture_verification_calls(source: dict, environment: dict) -> list[dict]:
    fixture = environment.get("fixture")
    if fixture is None:
        if source["source"].startswith("open_swe_traces/"):
            raise ValueError("Open-SWE requires its original repository fixture")
        return []
    if source["id"] not in fixture["source_ids"]:
        raise ValueError("Repository fixture belongs to another source trace")
    patch = fixture["test_patch"].encode()
    if hashlib.sha256(patch).hexdigest() != fixture["test_patch_sha256"]:
        raise ValueError("Benchmark test patch changed")
    encoded = base64.b64encode(patch).decode()
    program = (
        "import base64, subprocess; "
        "subprocess.run(['git', 'diff', '--exit-code', '--', 'tests'], check=True); "
        f"subprocess.run(['git', 'apply', '-'], input=base64.b64decode({encoded!r}), check=True); "
        f"subprocess.run(['bash', '-lc', {fixture['test_command']!r}], check=True)"
    )
    return [
        {
            "name": "bash",
            "arguments": {"command": shlex.quote(environment["recorder_python"]) + " -c " + shlex.quote(program)},
        }
    ]


def model_environment(environment: dict) -> dict:
    """Describe the fixture without supplying benchmark oracle tests to the model."""
    if "fixture" not in environment:
        return environment
    return {
        **environment,
        "fixture": {
            key: value for key, value in environment["fixture"].items() if key not in ("test_patch", "test_command")
        },
    }
