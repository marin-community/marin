# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run a trusted Harbor task script inside ScriptSpec and normalize its legacy reward."""

import argparse
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

from tasktrove_verify.file_ops.read import read_regular_bytes
from tasktrove_verify.grade import scored
from tasktrove_verify.modes.grade_script import parse_json_reward, parse_reward_number

VERDICT = "native-verdict.json"


def _native_reward(logs: Path, reward_key: str) -> tuple[float, dict[str, object]]:
    reward_json = logs / "reward.json"
    if reward_json.exists() or reward_json.is_symlink():
        reward, payload = parse_json_reward(read_regular_bytes(reward_json).decode(), reward_key)
        return reward, {key: value for key, value in payload.items() if key != reward_key}
    if reward_key != "reward":
        raise ValueError(f"named reward {reward_key!r} requires reward.json")
    reward_txt = logs / "reward.txt"
    if not reward_txt.exists():
        raise ValueError("native script wrote no reward file")
    return parse_reward_number(read_regular_bytes(reward_txt).decode().strip()), {}


def run_native(script_name: str, reward_key: str = "reward") -> dict:
    tests = Path(os.environ["TASKTROVE_TESTS_DIR"]).resolve()
    workspace = Path(os.environ["TASKTROVE_WORKSPACE"])
    legacy = Path(os.environ.get("TASKTROVE_NATIVE_LOGS_DIR", "/logs/verifier"))
    script = (tests / script_name).resolve()
    if not script.is_relative_to(tests) or not script.is_file():
        raise ValueError("native script must be a file under the trusted tests directory")
    legacy.mkdir(parents=True, exist_ok=True)
    for filename in ("reward.txt", "reward.json", "verdict.json"):
        (legacy / filename).unlink(missing_ok=True)
    completed = subprocess.run(["bash", str(script)], cwd=workspace, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"native script exited {completed.returncode}: {completed.stderr[-1000:]}")
    reward, metrics = _native_reward(legacy, reward_key)
    return asdict(scored(reward, native_metrics=metrics))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", help="relative source test script under the trusted tests directory")
    parser.add_argument("--reward-key", default="reward")
    args = parser.parse_args(argv)
    try:
        verdict = run_native(args.script, args.reward_key)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        verdict = {"status": "infra_error", "reward": 0.0, "detail": {"error": f"{type(error).__name__}: {error}"}}
    private = Path(os.environ["TASKTROVE_LOGS_DIR"])
    (private / VERDICT).write_text(json.dumps(verdict, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
