"""Run source-owned cohort scoring inside a bounded verifyit ScriptSpec."""

import dataclasses
import json
import math
import os
from pathlib import Path
import shlex
import sys
import tempfile

from verifyit.grade import Status, run
from verifyit.spec import ScriptSpec, render_spec

from skyrl_gym.envs.nemotron_ultra.genrm import grade_genrm_group
from skyrl_gym.envs.nemotron_ultra.judge import OpenAIJudge


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _validate(history, responses, principle, config):
    if not isinstance(history, list) or not history or not isinstance(responses, list) or len(responses) < 2:
        raise ValueError("GenRM requires a conversation and a comparison cohort")
    if any(not isinstance(response, dict) for response in responses):
        raise ValueError("GenRM response objects must be mappings")
    if not isinstance(principle, str) or not principle.strip():
        raise ValueError("GenRM principle must be nonempty")
    for name in (
        "default_score",
        "reasoning_bonus",
        "answer_bonus",
        "top_percentile",
        "group_reasoning_length_penalty_coeff",
        "group_answer_length_penalty_coeff",
        "temperature",
        "top_p",
        "genrm_parse_retry_sleep_seconds",
    ):
        if name in config and not _finite(config[name]):
            raise ValueError(f"Invalid finite GenRM configuration: {name}")
    if "group_answer_length_penalty_coeff" not in config:
        raise ValueError("Missing GenRM answer penalty policy")
    for name in (
        "max_concurrent_comparisons",
        "max_output_tokens",
        "genrm_parse_retries",
    ):
        if name in config and (
            type(config[name]) is not int or config[name] < (0 if name == "genrm_parse_retries" else 1)
        ):
            raise ValueError(f"Invalid GenRM integer configuration: {name}")


def _validate_scores(rewards, metrics, size):
    if not isinstance(rewards, list) or len(rewards) != size or not all(_finite(value) for value in rewards):
        raise ValueError("GenRM cohort reward vector is incomplete or nonfinite")
    if not isinstance(metrics, dict) or not all(
        isinstance(key, str) and _finite(value) for key, value in metrics.items()
    ):
        raise ValueError("GenRM metrics are nonfinite")


def grade_genrm_verifyit(*, conversation_history, response_objects, principle, judge, config):
    _validate(conversation_history, response_objects, principle, config)
    timeout = config.get("verifyit_timeout_seconds", 120.0)
    if not _finite(timeout) or not 0 < timeout <= 3600:
        raise ValueError("Invalid GenRM total deadline")
    with tempfile.TemporaryDirectory(prefix="skyrl-genrm-") as directory:
        root = Path(directory)
        payload = {
            "conversation_history": conversation_history,
            "response_objects": response_objects,
            "principle": principle,
            "judge": dataclasses.asdict(judge),
            "config": {
                **config,
                "verifyit_enabled": False,
                "verifyit_strict_json": True,
            },
        }
        (root / "input.json").write_text(json.dumps(payload, allow_nan=False))
        (root / "check.sh").write_text(
            "set -eu\nexec "
            + shlex.quote(sys.executable)
            + " -m skyrl_gym.envs.nemotron_ultra.genrm_verifyit "
            + shlex.quote(str(root / "input.json"))
            + "\n"
        )
        spec = root / "verifier.toml"
        spec.write_text(render_spec(ScriptSpec(path="check.sh", timeout=float(timeout), verdict_file="cohort.json")))
        result = run(spec, root)
        if result.status is not Status.SCORED:
            raise RuntimeError(f"GenRM cohort verification failed: {result.status}")
        rewards, metrics = (
            result.detail["cohort_rewards"],
            result.detail["cohort_metrics"],
        )
        _validate_scores(rewards, metrics, len(response_objects))
        return rewards, metrics


def _main():
    data = json.loads(Path(sys.argv[1]).read_text())
    _validate(
        data["conversation_history"],
        data["response_objects"],
        data["principle"],
        data["config"],
    )
    rewards, metrics = grade_genrm_group(
        conversation_history=data["conversation_history"],
        response_objects=data["response_objects"],
        principle=data["principle"],
        judge=OpenAIJudge(**{**data["judge"], "strict_completion": True}),
        config=data["config"],
    )
    _validate_scores(rewards, metrics, len(data["response_objects"]))
    # The scalar is only ScriptSpec's execution envelope. Training consumes the vector.
    result = {
        "schema_version": 1,
        "status": "scored",
        "reward": 0.0,
        "detail": {
            "cohort_rewards": rewards,
            "cohort_metrics": metrics,
            "comparison_count": len(data["response_objects"]),
            "runtime": "source_genrm",
        },
    }
    (Path(os.environ["VERIFYIT_LOGS_DIR"]) / "cohort.json").write_text(json.dumps(result, allow_nan=False))


if __name__ == "__main__":
    _main()
