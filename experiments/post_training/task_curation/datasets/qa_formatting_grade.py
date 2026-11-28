# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the original abstention or formatting rule against the final reply."""

import json
import os
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

from experiments.post_training.task_curation.datasets.nemotron_ultra.scorers.skyrl_gym.envs.nemotron_ultra import (
    format_verification,
)
from experiments.post_training.task_curation.datasets.qa_abstention_native import (
    _DEFAULT_JUDGE_PROMPT_TEMPLATE,
    _strip_thinking_traces,
    extract_boxed_answer,
    normalize_answer,
    parse_judge_grade,
)


def score_abstention(question: str, target: str, candidate: str, judge: Callable[[str], str]) -> tuple[float, dict]:
    answer = _strip_thinking_traces(candidate)
    boxed = extract_boxed_answer(answer)
    extracted = boxed if boxed is not None else answer
    if normalize_answer(extracted) == normalize_answer("[IDK]"):
        return 0.5, {"verdict": "abstain", "extracted_answer": extracted, "judge_output": None}
    prompt = _DEFAULT_JUDGE_PROMPT_TEMPLATE.format(question=question, target=target, predicted_answer=extracted)
    reply = judge(prompt)
    if not reply:
        raise ValueError("Native abstention judge returned an empty response")
    label = parse_judge_grade(_strip_thinking_traces(reply))
    return {"A": 1.0, "B": 0.0, "C": 0.5}[label], {
        "verdict": {"A": "correct", "B": "incorrect", "C": "abstain"}[label],
        "extracted_answer": extracted,
        "judge_output": reply,
    }


def request_grade(spec: dict, prompt: str) -> str:
    if os.environ["VERIFYIT_JUDGE_MODEL"] != spec["model"]:
        raise ValueError("Runtime judge differs from the pinned task model")
    payload = {
        "model": spec["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 64,
        "chat_template_kwargs": spec["chat_template_kwargs"],
    }
    request = urllib.request.Request(
        os.environ["VERIFYIT_JUDGE_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["VERIFYIT_JUDGE_API_KEY"]},
    )
    started = time.monotonic()
    for attempt in range(spec["transport_attempts"]):
        try:
            with urllib.request.urlopen(request, timeout=spec["request_timeout"]) as response:
                choice = json.load(response)["choices"][0]
            if choice["finish_reason"] != "stop" or not isinstance(choice["message"]["content"], str):
                raise ValueError("Native abstention judge response is incomplete")
            return choice["message"]["content"]
        except (urllib.error.URLError, TimeoutError) as error:
            retryable = not isinstance(error, urllib.error.HTTPError) or error.code in {408, 429, 500, 502, 503, 504}
            delay = min(2**attempt, 8)
            if (
                not retryable
                or attempt + 1 == spec["transport_attempts"]
                or time.monotonic() - started + delay >= spec["transport_budget"]
            ):
                raise
            time.sleep(delay)
    raise RuntimeError("Native abstention judge exhausted its transport attempts")


def main() -> None:
    root = Path("/tests")
    config = json.loads((root / "config.json").read_text())
    candidate = Path("/app/answer.txt").read_text()
    if config["family"] == "abstention":
        spec = tomllib.loads((root / "judge.toml").read_text())
        reward, detail = score_abstention(
            config["question"], config["answer"], candidate, lambda prompt: request_grade(spec, prompt)
        )
    elif config["family"] == "formatting":
        # Gym concatenates output_text parts; it does not strip text-based thinking tags.
        reward, detail = format_verification.grade_format(candidate, config["verifier"])
    else:
        raise ValueError(f"Unknown native scoring family: {config['family']}")
    Path("/app/verdict.json").write_text(json.dumps({"status": "scored", "reward": reward, "detail": detail}))


if __name__ == "__main__":
    main()
