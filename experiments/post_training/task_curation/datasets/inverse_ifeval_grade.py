# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run private VerifIF checks independently and require all source checks to pass."""

import json
import os
import re
import time
import tomllib
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path


def visible_answer(text: str) -> str:
    for tag in ("think", "thinking"):
        text = re.sub(rf"<{tag}>.*?</{tag}>", "", text, flags=re.DOTALL)
        text = re.sub(rf"^.*?</{tag}>", "", text, flags=re.DOTALL)
    return text.strip()


def native_verdict(text: str) -> bool:
    answer = visible_answer(text)
    yes, no = answer.rfind("[[YES]]"), answer.rfind("[[NO]]")
    if yes >= 0 or no >= 0:
        return yes > no
    last = answer.rsplit("\n", 1)[-1].strip().upper()
    if last in {"YES", "NO"}:
        return last == "YES"
    # A malformed provider response is unscored, including expected-NO rows.
    raise ValueError("VerifIF judge returned no conclusive YES/NO verdict")


def score_checks(checks: list[dict], candidate: str, judge: Callable[[str, str], str]) -> tuple[float, list[dict]]:
    answer = visible_answer(candidate)
    results = []
    for check in checks:
        said_yes = native_verdict(judge(check["content"], answer))
        passed = said_yes if check["pass_criteria"] == "YES" else not said_yes
        results.append(
            {"uid": check["uid"], "expected": check["pass_criteria"], "judge_said_yes": said_yes, "passed": passed}
        )
    if not results:
        raise ValueError("VerifIF task has no source checks")
    return float(all(row["passed"] for row in results)), results


def request_verdict(config: dict, spec: dict, question: str, candidate: str) -> str:
    prompt = config["native_judge_template"].format(question=question, model_response=candidate)
    payload = {
        "model": spec["model"],
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": "Evaluate the response."}],
        "temperature": spec["temperature"],
        "top_p": config["top_p"],
        "max_tokens": spec["max_completion_tokens"],
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
                raise ValueError("VerifIF judge completion is incomplete")
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
    raise RuntimeError("VerifIF judge exhausted its transport attempts")


def main() -> None:
    root = Path("/tests")
    config = json.loads((root / "config.json").read_text())
    spec = tomllib.loads((root / "judge.toml").read_text())
    if os.environ["VERIFYIT_JUDGE_MODEL"] != spec["model"]:
        raise ValueError("Runtime judge differs from the task's pinned model")
    reward, checks = score_checks(
        config["checks"],
        Path("/app/answer.txt").read_text(),
        lambda question, candidate: request_verdict(config, spec, question, candidate),
    )
    Path("/app/verdict.json").write_text(
        json.dumps({"status": "scored", "reward": reward, "detail": {"checks": checks, "aggregation": "all"}})
    )


if __name__ == "__main__":
    main()
