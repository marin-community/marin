# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the upstream answer-equivalence prompt and its unchanged result parser."""

import json
import os
import urllib.request
from pathlib import Path

from .scorers.judge_parser import parse_judge_response
from .scorers.prompts import GRADER_TEMPLATE

CONFIG = Path("/tests/config.json")
ANSWER = Path("/app/answer.txt")
VERDICT = Path("/app/verdict.json")


def main() -> None:
    config = json.loads(CONFIG.read_text())
    prompt = GRADER_TEMPLATE.format(
        question=config["question"], response=ANSWER.read_text(), correct_answer=config["answer"]
    )
    request = urllib.request.Request(
        os.environ["VERIFYIT_JUDGE_BASE_URL"].rstrip("/") + "/chat/completions",
        data=json.dumps(
            {
                "model": os.environ["VERIFYIT_JUDGE_MODEL"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "max_tokens": 4096,
            }
        ).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + os.environ["VERIFYIT_JUDGE_API_KEY"],
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.load(response)
    result = parse_judge_response(payload["choices"][0]["message"]["content"])
    if result["parse_error"]:
        raise ValueError("Original BrowseComp judge response has no conclusive correctness field")
    VERDICT.write_text(json.dumps({"status": "scored", "reward": float(result["correct"]), "detail": result}))


if __name__ == "__main__":
    main()
