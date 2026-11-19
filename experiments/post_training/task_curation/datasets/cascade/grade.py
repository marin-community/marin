# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the source NeMo terminal-answer graders without their HTTP server."""

import json
import multiprocessing
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from math_with_judge import _extract_last_boxed_answer, _run_math_verify_in_subprocess
from mcqa import MCQAResourcesServer
from pydantic import BaseModel
from structured_outputs import StructuredOutputsResourcesServer

MATH_TIMEOUT = 10.0


class Reply(BaseModel):
    output_text: str


class McqaRequest(BaseModel):
    """The source scorer's input fields; its output metadata is not a reward input."""

    options: list[dict[str, str | None]]
    expected_answer: str
    grading_mode: str = "strict_single_letter_boxed"
    template_metadata: dict[str, Any] | None = None
    response: Reply


def math_reward(answer: str, contract: dict) -> float:
    # The primary math_with_judge.yaml selects should_use_judge=false and the default
    # format_tolerant_answer_extraction=false. Preserve its required boxed-answer gate.
    candidate = _extract_last_boxed_answer(answer)
    if candidate is None or not candidate.strip():
        return 0.0
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_run_math_verify_in_subprocess, args=(contract["expected_answer"], answer, writer))
    process.start()
    writer.close()
    try:
        if not reader.poll(MATH_TIMEOUT):
            return 0.0
        reward, _ = reader.recv()
        return reward
    finally:
        reader.close()
        if process.is_alive():
            process.terminate()
        process.join()


def score(answer: str, contract: dict) -> float:
    agent = contract["agent_ref"]["name"]
    if agent == "mcqa_simple_agent":
        body = McqaRequest.model_validate({**contract, "response": {"output_text": answer}})
        server = MCQAResourcesServer(SimpleNamespace(grading_mode=None))
        return server._programmatic_score(body).reward
    if agent == "structured_outputs_simple_agent":
        server = StructuredOutputsResourcesServer(SimpleNamespace(xml_coerce_types=True))
        reward, _, _ = server.evaluate_structured_output_response(
            contract["schema_type"], contract["schema_str"], answer
        )
        return reward
    if agent == "math_with_judge_simple_agent":
        return math_reward(answer, contract)
    raise ValueError(f"Unsupported Cascade terminal grader: {agent}")


def main() -> None:
    print(score(Path("/app/answer.txt").read_text(), json.loads(Path("/tests/config.json").read_text())))


if __name__ == "__main__":
    main()
