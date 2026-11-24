# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ScienceAgentBench visual-judge requests and official response scoring.

The caller supplies the prompt from the pinned upstream judge source. This
module runs only in the grading environment, after candidate replay closes.
"""

import argparse
import base64
import json
import os
import re
import runpy
import sys
import tomllib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType

from openai import OpenAI
from openai.types.chat import ChatCompletion
from pydantic import ValidationError

MODEL = "MiniMaxAI/MiniMax-M3-MXFP8"
VENICE_MODEL = "qwen/qwen3-vl-235b-a22b-instruct"
COMPLETIONS = 3
REQUEST_TIMEOUT = 120.0
MAX_OUTPUT_TOKENS = 1000
SOURCE_JUDGE_MODULE = "gpt4_visual_judge"
SCORE_PATTERN = re.compile(r"\[FINAL SCORE\]: (\d{1,3})", re.DOTALL)


def score_figure(
    client: OpenAI,
    prompt: str,
    predicted_image: str,
    gold_image: str,
    *,
    model: str = MODEL,
    max_output_tokens: int = MAX_OUTPUT_TOKENS,
    record_response: Callable[[ChatCompletion], None] | None = None,
) -> tuple[list[str], float]:
    """Compare base64 PNGs, scoring text without a matching score marker as zero.

    Transport errors and absent completions propagate as infrastructure errors.
    The official parser uses the first matching marker without clamping its
    value; preserve that behavior, including values outside the prompted range
    and the three-digit prefix of a longer number.
    All three separate single-completion requests must succeed before returning
    an average. Retries belong to the caller.
    """
    openrouter = str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"
    if model == MODEL:
        request_model = "minimax/minimax-m3" if openrouter else model
        route, prompt_price, completion_price = "deepinfra/fp8", 0.28, 1.1
    elif model == VENICE_MODEL and openrouter:
        request_model = model
        route, prompt_price, completion_price = "venice/fp8", 0.21, 1.9
    else:
        raise RuntimeError(f"Unsupported ScienceAgentBench visual judge for this endpoint: {model}")
    provider = {
        "only": [route],
        "allow_fallbacks": False,
        "data_collection": "deny",
        "max_price": {"prompt": prompt_price, "completion": completion_price},
    }
    texts = []
    scores = []
    for _ in range(COMPLETIONS):
        response = client.with_options(timeout=REQUEST_TIMEOUT, max_retries=0).chat.completions.create(
            model=request_model,
            extra_body={"provider": provider} if openrouter else None,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{predicted_image}"}},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{gold_image}"}},
                    ],
                }
            ],
            temperature=0.2,
            max_tokens=max_output_tokens,
            n=1,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
        )
        # The SDK constructs responses without validating the provider's JSON schema.
        # Keep protocol errors distinct from candidate-output ValueError failures.
        try:
            response = ChatCompletion.model_validate(response.model_dump(), strict=True)
        except ValidationError as error:
            raise RuntimeError("Visual judge returned an invalid completion response") from error
        if record_response is not None:
            record_response(response)
        if len(response.choices) != 1 or response.choices[0].index != 0:
            raise RuntimeError("Visual judge did not return exactly one completion")
        text = response.choices[0].message.content
        if text is None:
            raise RuntimeError("Visual judge returned a completion without text")
        texts.append(text)
        match = SCORE_PATTERN.search(text)
        scores.append(int(match.group(1)) if match else 0)
    return texts, sum(scores) / COMPLETIONS


def encode_image(path: str) -> str:
    """Encode a source evaluator's PNG without decoding candidate image content."""
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def record_completion(receipts: list[dict], response: ChatCompletion) -> None:
    """Preserve each response in grader stdout even if a later call fails."""
    receipts.append(response.model_dump(mode="json", include={"id", "model", "provider", "usage"}))
    print(json.dumps({"judge_response": response.model_dump(mode="json")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", type=Path)
    parser.add_argument("judge_spec", type=Path)
    parser.add_argument("wrapper", type=Path)
    parser.add_argument("evaluator", type=Path)
    parser.add_argument("primary_output", type=Path)
    parser.add_argument("candidate_directory", type=Path)
    parser.add_argument("verdict", type=Path)
    args = parser.parse_args()
    settings = tomllib.loads(args.judge_spec.read_text())
    receipts = []
    with OpenAI(
        base_url=os.environ["VERIFYIT_JUDGE_BASE_URL"],
        api_key=os.environ["VERIFYIT_JUDGE_API_KEY"],
        max_retries=0,
    ) as client:
        # Preserve the pinned evaluator's figure selection and success threshold.
        # Only its judge transport is replaced; this module never runs in replay.
        judge = ModuleType(SOURCE_JUDGE_MODULE)
        judge.__dict__.update(
            encode_image=encode_image,
            score_figure=partial(
                score_figure,
                client,
                args.prompt.read_text(),
                model=os.environ["VERIFYIT_JUDGE_MODEL"],
                max_output_tokens=settings["max_completion_tokens"],
                record_response=partial(record_completion, receipts),
            ),
        )
        sys.modules[judge.__name__] = judge
        sys.argv = [
            str(path)
            for path in (args.wrapper, args.evaluator, args.primary_output, args.candidate_directory, args.verdict)
        ]
        runpy.run_path(str(args.wrapper), run_name="__main__")
        verdict = json.loads(args.verdict.read_text())
        verdict["detail"]["judge_receipts"] = receipts
        args.verdict.write_text(json.dumps(verdict, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
