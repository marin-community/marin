# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ScienceAgentBench visual-judge requests and official response scoring.

The caller supplies the prompt from the pinned upstream judge source. This
module runs only in the grading environment, after candidate replay closes.
"""

import re

from openai import OpenAI

MODEL = "gpt-4o-2024-05-13"
COMPLETIONS = 3
REQUEST_TIMEOUT = 120.0
SCORE_PATTERN = re.compile(r"\[FINAL SCORE\]: (\d{1,3})", re.DOTALL)


def score_figure(client: OpenAI, prompt: str, predicted_image: str, gold_image: str) -> tuple[list[str], float]:
    """Compare base64 PNGs, scoring text without a matching score marker as zero.

    Transport errors and absent completions propagate as infrastructure errors.
    The official parser uses the first matching marker without clamping its
    value; preserve that behavior, including values outside the prompted range
    and the three-digit prefix of a longer number.
    Retries belong to the caller so every attempt can be recorded.
    """
    response = client.with_options(timeout=REQUEST_TIMEOUT, max_retries=0).chat.completions.create(
        model=MODEL,
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
        max_tokens=1000,
        n=COMPLETIONS,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
    )
    if len(response.choices) != COMPLETIONS or {choice.index for choice in response.choices} != set(range(COMPLETIONS)):
        raise RuntimeError("Visual judge did not return all three distinct completions")
    texts = []
    scores = []
    for choice in response.choices:
        text = choice.message.content
        if text is None:
            raise RuntimeError("Visual judge returned a completion without text")
        texts.append(text)
        match = SCORE_PATTERN.search(text)
        scores.append(int(match.group(1)) if match else 0)
    return texts, sum(scores) / COMPLETIONS
