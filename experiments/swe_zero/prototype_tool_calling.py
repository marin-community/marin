# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Step 1: Quick prototype to test if Gemma 4 E2B/E4B can do multi-turn tool calls.

Usage:
    # Against a local vLLM server (e.g. on a TPU VM):
    python experiments/swe_zero/prototype_tool_calling.py \
        --api_base http://localhost:8000/v1 \
        --model google/gemma-4-E2B-it

    # Against any OpenAI-compatible endpoint:
    python experiments/swe_zero/prototype_tool_calling.py \
        --api_base http://<host>:<port>/v1 \
        --model google/gemma-4-E4B-it
"""

import argparse
import json
import logging
import sys

from openai import OpenAI

from experiments.swe_zero.scaffold import SYSTEM_PROMPT, TOOLS, RepoSnapshot, simulate_tool_response

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# A minimal SWE-bench-style task for testing tool calling
SAMPLE_TASK = """\
## Issue

There is a bug in the `calculate_discount` function in `pricing/discount.py`.
When a discount percentage of 0 is passed, it returns None instead of the original price.

## Relevant Interface

```python
def calculate_discount(price: float, discount_pct: float) -> float:
    \"\"\"Apply a discount percentage to a price and return the discounted price.\"\"\"
```

## Repository

- **Repo**: example/pricing-lib
- **Language**: Python
- **Base commit**: abc123
"""

SAMPLE_FILES = {
    "/pricing/discount.py": (
        """\
def calculate_discount(price: float, discount_pct: float) -> float:
    \"\"\"Apply a discount percentage to a price and return the discounted price.\"\"\"
    if discount_pct == 0:
        return None  # BUG: should return price
    if discount_pct < 0 or discount_pct > 100:
        raise ValueError(f"Invalid discount: {discount_pct}")
    return price * (1 - discount_pct / 100)
"""
    ),
    "/pricing/__init__.py": (
        """\
from pricing.discount import calculate_discount

__all__ = ["calculate_discount"]
"""
    ),
    "/tests/test_discount.py": (
        """\
import pytest
from pricing.discount import calculate_discount


def test_no_discount():
    assert calculate_discount(100.0, 0) == 100.0


def test_full_discount():
    assert calculate_discount(100.0, 100) == 0.0


def test_half_discount():
    assert calculate_discount(100.0, 50) == 50.0


def test_invalid_discount():
    with pytest.raises(ValueError):
        calculate_discount(100.0, 150)
"""
    ),
}


def run_prototype(api_base: str, model: str, api_key: str = "EMPTY") -> bool:
    """Run a quick multi-turn tool-calling test and report results."""
    client = OpenAI(base_url=api_base, api_key=api_key)
    snapshot = RepoSnapshot(files=dict(SAMPLE_FILES), repo_name="example/pricing-lib")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_TASK},
    ]

    logger.info("Testing model: %s at %s", model, api_base)
    logger.info("=" * 60)

    tool_call_count = 0
    finished = False

    for turn in range(15):
        logger.info("--- Turn %d ---", turn + 1)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=0.7,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error("API call failed: %s", e)
            return False

        choice = response.choices[0]
        message = choice.message

        # Log assistant response
        if message.content:
            logger.info("Assistant: %s", message.content[:200])

        # Build assistant message dict
        msg_dict: dict = {"role": "assistant"}
        if message.content:
            msg_dict["content"] = message.content
        if message.tool_calls:
            msg_dict["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(msg_dict)

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_call_count += 1
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                logger.info(
                    "Tool call #%d: %s(%s)",
                    tool_call_count,
                    tc.function.name,
                    json.dumps(args)[:200],
                )

                tool_response = simulate_tool_response(tc.function.name, args, snapshot)
                logger.info("Tool response: %s", tool_response[:200])

                messages.append(
                    {
                        "role": "tool",
                        "content": tool_response,
                        "tool_call_id": tc.id,
                    }
                )

                if tc.function.name == "finish":
                    finished = True
                    break

            if finished:
                break
        else:
            if choice.finish_reason == "stop":
                logger.info("Model stopped without tool calls.")
                break

    logger.info("=" * 60)
    logger.info("RESULTS:")
    logger.info("  Model: %s", model)
    logger.info("  Total turns: %d", turn + 1)
    logger.info("  Tool calls made: %d", tool_call_count)
    logger.info("  Finished cleanly: %s", finished)
    logger.info("  Token usage: %s", response.usage if response.usage else "N/A")

    if tool_call_count > 0 and finished:
        logger.info("SUCCESS: Model can issue multi-turn tool calls and finish.")
        return True
    elif tool_call_count > 0:
        logger.info("PARTIAL: Model issued tool calls but didn't finish cleanly.")
        return True
    else:
        logger.info("FAILURE: Model did not issue any tool calls.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test Gemma 4 multi-turn tool calling")
    parser.add_argument("--api_base", required=True, help="vLLM API base URL")
    parser.add_argument("--model", default="google/gemma-4-E2B-it", help="Model name")
    parser.add_argument("--api_key", default="EMPTY", help="API key (default: EMPTY for local)")
    args = parser.parse_args()

    success = run_prototype(args.api_base, args.model, args.api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
