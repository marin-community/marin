# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Step 1: Quick prototype to test if a code model can produce mini-swe-agent v1
style bash commands on a sample SWE-bench-style task.

Default model is `ricdomolm/mini-coder-1.7b` (Qwen3-1.7B fine-tuned on 400k
mini-swe-agent trajectories).

Usage:
    python experiments/swe_zero/prototype_tool_calling.py \
        --api_base http://localhost:8000/v1 \
        --model ricdomolm/mini-coder-1.7b
"""

import argparse
import logging
import sys

from openai import OpenAI

from experiments.swe_zero.scaffold import (
    SYSTEM_PROMPT,
    RepoSnapshot,
    extract_bash_command,
    simulate_bash,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_TASK = """\
Please solve this issue in the repository example/pricing-lib.

## Issue

There is a bug in the `calculate_discount` function in `pricing/discount.py`.
When a discount percentage of 0 is passed, it returns None instead of the original price.

## Repository Info

- **Language**: Python
- **Base commit**: abc123
- The repository is already cloned at `/repo`.
"""

SAMPLE_FILES = {
    "/repo/pricing/discount.py": (
        "def calculate_discount(price: float, discount_pct: float) -> float:\n"
        '    """Apply a discount percentage to a price and return the discounted price."""\n'
        "    if discount_pct == 0:\n"
        "        return None  # BUG: should return price\n"
        "    if discount_pct < 0 or discount_pct > 100:\n"
        '        raise ValueError(f"Invalid discount: {discount_pct}")\n'
        "    return price * (1 - discount_pct / 100)\n"
    ),
    "/repo/pricing/__init__.py": 'from pricing.discount import calculate_discount\n\n__all__ = ["calculate_discount"]\n',
    "/repo/tests/test_discount.py": (
        "import pytest\n"
        "from pricing.discount import calculate_discount\n\n\n"
        "def test_no_discount():\n"
        "    assert calculate_discount(100.0, 0) == 100.0\n\n\n"
        "def test_full_discount():\n"
        "    assert calculate_discount(100.0, 100) == 0.0\n\n\n"
        "def test_half_discount():\n"
        "    assert calculate_discount(100.0, 50) == 50.0\n\n\n"
        "def test_invalid_discount():\n"
        "    with pytest.raises(ValueError):\n"
        "        calculate_discount(100.0, 150)\n"
    ),
}


def run_prototype(api_base: str, model: str, api_key: str = "EMPTY") -> bool:
    """Run a multi-turn bash-based test in mini-swe-agent v2 style."""
    client = OpenAI(base_url=api_base, api_key=api_key)
    snapshot = RepoSnapshot(files=dict(SAMPLE_FILES), repo_name="example/pricing-lib")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": SAMPLE_TASK},
    ]

    logger.info("Testing model: %s at %s", model, api_base)
    logger.info("=" * 60)

    bash_count = 0
    finished = False

    for turn in range(15):
        logger.info("--- Turn %d ---", turn + 1)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=2048,
            )
        except Exception as e:
            logger.error("API call failed: %s", e)
            return False

        assistant_text = response.choices[0].message.content or ""
        logger.info("Assistant: %s", assistant_text[:300])

        messages.append({"role": "assistant", "content": assistant_text})

        bash_cmd = extract_bash_command(assistant_text)
        if bash_cmd:
            bash_count += 1
            logger.info("Bash #%d: %s", bash_count, bash_cmd[:200])

            if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in bash_cmd:
                finished = True
                break

            observation = simulate_bash(bash_cmd, snapshot)
            obs_text = f"OBSERVATION:\n{observation}" if observation else "OBSERVATION:\n(no output)"
            logger.info("Observation: %s", obs_text[:200])
            messages.append({"role": "user", "content": obs_text})
        else:
            if response.choices[0].finish_reason == "stop":
                break

    logger.info("=" * 60)
    logger.info("RESULTS:")
    logger.info("  Model: %s", model)
    logger.info("  Total turns: %d", turn + 1)
    logger.info("  Bash commands: %d", bash_count)
    logger.info("  Finished cleanly: %s", finished)

    if bash_count > 0:
        logger.info("SUCCESS: Model produces bash commands in mini-swe-agent v1 format.")
        return True
    else:
        logger.info("FAILURE: Model did not produce any bash commands.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test mini-coder model with mini-swe-agent v1 style")
    parser.add_argument("--api_base", required=True, help="API base URL")
    parser.add_argument("--model", default="ricdomolm/mini-coder-1.7b", help="Model name")
    parser.add_argument("--api_key", default="EMPTY", help="API key")
    args = parser.parse_args()

    success = run_prototype(args.api_base, args.model, args.api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
