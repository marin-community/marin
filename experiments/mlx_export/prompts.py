# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the parity prompt fixture for one source revision.

Renders each case with the source repo's tokenizer and chat template and
stores token ids, so the vLLM golden run and the MLX dump score identical
inputs. The long case is unrepeated Wikipedia text (WikiText-2, pinned) cut to
LONG_PROMPT_TOKENS, so positions past the sliding window exercise the window
boundary.

Usage (Mac):
  uv run python prompts.py --repo open-athena/Marin-Snowball-67B-A2B-v0.1 \
      --revision e4aa78c7c9729dea7238c2e7d1ac7fa57492e0a6 --out out/prompts.json
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

LONG_PROMPT_TOKENS = 8192
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
WIKITEXT_FILE = "wikitext-2-raw-v1/test-00000-of-00001.parquet"

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

ENGLISH_PASSAGE = (
    "The harbor town woke slowly under a grey sky. Fishing boats knocked "
    "against the pier while gulls argued over the morning's first catch. "
    "In the bakery on Front Street, the oven had been lit since four, and "
    "the smell of bread drifted as far as the lighthouse. Old Marek, who "
    "had kept the boats for forty years, watched the tide tables and "
    "shake his head at the wind. The children walked to school along the "
    "sea wall, counting the waves that broke over the stones. By noon the "
    "clouds thinned, and light fell across the water in long silver "
    "bands. The town had seen a hundred storms and a hundred springs, "
    "and it kept its own time, measured in tides rather than hours. "
) * 4

CODE_PROMPT = '''Complete this Python function so it returns the n-th Fibonacci number using iteration:

```python
def fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (fibonacci(0) == 0)."""
```

The function should raise ValueError for negative n and run in O(n) time.'''

# Full-width CJK punctuation is intentional.
MULTILINGUAL = (
    "Bonjour! Comment allez-vous aujourd'hui? J'espere que vous passez "
    "une excellente journee. 你好！你今天怎么样？我希望你有一个很棒的一天。 "  # noqa: RUF001
    "This is a mixed French and Chinese passage with a little English at "
    "the end. Quelle heure est-il? 现在几点了？ The answer should describe "  # noqa: RUF001
    "the weather in both French and Chinese."
)


def long_text() -> str:
    path = hf_hub_download(WIKITEXT_REPO, WIKITEXT_FILE, repo_type="dataset", revision=WIKITEXT_REVISION)
    return "".join(pq.read_table(path, columns=["text"]).column("text").to_pylist())


def build_cases(tokenizer) -> list[dict]:
    cases: list[dict] = []

    def add(case_id: str, ids: list[int], **meta):
        cases.append({"id": case_id, "prompt_token_ids": ids, **meta})
        print(f"{case_id}: {len(ids)} tokens")

    def render(conversation: list[dict], **kwargs) -> list[int]:
        text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False, **kwargs)
        return tokenizer.encode(text, add_special_tokens=False)

    def raw(text: str) -> list[int]:
        return [tokenizer.bos_token_id, *tokenizer.encode(text, add_special_tokens=False)]

    add("plain_short", raw(ENGLISH_PASSAGE), kind="raw")

    math_question = [
        {
            "role": "user",
            "content": (
                "A train leaves Ashford at 9:15 traveling at 62 km/h. "
                "A second train leaves Ashford at 10:05 on the same track "
                "traveling at 84 km/h. At what time does the second train "
                "catch the first? Show your reasoning."
            ),
        }
    ]
    add("chat_think", render(math_question, enable_thinking=True), kind="chat", enable_thinking=True)
    add("chat_nothink", render(math_question, enable_thinking=False), kind="chat", enable_thinking=False)

    weather_question = [{"role": "user", "content": "What's the weather in Paris right now?"}]
    add(
        "chat_tools",
        render(weather_question, tools=[WEATHER_TOOL], enable_thinking=False),
        kind="chat",
        enable_thinking=False,
        tools=True,
    )
    round_trip = [
        *weather_question,
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": {"city": "Paris", "unit": "celsius"}},
                }
            ],
        },
        {"role": "tool", "name": "get_weather", "content": "18C, light rain, wind 12 km/h"},
    ]
    add(
        "chat_tool_result",
        render(round_trip, tools=[WEATHER_TOOL], enable_thinking=False),
        kind="chat",
        enable_thinking=False,
        tools=True,
    )

    add("code", raw(CODE_PROMPT), kind="raw")
    add("multilingual", raw(MULTILINGUAL), kind="raw")

    long_ids = raw(long_text())
    assert len(long_ids) >= LONG_PROMPT_TOKENS, f"WikiText gave only {len(long_ids)} tokens"
    add("long_wikitext", long_ids[:LONG_PROMPT_TOKENS], kind="raw")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.repo, revision=args.revision)
    cases = build_cases(tokenizer)
    fixture = {"source_repo": args.repo, "source_revision": args.revision, "cases": cases}
    args.out.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {args.out} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
