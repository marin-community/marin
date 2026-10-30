# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Masking canary for the configured Axolotl strategy on real CL12 rows (pod-side).

Runs in the Axolotl task image with the fork's source first on ``PYTHONPATH``. For each
base, it tokenizes rows through the same strategy and settings a training payload uses
and compares the trained label positions with the configured template's ``{% generation %}``
mask (transformers' ``return_assistant_tokens_mask``): every assistant turn, reasoning
through its end-of-turn token, and nothing else.

That comparison cannot catch a template mask that itself leaves a token out, so the
canary also counts directly, from rendered token ids: each assistant header's first
turn-close token must be a trained label, and so must every ``</think>`` inside an
assistant turn. Exits non-zero on any mismatch or unlabeled token.

    python experiments/post_training/axolotl_sft/masking_check.py PAYLOAD_JSON_PATH DATA.jsonl.gz [ROWS]
"""

import gzip
import json
import sys

from axolotl.prompt_strategies import load as load_strategy
from axolotl.utils.dict import DictDefault
from transformers import AutoTokenizer

IGNORE = -100
THINK_CLOSE = "</think>"


def unlabeled_turn_tokens(input_ids: list[int], labels: list[int], header: list[int], close: int, think: int) -> dict:
    """Assistant turns and in-turn ``</think>`` tokens, and how many of each are not trained labels."""
    counts = {
        "assistant_turns": 0,
        "unclosed_turns": 0,
        "unlabeled_turn_close": 0,
        "think_close": 0,
        "unlabeled_think_close": 0,
    }
    for start in range(len(input_ids) - len(header) + 1):
        if input_ids[start : start + len(header)] != header:
            continue
        counts["assistant_turns"] += 1
        body = start + len(header)
        if close not in input_ids[body:]:
            counts["unclosed_turns"] += 1
            continue
        end = input_ids.index(close, body)
        counts["unlabeled_turn_close"] += labels[end] != close
        for position in range(body, end):
            if input_ids[position] == think:
                counts["think_close"] += 1
                counts["unlabeled_think_close"] += labels[position] != think
    return counts


def main() -> None:
    with open(sys.argv[1]) as handle:
        payload = json.load(handle)
    data_path = sys.argv[2]
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    config = payload["config"]
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision_of_model"])
    tokenizer.add_special_tokens({"eos_token": config["special_tokens"]["eos_token"]})
    cfg = DictDefault(config)
    dataset_config = dict(config["datasets"][0])
    strategy = load_strategy(dataset_config["type"], tokenizer, cfg, dataset_config)

    header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
    close = tokenizer.convert_tokens_to_ids(config["special_tokens"]["eos_token"])
    think = tokenizer.convert_tokens_to_ids(THINK_CLOSE)
    totals: dict[str, int] = {}
    mismatches = 0
    with gzip.open(data_path, "rt") as handle:
        for index, line in enumerate(handle):
            if index >= rows:
                break
            row = json.loads(line)
            actual = strategy.tokenize_prompt(row)
            labels = actual["labels"]
            expected = tokenizer.apply_chat_template(
                row["messages"],
                chat_template=config["chat_template_jinja"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            trained = [label != IGNORE for label in labels]
            counts = unlabeled_turn_tokens(actual["input_ids"], labels, header, close, think)
            # The header scan must find every assistant message, or it would undercount silently.
            counts["header_miscounts"] = counts["assistant_turns"] != sum(
                message["role"] == "assistant" for message in row["messages"]
            )
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
            if actual["input_ids"] != expected["input_ids"] or trained != [bool(m) for m in expected["assistant_masks"]]:
                mismatches += 1
                if mismatches <= 3:
                    ids = expected["input_ids"]
                    extra = tokenizer.decode(
                        [i for i, t, m in zip(ids, trained, expected["assistant_masks"], strict=False) if t and not m]
                    )
                    missing = tokenizer.decode(
                        [i for i, t, m in zip(ids, trained, expected["assistant_masks"], strict=False) if m and not t]
                    )
                    print(f"row {row['row_id']}: trained-not-expected {extra[:300]!r}")
                    print(f"row {row['row_id']}: expected-not-trained {missing[:300]!r}")
    print(json.dumps({"base": config["base_model"], "rows": rows, "mismatches": mismatches, "labels": totals}))
    unlabeled = sum(
        totals[key] for key in ("header_miscounts", "unclosed_turns", "unlabeled_turn_close", "unlabeled_think_close")
    )
    sys.exit(1 if mismatches or unlabeled else 0)


if __name__ == "__main__":
    main()
