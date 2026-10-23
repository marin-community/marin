# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Masking canary for the fork's chat_template strategy on real CL12 rows (pod-side).

Runs in the Axolotl task image with the fork's source first on ``PYTHONPATH``. For each
base, it tokenizes rows through the same strategy and settings a training payload uses
and compares the trained label positions with the shared template's ``{% generation %}``
mask (transformers' ``return_assistant_tokens_mask``): every assistant turn, reasoning
through its end-of-turn token, and nothing else. Exits non-zero on any mismatch.

    python experiments/post_training/axolotl_sft/masking_check.py PAYLOAD_JSON_PATH DATA.jsonl.gz [ROWS]
"""

import gzip
import json
import sys

from axolotl.prompt_strategies.chat_template import StrategyLoader
from axolotl.utils.dict import DictDefault
from transformers import AutoTokenizer

IGNORE = -100


def main() -> None:
    payload = json.load(open(sys.argv[1]))
    data_path = sys.argv[2]
    rows = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    config = payload["config"]
    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], revision=config["revision_of_model"])
    tokenizer.add_special_tokens({"eos_token": config["special_tokens"]["eos_token"]})
    cfg = DictDefault(config)
    dataset_config = dict(config["datasets"][0])
    strategy = StrategyLoader()(tokenizer, cfg, dataset_config)

    mismatches = 0
    with gzip.open(data_path, "rt") as handle:
        for index, line in enumerate(handle):
            if index >= rows:
                break
            row = json.loads(line)
            labels = strategy.tokenize_prompt(row)["labels"]
            expected = tokenizer.apply_chat_template(
                row["messages"],
                chat_template=config["chat_template_jinja"],
                tokenize=True,
                return_dict=True,
                return_assistant_tokens_mask=True,
            )
            trained = [label != IGNORE for label in labels]
            if trained != [bool(m) for m in expected["assistant_masks"]]:
                mismatches += 1
                if mismatches <= 3:
                    ids = expected["input_ids"]
                    extra = tokenizer.decode(
                        [i for i, t, m in zip(ids, trained, expected["assistant_masks"]) if t and not m]
                    )
                    missing = tokenizer.decode(
                        [i for i, t, m in zip(ids, trained, expected["assistant_masks"]) if m and not t]
                    )
                    print(
                        f"row {row['row_id']}: trained-not-expected {extra[:300]!r} expected-not-trained {missing[:300]!r}"
                    )
    print(json.dumps({"base": config["base_model"], "rows": rows, "mismatches": mismatches}))
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
