# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the token-level inputs for the Hero vLLM hill climb.

Writes one JSON file of token-id cases that every arm reads unchanged:

* ``gate``: short histories (chat math prompts and book excerpts) scored with prompt log-probs,
  replayed for greedy decode, and used for batch-invariance probes.
* ``long``: book text truncated to the long-context gate lengths.
* ``load``: chat-templated math problems for the RL-rollout-like throughput workload.

Run on a CPU host with Hugging Face access; upload the output to the TTL prefix with rclone.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from marin.datakit.chat_template import MARIN_CHAT_TEMPLATE
from transformers import AutoTokenizer

TOKENIZER = "marin-community/marin-tokenizer-tools"
TOKENIZER_REVISION = "3f6879acdaf3dae7874b2d48c7d11b9d8c6faf09"
SEED = 20261113
LONG_LENGTHS = (4096, 32768, 65536, 262144)
GATE_CHAT = 24
GATE_BOOK = 8
LOAD_PROMPTS = 1024


def _sha(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]


def _parquet(repo: str, filename: str, revision: str | None) -> list[dict]:
    path = hf_hub_download(repo, filename, repo_type="dataset", revision=revision)
    return pq.read_table(path).to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, revision=TOKENIZER_REVISION)
    rng = random.Random(SEED)

    gsm = _parquet("openai/gsm8k", "main/test-00000-of-00001.parquet", None)
    books = _parquet("emozilla/pg19", "data/test-00000-of-00001-29a571947c0b5ccc.parquet", None)

    def chat(question: str) -> list[int]:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            chat_template=MARIN_CHAT_TEMPLATE,
            tokenize=False,
            add_generation_prompt=True,
        )
        return tokenizer.encode(text, add_special_tokens=False)

    order = list(range(len(gsm)))
    rng.shuffle(order)
    gate_chat = [chat(gsm[i]["question"]) for i in order[:GATE_CHAT]]
    load = [chat(gsm[i]["question"]) for i in order[GATE_CHAT : GATE_CHAT + LOAD_PROMPTS]]

    book_tokens = []
    for book in sorted(books, key=lambda row: -len(row["text"])):
        book_tokens.append([tokenizer.bos_token_id, *tokenizer.encode(book["text"], add_special_tokens=False)])
        if len(book_tokens) >= GATE_BOOK + 1 and len(book_tokens[0]) > max(LONG_LENGTHS):
            break
    longest = book_tokens[0]
    assert len(longest) >= max(LONG_LENGTHS), len(longest)
    gate_book = []
    for tokens in book_tokens[1 : GATE_BOOK + 1]:
        length = rng.randint(512, 2048)
        start = rng.randint(1, len(tokens) - length)
        gate_book.append([tokenizer.bos_token_id, *tokens[start : start + length]])

    cases = {
        "tokenizer": {"name": TOKENIZER, "revision": TOKENIZER_REVISION},
        "sources": {"gsm8k": "openai/gsm8k main/test", "pg19": "emozilla/pg19 data/test"},
        "seed": SEED,
        "gate": (
            [{"case_id": f"chat{i:02d}", "tokens": tokens, "sha": _sha(tokens)} for i, tokens in enumerate(gate_chat)]
            + [{"case_id": f"book{i:02d}", "tokens": tokens, "sha": _sha(tokens)} for i, tokens in enumerate(gate_book)]
        ),
        "long": [{"case_id": f"long{n}", "tokens": longest[:n], "sha": _sha(longest[:n])} for n in LONG_LENGTHS],
        "load": [{"case_id": f"load{i:04d}", "tokens": tokens} for i, tokens in enumerate(load)],
    }
    args.output.write_text(json.dumps(cases))
    lengths = [len(c["tokens"]) for c in cases["load"]]
    print(
        json.dumps(
            {
                "gate": len(cases["gate"]),
                "gate_lengths": [len(c["tokens"]) for c in cases["gate"]],
                "long": [len(c["tokens"]) for c in cases["long"]],
                "load": len(lengths),
                "load_mean_prompt": sum(lengths) / len(lengths),
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
