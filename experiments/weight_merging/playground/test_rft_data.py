# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RFT split invariants: rejected answers stay out and questions never cross."""

from experiments.weight_merging.playground.rft_data import split_responses


class Tokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, **kwargs):
        return [2, 3]

    def encode(self, text, **kwargs):
        return [ord(char) for char in text]


def test_rft_selection_splits_questions_and_uses_only_verified_answers():
    items, responses = [], []
    for dataset in ("gsm8k", "mbpp"):
        for index in range(120):
            item_id = f"{dataset}/train/{index}"
            items.append({"id": item_id, "dataset": dataset, "prompt": "question", "tests": []})
            for sample in (0, 1):
                responses.append(
                    {
                        "item_id": item_id,
                        "sample_index": sample,
                        "completion": "accepted" if sample == 0 else "rejected",
                        "accepted": sample == 0,
                    }
                )
    data, records, stats = split_responses(items, responses, Tokenizer(), train_count=8, validation_count=8)
    for domain in ("math", "code"):
        assert len(data[domain]["train"]) == len(data[domain]["validation"]) == 8
        train_ids = {row["item_id"] for row in records[domain]["train"]}
        val_ids = {row["item_id"] for row in records[domain]["validation"]}
        assert train_ids.isdisjoint(val_ids)
        assert all(row["sample_index"] == 0 for split in records[domain].values() for row in split)
        assert stats[domain]["sampled"] == 240
        assert stats[domain]["verified"] == 120
