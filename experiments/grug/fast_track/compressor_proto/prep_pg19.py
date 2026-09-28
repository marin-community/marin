# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tokenize the locally cached PG-19 test split (100 books) for proto.py: a 4096-token byte-level BPE trained on
the train books, then book-disjoint eval (10), extra (30) and train (60) streams as int16 .npy files in the cwd."""

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

PG19_TEST = (
    "/Users/larry/.cache/huggingface/hub/datasets--emozilla--pg19-test/snapshots/"
    "c5e39bf32e33f9111323aa68d7d9000d22722035/data/test-00000-of-00001-29a571947c0b5ccc.parquet"
)
VOCAB = 4096
EOS_ID = 0


def main():
    texts = pq.read_table(PG19_TEST).column("text").to_pylist()
    order = np.random.default_rng(0).permutation(len(texts))
    splits = {"eval": order[:10], "extra": order[10:40], "train": order[40:]}
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(
        [texts[i] for i in splits["train"]], trainers.BpeTrainer(vocab_size=VOCAB, special_tokens=["<eos>"])
    )
    for name, books in splits.items():
        ids = []
        for i in books:
            ids.extend(tok.encode(texts[i]).ids)
            ids.append(EOS_ID)
        stream = np.array(ids, np.int16)
        np.save(f"pg19_{name}.npy", stream)
        print(name, len(books), stream.shape)


if __name__ == "__main__":
    main()
