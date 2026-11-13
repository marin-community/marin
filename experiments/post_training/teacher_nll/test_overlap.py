# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.post_training.teacher_nll import overlap


def naive(tokens):
    mod = 1 << 64
    return [
        sum((int(t) + 1) * pow(int(overlap.BASE), j, mod) for j, t in enumerate(tokens[i : i + overlap.GRAM])) % mod
        for i in range(len(tokens) - overlap.GRAM + 1)
    ]


def test_gram_hash_matches_naive_and_is_position_free():
    rng = np.random.default_rng(0)
    tokens = rng.integers(0, 150000, size=180)
    hasher = overlap.GramHasher(400)
    got = hasher(tokens)
    assert [int(h) for h in got] == naive(tokens)
    shifted = hasher(np.concatenate([rng.integers(0, 150000, size=37), tokens]))
    assert np.array_equal(shifted[37:], got)


def test_normalize_strips_render_tails():
    ours = np.array([5, 6, 92, 128009, 198, 128001])
    datakit = np.array([5, 6, 92, 128009, 220, 128001])
    assert overlap.normalize(ours).tolist() == overlap.normalize(datakit).tolist() == [5, 6, 92]


def test_row_index_finds_loss_grams_and_drops_boilerplate():
    rng = np.random.default_rng(1)
    hasher = overlap.GramHasher(1000)
    shared = rng.integers(0, 150000, size=60)

    def row(trial, body):
        ids = np.concatenate([[128000], body, shared, [128009, 128001]])
        weights = np.ones(len(ids), dtype=np.float32)
        weights[0] = 0
        return {"trial": trial, "task": trial.split("__")[0], "input_ids": ids, "loss_weights": weights}

    bodies = [rng.integers(0, 150000, size=120) for _ in range(3)]
    rows = [row(f"t{i}__a", b) for i, b in enumerate(bodies)]
    # A second attempt of task t1 shares its body: attempts of one task keep their grams.
    rows.append(row("t1__b", bodies[1]))
    index = overlap.RowIndex(rows, hasher)
    doc = np.concatenate([rng.integers(0, 150000, size=40), bodies[1][10:80], rng.integers(0, 150000, size=40)])
    found, qidx = index.lookup(hasher(doc))
    assert len(found) == 70 - overlap.GRAM + 1
    counts = index.row_counts(np.unique(qidx))
    assert counts.tolist() == [0, 21, 0, 21]
    assert len(index.lookup(hasher(shared))[0]) == 0
    norm = overlap.normalize(rows[2]["input_ids"])
    assert index.exact[len(norm)][overlap.digest(norm)] == 2
