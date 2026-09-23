# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import sys
from types import SimpleNamespace

import numpy as np

from experiments.datakit.cluster.domain.v0 import train


def test_train_centroids_without_coarse_views_skips_linkage(monkeypatch):
    embeddings = np.ones((1, train.LUXICAL_DIM), dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    saved: list[str] = []

    class FakeKmeans:
        def __init__(self, **kwargs):
            self.centroids = embeddings
            self.obj = [1.0]

        def train(self, values):
            assert values is embeddings

    monkeypatch.setattr(train, "_load_sample_parquet", lambda _: embeddings)
    monkeypatch.setattr(train, "_save_npy", lambda _array, _path, name: saved.append(name))
    monkeypatch.setattr(train, "open_url", lambda *_args: contextlib.nullcontext(io.StringIO()))
    monkeypatch.setitem(
        sys.modules,
        "faiss",
        SimpleNamespace(Kmeans=FakeKmeans, omp_set_num_threads=lambda _threads: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "threadpoolctl",
        SimpleNamespace(threadpool_limits=lambda **_kwargs: contextlib.nullcontext()),
    )
    monkeypatch.setattr(
        "scipy.cluster.hierarchy.linkage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("linkage must not run")),
    )

    train.train_centroids(
        output_path="unused",
        sample_path="unused",
        k_train=1,
        k_views=(),
        n_threads=1,
    )

    assert saved == ["centroids_1.npy"]
