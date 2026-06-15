# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the 4plus-only decon val build's pure derivation logic."""

import numpy as np

from scripts.analysis.build_decon_val_4plus_only import WINDOW, compute_fully_contained


def _ends_from_lengths(lengths: list[int]) -> np.ndarray:
    return np.cumsum(np.asarray(lengths, dtype=np.int64))


def test_fully_contained_single_window_membership():
    # Four docs, each shorter than one window, packed back-to-back so each lands
    # in a distinct window 0,1,2,3. Val windows = {1, 3}. Docs in windows 1 and 3
    # are fully contained; docs in 0 and 2 are not.
    lengths = [WINDOW, WINDOW, WINDOW, WINDOW]
    ends = _ends_from_lengths(lengths)
    val_windows = np.array([1, 3], dtype=np.int64)
    fully = compute_fully_contained(ends, val_windows)
    assert fully.tolist() == [1, 3]


def test_fully_contained_doc_spanning_two_windows():
    # A doc that straddles a window boundary is contained only if BOTH windows
    # it touches are validation windows.
    # doc0: tokens [0, 100)        -> window 0
    # doc1: tokens [100, WINDOW+50) -> windows 0 and 1 (straddles boundary)
    # doc2: rest of window 1        -> window 1
    lengths = [100, WINDOW + 50 - 100, WINDOW - 50]
    ends = _ends_from_lengths(lengths)
    # Only window 1 is val: doc1 (touches 0 and 1) is NOT contained; doc2 is.
    assert compute_fully_contained(ends, np.array([1], dtype=np.int64)).tolist() == [2]
    # Both windows 0 and 1 val: doc1 now fully contained too.
    assert compute_fully_contained(ends, np.array([0, 1], dtype=np.int64)).tolist() == [0, 1, 2]


def test_fully_contained_excludes_partial_window_overlap():
    # A non-val window sandwiched between val windows must not be counted as
    # contained even though its endpoints' neighbors are val.
    lengths = [WINDOW, WINDOW, WINDOW]
    ends = _ends_from_lengths(lengths)
    # windows 0 and 2 val, window 1 not -> only docs in 0 and 2 contained.
    assert compute_fully_contained(ends, np.array([0, 2], dtype=np.int64)).tolist() == [0, 2]
