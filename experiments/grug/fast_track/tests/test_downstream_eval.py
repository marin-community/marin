# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import math

import pyarrow as pa
import pytest

from experiments.grug.fast_track.downstream_eval import collapse_whitespace, downstream_metrics, encode_request


class _CharTokenizer:
    """One token per character, with a merge of ``" a"`` into one token, so boundaries can shift."""

    bos_token_id = 0
    eos_token_id = 1

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids, index = [], 0
        while index < len(text):
            if text.startswith(" a", index):
                ids.append(2)
                index += 2
            else:
                ids.append(10 + ord(text[index]))
                index += 1
        return ids


def test_encode_request_scores_the_continuation_tokens_after_bos():
    request = encode_request(_CharTokenizer(), "Q:", " ab")

    assert request.tokens[0] == 0
    # The context is "Q:" (2 tokens after BOS); " a" merges, so the continuation is 2 tokens.
    assert request.continuation_start == 3
    assert len(request.tokens) - request.continuation_start == 2
    assert request.continuation_bytes == 3


def test_encode_request_splits_when_the_boundary_merges():
    # "x " + "a" tokenizes "x a" with a merged " a", so the context prefix does not match.
    request = encode_request(_CharTokenizer(), "x ", "a")

    assert request.tokens[1 : request.continuation_start] == tuple(_CharTokenizer().encode("x "))
    assert request.tokens[request.continuation_start :] == tuple(_CharTokenizer().encode("a"))


def _records(rows):
    return pa.Table.from_pylist(
        [
            {
                "task": task,
                "example_id": example,
                "choice": choice,
                "label": label,
                "log_likelihood": ll,
                "greedy": greedy,
                "tokens": 1,
                "bytes": nbytes,
                "chars": nbytes,
            }
            for task, example, choice, label, ll, greedy, nbytes in rows
        ]
    )


def test_downstream_metrics_accuracy_normalization_and_bpb():
    records = _records(
        [
            # Raw log-likelihood picks the short wrong choice; per-character normalization picks the long right one.
            ("arc_easy", "a", 0, 1, -2.0, False, 2),
            ("arc_easy", "a", 1, 1, -3.0, False, 12),
            ("arc_easy", "b", 0, 0, -1.0, True, 4),
            ("arc_easy", "b", 1, 0, -5.0, False, 4),
            ("lambada", "c", 0, 0, -4.0, True, 8),
            ("lambada", "d", 0, 0, -4.0, False, 8),
        ]
    )

    metrics = downstream_metrics(records)

    assert metrics["arc_easy"]["acc"] == pytest.approx(0.5)
    assert metrics["arc_easy"]["acc_norm"] == pytest.approx(1.0)
    expected_bpb = (3.0 / (12 * math.log(2)) + 1.0 / (4 * math.log(2))) / 2
    assert metrics["arc_easy"]["correct_bpb"] == pytest.approx(expected_bpb)
    # A single-choice task has no choice metrics; its accuracy is greedy recovery.
    assert metrics["lambada"]["acc"] == pytest.approx(0.5)
    assert "acc_norm" not in metrics["lambada"]


def test_collapse_whitespace_keeps_words_and_paragraphs():
    text = "a  b \t c  \n\n\n\n  d\n e  "

    assert collapse_whitespace(text) == "a b c\n\nd\ne"
