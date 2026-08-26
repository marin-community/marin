# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import datasets
import pytest

from experiments.evaluation.tasks.mrcr.utils import (
    mrcr_bin,
    process_docs,
    process_results,
    process_smoke_docs,
    render_prompt,
    score_response,
)


def _row(needles: int, repeated_words: int, suffix: str) -> dict:
    nonce = f"nonce{needles}{suffix}"
    messages = [
        {"role": "user", "content": "word " * repeated_words},
        {"role": "assistant", "content": "distractor"},
        {"role": "user", "content": f"Prepend {nonce} to the requested answer."},
    ]
    return {
        "prompt": json.dumps(messages),
        "answer": f"{nonce}target response",
        "random_string_to_prepend": nonce,
        "n_needles": needles,
    }


def test_render_prompt_preserves_roles_and_leaves_nonce_for_generation():
    row = _row(2, 5_000, "a")

    prompt = render_prompt(row)

    assert prompt.startswith("User: word word ")
    assert "\nAssistant: distractor\nUser: Prepend nonce2a" in prompt
    assert prompt.endswith("\nAssistant: ")
    assert not prompt.endswith(row["random_string_to_prepend"])


@pytest.mark.parametrize(
    ("response", "expected_score", "expected_prefix_hit"),
    [
        ("nonce2atarget response", 1.0, 1.0),
        ("target response", 0.0, 0.0),
        ("nonce2adifferent", 1 / 3, 1.0),
    ],
)
def test_score_response_requires_nonce_then_scores_response_body(response, expected_score, expected_prefix_hit):
    score, prefix_hit = score_response(response, "nonce2atarget response", "nonce2a")

    assert score == expected_score
    assert prefix_hit == expected_prefix_hit


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        (4_095, None),
        (4_096, 8_192),
        (8_192, 8_192),
        (8_193, 16_384),
        (262_144, 262_144),
        (262_145, None),
    ],
)
def test_mrcr_bin_matches_official_boundaries(tokens, expected):
    assert mrcr_bin(tokens) == expected


def test_smoke_selects_one_example_per_cell_and_full_order_round_robins_cells():
    rows = [
        _row(needles, repeated_words, suffix)
        for suffix in ("a", "b")
        for repeated_words in (5_000, 10_000)
        for needles in (2, 4, 8)
    ]
    source = datasets.Dataset.from_list(rows)

    smoke = process_smoke_docs(source)
    full = process_docs(source)

    smoke_cells = list(zip(smoke["mrcr_bin_upper"], smoke["n_needles"], strict=True))
    first_full_cells = list(zip(full[:6]["mrcr_bin_upper"], full[:6]["n_needles"], strict=True))
    assert len(smoke) == 6
    assert len(set(smoke_cells)) == 6
    assert first_full_cells == smoke_cells
    assert len(full) == 12


def test_process_results_emits_aggregate_prefix_and_matching_cell_metric():
    row = _row(8, 5_000, "a")
    row["mrcr_bin_upper"] = 8_192

    result = process_results(row, [row["answer"]])

    assert result == {
        "mrcr_accuracy": 1.0,
        "prefix_hit_rate": 1.0,
        "mrcr_8192_8needle": 1.0,
    }
