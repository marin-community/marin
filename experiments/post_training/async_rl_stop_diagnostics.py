# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Nonexclusive diagnostics of retained finalized evaluation responses.

These measurements describe parser location and repetition. They do not infer
raw generation tokens or attribute truncation to a cause.
"""

import re
from collections.abc import Sequence

ANSWER = re.compile(r"#### (\-?[0-9\.\,]+)")
REPETITION_THRESHOLDS = (0.25, 0.5, 0.75)


def repeated_window_fraction(tokens: Sequence[int], width: int = 16) -> float:
    """Duplicate window occurrences divided by all consecutive windows."""
    if width <= 0:
        raise ValueError("Window width must be positive")
    windows = [tuple(tokens[i : i + width]) for i in range(max(0, len(tokens) - width + 1))]
    return 1 - len(set(windows)) / len(windows) if windows else 0.0


def parser_inside_thinking(prompt_ids, response_ids, output_response, decoder, start_id, end_id):
    """Locate the native first answer marker using token-defined thinking state.

    Decoding is checked against the retained response. A missing closing token
    means thinking remains open only if the actual prompt or response opened it.
    """
    if decoder.decode(response_ids, skip_special_tokens=False) != output_response:
        raise ValueError("Pinned decoder differs from retained response")
    match = ANSWER.search(output_response)
    if match is None:
        return None
    thinking = False
    for token in prompt_ids:
        if token == start_id:
            thinking = True
        elif token == end_id:
            thinking = False
    for index, token in enumerate(response_ids):
        if token not in (start_id, end_id):
            continue
        prefix = decoder.decode(response_ids[: index + 1], skip_special_tokens=False)
        if not output_response.startswith(prefix):
            raise ValueError("Special-token prefix decode is not stable")
        if len(prefix) > match.start():
            break
        thinking = token == start_id
    return thinking


def fraction(numerator: int, denominator: int) -> dict:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "fraction": numerator / denominator if denominator else None,
    }


def summarize(rows: list[dict]) -> dict:
    """Keep overlapping outcomes and each denominator visible."""
    groups = {
        "all": rows,
        "length": [r for r in rows if r["stop_reason"] == "length"],
        "rewarded": [r for r in rows if r["reward"] == 1],
        "rewarded_length": [r for r in rows if r["reward"] == 1 and r["stop_reason"] == "length"],
    }
    output = {}
    for name, selected in groups.items():
        markers = [r for r in selected if r["parser_inside_thinking"] is not None]
        eos_known = [r for r in selected if r["no_effective_eos"] is not None]
        output[name] = {
            "rows": len(selected),
            "parser_inside_thinking_among_rows": fraction(
                sum(r["parser_inside_thinking"] is True for r in selected), len(selected)
            ),
            "parser_inside_thinking_among_marker_rows": fraction(
                sum(r["parser_inside_thinking"] is True for r in markers), len(markers)
            ),
            "no_effective_eos": fraction(sum(r["no_effective_eos"] for r in eos_known), len(eos_known)),
            "effective_eos_unknown_rows": len(selected) - len(eos_known),
            "repetition_duplicate_window_fraction_ge": {
                str(threshold): fraction(
                    sum(r["repeated_16gram_fraction"] >= threshold for r in selected), len(selected)
                )
                for threshold in REPETITION_THRESHOLDS
            },
            "parser_and_repetition_ge_half": fraction(
                sum(r["parser_inside_thinking"] is True and r["repeated_16gram_fraction"] >= 0.5 for r in selected),
                len(selected),
            ),
        }
    return output
