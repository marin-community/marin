# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from experiments.grug.moe_hero_ep.ops.forward_goldens import GoldenRequest, build_inputs, golden_spec
from experiments.grug.moe_hero_ep.ops.vibe_check.completions import Checkpoint


class _Tokenizer:
    eos_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        tokens = [ord(character) % 251 + 1 for character in text]
        return [252, *tokens] if add_special_tokens else tokens


def _request(mode: str) -> GoldenRequest:
    return GoldenRequest(
        checkpoint=Checkpoint(
            uri="s3://fixture/checkpoint",
            run_id="hero",
            step=108000,
            timestamp="2026-09-15T16:01:28.296153",
            metadata_digest="0" * 64,
        ),
        spec=golden_spec(mode),
        source_revision="0" * 40,
        target_cluster="cw-us-east-08a",
    )


def test_required_input_bank_preserves_padding_alignment_and_boundaries() -> None:
    request = _request("required")

    arrays, cases = build_inputs(request, _Tokenizer())

    assert arrays["tokens"].shape == (32, 4096)
    assert arrays["valid_lengths"].tolist() == [case.valid_length for case in request.spec.cases]
    assert np.array_equal(arrays["token_validity"], arrays["segment_ids"] >= 0)
    assert np.all(arrays["segment_ids"][~arrays["token_validity"]] == -1)
    assert np.all(arrays["score_mask"] <= arrays["token_validity"])
    assert not np.array_equal(arrays["score_mask"], arrays["token_validity"])
    score_rows = np.argwhere(arrays["score_mask"])
    assert np.array_equal(arrays["prediction_positions"], score_rows[:, 1] - 1)
    assert np.array_equal(arrays["target_token_ids"], arrays["tokens"][score_rows[:, 0], score_rows[:, 1]])
    assert {case["valid_length"] for case in cases} >= {2047, 2048, 2049, 4095, 4096}

    boundary_rows = [
        row
        for row, case in enumerate(request.spec.cases)
        if case.id.startswith(("local-window-minus-one", "local-window-exact", "local-window-plus-one"))
        and case.id.endswith("repeat-0")
    ]
    shortest = min(request.spec.cases[row].valid_length for row in boundary_rows)
    for row in boundary_rows[1:]:
        assert np.array_equal(arrays["tokens"][boundary_rows[0], :shortest], arrays["tokens"][row, :shortest])


def test_smoke_input_bank_keeps_full_hero_batch_with_short_sequences() -> None:
    arrays, cases = build_inputs(_request("smoke"), _Tokenizer())

    assert arrays["tokens"].shape == (32, 64)
    assert len(cases) == 32
    assert arrays["token_validity"].all()
