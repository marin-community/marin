# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior checks for the structure-closure continuation transport."""

import hashlib
import importlib
import io
import json

import pytest
from marin.evaluation.records import TaskCoverage

tournament = importlib.import_module("experiments.evaluation.campaigns.eval-campaign-09-25.structure_closure_tournament")
refresh = importlib.import_module(
    "experiments.evaluation.campaigns.eval-campaign-09-25.analysis.structure_closure_tournament.refresh"
)


def test_score_display_uses_three_decimal_places() -> None:
    assert refresh._score_text(0) == "0.000"
    assert refresh._score_text(0.001) == "0.001"
    assert refresh._score_text(0.01004) == "0.010"
    assert refresh._score_text(0.9501) == "0.950"
    assert refresh._score_text(1) == "1.000"


def test_tournament_cell_requires_ninety_percent_scored_coverage() -> None:
    coverage = TaskCoverage(n_benchmark=704, n_attempted=704, n_scored=634)
    assert refresh._acceptable_coverage(coverage)
    assert not refresh._acceptable_coverage(coverage.model_copy(update={"n_scored": 633}))
    assert not refresh._acceptable_coverage(coverage.model_copy(update={"n_attempted": 633}))


def test_completion_sends_raw_tokens_and_retains_special_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = []

    def respond(request, timeout):
        requests.append((request, timeout))
        return io.BytesIO(b'{"choices":[{"text":"</tool_call><|eot_id|>","finish_reason":"stop"}]}')

    monkeypatch.setattr(tournament, "urlopen", respond)
    text, reason = tournament._completion("http://localhost/v1", "grug", [11, 12], "secret")

    request, timeout = requests[0]
    payload = json.loads(request.data)
    assert (text, reason) == ("</tool_call><|eot_id|>", "stop")
    assert payload["prompt"] == [11, 12]
    assert "messages" not in payload
    assert payload["skip_special_tokens"] is False
    assert payload["spaces_between_special_tokens"] is False
    assert payload["max_tokens"] == 1024
    assert request.get_header("Authorization") == "Bearer secret"
    assert timeout == tournament.REQUEST_TIMEOUT


def test_other_tokenizer_reencodes_identical_prompt_text(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "<tool_call>{"
    case = {
        "case_id": "case-1",
        "prompt_token_ids": [101, 102],
        "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }

    class Tokenizer:
        def __init__(self, tokens):
            self.tokens = tokens

        def decode(self, tokens, **kwargs):
            assert tokens == self.tokens
            return text

        def encode(self, value, **kwargs):
            assert value == text
            return self.tokens

    monkeypatch.setattr(tournament, "_cases", lambda: [case])
    monkeypatch.setattr(
        tournament.AutoTokenizer,
        "from_pretrained",
        lambda model, revision: (
            Tokenizer([101, 102]) if model == tournament.REFERENCE_TOKENIZER else Tokenizer([201, 202, 203])
        ),
    )
    tournament._prompt_tokens.cache_clear()
    model = "open-athena/Grug-67B-A2B-Merge-NUPA-BFCL-Shared100-2026.10.09"
    assert tournament._prompt_tokens(model, "test-revision") == {"case-1": [201, 202, 203]}
    tournament._prompt_tokens.cache_clear()
