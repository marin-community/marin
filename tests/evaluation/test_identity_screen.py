# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import copy
import json

import pytest
import requests
from click.testing import CliRunner
from marin.evaluation.identity import (
    IdentityRules,
    NativeCaptureError,
    boundary_view,
    capture_native_response,
    decode_views,
    prescreen,
)

from experiments.evaluation import identity_screen as cli


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["tokenize"] and kwargs["add_generation_prompt"]
        assert not kwargs["continue_final_message"]
        return [100, 101]

    def decode(self, ids, *, skip_special_tokens, clean_up_tokenization_spaces):
        assert not clean_up_tokenization_spaces
        names = {2: "<START>", 3: "<END>", 9: "<EOT>", 10: "ChatGPT", 20: "Marin", 21: "Community"}
        return " ".join(names.get(i, str(i)) for i in ids if not (skip_special_tokens and i in (2, 3, 9)))


def response(value):
    result = requests.Response()
    result.status_code = 200
    result._content = json.dumps(value).encode()
    return result


def body():
    return {
        "model": "fixture",
        "messages": [{"role": "user", "content": "Who are you?"}],
        "temperature": 0.0,
        "max_tokens": 4096,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_capture_changes_only_native_flags_and_retains_provider_fields(monkeypatch):
    original = body()
    original["tools"] = [{"type": "function", "function": {"name": "lookup"}}]
    frozen = copy.deepcopy(original)
    seen = []
    result = {
        "prompt_token_ids": [100, 101],
        "choices": [
            {
                "token_ids": [2, 10, 3, 20, 21, 9],
                "message": {"content": "RAW", "reasoning_content": "PROVIDER-R"},
                "finish_reason": "stop",
            }
        ],
    }

    def post(self, url, **kwargs):
        seen.append((url, copy.deepcopy(kwargs["json"])))
        return response({"tokens": [100, 101]} if url.endswith("/tokenize") else result)

    monkeypatch.setattr(requests.Session, "post", post)
    with requests.Session() as session:
        captured = capture_native_response(session, "http://fixture", original, expected_prompt_ids=[100, 101])
    assert original == frozen
    assert seen[0][1]["tools"] == original["tools"]
    assert seen[0][1]["messages"] == original["messages"]
    assert seen[1][1] == {**original, "skip_special_tokens": False, "return_token_ids": True}
    assert captured["response"] == result
    views = decode_views(Tokenizer(), captured["generated_ids"], thinking=True, start_id=2, end_id=3)
    assert views["native_final_text"] == "Marin Community <EOT>"
    assert "ChatGPT" in views["whole_id_proxy_text"]
    labels = prescreen(views["native_final_text"], IdentityRules("Marin", "Community", "ChatGPT"))
    assert labels["name_match"] and labels["creator_match"]
    assert labels["foreign_self_claim_match"] is None and labels["needs_manual_review"]


@pytest.mark.parametrize(
    "failure", ["tokenize", "returned_prefix", "empty_ids", "bool_id", "missing_ids", "two_choices"]
)
def test_capture_failure_preserves_evidence_without_retries(monkeypatch, failure):
    calls = []

    def post(self, url, **kwargs):
        calls.append(url)
        if url.endswith("/tokenize"):
            return response({"tokens": [999] if failure == "tokenize" else [100, 101]})
        choice = {"token_ids": [2, 3, 20], "message": {"content": "raw"}}
        if failure == "empty_ids":
            choice["token_ids"] = []
        elif failure == "bool_id":
            choice["token_ids"] = [True]
        elif failure == "missing_ids":
            choice.pop("token_ids")
        return response(
            {
                "prompt_token_ids": [999] if failure == "returned_prefix" else [100, 101],
                "choices": [choice, choice] if failure == "two_choices" else [choice],
            }
        )

    monkeypatch.setattr(requests.Session, "post", post)
    with requests.Session() as session, pytest.raises(NativeCaptureError) as error:
        capture_native_response(session, "http://fixture", body(), expected_prompt_ids=[100, 101])
    assert len(calls) == (1 if failure == "tokenize" else 2)
    assert "tokenize_response" in error.value.evidence
    if failure != "tokenize":
        assert error.value.evidence["response"]["choices"][0]["message"]["content"] == "raw"


@pytest.mark.parametrize(
    "ids,closed,duplicate,first",
    [
        ([2, 10, 3, 20], True, False, 2),
        ([7, 2, 3, 20], True, False, 2),
        ([2, 10], False, False, None),
        ([3, 20], False, False, 0),
        ([3, 2, 20], False, False, 0),
        ([2, 3, 20, 3], False, True, 1),
    ],
)
def test_first_generated_end_is_a_view_not_protocol_admission(ids, closed, duplicate, first):
    snapshot = list(ids)
    view = boundary_view(ids, start_id=2, end_id=3)
    assert ids == snapshot
    assert view["model_opened_and_closed"] == closed
    assert view["duplicate_generated_boundaries"] == duplicate
    assert view["first_generated_end"] == first
    assert view["canonical_leading_start"] == (closed and ids[0] == 2)
    assert view["answer_ids"] == (None if first is None else ids[first + 1 :])


def test_missing_end_is_unavailable_not_wrong_identity_and_off_uses_full_output():
    tok = Tokenizer()
    rules = IdentityRules("Marin", "Community")
    on = decode_views(tok, [2, 20, 21], thinking=True, start_id=2, end_id=3)
    assert on["native_final_text"] is None
    labels = prescreen(on["native_final_text"], rules)
    assert labels["answer_available"] is False and labels["name_match"] is None
    off = decode_views(tok, [20, 21, 9], thinking=False, start_id=2, end_id=3)
    assert off["native_final_text"] == "Marin Community <EOT>"
    assert prescreen(off["native_final_text"], rules)["name_match"]


def test_prompt_owned_start_cannot_supply_a_generated_start(monkeypatch):
    def post(self, url, **kwargs):
        return response(
            {"tokens": [100, 2]}
            if url.endswith("/tokenize")
            else {"prompt_token_ids": [100, 2], "choices": [{"token_ids": [3, 20]}]}
        )

    monkeypatch.setattr(requests.Session, "post", post)
    with requests.Session() as session:
        result = capture_native_response(session, "http://fixture", body(), expected_prompt_ids=[100, 2])
    view = boundary_view(result["generated_ids"], start_id=2, end_id=3)
    assert view["generated_start_positions"] == []
    assert view["end_without_preceding_generated_start"] and not view["model_opened_and_closed"]


def test_cli_writes_raw_and_native_views_from_explicit_requests(tmp_path, monkeypatch):
    monkeypatch.setattr(cli.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(
        requests.Session,
        "post",
        lambda self, url, **kw: response(
            {"tokens": [100, 101]}
            if url.endswith("/tokenize")
            else {
                "prompt_token_ids": [100, 101],
                "choices": [
                    {"token_ids": [2, 10, 3, 20, 21, 9], "message": {"content": "RAW", "reasoning_content": ""}}
                ],
            }
        ),
    )
    source = tmp_path / "requests.jsonl"
    source.write_text(json.dumps({"id": "case", "request": body()}) + "\n")
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({"name_pattern": "Marin", "creator_pattern": "Community"}))
    output = tmp_path / "results.jsonl"
    args = [
        "--base-url",
        "http://fixture",
        "--model",
        "new-target",
        "--tokenizer",
        str(tmp_path),
        "--requests",
        str(source),
        "--rules",
        str(rules),
        "--start-id",
        "2",
        "--end-id",
        "3",
        "--output",
        str(output),
    ]
    result = CliRunner().invoke(cli.main, args)
    assert result.exit_code == 0, result.output
    provenance, row = [json.loads(x) for x in output.read_text().splitlines()]
    assert provenance["record_type"] == "provenance"
    assert row["source_request"]["model"] == "fixture"
    assert row["capture"]["request"]["model"] == "new-target"
    assert row["capture"]["response"]["choices"][0]["message"] == {"content": "RAW", "reasoning_content": ""}
    assert row["prescreen"]["name_match"] and row["views"]["boundary"]["model_opened_and_closed"]
    assert CliRunner().invoke(cli.main, args).exit_code != 0  # Never overwrite an existing capture.
