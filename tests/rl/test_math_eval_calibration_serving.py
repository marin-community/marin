# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
import time
from itertools import pairwise
from types import SimpleNamespace

import pytest
from tokenizers import AddedToken, Regex, Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split

from experiments.post_training.math_eval import calibration_serving as serving
from experiments.post_training.math_eval.calibration_protocol import CalibrationProtocol
from experiments.post_training.math_eval.contract import QWEN, SNOWBALL
from experiments.post_training.math_eval.pool import prompt_hash
from experiments.post_training.math_eval.snowball_checkpoint_protocol import (
    SnowballHeldoutProtocol,
    snowball_protocol_receipt,
    snowball_request,
    snowball_rows,
)


def endpoint_fixture(monkeypatch, *, duplicate=False):
    vocab = {chr(i): i for i in range(1, 128)} | {"<eos>": 0}
    decoder = Tokenizer(WordLevel(vocab, unk_token="?"))
    decoder.pre_tokenizer = Split(Regex(""), behavior="isolated")
    decoder.decoder = Fuse()
    decoder.add_special_tokens([AddedToken("<eos>", special=True)])
    items = []
    for index in range(3):
        problem = f"Question {index}: 2+3?"
        items.append(
            {
                "problem": problem,
                "prompt_sha256": prompt_hash(problem),
                "prompt_template_id_qwen": QWEN.template_id,
                "prompt_template_id_snowball": SNOWBALL.template_id,
                "gold": "5",
                "env_class": "aime",
                "bin": "fixture",
            }
        )
    state = {"active": 0, "completed": 0, "calls": 0, "requests": []}
    lock = threading.Lock()

    def post(_url, *, json, timeout):
        assert timeout == 600
        with lock:
            state["active"] += 1
            state["calls"] += 1
            state["requests"].append(json)
            call = state["calls"]
        time.sleep(0.01 if call % 2 else 0.03)
        text = "Answer: 5"
        tokens = [*decoder.encode(text).ids, 0]
        choices = [
            {
                "index": i,
                "text": text,
                "token_ids": tokens,
                "prompt_token_ids": json["prompt"],
                "finish_reason": "stop",
                "stop_reason": None,
            }
            for i in range(json["n"])
        ]
        response = {
            "model": json["model"],
            "id": "duplicate" if duplicate else str(call),
            "choices": choices,
            "usage": {
                "prompt_tokens": len(json["prompt"]),
                "completion_tokens": len(tokens) * json["n"],
                "total_tokens": len(json["prompt"]) + len(tokens) * json["n"],
            },
        }
        with lock:
            state["active"] -= 1
            state["completed"] += 1
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: response)

    monkeypatch.setattr(serving.requests, "post", post)
    session = SimpleNamespace(
        model=SimpleNamespace(endpoint=SimpleNamespace(base_url="http://fixture/v1")), check_alive=lambda: None
    )
    return decoder, items, session, state


def test_calibration_panels_drain_all_http_work_and_keep_namespaces(tmp_path, monkeypatch):
    decoder, items, session, state = endpoint_fixture(monkeypatch)
    seen = set()
    timings = []
    for protocol in [CalibrationProtocol("dev_greedy", 1, 1), CalibrationProtocol("heldout_stochastic", 4)]:
        rows, timing = serving.run_panel(
            items,
            decoder,
            protocol,
            session,
            api_model="checkpoint-fixture",
            output_uri=str(tmp_path / protocol.identity),
            provenance={},
            seen_response_ids=seen,
        )
        assert len(rows) == 3 * protocol.samples
        assert state["active"] == 0 and state["completed"] == state["calls"]
        assert timing["requests_returned"] == 3 and timing["http_executor_drained"]
        assert [row["row_ordinal"] for row in rows] == list(range(3 * protocol.samples))
        assert all(row["score"] == 1 and row["native_contract_correct"] for row in rows)
        timings.append(timing)
    assert timings[0]["finished_at_ms"] <= timings[1]["started_at_ms"]
    assert state["calls"] == len(seen) == 6


def test_calibration_duplicate_response_fails_and_preserves_evidence(tmp_path, monkeypatch):
    decoder, items, session, state = endpoint_fixture(monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match="reused"):
        serving.run_panel(
            items,
            decoder,
            CalibrationProtocol("dev_greedy", 1),
            session,
            api_model="checkpoint-fixture",
            output_uri=str(tmp_path / "panel"),
            provenance={"binding": "fixture"},
            seen_response_ids=set(),
        )
    assert state["active"] == 0 and state["completed"] == 3
    receipt = json.loads((tmp_path / "panel" / "validation-failure.json").read_text())
    assert receipt["valid"] is False and receipt["response"]["id"] == "duplicate"
    assert not (tmp_path / "panel" / "generation.json").exists()


@pytest.mark.parametrize("seed", [17, 29, 43])
def test_snowball_existing_http_runner_forwards_seed_and_drains_k1_k8(tmp_path, monkeypatch, seed):
    decoder, items, session, state = endpoint_fixture(monkeypatch)
    seen = set()
    panels = []
    for panel in ("math_greedy", "math_stochastic", "platinum_greedy"):
        protocol = SnowballHeldoutProtocol(panel, seed)
        rows, timing = serving.run_panel(
            items,
            decoder,
            protocol,
            session,
            api_model="trained-u100",
            output_uri=str(tmp_path / panel),
            provenance={},
            seen_response_ids=seen,
            request_adapter=snowball_request,
            rows_adapter=snowball_rows,
            receipt_adapter=snowball_protocol_receipt,
        )
        assert len(rows) == 3 * protocol.samples and state["active"] == 0
        assert timing["http_executor_drained"] and state["completed"] == state["calls"]
        assert [(row["uid"], row["sample_index"]) for row in rows] == [
            (item["prompt_sha256"], index) for item in items for index in range(protocol.samples)
        ]
        panels.append(timing)
    assert all(a["finished_at_ms"] <= b["started_at_ms"] for a, b in pairwise(panels))
    assert len(state["requests"]) == 9
    assert [(request["n"], request["temperature"], request["top_p"]) for request in state["requests"]] == (
        [(1, 0.0, 1.0)] * 3 + [(8, 0.6, 0.95)] * 3 + [(1, 0.0, 1.0)] * 3
    )
    assert all(request["seed"] == seed and request["max_tokens"] == 4096 for request in state["requests"])
