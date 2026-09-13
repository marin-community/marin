# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import inspect
import json
import signal
import threading

import pytest
from tokenizers import Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel

from experiments.post_training.math_eval import semantic_worker
from experiments.post_training.math_eval.scoring import score_row, semantic_answer


def fixture_worker(connection):
    """Real spawned process exercising the IPC boundary and hard cleanup path."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    connection.send_bytes(json.dumps({"ready": True, "source_sha256": "fixture"}).encode())
    request = json.loads(connection.recv_bytes())
    mode = request["arguments"][0]
    if mode in {"hang", "#### 5"}:
        threading.Event().wait()
    elif mode == "crash":
        connection.close()
        return
    elif mode == "foreign":
        connection.send_bytes(json.dumps({"request_sha256": "foreign", "result": [1, "parsed", "fixture"]}).encode())
    else:
        connection.send_bytes(
            json.dumps({"request_sha256": request["request_sha256"], "result": [1, "parsed", "fixture"]}).encode()
        )
    threading.Event().wait()


IDENTITY = {"uid": "fixture-row", "prompt_sha256": "a" * 64, "response_sha256": "b" * 64}


def test_semantic_worker_preserves_actual_scorer_results_and_input_identity():
    digest = hashlib.sha256(inspect.getsource(semantic_answer).encode()).hexdigest()
    examples = [("42", "resolved", "42"), ("42", "resolved", "41"), (None, "missing", "42")]
    with semantic_worker.SemanticWorker(
        source_sha256=digest, startup_timeout=60, row_timeout=30, cleanup_timeout=1
    ) as worker:
        for args in examples:
            assert worker.score(*args, identity=IDENTITY) == semantic_answer(*args)
        assert len({row["request_sha256"] for row in worker.receipts}) == 3
        assert all(row["prompt_sha256"] == IDENTITY["prompt_sha256"] for row in worker.receipts)


def test_semantic_timeout_is_unresolved_and_restarts_reaped_process(monkeypatch):
    monkeypatch.setattr(semantic_worker, "_worker_main", fixture_worker)
    with semantic_worker.SemanticWorker(
        source_sha256="fixture", startup_timeout=60, row_timeout=0.1, cleanup_timeout=0.1
    ) as worker:
        result = worker.score("hang", "resolved", "gold", identity=IDENTITY)
        assert result == (None, "semantic_worker_timeout", "unresolved")
        assert worker.score("success", "resolved", "gold", identity=IDENTITY) == (1, "parsed", "fixture")
        assert [row["status"] for row in worker.receipts] == ["semantic_worker_timeout", "parsed"]


@pytest.mark.parametrize("mode", ["crash", "foreign"])
def test_semantic_unexpected_worker_failure_does_not_become_a_score(monkeypatch, mode):
    monkeypatch.setattr(semantic_worker, "_worker_main", fixture_worker)
    with semantic_worker.SemanticWorker(
        source_sha256="fixture", startup_timeout=60, row_timeout=10, cleanup_timeout=0.1
    ) as worker:
        with pytest.raises((EOFError, ValueError)):
            worker.score(mode, "resolved", "gold", identity=IDENTITY)
        assert worker.receipts == []


def test_semantic_expiry_preserves_native_completed_correctness(monkeypatch):
    monkeypatch.setattr(semantic_worker, "_worker_main", fixture_worker)
    decoder = Tokenizer(WordLevel({chr(i): i for i in range(128)}, unk_token="?"))
    decoder.decoder = Fuse()
    tokens = list(map(ord, "#### 5"))
    row = {
        "uid": "frozen-row",
        "env_class": "gsm8k",
        "score": [0, 1],
        "response_ids": tokens,
        "response_length": len(tokens),
        "prompt_token_ids": [],
        "stop_reason": "stop",
        "output_response": "#### 5",
        "env_extras": {
            "reward_model": {"ground_truth": "5"},
            "reward_spec": {"ground_truth": "5"},
            "extra_info": {"prompt_sha256": "a" * 64},
        },
    }
    with semantic_worker.SemanticWorker(
        source_sha256="fixture", startup_timeout=60, row_timeout=0.1, cleanup_timeout=0.1
    ) as worker:
        result = score_row(row, decoder, model="qwen", thinking=False, semantic_worker=worker)
    assert result.score_contract == result.contract_correct == result.score_contract_completed == 1
    assert result.native_reward_tokens == (0, 1)
    assert result.score_semantic is None and result.format_gap is None
    assert result.semantic_status == "semantic_worker_timeout"
