# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from finestore.cache import PersistentKvCache

from taskcompendium.models import Source, TaskSpec
from taskcompendium.pipeline.models import RawRow, ReviewRubric, ReviewStatus
from taskcompendium.pipeline.query_cache import cached_request_output
from taskcompendium.pipeline.review import BatchReviewer, ChatReviewer, valid_review_completion
from taskcompendium.pipeline.review_requests import RequestObservation, RequestOutput

from .pipeline_stages import svamp_row_task
from .test_pipeline import response


class UnusedBatchClient:
    def upload(self, *_args):
        raise AssertionError("Direct review or an exact cache hit must not upload files")

    def create(self, *_args):
        raise AssertionError("Direct review or an exact cache hit must not create a batch")

    def wait(self, *_args):
        raise AssertionError("No prior submitted batch exists")

    def output(self, *_args):
        raise AssertionError("No prior submitted batch exists")


class ReviewChat:
    def __init__(self):
        self.requests = []

    def complete(self, body):
        self.requests.append(body)
        task_id = json.loads(body["messages"][1]["content"])["id"]
        verdict = {
            "task_id": task_id,
            "quality": "good",
            "confidence": "high",
            "reference_status": "consistent",
            "defects": [],
            "evidence": "Two apples is the supplied count.",
        }
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-real-response",
                                "type": "function",
                                "function": {"name": "review_task", "arguments": json.dumps(verdict)},
                            }
                        ],
                    },
                }
            ]
        }


def test_direct_review_reuses_exact_cache_across_modes_and_source_ids(tmp_path):
    task = svamp_row_task(
        RawRow(
            "source-a",
            Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            {"Body": "Ada has two apples.", "Question": "How many apples?", "Answer": "2"},
        )
    )
    assert isinstance(task, TaskSpec)
    rubric = ReviewRubric("fixture", "1", ("Check the supplied count.",))
    chat = ReviewChat()
    batches = UnusedBatchClient()
    cache = str(tmp_path / "cache")
    abandoned = tmp_path / "cache/pending-batches/old/requests/reserved.json"
    abandoned.parent.mkdir(parents=True)
    abandoned.write_text('{"state": "reserved", "attempts": ["missing.json"]}')
    original = abandoned.read_bytes()
    direct = ChatReviewer(chat, "fixture", "deployment", query_cache_root=cache)
    first = direct.review([task], rubric).reviews
    second = direct.review([task.model_copy(update={"id": "source-b"})], rubric).reviews
    batch = BatchReviewer(batches, "fixture", "deployment", query_cache_root=cache)
    third = batch.review([task.model_copy(update={"id": "source-c"})], rubric).reviews
    assert [row.status for row in first + second + third] == [ReviewStatus.REVIEWED] * 3
    assert [row.task_id for row in first + second + third] == ["source-a", "source-b", "source-c"]
    assert len(chat.requests) == 1
    assert abandoned.read_bytes() == original


@pytest.mark.parametrize("corruption", ["malformed", "mismatched", "invalid_output", "invalid_shape"])
def test_invalid_cache_entry_does_not_block_inference(tmp_path, corruption):
    calls = []
    requests = [{"custom_id": "task", "body": {"prompt": "Count apples."}}]
    raw = json.dumps(response("task"))

    def submit(selected):
        calls.append(selected)
        return RequestOutput(tuple(selected), [RequestObservation(("task",), raw)])

    cache_root = str(tmp_path / "cache")
    options = dict(
        cache_root=cache_root,
        model_revision="pinned",
        valid_completion=valid_review_completion,
        submit=submit,
    )
    first = cached_request_output(requests, **options)
    assert first.output == raw
    key = first.cache_keys["task"]
    cache = PersistentKvCache.at(cache_root)
    envelope = json.loads(cache.load_many([key])[key])
    cache.close()
    if corruption == "malformed":
        corrupted = b"invalid JSON"
    else:
        if corruption == "mismatched":
            envelope["identity"]["model_revision"] = "different"
        elif corruption == "invalid_shape":
            envelope["raw_output"] = json.dumps({"custom_id": "task", "response": []})
        else:
            envelope["raw_output"] = "invalid response"
        corrupted = json.dumps(envelope).encode()
    cache = PersistentKvCache.at(cache_root)
    cache.store(key, corrupted)
    cache.close()

    assert cached_request_output(requests, **options).output == raw
    assert calls == [requests, requests]
    assert cached_request_output(requests, **options).output == raw
    assert len(calls) == 2


@pytest.mark.parametrize("recover", [False, True])
def test_direct_provider_failures_have_finite_neutral_retries(tmp_path, recover):
    task = svamp_row_task(
        RawRow(
            "task",
            Source(dataset="fixture", revision="1", row="0", importer_revision="1"),
            {"Body": "Ada has two apples.", "Question": "How many apples?", "Answer": "2"},
        )
    )
    assert isinstance(task, TaskSpec)
    rubric = ReviewRubric("fixture", "1", ("Check the supplied count.",))

    class TransientChat(ReviewChat):
        attempts = 0

        def complete(self, body):
            self.attempts += 1
            if not recover or self.attempts < 3:
                raise TimeoutError("Provider unavailable")
            return super().complete(body)

    chat = TransientChat()
    reviewer = ChatReviewer(chat, "fixture", "deployment", query_cache_root=str(tmp_path / "cache"))
    result = reviewer.review([task], rubric)
    record = result.reviews[0]
    assert chat.attempts == 3
    assert record.status == (ReviewStatus.REVIEWED if recover else ReviewStatus.UNAVAILABLE)
    assert (record.verdict is not None) == recover
    assert len(result.attempts) == 3
    assert all(attempt.requests.observations for attempt in result.attempts)


@pytest.mark.parametrize("failure", ["unreadable", "store", "close"])
def test_cache_storage_failure_does_not_discard_provider_response(tmp_path, monkeypatch, failure):
    cache_root = tmp_path / "cache"
    if failure == "unreadable":
        cache_root.write_text("not a FineStore archive")
    else:

        def fail(*_args, **_kwargs):
            raise OSError("Cache storage unavailable")

        monkeypatch.setattr(PersistentKvCache, failure, fail)
    requests = [{"custom_id": "task", "body": {"prompt": "Count apples."}}]
    raw = json.dumps({"custom_id": "task", "answer": "2"})
    submitted = []

    def submit(selected):
        submitted.extend(selected)
        return RequestOutput(tuple(selected), [RequestObservation(("task",), raw)])

    result = cached_request_output(
        requests,
        cache_root=str(cache_root),
        model_revision="pinned",
        valid_completion=lambda result, _task_id: result == raw,
        submit=submit,
    )
    assert result.output == raw
    assert submitted == requests
    assert result.observations[0].output == raw
