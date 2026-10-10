# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from taskcompendium.pipeline.chat_requests import MAX_DIRECT_CONCURRENT_REQUESTS, chat_output
from taskcompendium.pipeline.models import ReviewStatus
from taskcompendium.pipeline.review import review_records

from .test_pipeline import response


@dataclass
class ChatService:
    requests: list[dict] = field(default_factory=list)

    def complete(self, body):
        self.requests.append(body)
        return response(body["task_id"])["response"]["body"]


def test_direct_oversized_request_isolated_with_utf8_bytes_and_evidence():
    service = ChatService()
    requests = [
        {"custom_id": "oversized", "body": {"task_id": "oversized", "prompt": "é" * 50}},
        {"custom_id": "small", "body": {"task_id": "small"}},
    ]
    oversized_bytes = len(json.dumps(requests[0], ensure_ascii=False, separators=(",", ":")).encode()) + 1
    output = chat_output(service, requests, max_concurrent=2, max_batch_bytes=oversized_bytes - 1)
    assert [record.status for record in review_records(output.output, ["oversized", "small"])] == [
        ReviewStatus.UNAVAILABLE,
        ReviewStatus.REVIEWED,
    ]
    assert service.requests == [requests[1]["body"]]
    assert output.requests == tuple(requests)
    assert json.loads(output.observations[1].output)["response"]["body"] == response("small")["response"]["body"]


def test_concurrent_direct_invocations_share_process_admission_limit():
    lock = threading.Lock()
    saturated = threading.Event()
    release = threading.Event()

    class BlockingChat:
        active = 0
        peak = 0
        completed = 0

        def complete(self, body):
            with lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
                if self.active == MAX_DIRECT_CONCURRENT_REQUESTS:
                    saturated.set()
            assert release.wait(timeout=10)
            with lock:
                self.active -= 1
                self.completed += 1
            return response(body["task_id"])["response"]["body"]

    service = BlockingChat()
    requests = [{"custom_id": str(index), "body": {"task_id": str(index)}} for index in range(16)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(chat_output, service, requests, max_concurrent=MAX_DIRECT_CONCURRENT_REQUESTS)
            for _ in range(2)
        ]
        try:
            assert saturated.wait(timeout=10)
        finally:
            release.set()
        outputs = [future.result() for future in futures]
    assert service.peak == MAX_DIRECT_CONCURRENT_REQUESTS
    assert service.completed == 32
    assert all(
        record.status == ReviewStatus.REVIEWED
        for output in outputs
        for record in review_records(output.output, [str(index) for index in range(16)])
    )
