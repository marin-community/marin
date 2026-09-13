# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Contain advisory semantic comparison in a process with a wall-clock bound."""

import hashlib
import inspect
import json
import logging
import multiprocessing
import threading
import time

from experiments.post_training.math_eval.scoring import semantic_answer

MAX_MESSAGE_BYTES = 1048576
logger = logging.getLogger(__name__)


def _encode(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Semantic IPC message exceeds its byte bound")
    return payload


def _worker_main(connection):
    source_sha = hashlib.sha256(inspect.getsource(semantic_answer).encode()).hexdigest()
    connection.send_bytes(_encode({"ready": True, "source_sha256": source_sha}))
    while True:
        request = json.loads(connection.recv_bytes(MAX_MESSAGE_BYTES))
        if request is None:
            return
        try:
            result = semantic_answer(*request["arguments"])
            response = {"request_sha256": request["request_sha256"], "result": result}
        except Exception as error:
            # Error type only: exception text may contain a mathematical response.
            response = {"request_sha256": request["request_sha256"], "error_type": type(error).__name__}
        connection.send_bytes(_encode(response))


class SemanticWorker:
    """Keep successful semantic results exact; classify wall-time expiry as unresolved.

    Startup/import time has a separate deadline. A timeout destroys the worker;
    the next call starts a fresh process. Unexpected worker failures fail the
    audit rather than becoming scores. Receipts contain identities, not text.
    """

    def __init__(self, *, source_sha256, startup_timeout, row_timeout, cleanup_timeout):
        if min(startup_timeout, row_timeout, cleanup_timeout) <= 0:
            raise ValueError("Semantic execution bounds must be positive")
        self.source_sha256 = source_sha256
        self.startup_timeout = startup_timeout
        self.row_timeout = row_timeout
        self.cleanup_timeout = cleanup_timeout
        self.connection = None
        self.process = None
        self.receipts = []

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, *_):
        self.close()

    def _start(self):
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        self.connection = parent
        self.process = context.Process(target=_worker_main, args=(child,))
        self.process.start()
        child.close()
        try:
            if not parent.poll(self.startup_timeout):
                raise TimeoutError("Semantic worker startup exceeded its independent bound")
            ready = json.loads(parent.recv_bytes(MAX_MESSAGE_BYTES))
            if ready != {"ready": True, "source_sha256": self.source_sha256}:
                raise ValueError("Semantic worker source identity differs")
        except BaseException:
            self.close()
            raise

    def close(self):
        process = self.process
        if self.connection is not None:
            self.connection.close()
            self.connection = None
        if process is not None:
            if process.is_alive():
                process.terminate()
                process.join(self.cleanup_timeout)
            if process.is_alive():
                process.kill()
                process.join(self.cleanup_timeout)
            if process.is_alive():
                raise RuntimeError("Semantic subprocess could not be reaped")
            process.join()
            process.close()
            self.process = None

    def score(self, segment, boundary, gold, *, identity):
        if set(identity) != {"uid", "prompt_sha256", "response_sha256"}:
            raise ValueError("Semantic request lacks stable row identity")
        arguments = [segment, boundary, gold]
        request_sha = hashlib.sha256(_encode({"identity": identity, "arguments": arguments})).hexdigest()
        payload = _encode({"request_sha256": request_sha, "arguments": arguments})
        if self.process is None:
            self._start()
        start = time.monotonic()
        send_errors = []

        def send():
            try:
                self.connection.send_bytes(payload)
            except Exception as error:
                send_errors.append(error)

        sender = threading.Thread(target=send, daemon=True)
        try:
            sender.start()
            sender.join(self.row_timeout)
            remaining = max(0, self.row_timeout - (time.monotonic() - start))
            if sender.is_alive() or (not send_errors and not self.connection.poll(remaining)):
                self.close()
                sender.join(self.cleanup_timeout)
                if sender.is_alive():
                    raise RuntimeError("Semantic IPC sender did not stop after worker cleanup")
                result = (None, "semantic_worker_timeout", "unresolved")
            else:
                if send_errors:
                    raise RuntimeError("Semantic request transport failed") from send_errors[0]
                response = json.loads(self.connection.recv_bytes(MAX_MESSAGE_BYTES))
                if response.get("request_sha256") != request_sha or "error_type" in response:
                    raise ValueError("Semantic worker failed or returned a foreign request")
                result = response.get("result")
                if not isinstance(result, list) or len(result) != 3:
                    raise ValueError("Malformed semantic worker result")
                if result[0] not in (None, 0.0, 1.0) or not all(isinstance(value, str) for value in result[1:]):
                    raise ValueError("Invalid semantic worker score or status")
                result = tuple(result)
        except BaseException:
            self.close()
            raise
        receipt = {
            **identity,
            "request_sha256": request_sha,
            "semantic_source_sha256": self.source_sha256,
            "status": result[1],
            "elapsed_seconds": time.monotonic() - start,
            "row_timeout_seconds": self.row_timeout,
        }
        self.receipts.append(receipt)
        if result[1] == "semantic_worker_timeout" or len(self.receipts) % 128 == 0:
            logger.info("SEMANTIC_WORKER_PROGRESS %s", json.dumps({"rows": len(self.receipts), **receipt}))
        return result
