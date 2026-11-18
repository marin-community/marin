# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A bounded private HTTP listener for the isolated application proofs."""

import threading
import time
from contextlib import contextmanager

import uvicorn


@contextmanager
def serve(app, port: int):
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", access_log=False))
    thread = threading.Thread(target=server.run)
    thread.start()
    try:
        deadline = time.monotonic() + 15
        while not server.started:
            if not thread.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("The private HTTP service did not start")
            time.sleep(0.05)
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("The private HTTP service did not shut down")
