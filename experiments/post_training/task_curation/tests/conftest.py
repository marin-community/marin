# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import threading

import pytest

from experiments.post_training.task_curation.tests.local_grader import LocalGraderMachines, local_grader_machines
from experiments.post_training.task_curation.tests.review_server import ReviewHTTPServer


@pytest.fixture(scope="session")
def machines() -> LocalGraderMachines:
    """Docker machines running the local grader image; requesting tests skip without Docker or the image."""
    return local_grader_machines()


@pytest.fixture
def review_server():
    server = ReviewHTTPServer()
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        thread.join()
        server.server_close()
