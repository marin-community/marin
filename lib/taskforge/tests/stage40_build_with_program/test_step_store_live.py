# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One store round trip against a real object-store root.

Run by hand with the bucket's credentials configured for rigging (``configure_coreweave_s3`` reads
them from the environment) and the root as ``TASKFORGE_STEP_STORE_LIVE_ROOT``::

    TASKFORGE_STEP_STORE_LIVE_ROOT=s3://<bucket>/<scratch prefix> uv run --package marin-taskforge \\
        --frozen --group test pytest -c lib/taskforge/pyproject.toml \\
        lib/taskforge/tests/stage40_build_with_program/test_step_store_live.py -m manual

The test leaves nothing under the root. Last run 2026-10-13 against CoreWeave S3: store 0.37 s,
hit load 0.04 s, no temporary objects left behind.
"""

import os
import uuid
from dataclasses import dataclass

import pytest
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.s3_compat import configure_coreweave_s3

from taskforge.stage40_build_with_program.step_store import SDK_VERSION, RLEnvBuildStepKey, RLEnvBuildStepStore

ROOT_ENV = "TASKFORGE_STEP_STORE_LIVE_ROOT"


@dataclass(frozen=True)
class Numbers:
    value: int
    text: str


@pytest.mark.manual
def test_store_round_trips_on_an_object_store_root():
    parent = os.environ.get(ROOT_ENV)
    if not parent:
        pytest.skip(f"{ROOT_ENV} is not set")
    configure_coreweave_s3()
    root = f"{parent.rstrip('/')}/{uuid.uuid4().hex[:8]}"
    fs, _ = url_to_fs(root)
    store = RLEnvBuildStepStore(root, "live--00")
    key = RLEnvBuildStepKey(SDK_VERSION, "a" * 64, {"proposal": "b" * 64}, "c" * 64)
    try:
        assert store.load(key, "numbers", Numbers) is None
        stored = store.store(key, "numbers", Numbers(7, "seven"))
        loaded = store.load(key, "numbers", Numbers)
        assert loaded is not None
        record, output = loaded
        assert output == Numbers(7, "seven")
        assert record == stored
        objects = fs.find(root)
        assert len(objects) == 2 and not any(".tmp." in path for path in objects)
    finally:
        fs.rm(root, recursive=True)
