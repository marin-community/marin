# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Physical-host slots must coordinate workers with private mount namespaces."""

from concurrent.futures import ThreadPoolExecutor
from uuid import UUID

import pytest
from rigging.filesystem.conditional_object import ConditionalWriteError, conditional_object

from experiments.post_training.task_curation.local_host_admission import (
    LocalHostAdmission,
    LocalHostAdmissionConfig,
)


def host_pool(tmp_path, worker, *, host=1, maximum=2):
    def storage(path):
        return conditional_object(str(tmp_path / path.removeprefix("s3://controls/")))

    return LocalHostAdmission(
        LocalHostAdmissionConfig(control_root="s3://controls/admission", max_trials_per_host=maximum),
        UUID(int=host),
        f"node-{host}",
        worker,
        storage,
    )


def test_concurrent_worker_namespaces_share_one_physical_host_cap(tmp_path):
    workers = [host_pool(tmp_path, f"worker-{index}") for index in range(12)]
    with ThreadPoolExecutor(max_workers=12) as executor:
        claims = list(executor.map(lambda worker: worker.try_claim(worker.worker_id), workers))
    acquired = [claim for claim in claims if claim is not None]
    assert len(acquired) == 2
    assert len({claim.control.path for claim in acquired}) == 2
    assert all(worker.try_claim("later-task") is None for worker in workers)
    assert host_pool(tmp_path, "another-host", host=2).try_claim("independent") is not None
    for claim in acquired:
        claim.release()
    assert workers[0].try_claim("after-release") is not None


def test_claims_do_not_expire_and_a_stale_holder_cannot_release_a_successor(tmp_path):
    first = host_pool(tmp_path, "first", maximum=1)
    second = host_pool(tmp_path, "second", maximum=1)
    claim = first.try_claim("original-task")
    assert claim is not None
    assert second.try_claim("replacement-task") is None
    claim.release()
    successor = second.try_claim("replacement-task")
    assert successor is not None
    with pytest.raises(ConditionalWriteError):
        claim.release()
    assert first.try_claim("third-task") is None
    assert successor.record.task_id == "replacement-task"


def test_admission_wait_deadline_never_enters_source_execution(tmp_path):
    pool = host_pool(tmp_path, "owner", maximum=1)
    held = pool.try_claim("other-task")
    assert held is not None
    clock = [0.0]

    def sleep(delay):
        clock[0] += delay

    executed = []
    with pytest.raises(TimeoutError):
        with pool.claim("queued-task", deadline=3, now=lambda: clock[0], sleep=sleep):
            executed.append("source")
    assert executed == []
    assert pool.try_claim("still-held") is None
    held.release()
    with pool.claim("available-task", deadline=4, now=lambda: clock[0], sleep=sleep):
        executed.append("source")
    assert executed == ["source"]


def test_inconsistent_boot_identity_cannot_open_a_second_pool_on_the_same_node(tmp_path):
    first = host_pool(tmp_path, "first", maximum=1)
    held = first.try_claim("running")
    assert held is not None
    changed_boot = LocalHostAdmission(first.config, UUID(int=2), first.node_id, "another-pod", first.object_factory)
    with pytest.raises(ValueError, match="physical-host admission contract"):
        changed_boot.try_claim("queued")
    assert first.try_claim("still-queued") is None
