# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Per-allocation sealing: a held allocation is uploaded unsealed and resumed; a sealed one never reruns."""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("taskforge")

from shellbox.machine import Backend
from taskforge.sandbox.factories import MachineHost

from experiments.post_training.capability_driven_envs.acceptance_measure import driver
from experiments.post_training.capability_driven_envs.acceptance_measure.pipeline import Host
from experiments.post_training.capability_driven_envs.acceptance_measure.records import Allocation

FIXTURE = (
    Path(__file__).parent / "data" / "acceptance_measure" / "d01.uncertainty.statistics.point-estimation.proposal.json"
)
HOST = Host(MachineHost.LAPTOP, (Backend.SHELLSIM,), None, None, "http://localhost:1/v1", None, None, None)


def allocation() -> Allocation:
    sha = hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
    return Allocation("B", "d01.uncertainty.statistics.point-estimation", "d01", 0, 23, str(FIXTURE), sha)


def test_a_held_allocation_resumes_from_its_uploaded_tree_and_seals_once(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver, "MODULE", "experiments.post_training.capability_driven_envs.tests.acceptance_measure_fake_child"
    )
    store = driver.Store(str(tmp_path / "out"))
    prefix = f"allocations/{allocation().key}"

    def run(terminal: str) -> str:
        monkeypatch.setenv("FAKE_TERMINAL", terminal)
        return asyncio.run(driver.run_one(store, allocation(), HOST, tmp_path / "work", "m" * 64))

    assert run("held") == "unsealed"
    assert not store.exists(f"{prefix}/SEALED.json")
    assert store.read(f"{prefix}/tree/progress.txt") == b"ran\n"

    assert run("accepted") == "sealed_accepted"
    assert store.read(f"{prefix}/tree/progress.txt") == b"ran\nran\n", "the second try resumed the uploaded tree"
    seal = json.loads(store.read(f"{prefix}/SEALED.json"))
    assert seal["terminal"] == "accepted" and seal["attempt"] == 1
    assert seal["records_sha256"] == hashlib.sha256(store.read(f"{prefix}/records.jsonl")).hexdigest()
    attempts = [json.loads(store.read(f"{prefix}/attempts/{n}.json")) for n in range(2)]
    assert [a["terminal"]["terminal"] for a in attempts] == ["held", "accepted"] and attempts[1]["restored_files"] == 2

    assert run("accepted") == "sealed_before"


def test_an_allocation_that_never_seals_stops_after_its_attempt_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver, "MODULE", "experiments.post_training.capability_driven_envs.tests.acceptance_measure_fake_child"
    )
    monkeypatch.setenv("FAKE_TERMINAL", "held")
    store = driver.Store(str(tmp_path / "out"))

    outcomes = [asyncio.run(driver.run_one(store, allocation(), HOST, tmp_path / "work", "m" * 64)) for _ in range(4)]

    assert outcomes == ["unsealed"] * driver.MAX_ATTEMPTS + ["attempts_exhausted"]


def test_the_shard_dispatches_every_allocation_with_a_sliding_window(tmp_path, monkeypatch):
    monkeypatch.setattr(
        driver, "MODULE", "experiments.post_training.capability_driven_envs.tests.acceptance_measure_fake_child"
    )
    monkeypatch.setenv("FAKE_TERMINAL", "rejected")
    store = driver.Store(str(tmp_path / "out"))
    base = allocation()
    allocations = [Allocation(arm, base.capability, "d01", 0, 23, base.proposal, base.sha256) for arm in ("A", "B", "C")]

    outcomes = asyncio.run(driver.run_shard(store, allocations, HOST, tmp_path / "work", "m" * 64, concurrency=2))

    assert outcomes == {a.key: "sealed_rejected" for a in allocations}
