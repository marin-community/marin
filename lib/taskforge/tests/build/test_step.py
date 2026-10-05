# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from taskforge.build.author import load_module
from taskforge.build.run import item_id_for
from taskforge.build.sdk import Build
from taskforge.build.step import CacheStatus, StepCache

PROGRAM = """
import asyncio

PROMPT = "first"


def render(text: str) -> str:
    return PROMPT + ":" + text


@step(StepRole.OTHER)
async def write(b: Build, name: str, delay: float) -> str:
    await asyncio.sleep(delay)
    b.emit(name, render(name).encode())
    return render(name)


async def both(b: Build) -> tuple[str, str]:
    return tuple(await asyncio.gather(write(b, "slow", 0.02), write(b, "fast", 0.0)))
"""


async def run_both(source: str, proposal, tmp_path, services) -> tuple[tuple[str, str], Build, StepCache]:
    module = load_module(source)
    cache = StepCache(root=tmp_path / "cache", item_id=item_id_for(proposal))
    async with services() as s:
        b = Build(proposal, cache.item_id, s, cache, tmp_path / "scratch", 0)
        return await module.both(b), b, cache


async def test_concurrent_steps_keep_their_own_resources_across_hits(proposal, tmp_path, services):
    outputs, b, _ = await run_both(PROGRAM, proposal, tmp_path, services)
    assert outputs == ("first:slow", "first:fast")
    first = {r.name: b.read(r.blob) for r in b.resources}

    again, b, cache = await run_both(PROGRAM, proposal, tmp_path, services)

    assert again == outputs
    assert {r.status for r in cache.records} == {CacheStatus.HIT}
    assert {r.name: b.read(r.blob) for r in b.resources} == first == {"slow": b"first:slow", "fast": b"first:fast"}


async def test_editing_a_constant_the_step_reads_invalidates_it(proposal, tmp_path, services):
    await run_both(PROGRAM, proposal, tmp_path, services)

    outputs, _, cache = await run_both(PROGRAM.replace('"first"', '"second"'), proposal, tmp_path, services)

    assert outputs == ("second:slow", "second:fast")
    assert {r.status for r in cache.records} == {CacheStatus.MISS}


DATA_PROGRAM = """
from dataclasses import dataclass


@dataclass(frozen=True)
class Key:
    value: int


ANSWERS = {"x": [1, 2]}
KEY = Key(3)


@step(StepRole.OTHER)
async def answer(b: Build) -> str:
    return f"{ANSWERS['x']}:{KEY.value}"


async def both(b: Build) -> tuple[str, str]:
    return (await answer(b), "")
"""


@pytest.mark.parametrize(
    "edit",
    [("[1, 2]", "[1, 5]"), ("Key(3)", "Key(4)")],
    ids=["dict-constant", "dataclass-instance-constant"],
)
async def test_editing_a_data_constant_the_step_reads_invalidates_it(edit, proposal, tmp_path, services):
    first, _, _ = await run_both(DATA_PROGRAM, proposal, tmp_path, services)

    outputs, _, cache = await run_both(DATA_PROGRAM.replace(*edit), proposal, tmp_path, services)

    assert outputs != first
    assert {r.status for r in cache.records} == {CacheStatus.MISS}
