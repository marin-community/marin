# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The ``train`` box on a SkyRL training job: tasks go into the job's feed, and its graded groups come back.

The feed is an object-store prefix that the loop and the SkyRL job share (``data.feed.root`` of the job, see
``skyrl_train.dataset.task_feed`` in ``vendor/skyrl``):

- ``chunks/<first seq>.parquet``: the tasks in the order of training, with the columns ``seq`` and ``task`` (TaskSpec
  JSON). The loop writes them.
- ``graded/<seq>.json``: the graded attempts of one task, as a ``GradedGroup``. The job writes them.
- ``skipped/<seq>.json``: a task that the job did not run, and why. The job writes them.
- ``policy_endpoint.json``: the OpenAI-compatible endpoint of the live policy, which the Taskforge gate uses.

The box keeps at most ``lookahead`` tasks in the feed that did not come back, so the shares of ``mix`` apply to each
training batch. A task that comes back late after a resume of the job, or twice, is dropped.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath
from rolloutengine.models import GradedRollout
from taskcompendium.models import TaskSpec
from transformers import AutoTokenizer

from experiments.post_training.baby_rsi.policy import Sampling, VllmPolicy
from experiments.post_training.baby_rsi.rollouts import TaskRollouts

logger = logging.getLogger(__name__)

CHUNKS = "chunks"
GRADED = "graded"
SKIPPED = "skipped"
ENDPOINT = "policy_endpoint.json"
SEQ_DIGITS = 12
SKYRL_MODEL_PREFIX = "skyrl-step-"


class GradedGroup(BaseModel):
    """The graded attempts of one feed task, from the policy version ``policy_step``, without token-level data."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64", frozen=True)
    uid: str
    policy_step: int
    rollouts: tuple[GradedRollout, ...]


class Skipped(BaseModel):
    uid: str
    reason: str


@dataclass(frozen=True)
class FeedConfig:
    root: StoragePath
    lookahead: int
    """Tasks in the feed that did not come back, at most: about (staleness + 2) training batches."""
    chunk_rows: int
    poll_seconds: float
    groups: int
    """Graded groups after which the box ends: the job's steps times its prompts per step."""


def seq_name(seq: int, suffix: str) -> str:
    return f"{seq:0{SEQ_DIGITS}d}{suffix}"


def written_tasks(root: StoragePath) -> dict[str, TaskSpec]:
    """The tasks in the feed of ``root`` by uid, for example of a run that a preemption stopped."""
    tasks = {}
    for chunk in _listing(root / CHUNKS):
        with chunk.open("rb") as handle:
            table = pq.read_table(handle)
        for seq, task in zip(table["seq"].to_pylist(), table["task"].to_pylist(), strict=True):
            tasks[str(seq)] = TaskSpec.model_validate_json(task)
    return tasks


def returned_uids(root: StoragePath) -> set[str]:
    return {_uid(path) for path in _listing(root / GRADED) + _listing(root / SKIPPED)}


class FeedWriter:
    """Writes tasks to the feed in chunks; continues after the last task of the feed."""

    def __init__(self, root: StoragePath, chunk_rows: int, next_seq: int):
        self.root = root
        self.chunk_rows = chunk_rows
        self.next_seq = next_seq
        self._rows: list[tuple[int, str]] = []

    def append(self, task: TaskSpec) -> str:
        seq = self.next_seq
        self.next_seq += 1
        self._rows.append((seq, task.model_dump_json()))
        if len(self._rows) >= self.chunk_rows:
            self.flush()
        return str(seq)

    def flush(self) -> None:
        if not self._rows:
            return
        seqs, tasks = zip(*self._rows, strict=True)
        table = pa.table({"seq": pa.array(seqs, pa.int64()), "task": pa.array(tasks, pa.string())})
        with (self.root / CHUNKS / seq_name(seqs[0], ".parquet")).open("wb") as handle:
            pq.write_table(table, handle)
        self._rows = []


async def skyrl_train(tasks: AsyncIterable[TaskSpec], config: FeedConfig) -> AsyncIterator[TaskRollouts]:
    """Write ``tasks`` into the feed of a SkyRL job and yield the job's graded groups as they come."""
    written = await asyncio.to_thread(written_tasks, config.root)
    returned = await asyncio.to_thread(returned_uids, config.root)
    pending = {uid: task for uid, task in written.items() if uid not in returned}
    feed = FeedWriter(config.root, config.chunk_rows, next_seq=max(map(int, written), default=-1) + 1)
    room = asyncio.Semaphore(max(config.lookahead - len(pending), 0))
    logger.info("The feed has %d tasks, %d of them pending", len(written), len(pending))

    async def write() -> None:
        async for task in tasks:
            if room.locked():
                # The job must see the written tasks before the box waits for one to come back.
                await asyncio.to_thread(feed.flush)
            await room.acquire()
            pending[feed.append(task)] = task
        await asyncio.to_thread(feed.flush)

    writer = asyncio.create_task(write())
    count = 0
    try:
        async for record in _returned(config.root, config.poll_seconds, seen=returned, writer=writer):
            task = pending.pop(record.uid, None)
            if task is None:
                continue
            room.release()
            if isinstance(record, Skipped):
                logger.warning("The job skipped %s (%s): %s", record.uid, task.id, record.reason)
                continue
            yield TaskRollouts(task, record.rollouts, f"{SKYRL_MODEL_PREFIX}{record.policy_step}")
            count += 1
            if count >= config.groups:
                return
    finally:
        writer.cancel()


async def _returned(
    root: StoragePath, poll_seconds: float, *, seen: set[str], writer: asyncio.Task[None]
) -> AsyncIterator[GradedGroup | Skipped]:
    """The new graded and skipped records of the feed, every ``poll_seconds``; raises the writer's failure."""
    while True:
        if writer.done() and not writer.cancelled() and writer.exception() is not None:
            raise RuntimeError("The feed writer failed") from writer.exception()
        for directory, model in ((GRADED, GradedGroup), (SKIPPED, Skipped)):
            paths = await asyncio.to_thread(_listing, root / directory)
            for path in paths:
                uid = _uid(path)
                if uid in seen:
                    continue
                seen.add(uid)
                yield model.model_validate_json(await asyncio.to_thread(path.read_text))
        await asyncio.sleep(poll_seconds)


def _uid(path: StoragePath) -> str:
    return str(int(path.name.split(".")[0]))


def _listing(directory: StoragePath) -> list[StoragePath]:
    """The files in ``directory``, sorted, from a fresh listing: the object-store client caches listings."""
    fs, path = url_to_fs(str(directory))
    if not fs.exists(path):
        return []
    fs.invalidate_cache(path)
    names = sorted(child.rsplit("/", 1)[-1] for child in fs.ls(path, detail=False))
    return [directory / name for name in names]


async def live_policy(
    root: StoragePath, tokenizer: str, sampling: Sampling, max_model_len: int, **kwargs: Any
) -> VllmPolicy:
    """The live policy of the SkyRL job of ``root``, for the Taskforge gate; waits until the job serves it."""
    path = root / ENDPOINT
    while not await asyncio.to_thread(path.exists):
        logger.info("Waiting for %s", path)
        await asyncio.sleep(30)
    endpoint = json.loads(await asyncio.to_thread(path.read_text))
    logger.info("The live policy is %s at %s", endpoint["model"], endpoint["base_url"])
    base_url = endpoint["base_url"].removesuffix("/").removesuffix("/v1")
    return VllmPolicy(
        base_url, AutoTokenizer.from_pretrained(tokenizer), sampling, max_model_len, model=endpoint["model"], **kwargs
    )
