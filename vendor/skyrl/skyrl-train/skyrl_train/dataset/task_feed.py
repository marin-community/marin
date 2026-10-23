"""A task feed that a producer extends while training runs, and the records that the run writes back to it.

The feed root holds these objects:

- ``chunks/<first_seq:012d>.parquet``: written by the producer, one atomic object each. Columns ``seq`` (int64,
  contiguous from 0 across chunks) and ``task`` (a ``TaskSpec`` JSON).
- ``graded/<seq:012d>.json``: a ``GradedGroup`` for each generated group, written before batch selection.
- ``skipped/<seq:012d>.json``: a ``SkippedRow`` for a row that the run cannot train.
- ``policy_endpoint.json``: a ``PolicyEndpoint``, the OpenAI-compatible endpoint of the live policy.

The uid of a row is ``str(seq)``.
"""

import bisect
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import pyarrow.parquet as pq
from loguru import logger
from pydantic import BaseModel, ConfigDict
from rolloutengine.harness.assistant_prefix import ASSISTANT_PREFIX_HARNESS
from rolloutengine.models import GradedRollout, MachineRuntimeSpec, TaskRolloutSpec
from shellbox.machine import Backend, CommandSemantics, NetworkPolicy
from skyrl_gym.rollout_specs import SHELLBOX_HARNESS, rollout_spec
from taskcompendium.models import ScriptGrader, TaskSpec, TextMessage, VerifyitGrader
from transformers import PreTrainedTokenizerBase

from marinskyrl.remote_io import filesystem_and_path
from marinskyrl.resource_locator import join_resource_path
from skyrl_train.dataset.tasks import TASK_ROLLOUT_COLUMN, TaskDataset, task_prompt, task_prompt_token_count
from skyrl_train.io import io

CHUNKS = "chunks"
GRADED = "graded"
SKIPPED = "skipped"
POLICY_ENDPOINT = "policy_endpoint.json"


class GradedGroup(BaseModel):
    """The graded attempts of one feed row, without token-level data."""

    model_config = ConfigDict(ser_json_bytes="base64", val_json_bytes="base64", frozen=True)

    uid: str
    policy_step: int
    rollouts: tuple[GradedRollout, ...]


class SkippedRow(BaseModel):
    """A feed row that the run does not train, and why."""

    model_config = ConfigDict(frozen=True)

    uid: str
    reason: str


class PolicyEndpoint(BaseModel):
    """The live policy's OpenAI-compatible endpoint."""

    model_config = ConfigDict(frozen=True)

    base_url: str
    model: str


def record_path(root: str, kind: str, uid: str) -> str:
    return join_resource_path(root, kind, f"{int(uid):012d}.json")


def stripped_rollout(graded: GradedRollout) -> GradedRollout:
    """Remove the token IDs, the log probabilities and the steps of each conversation; keep the messages."""
    conversations = tuple(
        replace(conversation, prompt_token_ids=(), response_token_ids=(), loss_mask=(), logprobs=None, steps=())
        for conversation in graded.rollout.conversations
    )
    return replace(graded, rollout=replace(graded.rollout, conversations=conversations))


def graded_group(uid: str, policy_step: int, rollouts: Sequence[GradedRollout]) -> GradedGroup:
    return GradedGroup(uid=uid, policy_step=policy_step, rollouts=tuple(map(stripped_rollout, rollouts)))


def write_graded_group(root: str, group: GradedGroup) -> None:
    io.write_bytes_atomic(record_path(root, GRADED, group.uid), group.model_dump_json().encode())


def write_skipped(root: str, uid: str, reason: str) -> None:
    logger.warning("Skipping feed row {}: {}", uid, reason)
    io.write_bytes_atomic(
        record_path(root, SKIPPED, uid), SkippedRow(uid=uid, reason=reason).model_dump_json().encode()
    )


def write_policy_endpoint(root: str, endpoint: PolicyEndpoint) -> None:
    io.write_bytes_atomic(join_resource_path(root, POLICY_ENDPOINT), endpoint.model_dump_json().encode())


def machine_runtime(backend: str) -> MachineRuntimeSpec:
    """A machine of ``backend`` with no network and the backend's own defaults."""
    return MachineRuntimeSpec(
        backend=backend,
        network=NetworkPolicy.DENY,
        cpus=None,
        memory_mb=None,
        storage_mb=None,
        gpus=0,
        user=None,
        startup_timeout=None,
        cleanup_timeout=None,
    )


def needs_machine(task: TaskSpec) -> bool:
    requirements = task.environment_requirements
    return bool(requirements.capabilities or requirements.setup_commands or task.resources.worker or task.output_paths)


def lower_feed_task(task: TaskSpec, session: Mapping[str, Any]) -> TaskRolloutSpec:
    """The rollout spec of a feed task.

    A task whose context ends with an assistant text message continues it with the assistant-prefix harness. Every
    other task gets the Shellbox shell harness with ``session``. A task that needs a machine gets a bubblewrap machine
    for Linux process semantics, else a ShellSim machine. A script grader, or a verifyit grader with an
    environment, grades in a bubblewrap machine. Machines have no network.
    """
    last = task.context.events[-1] if task.context.events else None
    assistant_prefix = isinstance(last, TextMessage) and last.role == "assistant"
    task_machine = None
    if needs_machine(task):
        semantics = task.environment_requirements.command_semantics
        task_machine = machine_runtime(
            Backend.LOCAL if semantics == CommandSemantics.LINUX_PROCESS else Backend.SHELLSIM
        )
    grader = task.grader
    graded_in_machine = isinstance(grader, ScriptGrader) or (
        isinstance(grader, VerifyitGrader) and grader.environment is not None
    )
    return rollout_spec(
        task,
        {**session, "task_session": ASSISTANT_PREFIX_HARNESS if assistant_prefix else SHELLBOX_HARNESS},
        task_machine=task_machine,
        verifier_machine=machine_runtime(Backend.LOCAL) if graded_in_machine else None,
    )


@dataclass(frozen=True)
class _FeedRow:
    seq: int
    spec_json: str
    env_class: str
    data_source: str


class FeedTaskDataset(TaskDataset):
    """Task rows that a producer appends under ``root`` while training runs, in sequence order.

    ``poll`` reads the chunks that follow the last one read, at most once each ``poll_seconds``. A row that does
    not parse, cannot be lowered, or has a prompt longer than ``max_prompt_length`` is not added; its
    ``skipped`` record is written instead. The constructor waits until ``min_rows`` rows are available.
    """

    def __init__(
        self,
        root: str,
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        *,
        session: Mapping[str, Any],
        poll_seconds: float,
        min_rows: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.root = root
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.session = dict(session)
        self.poll_seconds = poll_seconds
        self._clock = clock
        self._rows: list[_FeedRow] = []
        self._seqs: list[int] = []
        self._end_seq = 0
        self._polled_at = -math.inf
        self.poll(force=True)
        while len(self._rows) < min_rows:
            logger.info("Task feed {} has {} of {} rows to start; waiting", root, len(self._rows), min_rows)
            sleep(poll_seconds)
            self.poll(force=True)

    def poll(self, *, force: bool = False) -> None:
        """Read the chunks that start at the next sequence number, in order."""
        now = self._clock()
        if not force and now - self._polled_at < self.poll_seconds:
            return
        self._polled_at = now
        chunks_root = join_resource_path(self.root, CHUNKS)
        filesystem, path = filesystem_and_path(chunks_root)
        filesystem.invalidate_cache(path)
        if not filesystem.exists(path):
            return
        names = (name.rsplit("/", 1)[-1] for name in filesystem.ls(path, detail=False))
        chunks = {int(name.removesuffix(".parquet")): name for name in names if name.endswith(".parquet")}
        while self._end_seq in chunks:
            self._read_chunk(join_resource_path(chunks_root, chunks[self._end_seq]))
        if any(first > self._end_seq for first in chunks):
            logger.warning("Task feed {} has no chunk that starts at seq {}", self.root, self._end_seq)

    def _read_chunk(self, path: str) -> None:
        with io.open_file(path, "rb") as source:
            table = pq.read_table(source, columns=["seq", "task"])
        seqs = table.column("seq").to_pylist()
        if seqs != list(range(self._end_seq, self._end_seq + len(seqs))) or not seqs:
            raise ValueError(f"Feed chunk {path} does not hold contiguous rows from seq {self._end_seq}")
        rows = []
        for seq, task_json in zip(seqs, table.column("task").to_pylist(), strict=True):
            uid = str(seq)
            try:
                lowered = lower_feed_task(TaskSpec.model_validate_json(task_json), self.session)
                prompt = task_prompt(lowered)
                length = task_prompt_token_count(self.tokenizer, prompt, "prompt")
            except Exception as error:
                write_skipped(self.root, uid, f"cannot lower the task: {type(error).__name__}: {error}")
                continue
            if length > self.max_prompt_length:
                write_skipped(self.root, uid, f"prompt has {length} tokens, more than {self.max_prompt_length}")
                continue
            rows.append(_FeedRow(seq, prompt[TASK_ROLLOUT_COLUMN], prompt["env_class"], prompt["data_source"]))
        self._rows.extend(rows)
        self._seqs.extend(row.seq for row in rows)
        self._end_seq = seqs[-1] + 1
        logger.info("Read {} feed rows from {}; {} rows in total", len(rows), path, len(self._rows))

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int):
        # ``collate_fn`` reads only the lowered task, and the worker renders the messages, so the row has none.
        row = self._rows[index]
        extras = {TASK_ROLLOUT_COLUMN: row.spec_json, "data_source": row.data_source}
        return None, row.env_class, extras, str(row.seq)

    def uid(self, index: int) -> str:
        return str(self._rows[index].seq)

    def position(self, seq: int) -> int:
        """The index of the first row whose sequence number is ``seq`` or more."""
        return bisect.bisect_left(self._seqs, seq)

    def next_seq(self, index: int) -> int:
        """The sequence number of row ``index``, or the next sequence number to read for the end."""
        return self._seqs[index] if index < len(self._seqs) else self._end_seq
