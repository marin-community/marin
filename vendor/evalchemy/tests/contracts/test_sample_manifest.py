"""Identity and streaming-checkpoint behavior at the source execution boundary."""

import asyncio

import pytest

from eval.contracts.sample_manifest import SampleIdentityError, SampleManifest, SampleRequest
from eval.runtime.checkpoints import CheckpointRow, ResumeStream, completed_operation, input_digest
from eval.runtime.resources import SerialIO


@pytest.fixture
def serial_io():
    owner = SerialIO()
    try:
        yield owner
    finally:
        owner.close()


class _Checkpoints:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.written = []
        self.closed = False

    def records(self):
        yield from self.rows

    def record(self, row):
        self.written.append(row)

    def finalize(self):
        self.closed = True


@pytest.mark.parametrize("source_id", ["sample_0", 7, {"split": "test", "id": [3, "a"]}])
def test_source_id_types_round_trip_without_integer_coercion(source_id):
    manifest = SampleManifest("CruxEval")
    [entry] = manifest.plan_batch([SampleRequest(source_id=source_id, ordinal=0)])
    manifest.mark_generated([entry], ["answer"])

    assert entry.source_id == source_id
    assert manifest.generated_sample_count == 1


def test_source_identity_cannot_move_to_a_different_ordinal():
    manifest = SampleManifest("task")
    manifest.plan_batch([SampleRequest(source_id="stable", ordinal=0)])

    with pytest.raises(SampleIdentityError, match="ordinal"):
        manifest.plan_batch([SampleRequest(source_id="stable", ordinal=1)])


def test_stream_refuses_changed_admitted_item_before_model_execution():
    identity = {"source_id": "old", "ordinal": 0}
    store = _Checkpoints([CheckpointRow(0, input_digest(identity), {"phase": "generate"}, {"input_digest": "x", "output": "saved"})])
    stream = ResumeStream(store)

    with pytest.raises(ValueError, match="checkpoint source changed"):
        stream.admit(0, {"source_id": "new", "ordinal": 0})


def test_stream_refuses_changed_operation_input_without_execution(serial_io):
    store = _Checkpoints()
    journal = ResumeStream(store).admit(0, {"source_id": "stable"})
    journal.rows[input_digest({"phase": "generate"})] = {"input_digest": input_digest({"prompt": "old"}), "output": "saved"}
    called = False

    async def execute():
        nonlocal called
        called = True
        return "new"

    with pytest.raises(ValueError, match="checkpoint input changed"):
        asyncio.run(completed_operation(journal, {"phase": "generate"}, {"prompt": "new"}, execute, io=serial_io, successful=lambda _: True))
    assert called is False


def test_stream_records_only_successful_completed_operations(serial_io):
    store = _Checkpoints()
    stream = ResumeStream(store)
    journal = stream.admit(0, {"source_id": "stable"})

    async def execute():
        return "answer"

    assert asyncio.run(completed_operation(journal, {"phase": "generate"}, {"prompt": "q"}, execute, io=serial_io, successful=lambda _: True)) == "answer"
    assert store.written[0].position == 0
    assert store.written[0].payload["output"] == "answer"
