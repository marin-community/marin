# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Non-training checks for the measured interval and unchanged FLOP ledger."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.grug.belay_tpu import useful_mfu as mfu


def test_packed_batch_counts_match_document_oracle():
    segments = np.asarray([[4, 4, 8, 8, 8, 9, 9], [2, 2, 2, 3, 3, 3, 3]], dtype=np.int32)
    weights = np.asarray([[1, 1, 0, 1, 1, 0, 1], [0, 1, 1, 1, 0, 1, 1]], dtype=np.float32)
    expected_pairs = 0
    for sequence, mask in zip(segments, weights, strict=True):
        start = 0
        for index, (segment, valid) in enumerate(zip(sequence, mask, strict=True)):
            if index == 0 or segment != sequence[index - 1]:
                start = index
            expected_pairs += int(valid) * (index - start + 1)
    queries, pairs, binary = mfu.batch_counts(weights, segments)
    assert int(queries) == int(weights.sum())
    assert int(pairs) == expected_pairs
    assert bool(binary)


@pytest.mark.parametrize("pipeline_depth", [0, 1])
@pytest.mark.parametrize("initial_step", [0, 5187])
def test_windows_include_loading_and_callbacks_without_adjacent_updates(
    monkeypatch, tmp_path, pipeline_depth, initial_step
):
    # The driver advances a clock and scalar metadata only; there is no model or optimizer.
    clock = [0.0]
    queued = []
    synchronized = []
    monkeypatch.setenv("IRIS_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(mfu, "WINDOW_STARTS", (2, 5))
    monkeypatch.setattr(mfu, "WINDOW_STEPS", 3)
    monkeypatch.setattr(mfu, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(mfu.jax, "process_index", lambda: 0)
    monkeypatch.setattr(mfu.jax, "device_count", lambda: 16)
    monkeypatch.setattr(mfu.jax, "local_devices", lambda: [])
    monkeypatch.setattr(mfu.jax, "block_until_ready", lambda state: synchronized.append(state))
    monkeypatch.setattr(mfu.multihost_utils, "process_allgather", lambda value: value)
    monkeypatch.setattr(mfu.tracker, "log", lambda *args, **kwargs: None)

    def synchronize(_):
        clock[0] += 0.25

    def queue_counts(step, _):
        queued.append(step)
        return np.int32(500_000), np.int32(900_000 + step), np.bool_(True)

    monkeypatch.setattr(mfu.multihost_utils, "sync_global_devices", synchronize)
    monkeypatch.setattr(mfu, "batch_counts", queue_counts)
    meter = mfu.UsefulMfuWindows(initial_step=initial_step)
    pending = None

    def finish(record, state):
        step, counts = record
        clock[0] += 0.3  # All host callbacks and logging stay inside the window.
        meter.observe(step, state, counts, {"moe/dropped_assignments": np.array([2, 3], np.int32)})

    for step in range(initial_step, initial_step + 9):
        if meter.boundary_before(step):
            if pending is not None:
                finish(pending, step)
                pending = None
            meter.begin_if_due(step, step)
        clock[0] += 0.2  # Loading is after the start fence and before the end fence.
        if pending is not None and pipeline_depth == 0:
            finish(pending, step)
            pending = None
        batch = SimpleNamespace(
            tokens=SimpleNamespace(shape=(128, 4096)),
            loss_weight=step,
            attn_mask=SimpleNamespace(segment_ids=("segments",)),
        )
        clock[0] += 0.5
        counts = meter.prepare_counts(step, batch)
        if pending is not None:
            finish(pending, None)  # A following dispatch has donated this intermediate state.
        pending = (step, counts)
    finish(pending, initial_step + 9)

    rows = [json.loads(line) for line in meter.output.read_text().splitlines()]
    updates = [row for row in rows if row["stage"] == "useful_mfu_step"]
    windows = [row for row in rows if row["stage"] == "useful_mfu_window"]
    assert [row["step"] for row in updates] == list(range(initial_step + 2, initial_step + 8))
    assert queued == [initial_step, *range(initial_step + 2, initial_step + 8)]
    assert synchronized == [initial_step + 2, initial_step + 5, initial_step + 5, initial_step + 8]
    assert len(windows) == 2
    for index, window in enumerate(windows):
        first = initial_step + 2 + 3 * index
        flops = sum(mfu.useful_flops(500_000, 900_000 + step, 5) for step in range(first, first + 3))
        assert window["timing_protocol"] == 2
        assert window["first_step"] == first
        assert window["last_step"] == first + 2
        assert window["seconds"] == pytest.approx(3.25)
        assert window["useful_flops_lower_bound"] == flops
        assert window["useful_mfu_lower_bound"] == pytest.approx(flops / 3.25 / (16 * mfu.BF16_PEAK_PER_V4_CHIP))


@pytest.mark.parametrize("backend", ["xla", "pallas", "causal"])
def test_attention_backend_preserves_frozen_ledger_but_model_changes_do_not(backend):
    # The causal backend previously failed before the first continuation update.
    legacy = {"hidden_dim": 512, "num_experts_per_token": 10}
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    source = {**legacy, "tpu_attention": backend}
    assert mfu.model_fingerprint(source) == expected
    assert mfu.model_fingerprint({**source, "hidden_dim": 256}) != expected
    assert mfu.model_fingerprint({**source, "num_experts_per_token": 8}) != expected
    with pytest.raises(ValueError, match="Unknown attention backend"):
        mfu.model_fingerprint({**source, "tpu_attention": "unverified"})
