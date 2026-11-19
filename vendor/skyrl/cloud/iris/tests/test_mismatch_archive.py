"""Round trips and interrupted writes for mismatch archives."""

import pytest
from finestore.reader import ReadView
from finestore.rl.mismatch_probe import MANIFEST_TABLE, SCORES_TABLE, ManifestRow, ScoreRow

from skyrl_train.mismatch_probe.archive import MismatchArchive, read_frozen_probe


def test_route_bank_larger_than_default_transaction_limit_round_trips(tmp_path):
    # Forty 4K captures of 48 layers x eight uint16 routes exceed the real
    # 100 MB transaction cap. Reuse the immutable bytes in memory.
    routes = bytes(4096 * 48 * 8 * 2)
    scores = [
        ScoreRow(
            probe_hash="route-bank",
            sample_id=f"response-{index}",
            scorer="trainer",
            mode="router_replay",
            update=0,
            global_step=0,
            logprobs=[-float(index + 1)] * 4096,
            expert_choices=routes,
            expert_choices_shape=[4096, 48, 8],
            expert_choices_dtype="uint16",
        )
        for index in range(40)
    ]
    manifest = ManifestRow(
        archive=str(tmp_path),
        status="complete",
        probe_hash="route-bank",
        checkpoint_path="/staged/policy",
        runtime_commit="pinned",
        tokenizer_fingerprint="pinned",
        starting_global_step=0,
        scored_updates=[0],
        scored_global_steps=[0],
        architecture="ToyMoeForCausalLM",
        vllm_enforce_eager=False,
        optimizer_steps_per_update=1,
        seed=42,
        bootstrap_seed=17,
        created_at_utc="2026-11-03T00:00:00Z",
        config_json="{}",
        software_json="{}",
        hardware_json="{}",
        batch_layout_json="{}",
        timing_json="{}",
        step_metrics_json="{}",
    )
    archive = MismatchArchive(str(tmp_path), writer_id="archive-test")
    try:
        archive.write(scores=scores, manifest=manifest)
    finally:
        archive.close()

    view = ReadView(str(tmp_path))
    restored = view.scan(SCORES_TABLE).to_pylist()
    assert len(restored) == len(scores)
    assert {row["sample_id"] for row in restored} == {row.sample_id for row in scores}
    assert all(row["expert_choices"] == routes for row in restored)
    last = next(row for row in restored if row["sample_id"] == "response-39")
    assert last["logprobs"] == [-40.0] * 4096
    assert view.scan(MANIFEST_TABLE).to_pylist()[0]["status"] == "complete"


def test_reuse_probe_rejects_an_interrupted_archive(tmp_path):
    archive = MismatchArchive(str(tmp_path), writer_id="interrupted")
    try:
        archive.write(
            scores=[
                ScoreRow(
                    probe_hash="partial",
                    sample_id="response",
                    scorer="vllm.generate",
                    update=0,
                    global_step=0,
                    logprobs=[-1.0],
                )
            ]
        )
    finally:
        archive.close()

    with pytest.raises(ValueError, match="no manifest table"):
        read_frozen_probe(str(tmp_path))
