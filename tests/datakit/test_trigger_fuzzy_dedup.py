# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from marin.execution.step_status import STATUS_SUCCESS, StatusFile, get_status_path

from experiments.datakit.reference_pipeline import build_fuzzy_dedup_steps
from experiments.datakit.scripts.trigger_fuzzy_dedup import _clear_completed_status, parse_args


def test_shard_and_worker_flags_reach_the_fields_the_launcher_reads():
    """The launcher reads these fields; a renamed dest silently changes the graph shape."""
    args = parse_args(
        [
            "--dedup-input-shards",
            "512",
            "--dedup-reduce-shards",
            "256",
            "--dedup-max-workers",
            "8",
            "--minhash-max-workers",
            "64",
        ]
    )

    assert args.dedup_input_shards == 512
    assert args.dedup_reduce_shards == 256
    assert args.dedup_max_workers == 8
    assert args.minhash_max_workers == 64


@pytest.mark.parametrize(
    "flag",
    ["--dedup-input-shards", "--dedup-reduce-shards", "--dedup-max-workers", "--minhash-max-workers"],
)
def test_non_positive_counts_are_rejected_by_the_flag_the_user_typed(flag, capsys):
    with pytest.raises(SystemExit):
        parse_args([flag, "0"])

    assert flag in capsys.readouterr().err


def test_continuation_flags_default_to_off():
    """A plain launch must not pin the output tree or overwrite a completed run."""
    args = parse_args([])

    assert args.dedup_output_path is None
    assert args.rerun_completed is False


def test_pinned_output_path_reaches_the_dedup_step(tmp_path):
    pinned = str(tmp_path / "dedup_709f5997")
    steps = build_fuzzy_dedup_steps({}, dedup_output_path=pinned)

    assert steps.dedup.output_path == pinned


def test_clearing_a_completed_status_lets_the_step_run_again(tmp_path):
    output_path = str(tmp_path / "dedup")
    StatusFile(output_path, worker_id="test").write_status(STATUS_SUCCESS)
    assert StatusFile(output_path, worker_id="test").status == STATUS_SUCCESS

    _clear_completed_status(output_path)

    assert StatusFile(output_path, worker_id="test").status is None
    assert not Path(get_status_path(output_path)).exists()


def test_clearing_an_absent_status_is_a_no_op(tmp_path):
    _clear_completed_status(str(tmp_path / "never-run"))
