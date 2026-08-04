# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.datakit.scripts.trigger_fuzzy_dedup import parse_args


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
