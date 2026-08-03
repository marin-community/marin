# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.datakit.scripts.trigger_fuzzy_dedup import parse_args


def test_default_launch_limits_parallel_steps_and_minhash_workers():
    args = parse_args([])

    assert args.max_concurrent == 16
    assert args.minhash_max_workers == 1024
    assert args.coordinator_cpu == 1.0
    assert args.coordinator_ram == "3g"
    assert args.dedup_max_workers == 48
    assert args.dedup_max_parallelism == 2544
    assert args.dedup_reduce_shards == 1248
    assert args.dedup_worker_cpu == 120.0
    assert args.dedup_worker_ram == "850g"
    assert args.dedup_worker_disk == "1t"
    assert args.dedup_map_task_cpu == 2.0
    assert args.dedup_map_task_ram == "16g"
    assert args.dedup_reduce_task_cpu == 30.0
    assert args.dedup_reduce_task_ram == "40g"
    assert args.dedup_reduce_task_disk == "96g"
