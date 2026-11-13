# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evaluation.catalog import evaluation_benchmarks


def test_generation_benchmarks_declare_a_seed():
    # Without a seed, a served model that samples returns different replies on every run.
    unseeded = []
    for benchmark in evaluation_benchmarks().values():
        for variant in benchmark.variants:
            for source in benchmark.build(variant):
                protocol = source.config.protocol
                if protocol.sampling is not None and protocol.trials.seed is None:
                    unseeded.append((benchmark.name, variant.protocol, source.info.id))
    assert unseeded == []
