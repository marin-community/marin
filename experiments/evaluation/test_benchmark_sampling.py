# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evaluation.catalog import evaluation_benchmarks


def test_generation_benchmarks_declare_repeatable_sampling():
    # A missing temperature or seed falls back to the served model's default, which for a model
    # without sampling defaults is unseeded temperature 1.0: repeated runs then disagree.
    undeclared = []
    for benchmark in evaluation_benchmarks().values():
        for variant in benchmark.variants:
            for source in benchmark.build(variant):
                protocol = source.config.protocol
                if protocol.sampling is None:
                    continue
                temperature = protocol.sampling.temperature
                sampled_without_seed = bool(temperature) and protocol.trials.seed is None
                if temperature is None or sampled_without_seed:
                    undeclared.append((benchmark.name, variant.protocol, source.info.id))
    assert undeclared == []
