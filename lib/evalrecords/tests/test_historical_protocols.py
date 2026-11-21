# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Historical Harbor datasets read as the curated benchmarks that replaced them."""

import pytest
from evalrecords.historical_protocols import harbor_benchmark
from evalrecords.records import HarborRef


@pytest.mark.parametrize(
    "dataset, benchmark",
    [
        ("hf://DCAgent2/terminal_bench_2", "terminal-bench-2.0"),
        # Early Harbor records omitted the hf:// prefix.
        ("DCAgent2/terminal_bench_2", "terminal-bench-2.0"),
        ("openthoughts-tblite", "ot-tblite"),
        ("hf://DCAgent/dev_set_v2", "ot-tblite"),
        ("swebenchpro", "swebenchpro"),
    ],
)
def test_harbor_dataset_reads_as_its_curated_benchmark(dataset, benchmark):
    assert harbor_benchmark(HarborRef(dataset=dataset, version="1", agent="terminus-2", env="daytona")) == benchmark
