# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import re

import pandas as pd
import pytest
from mariner.cli import INPUT_ERROR_STATUS, main
from mariner.tables import format_scaling_main, scaling_main_table

NUMBER = re.compile(r"\d+\.\d+")
# tab:scaling-main as sections/results.tex:22-25 prints it: each row's 11 numbers in print order, the 3e18 means and
# standard deviations, the losses at 2e19 to 1e21 FLOPs, and the accuracy (%).
PRINTED_SCALING_MAIN = {
    "Proportional": ["1.038", "0.001", "0.902", "0.770", "0.721", "1.198", "0.004", "0.972", "0.754", "0.673", "30.1"],
    "UniMax-8": ["1.021", "0.002", "0.892", "0.762", "0.715", "1.133", "0.004", "0.918", "0.701", "0.625", "32.7"],
    "Olmix": ["1.002", "0.001", "0.873", "0.745", "0.700", "1.092", "0.004", "0.892", "0.676", "0.607", "35.1"],
    "MARINER": ["0.982", "0.001", "0.854", "0.729", "0.686", "1.068", "0.005", "0.868", "0.656", "0.589", "36.4"],
}
# The small ladder's MARINER row (conftest.small_ladder_snapshot): each objective's 3e18 mean (base + 0.001) and SD,
# its losses 0.1 BPB apart, and the mean of its two accuracies, 33 and 35. None of its off-objective values appears.
SMALL_LADDER_MARINER = ["1.301", "0.001", "1.200", "1.100", "1.000", "1.311", "0.001", "1.210", "1.110", "1.010", "34.0"]


def printed_rows(text: str) -> dict[str, list[str]]:
    """The numbers of each mixture's line of ``format_scaling_main``'s text, by the mixture's printed name."""
    rows = {}
    for line in text.splitlines():
        name = line.split(maxsplit=1)[0] if line.strip() else ""
        if name in PRINTED_SCALING_MAIN:
            rows[name] = NUMBER.findall(line)
    return rows


def test_cli_tables_prints_the_table(capsys, small_ladder_snapshot):
    assert main(["tables", "--data", str(small_ladder_snapshot)]) == 0
    printed = capsys.readouterr().out
    rows = printed_rows(printed)
    assert list(rows) == list(PRINTED_SCALING_MAIN)  # paper order
    assert rows["MARINER"] == SMALL_LADDER_MARINER


def test_cli_tables_names_a_column_the_ladder_lacks(capsys, small_ladder_snapshot):
    ladder = small_ladder_snapshot / "data" / "ladder" / "ladder.csv"
    pd.read_csv(ladder).drop(columns="bpb__obe_easy").to_csv(ladder, index=False)
    assert main(["tables", "--data", str(small_ladder_snapshot)]) == INPUT_ERROR_STATUS
    error = capsys.readouterr().err
    assert "ladder" in error and "bpb__obe_easy" in error


@pytest.mark.data_integration
@pytest.mark.timeout(60)
def test_scaling_main_prints_the_papers_numbers(snapshot):
    assert printed_rows(format_scaling_main(scaling_main_table(snapshot))) == PRINTED_SCALING_MAIN
