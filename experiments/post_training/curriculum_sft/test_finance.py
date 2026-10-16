# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import random
from fractions import Fraction

import pytest
from rigging.filesystem.storage_path import StoragePath
from zephyr.readers import load_parquet

from experiments.post_training.curriculum_sft.finance import (
    REPORTING_ANALYSIS,
    REPORTING_DISCLOSURES,
    WORKING_CAPITAL,
    FinanceProblemsConfig,
    LineItem,
    finance_problem,
    generate_company,
    rounded,
    write_finance_problems,
)
from experiments.post_training.curriculum_sft.generation import PROBLEMS_FILENAME


@pytest.mark.parametrize("seed", range(50))
def test_generated_statements_are_internally_consistent(seed):
    company = generate_company(random.Random(seed))

    for year in company.fiscal_years:
        assert company.value(LineItem.TOTAL_ASSETS, year) == company.value(LineItem.TOTAL_LIABILITIES_AND_EQUITY, year)
    for year in company.reported_years:
        prior = year - 1
        assert company.value(LineItem.GROSS_PROFIT, year) - company.value(LineItem.SGA, year) - company.value(
            LineItem.RND, year
        ) == company.value(LineItem.OPERATING_INCOME, year)
        assert company.value(LineItem.RETAINED_EARNINGS, year) - company.value(
            LineItem.RETAINED_EARNINGS, prior
        ) == company.value(LineItem.NET_INCOME, year) + company.value(LineItem.DIVIDENDS, year)
        net_cash_flow = sum(
            company.value(item, year)
            for item in (LineItem.OPERATING_CASH_FLOW, LineItem.INVESTING_CASH_FLOW, LineItem.FINANCING_CASH_FLOW)
        )
        assert company.value(LineItem.CASH, prior) + net_cash_flow == company.value(LineItem.CASH, year)
        assert company.value(LineItem.ENDING_CASH, year) == company.value(LineItem.CASH, year)


def test_reference_answers_match_hand_computed_values():
    quick = finance_problem(WORKING_CAPITAL, 2, seed=17)
    # (356,179 cash + 71,081 short-term investments + 224,605 receivables) / 371,523 current liabilities = 1.7546
    for amount in ("356,179", "71,081", "224,605", "371,523"):
        assert amount in quick["problem"]
    assert (quick["template"], quick["answer"], quick["yes_no"]) == ("quick_ratio_above", "1.75", "yes")
    assert "exceed 1.5" in quick["problem"]

    payable_days = finance_problem(WORKING_CAPITAL, 3, seed=17)
    # 365 x ((81,928 + 90,372) / 2) / 992,281 cost of goods sold = 31.689
    for amount in ("81,928", "90,372", "992,281"):
        assert amount in payable_days["problem"]
    assert (payable_days["template"], payable_days["answer"]) == ("days_payable_outstanding", "31.7")

    free_cash_flow = finance_problem(REPORTING_DISCLOSURES, 1, seed=17)
    # 234,158 operating cash flow - 98,340 capex, in thousands, is $135.818 million.
    for amount in ("234,158", "(98,340)"):
        assert amount in free_cash_flow["problem"]
    assert (free_cash_flow["template"], free_cash_flow["answer"]) == ("free_cash_flow", "135.82")


def test_rounding_is_half_up_not_bankers():
    assert rounded(Fraction(1, 8), 2) == "0.13"
    assert rounded(Fraction(-25, 10), 0) == "-3"


def test_problem_artifact_is_deterministic_for_a_seed(tmp_path):
    def written_rows(name: str, seed: int) -> list[dict]:
        write_finance_problems(
            FinanceProblemsConfig(
                output_path=str(tmp_path / name), capability_id=REPORTING_ANALYSIS, count=12, seed=seed
            )
        )
        return list(load_parquet(str(StoragePath(str(tmp_path / name)) / PROBLEMS_FILENAME)))

    first = written_rows("first", seed=5)

    assert first == written_rows("second", seed=5)
    assert [row["problem"] for row in first] != [row["problem"] for row in written_rows("other", seed=6)]
    assert all(row["accepted"] for row in first)
    assert len({row["template"] for row in first}) == 6
