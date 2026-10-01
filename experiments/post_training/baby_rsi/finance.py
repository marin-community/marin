# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate finance curriculum problems whose reference answers are computed, not authored.

Each problem is a 10-K-style excerpt for a synthetic company: three years of income and cash-flow
statements, two year-end balance sheets, and a short MD&A paragraph, followed by one question. A
seeded simulation builds four years of linked statements, so the balance sheet balances, net income
less dividends rolls retained earnings forward, and the cash-flow statement reconciles to balance
sheet cash. Question templates cover the ratio families and answer shapes that FinanceBench tests:
a number with stated units and rounding, or a yes/no judgment plus the number behind it. Each
question states the definition it expects, and the reference answer is computed exactly from the
displayed integers, so no model or solver has to agree with it.

The problems artifact has the schema that ``self_distill.self_distill_step`` reads.
"""

import json
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from enum import StrEnum
from fractions import Fraction
from typing import Any

import pyarrow as pa
from marin.execution.artifact import Artifact
from marin.execution.lazy import ArtifactStep, StepContext
from marin.experiment.namespacing import user_owned_name
from rigging.filesystem.storage_path import StoragePath

from experiments.post_training.baby_rsi.generation import MANIFEST_FILENAME, PROBLEMS_FILENAME, write_table

logger = logging.getLogger(__name__)

REPORTING_ANALYSIS = "d27.reporting.analysis"
REPORTING_DISCLOSURES = "d27.reporting.disclosures"
WORKING_CAPITAL = "d27.corporate.working_capital"

DAYS_PER_YEAR = 365
BOXED_INSTRUCTION = "Put only the final number, without units, in \\boxed{}."

FINANCE_PROBLEM_SCHEMA = pa.schema(
    [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("capability_id", pa.string(), nullable=False),
        pa.field("template", pa.string(), nullable=False),
        pa.field("fiscal_year", pa.int64(), nullable=False),
        pa.field("units", pa.string(), nullable=False),
        pa.field("yes_no", pa.string()),
        pa.field("problem", pa.string(), nullable=False),
        pa.field("answer", pa.string(), nullable=False),
        pa.field("accepted", pa.bool_(), nullable=False),
    ]
)


class Units(StrEnum):
    THOUSANDS = "thousands"
    MILLIONS = "millions"


class TableStyle(StrEnum):
    MARKDOWN = "markdown"
    FIXED_WIDTH = "fixed_width"


class LineItem(StrEnum):
    REVENUE = "revenue"
    COST_OF_REVENUE = "cost_of_revenue"
    GROSS_PROFIT = "gross_profit"
    SGA = "sga"
    RND = "rnd"
    OPERATING_INCOME = "operating_income"
    INTEREST_EXPENSE = "interest_expense"
    PRETAX_INCOME = "pretax_income"
    INCOME_TAX = "income_tax"
    NET_INCOME = "net_income"
    CASH = "cash"
    SHORT_TERM_INVESTMENTS = "short_term_investments"
    RECEIVABLES = "receivables"
    INVENTORIES = "inventories"
    PREPAID = "prepaid"
    CURRENT_ASSETS = "current_assets"
    PPE = "ppe"
    GOODWILL = "goodwill"
    OTHER_ASSETS = "other_assets"
    TOTAL_ASSETS = "total_assets"
    PAYABLES = "payables"
    ACCRUED = "accrued"
    CURRENT_DEBT = "current_debt"
    CURRENT_LIABILITIES = "current_liabilities"
    LONG_TERM_DEBT = "long_term_debt"
    OTHER_LIABILITIES = "other_liabilities"
    TOTAL_LIABILITIES = "total_liabilities"
    PAID_IN_CAPITAL = "paid_in_capital"
    RETAINED_EARNINGS = "retained_earnings"
    TREASURY_STOCK = "treasury_stock"
    TOTAL_EQUITY = "total_equity"
    TOTAL_LIABILITIES_AND_EQUITY = "total_liabilities_and_equity"
    DEPRECIATION = "depreciation"
    STOCK_COMPENSATION = "stock_compensation"
    CHANGE_RECEIVABLES = "change_receivables"
    CHANGE_INVENTORIES = "change_inventories"
    CHANGE_PREPAID = "change_prepaid"
    CHANGE_PAYABLES = "change_payables"
    CHANGE_ACCRUED = "change_accrued"
    OPERATING_CASH_FLOW = "operating_cash_flow"
    CAPEX = "capex"
    INVESTMENT_PURCHASES = "investment_purchases"
    INVESTING_CASH_FLOW = "investing_cash_flow"
    DEBT_PROCEEDS = "debt_proceeds"
    DEBT_REPAYMENTS = "debt_repayments"
    DIVIDENDS = "dividends"
    REPURCHASES = "repurchases"
    FINANCING_CASH_FLOW = "financing_cash_flow"
    ENDING_CASH = "ending_cash"


# Filings name the same line item differently; each company draws one label per item.
LABEL_VARIANTS: dict[LineItem, tuple[str, ...]] = {
    LineItem.REVENUE: ("Revenue", "Net sales", "Total revenues", "Net revenues"),
    LineItem.COST_OF_REVENUE: ("Cost of revenue", "Cost of sales", "Cost of goods sold"),
    LineItem.GROSS_PROFIT: ("Gross profit", "Gross margin"),
    LineItem.SGA: ("Selling, general and administrative", "SG&A expenses", "Selling and administrative expenses"),
    LineItem.RND: ("Research and development", "R&D expenses"),
    LineItem.OPERATING_INCOME: ("Operating income", "Income from operations", "Operating profit"),
    LineItem.INTEREST_EXPENSE: ("Interest expense, net", "Interest expense"),
    LineItem.PRETAX_INCOME: ("Income before income taxes", "Earnings before income taxes"),
    LineItem.INCOME_TAX: ("Provision for income taxes", "Income tax expense"),
    LineItem.NET_INCOME: ("Net income", "Net earnings"),
    LineItem.CASH: ("Cash and cash equivalents",),
    LineItem.SHORT_TERM_INVESTMENTS: ("Short-term investments", "Marketable securities"),
    LineItem.RECEIVABLES: ("Accounts receivable, net", "Trade receivables, net of allowances", "Receivables, net"),
    LineItem.INVENTORIES: ("Inventories", "Inventories, net"),
    LineItem.PREPAID: ("Prepaid expenses and other current assets", "Other current assets"),
    LineItem.CURRENT_ASSETS: ("Total current assets",),
    LineItem.PPE: ("Property and equipment, net", "Property, plant and equipment, net"),
    LineItem.GOODWILL: ("Goodwill",),
    LineItem.OTHER_ASSETS: ("Other long-term assets", "Other assets"),
    LineItem.TOTAL_ASSETS: ("Total assets",),
    LineItem.PAYABLES: ("Accounts payable", "Trade accounts payable"),
    LineItem.ACCRUED: ("Accrued liabilities", "Accrued expenses and other current liabilities"),
    LineItem.CURRENT_DEBT: ("Current portion of long-term debt", "Short-term borrowings and current maturities"),
    LineItem.CURRENT_LIABILITIES: ("Total current liabilities",),
    LineItem.LONG_TERM_DEBT: ("Long-term debt, net of current portion", "Long-term debt"),
    LineItem.OTHER_LIABILITIES: ("Other long-term liabilities",),
    LineItem.TOTAL_LIABILITIES: ("Total liabilities",),
    LineItem.PAID_IN_CAPITAL: ("Common stock and additional paid-in capital",),
    LineItem.RETAINED_EARNINGS: ("Retained earnings",),
    LineItem.TREASURY_STOCK: ("Treasury stock, at cost",),
    LineItem.TOTAL_EQUITY: ("Total stockholders' equity", "Total shareholders' equity"),
    LineItem.TOTAL_LIABILITIES_AND_EQUITY: (
        "Total liabilities and stockholders' equity",
        "Total liabilities and equity",
    ),
    LineItem.DEPRECIATION: ("Depreciation and amortization",),
    LineItem.STOCK_COMPENSATION: ("Stock-based compensation", "Share-based compensation expense"),
    LineItem.CHANGE_RECEIVABLES: ("Accounts receivable",),
    LineItem.CHANGE_INVENTORIES: ("Inventories",),
    LineItem.CHANGE_PREPAID: ("Prepaid expenses and other assets",),
    LineItem.CHANGE_PAYABLES: ("Accounts payable",),
    LineItem.CHANGE_ACCRUED: ("Accrued liabilities",),
    LineItem.OPERATING_CASH_FLOW: ("Net cash provided by operating activities", "Cash flows from operating activities"),
    LineItem.CAPEX: ("Purchases of property and equipment", "Capital expenditures", "Additions to property and plant"),
    LineItem.INVESTMENT_PURCHASES: (
        "Net purchases of short-term investments",
        "Purchases of marketable securities, net",
    ),
    LineItem.INVESTING_CASH_FLOW: ("Net cash used in investing activities",),
    LineItem.DEBT_PROCEEDS: ("Proceeds from borrowings", "Proceeds from issuance of long-term debt"),
    LineItem.DEBT_REPAYMENTS: ("Repayments of long-term debt", "Principal payments on debt"),
    LineItem.DIVIDENDS: ("Dividends paid", "Cash dividends paid to shareholders"),
    LineItem.REPURCHASES: ("Repurchases of common stock", "Purchases of treasury stock"),
    LineItem.FINANCING_CASH_FLOW: ("Net cash (used in) provided by financing activities",),
    LineItem.ENDING_CASH: ("Cash and cash equivalents, end of year",),
}

INCOME_STATEMENT = (
    LineItem.REVENUE,
    LineItem.COST_OF_REVENUE,
    LineItem.GROSS_PROFIT,
    LineItem.SGA,
    LineItem.RND,
    LineItem.OPERATING_INCOME,
    LineItem.INTEREST_EXPENSE,
    LineItem.PRETAX_INCOME,
    LineItem.INCOME_TAX,
    LineItem.NET_INCOME,
)
BALANCE_SHEET = (
    LineItem.CASH,
    LineItem.SHORT_TERM_INVESTMENTS,
    LineItem.RECEIVABLES,
    LineItem.INVENTORIES,
    LineItem.PREPAID,
    LineItem.CURRENT_ASSETS,
    LineItem.PPE,
    LineItem.GOODWILL,
    LineItem.OTHER_ASSETS,
    LineItem.TOTAL_ASSETS,
    LineItem.PAYABLES,
    LineItem.ACCRUED,
    LineItem.CURRENT_DEBT,
    LineItem.CURRENT_LIABILITIES,
    LineItem.LONG_TERM_DEBT,
    LineItem.OTHER_LIABILITIES,
    LineItem.TOTAL_LIABILITIES,
    LineItem.PAID_IN_CAPITAL,
    LineItem.RETAINED_EARNINGS,
    LineItem.TREASURY_STOCK,
    LineItem.TOTAL_EQUITY,
    LineItem.TOTAL_LIABILITIES_AND_EQUITY,
)
CASH_FLOW_STATEMENT = (
    LineItem.NET_INCOME,
    LineItem.DEPRECIATION,
    LineItem.STOCK_COMPENSATION,
    LineItem.CHANGE_RECEIVABLES,
    LineItem.CHANGE_INVENTORIES,
    LineItem.CHANGE_PREPAID,
    LineItem.CHANGE_PAYABLES,
    LineItem.CHANGE_ACCRUED,
    LineItem.OPERATING_CASH_FLOW,
    LineItem.CAPEX,
    LineItem.INVESTMENT_PURCHASES,
    LineItem.INVESTING_CASH_FLOW,
    LineItem.DEBT_PROCEEDS,
    LineItem.DEBT_REPAYMENTS,
    LineItem.DIVIDENDS,
    LineItem.REPURCHASES,
    LineItem.FINANCING_CASH_FLOW,
    LineItem.ENDING_CASH,
)


@dataclass(frozen=True)
class Industry:
    description: str
    name_words: tuple[str, ...]
    gross_margin: tuple[float, float]
    operating_margin: tuple[float, float]
    rnd_share_of_opex: tuple[float, float]
    receivable_days: tuple[float, float]
    inventory_days: tuple[float, float]
    payable_days: tuple[float, float]
    capex_share: tuple[float, float]
    stock_compensation_share: tuple[float, float]
    growth_drivers: tuple[str, ...]


INDUSTRIES = (
    Industry(
        "a specialty retailer of outdoor apparel and equipment",
        ("Outfitters", "Home Stores", "Apparel"),
        (0.30, 0.42),
        (0.06, 0.11),
        (0.0, 0.0),
        (2, 8),
        (60, 110),
        (30, 55),
        (0.02, 0.05),
        (0.002, 0.008),
        ("comparable-store sales and store count", "e-commerce traffic and pricing"),
    ),
    Industry(
        "a provider of subscription business software",
        ("Software", "Systems", "Analytics"),
        (0.68, 0.82),
        (0.08, 0.25),
        (0.25, 0.40),
        (45, 75),
        (0, 0),
        (15, 40),
        (0.02, 0.06),
        (0.03, 0.08),
        ("new subscriptions and seat counts", "renewal pricing and add-on module sales"),
    ),
    Industry(
        "a manufacturer of industrial pumps and flow-control components",
        ("Industries", "Machinery", "Components"),
        (0.25, 0.38),
        (0.08, 0.16),
        (0.10, 0.25),
        (45, 70),
        (60, 120),
        (35, 70),
        (0.03, 0.07),
        (0.003, 0.010),
        ("aftermarket parts demand and pricing", "shipments to energy and water customers"),
    ),
    Industry(
        "an operator of casual-dining restaurants",
        ("Restaurants", "Hospitality", "Kitchens"),
        (0.18, 0.30),
        (0.06, 0.12),
        (0.0, 0.0),
        (3, 10),
        (5, 12),
        (15, 30),
        (0.05, 0.09),
        (0.002, 0.006),
        ("menu pricing and restaurant count", "guest traffic and catering sales"),
    ),
    Industry(
        "a designer and manufacturer of analog and power semiconductors",
        ("Semiconductor", "Microdevices", "Silicon"),
        (0.45, 0.65),
        (0.12, 0.30),
        (0.40, 0.60),
        (35, 60),
        (80, 150),
        (30, 60),
        (0.06, 0.15),
        (0.02, 0.05),
        ("automotive and industrial demand", "new product ramps and product mix"),
    ),
    Industry(
        "a maker of packaged foods and household consumer products",
        ("Brands", "Foods", "Consumer Products"),
        (0.35, 0.50),
        (0.10, 0.18),
        (0.03, 0.08),
        (25, 45),
        (45, 90),
        (50, 90),
        (0.03, 0.06),
        (0.004, 0.012),
        ("pricing and distribution", "snack and household cleaning volumes"),
    ),
)

NAME_ROOTS = (
    "Halvorsen",
    "Bristow",
    "Carraway",
    "Delmont",
    "Everly",
    "Fairhaven",
    "Granite Peak",
    "Harlow",
    "Ironwood",
    "Juniper",
    "Kestrel",
    "Larkspur",
    "Meridian",
    "Northgate",
    "Oakridge",
    "Pinecrest",
    "Redwater",
    "Silverline",
    "Tamarack",
    "Westbrook",
)
NAME_SUFFIXES = ("Inc.", "Corporation", "Holdings, Inc.", "Co.")
# (month and day, calendar-year offset): retailers often end fiscal 2023 in early 2024.
FISCAL_YEAR_ENDS = (("December 31", 0), ("June 30", 0), ("September 30", 0), ("January 31", 1))
YEAR_LABELS = ("FY{year}", "Fiscal {year}", "{year}")


@dataclass(frozen=True)
class Company:
    """Four fiscal years of linked statements; the earliest year only supplies opening balances."""

    name: str
    industry: Industry
    units: Units
    table_style: TableStyle
    year_label: str
    fiscal_year_end: tuple[str, int]
    labels: dict[LineItem, str]
    values: dict[int, dict[LineItem, int]]
    revolver_capacity: int

    @property
    def fiscal_years(self) -> list[int]:
        return sorted(self.values)

    @property
    def reported_years(self) -> list[int]:
        """Fiscal years with an income statement and cash-flow statement, newest first."""
        return self.fiscal_years[:0:-1]

    @property
    def balance_sheet_years(self) -> list[int]:
        return self.fiscal_years[:1:-1]

    def value(self, item: LineItem, year: int) -> int:
        return self.values[year][item]

    def period(self, year: int) -> str:
        return self.year_label.format(year=year)

    def year_end(self, year: int) -> str:
        month_day, offset = self.fiscal_year_end
        return f"{month_day}, {year + offset}"


def _uniform(rng: random.Random, bounds: tuple[float, float]) -> float:
    return rng.uniform(*bounds)


def _operating_balances(revenue: int, cost: int, rates: dict[LineItem, float]) -> dict[LineItem, int]:
    return {
        LineItem.RECEIVABLES: round(revenue * rates[LineItem.RECEIVABLES] / DAYS_PER_YEAR),
        LineItem.INVENTORIES: round(cost * rates[LineItem.INVENTORIES] / DAYS_PER_YEAR),
        LineItem.PREPAID: round(revenue * rates[LineItem.PREPAID]),
        LineItem.PAYABLES: round(cost * rates[LineItem.PAYABLES] / DAYS_PER_YEAR),
        LineItem.ACCRUED: round(revenue * rates[LineItem.ACCRUED]),
    }


def _with_totals(balances: dict[LineItem, int]) -> dict[LineItem, int]:
    current_assets = sum(
        balances[item]
        for item in (
            LineItem.CASH,
            LineItem.SHORT_TERM_INVESTMENTS,
            LineItem.RECEIVABLES,
            LineItem.INVENTORIES,
            LineItem.PREPAID,
        )
    )
    total_assets = (
        current_assets + balances[LineItem.PPE] + balances[LineItem.GOODWILL] + balances[LineItem.OTHER_ASSETS]
    )
    current_liabilities = balances[LineItem.PAYABLES] + balances[LineItem.ACCRUED] + balances[LineItem.CURRENT_DEBT]
    total_liabilities = current_liabilities + balances[LineItem.LONG_TERM_DEBT] + balances[LineItem.OTHER_LIABILITIES]
    equity = (
        balances[LineItem.PAID_IN_CAPITAL] + balances[LineItem.RETAINED_EARNINGS] + balances[LineItem.TREASURY_STOCK]
    )
    return {
        **balances,
        LineItem.CURRENT_ASSETS: current_assets,
        LineItem.TOTAL_ASSETS: total_assets,
        LineItem.CURRENT_LIABILITIES: current_liabilities,
        LineItem.TOTAL_LIABILITIES: total_liabilities,
        LineItem.TOTAL_EQUITY: equity,
        LineItem.TOTAL_LIABILITIES_AND_EQUITY: total_liabilities + equity,
    }


def _simulate(industry: Industry, units: Units, last_year: int, rng: random.Random) -> dict[int, dict[LineItem, int]]:
    revenue = round(rng.uniform(80_000, 4_000_000) if units is Units.THOUSANDS else rng.uniform(400, 60_000))
    gross_margin = _uniform(rng, industry.gross_margin)
    operating_margin = _uniform(rng, industry.operating_margin)
    rnd_share = _uniform(rng, industry.rnd_share_of_opex)
    rates = {
        LineItem.RECEIVABLES: _uniform(rng, industry.receivable_days),
        LineItem.INVENTORIES: _uniform(rng, industry.inventory_days),
        LineItem.PREPAID: rng.uniform(0.01, 0.03),
        LineItem.PAYABLES: _uniform(rng, industry.payable_days),
        LineItem.ACCRUED: rng.uniform(0.04, 0.12),
    }
    capex_share = _uniform(rng, industry.capex_share)
    stock_compensation_share = _uniform(rng, industry.stock_compensation_share)
    depreciation_rate = rng.uniform(0.08, 0.15)
    investment_share = rng.choice((0.0, rng.uniform(0.02, 0.15)))
    interest_rate = rng.uniform(0.03, 0.06)
    tax_rate = rng.uniform(0.19, 0.26)
    payout = rng.choice((0.0, rng.uniform(0.2, 0.5)))
    current_debt_share = rng.uniform(0.05, 0.15)
    minimum_cash_share = rng.uniform(0.03, 0.06)
    target_cash_share = rng.uniform(0.08, 0.2)
    repurchase_share = rng.choice((0.0, rng.uniform(0.5, 1.0)))

    cost = round(revenue * (1 - gross_margin))
    debt = round(revenue * rng.uniform(0.0, 0.35))
    current_debt = round(debt * current_debt_share)
    opening = {
        LineItem.CASH: round(revenue * rng.uniform(0.05, 0.2)),
        LineItem.SHORT_TERM_INVESTMENTS: round(revenue * investment_share),
        **_operating_balances(revenue, cost, rates),
        LineItem.PPE: round(revenue * rng.uniform(0.15, 0.6)),
        LineItem.GOODWILL: round(revenue * rng.choice((0.0, rng.uniform(0.05, 0.4)))),
        LineItem.OTHER_ASSETS: round(revenue * rng.uniform(0.02, 0.08)),
        LineItem.CURRENT_DEBT: current_debt,
        LineItem.LONG_TERM_DEBT: debt - current_debt,
        LineItem.OTHER_LIABILITIES: round(revenue * rng.uniform(0.02, 0.08)),
        LineItem.PAID_IN_CAPITAL: round(revenue * rng.uniform(0.05, 0.2)),
        LineItem.RETAINED_EARNINGS: 0,
        LineItem.TREASURY_STOCK: 0,
    }
    # Retained earnings is the plug that balances the opening balance sheet; later years roll it forward.
    unbalanced = _with_totals(opening)
    opening[LineItem.RETAINED_EARNINGS] = unbalanced[LineItem.TOTAL_ASSETS] - unbalanced[LineItem.TOTAL_LIABILITIES]
    opening[LineItem.RETAINED_EARNINGS] -= opening[LineItem.PAID_IN_CAPITAL]
    years = {last_year - 3: _with_totals(opening)}

    for year in range(last_year - 2, last_year + 1):
        prior = years[year - 1]
        growth = rng.uniform(0.03, 0.25) if rng.random() < 0.75 else -rng.uniform(0.03, 0.12)
        revenue = round(revenue * (1 + growth))
        cost = round(revenue * (1 - gross_margin - rng.uniform(-0.02, 0.02)))
        gross_profit = revenue - cost
        operating_income = round(revenue * (operating_margin + rng.uniform(-0.015, 0.015)))
        operating_expenses = gross_profit - operating_income
        rnd = round(operating_expenses * rnd_share)
        interest = round((prior[LineItem.CURRENT_DEBT] + prior[LineItem.LONG_TERM_DEBT]) * interest_rate)
        pretax = operating_income - interest
        tax = round(pretax * tax_rate)
        net_income = pretax - tax

        balances = _operating_balances(revenue, cost, rates)
        depreciation = round(prior[LineItem.PPE] * depreciation_rate)
        capex = round(revenue * capex_share)
        stock_compensation = round(revenue * stock_compensation_share)
        investments = round(revenue * investment_share)
        changes = {item: balances[item] - prior[item] for item in balances}
        operating_cash_flow = (
            net_income
            + depreciation
            + stock_compensation
            - changes[LineItem.RECEIVABLES]
            - changes[LineItem.INVENTORIES]
            - changes[LineItem.PREPAID]
            + changes[LineItem.PAYABLES]
            + changes[LineItem.ACCRUED]
        )
        investment_purchases = investments - prior[LineItem.SHORT_TERM_INVESTMENTS]
        dividends = round(max(net_income, 0) * payout)
        repayments = prior[LineItem.CURRENT_DEBT]
        cash_before_borrowing = (
            prior[LineItem.CASH] + operating_cash_flow - capex - investment_purchases - dividends - repayments
        )
        # Borrow only what keeps cash at the company's minimum operating balance; buybacks spend part of
        # the cash above a target balance, so liquidity ratios do not grow without bound.
        proceeds = max(0, round(revenue * minimum_cash_share) - cash_before_borrowing)
        repurchases = round(max(0, cash_before_borrowing - round(revenue * target_cash_share)) * repurchase_share)
        debt = prior[LineItem.CURRENT_DEBT] + prior[LineItem.LONG_TERM_DEBT] - repayments + proceeds
        current_debt = round(debt * current_debt_share)
        cash = cash_before_borrowing + proceeds - repurchases
        years[year] = {
            LineItem.REVENUE: revenue,
            LineItem.COST_OF_REVENUE: cost,
            LineItem.GROSS_PROFIT: gross_profit,
            LineItem.SGA: operating_expenses - rnd,
            LineItem.RND: rnd,
            LineItem.OPERATING_INCOME: operating_income,
            LineItem.INTEREST_EXPENSE: interest,
            LineItem.PRETAX_INCOME: pretax,
            LineItem.INCOME_TAX: tax,
            LineItem.NET_INCOME: net_income,
            **_with_totals(
                {
                    LineItem.CASH: cash,
                    LineItem.SHORT_TERM_INVESTMENTS: investments,
                    **balances,
                    LineItem.PPE: prior[LineItem.PPE] + capex - depreciation,
                    LineItem.GOODWILL: prior[LineItem.GOODWILL],
                    LineItem.OTHER_ASSETS: prior[LineItem.OTHER_ASSETS],
                    LineItem.CURRENT_DEBT: current_debt,
                    LineItem.LONG_TERM_DEBT: debt - current_debt,
                    LineItem.OTHER_LIABILITIES: prior[LineItem.OTHER_LIABILITIES],
                    LineItem.PAID_IN_CAPITAL: prior[LineItem.PAID_IN_CAPITAL] + stock_compensation,
                    LineItem.RETAINED_EARNINGS: prior[LineItem.RETAINED_EARNINGS] + net_income - dividends,
                    LineItem.TREASURY_STOCK: prior[LineItem.TREASURY_STOCK] - repurchases,
                }
            ),
            LineItem.DEPRECIATION: depreciation,
            LineItem.STOCK_COMPENSATION: stock_compensation,
            LineItem.CHANGE_RECEIVABLES: -changes[LineItem.RECEIVABLES],
            LineItem.CHANGE_INVENTORIES: -changes[LineItem.INVENTORIES],
            LineItem.CHANGE_PREPAID: -changes[LineItem.PREPAID],
            LineItem.CHANGE_PAYABLES: changes[LineItem.PAYABLES],
            LineItem.CHANGE_ACCRUED: changes[LineItem.ACCRUED],
            LineItem.OPERATING_CASH_FLOW: operating_cash_flow,
            LineItem.CAPEX: -capex,
            LineItem.INVESTMENT_PURCHASES: -investment_purchases,
            LineItem.INVESTING_CASH_FLOW: -capex - investment_purchases,
            LineItem.DEBT_PROCEEDS: proceeds,
            LineItem.DEBT_REPAYMENTS: -repayments,
            LineItem.DIVIDENDS: -dividends,
            LineItem.REPURCHASES: -repurchases,
            LineItem.FINANCING_CASH_FLOW: proceeds - repayments - dividends - repurchases,
            LineItem.ENDING_CASH: cash,
        }
    return years


def generate_company(rng: random.Random) -> Company:
    """Draw one synthetic company with internally consistent statements."""
    industry = rng.choice(INDUSTRIES)
    units = rng.choice(tuple(Units))
    last_year = rng.randint(2019, 2025)
    values = _simulate(industry, units, last_year, rng)
    return Company(
        name=f"{rng.choice(NAME_ROOTS)} {rng.choice(industry.name_words)} {rng.choice(NAME_SUFFIXES)}",
        industry=industry,
        units=units,
        table_style=rng.choice(tuple(TableStyle)),
        year_label=rng.choice(YEAR_LABELS),
        fiscal_year_end=rng.choice(FISCAL_YEAR_ENDS),
        labels={item: rng.choice(variants) for item, variants in LABEL_VARIANTS.items()},
        values=values,
        revolver_capacity=round(values[last_year][LineItem.REVENUE] * rng.uniform(0.05, 0.25), -2),
    )


def _amount(value: int) -> str:
    return f"({-value:,})" if value < 0 else f"{value:,}"


def _table(company: Company, items: tuple[LineItem, ...], headers: list[str], years: list[int]) -> str:
    rows = [
        [company.labels[item], *(_amount(company.value(item, year)) for year in years)]
        for item in items
        if any(company.value(item, year) for year in years)
    ]
    if company.table_style is TableStyle.MARKDOWN:
        lines = [f"| | {' | '.join(headers)} |", f"|---|{'---:|' * len(headers)}"]
        lines += [f"| {' | '.join(row)} |" for row in rows]
        return "\n".join(lines)
    label_width = max(len(row[0]) for row in rows)
    width = max(len(cell) for row in [headers, *(row[1:] for row in rows)] for cell in row) + 2
    lines = [" " * label_width + "".join(header.rjust(width) for header in headers)]
    lines += [row[0].ljust(label_width) + "".join(cell.rjust(width) for cell in row[1:]) for row in rows]
    return "\n".join(lines)


def _dollars(company: Company, value: int) -> str:
    """Describe an amount the way MD&A prose does, in rounded millions or billions."""
    millions = value / 1000 if company.units is Units.THOUSANDS else value
    return f"${millions / 1000:.2f} billion" if millions >= 1000 else f"${millions:.1f} million"


def _mdna(company: Company) -> str:
    year, prior = company.reported_years[:2]
    revenue, prior_revenue = company.value(LineItem.REVENUE, year), company.value(LineItem.REVENUE, prior)
    margin = 100 * company.value(LineItem.GROSS_PROFIT, year) / revenue
    prior_margin = 100 * company.value(LineItem.GROSS_PROFIT, prior) / prior_revenue
    direction = "increased" if revenue > prior_revenue else "decreased"
    liquidity = company.value(LineItem.CASH, year) + company.value(LineItem.SHORT_TERM_INVESTMENTS, year)
    driver = company.industry.growth_drivers[year % len(company.industry.growth_drivers)]
    return (
        f"{company.labels[LineItem.REVENUE]} {direction} to {_dollars(company, revenue)} in "
        f"{company.period(year)} from {_dollars(company, prior_revenue)} in {company.period(prior)}, "
        f"reflecting changes in {driver}. Gross margin was {margin:.1f}% of {company.labels[LineItem.REVENUE].lower()}, "
        f"compared with {prior_margin:.1f}% in the prior year. As of {company.year_end(year)}, we held "
        f"{_dollars(company, liquidity)} of cash, cash equivalents and short-term investments and had "
        f"{_dollars(company, company.revolver_capacity)} of undrawn capacity under our revolving credit facility."
    )


def render_filing(company: Company) -> str:
    """Render the company's statements and MD&A as a 10-K excerpt."""
    reported = company.reported_years
    balance_years = company.balance_sheet_years
    latest = reported[0]
    units = f"(in {company.units}, except where noted)"
    return "\n\n".join(
        [
            f"The following excerpts are from the annual report on Form 10-K of {company.name}, "
            f"{company.industry.description}, for {company.period(latest)}. Fiscal year {latest} ended "
            f"{company.year_end(latest)}. Amounts in the tables are in {company.units} of U.S. dollars.",
            f"CONSOLIDATED STATEMENTS OF OPERATIONS {units}",
            _table(company, INCOME_STATEMENT, [company.period(year) for year in reported], reported),
            f"CONSOLIDATED BALANCE SHEETS {units}",
            _table(company, BALANCE_SHEET, [company.year_end(year) for year in balance_years], balance_years),
            f"CONSOLIDATED STATEMENTS OF CASH FLOWS {units}",
            _table(company, CASH_FLOW_STATEMENT, [company.period(year) for year in reported], reported),
            "MANAGEMENT'S DISCUSSION AND ANALYSIS (excerpt)",
            _mdna(company),
        ]
    )


class Template(StrEnum):
    OPERATING_MARGIN = "operating_margin"
    OPERATING_MARGIN_ABOVE = "operating_margin_above"
    RETURN_ON_ASSETS = "return_on_assets"
    CAPEX_TO_REVENUE = "capex_to_revenue"
    CAPEX_INTENSITY_ABOVE = "capex_intensity_above"
    YEAR_OVER_YEAR_GROWTH = "year_over_year_growth"
    CAPEX_AMOUNT = "capex_amount"
    FREE_CASH_FLOW = "free_cash_flow"
    TOTAL_DEBT = "total_debt"
    DIVIDENDS_PAID = "dividends_paid"
    CURRENT_RATIO = "current_ratio"
    CURRENT_RATIO_ABOVE = "current_ratio_above"
    QUICK_RATIO_ABOVE = "quick_ratio_above"
    DAYS_PAYABLE_OUTSTANDING = "days_payable_outstanding"
    INVENTORY_TURNOVER = "inventory_turnover"
    OPERATING_CASH_FLOW_RATIO = "operating_cash_flow_ratio"
    CASH_CONVERSION_CYCLE = "cash_conversion_cycle"


@dataclass(frozen=True)
class Question:
    template: Template
    fiscal_year: int
    text: str
    answer: str
    """The exact rounded number the question asks for."""
    yes_no: str | None = None


def rounded(value: Fraction, places: int) -> str:
    """Round half away from zero to ``places`` decimals."""
    exact = Decimal(value.numerator) / Decimal(value.denominator)
    return str(exact.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


def _percent(numerator: int, denominator: int) -> Fraction:
    return Fraction(100 * numerator, denominator)


def _millions(company: Company, value: int) -> Fraction:
    return Fraction(value, 1000) if company.units is Units.THOUSANDS else Fraction(value)


def _average(company: Company, item: LineItem, year: int) -> Fraction:
    return Fraction(company.value(item, year) + company.value(item, year - 1), 2)


PERCENT_INSTRUCTION = "Express it as a percentage rounded to one decimal place (write 12.3 for 12.3%)."
MILLIONS_INSTRUCTION = "Give the amount in millions of U.S. dollars, rounded to two decimal places."
AVERAGE_NOTE = "Average balances are the mean of the balances at the start and end of the fiscal year."


def _yes_no_question(
    template: Template,
    year: int,
    value: Fraction,
    places: int,
    thresholds: tuple[float, ...],
    ask: str,
    rng: random.Random,
) -> Question | None:
    # Take the conventional threshold just below or just above the value, so yes and no are both common.
    below = [threshold for threshold in thresholds if threshold < value]
    above = [threshold for threshold in thresholds if threshold > value]
    threshold = rng.choice([side for side in (below[-1:], above[:1]) if side])[0]
    answer = rounded(value, places)
    # A value that rounds onto the threshold makes the yes/no part ambiguous.
    if Decimal(answer) == Decimal(str(threshold)):
        return None
    yes_no = "yes" if value > Fraction(str(threshold)) else "no"
    text = ask.format(threshold=threshold) + f" Answer yes or no, then give the value. {BOXED_INSTRUCTION}"
    return Question(template, year, text, answer, yes_no)


def _operating_margin(company: Company, rng: random.Random) -> Question:
    year = rng.choice(company.reported_years)
    margin = _percent(company.value(LineItem.OPERATING_INCOME, year), company.value(LineItem.REVENUE, year))
    text = (
        f"What was {company.name}'s operating margin in {company.period(year)}? Define operating margin as "
        f"operating income divided by total revenue. {PERCENT_INSTRUCTION} {BOXED_INSTRUCTION}"
    )
    return Question(Template.OPERATING_MARGIN, year, text, rounded(margin, 1))


def _operating_margin_above(company: Company, rng: random.Random) -> Question | None:
    year = rng.choice(company.reported_years)
    margin = _percent(company.value(LineItem.OPERATING_INCOME, year), company.value(LineItem.REVENUE, year))
    return _yes_no_question(
        Template.OPERATING_MARGIN_ABOVE,
        year,
        margin,
        1,
        (5, 10, 15, 20, 25, 30),
        f"Did {company.name} earn an operating margin (operating income / total revenue) above {{threshold}}% in "
        f"{company.period(year)}? Report the margin as a percentage rounded to one decimal place.",
        rng,
    )


def _return_on_assets(company: Company, rng: random.Random) -> Question:
    year = company.reported_years[0]
    value = 100 * company.value(LineItem.NET_INCOME, year) / _average(company, LineItem.TOTAL_ASSETS, year)
    text = (
        f"Calculate {company.name}'s return on assets for {company.period(year)}, defined as net income divided "
        f"by average total assets. {AVERAGE_NOTE} {PERCENT_INSTRUCTION} {BOXED_INSTRUCTION}"
    )
    return Question(Template.RETURN_ON_ASSETS, year, text, rounded(value, 1))


def _capex_to_revenue(company: Company, rng: random.Random) -> Question:
    year = rng.choice(company.reported_years)
    value = _percent(-company.value(LineItem.CAPEX, year), company.value(LineItem.REVENUE, year))
    text = (
        f"What was {company.name}'s capital expenditure as a share of revenue in {company.period(year)}? Use "
        f"purchases of property and equipment from the cash-flow statement as capital expenditure. "
        f"{PERCENT_INSTRUCTION} {BOXED_INSTRUCTION}"
    )
    return Question(Template.CAPEX_TO_REVENUE, year, text, rounded(value, 1))


def _capex_intensity_above(company: Company, rng: random.Random) -> Question | None:
    year = rng.choice(company.reported_years)
    value = _percent(-company.value(LineItem.CAPEX, year), company.value(LineItem.REVENUE, year))
    return _yes_no_question(
        Template.CAPEX_INTENSITY_ABOVE,
        year,
        value,
        1,
        (2, 3, 5, 8, 10, 12),
        f"Treat a business as capital-intensive when capital expenditure (purchases of property and equipment) "
        f"exceeds {{threshold}}% of revenue. Was {company.name} capital-intensive in {company.period(year)}? "
        "Report capital expenditure as a percentage of revenue, rounded to one decimal place.",
        rng,
    )


def _year_over_year_growth(company: Company, rng: random.Random) -> Question | None:
    year = rng.choice(company.reported_years[:2])
    item = rng.choice((LineItem.REVENUE, LineItem.OPERATING_INCOME, LineItem.NET_INCOME))
    current, prior = company.value(item, year), company.value(item, year - 1)
    if prior <= 0:
        return None
    growth = _percent(current - prior, prior)
    # Near-zero growth makes the relative answer tolerance meaningless after rounding.
    if abs(growth) < 2:
        return None
    name = {
        LineItem.REVENUE: "total revenue",
        LineItem.OPERATING_INCOME: "operating income",
        LineItem.NET_INCOME: "net income",
    }[item]
    text = (
        f"By what percentage did {company.name}'s {name} change from {company.period(year - 1)} to "
        f"{company.period(year)}? Compute (current year - prior year) / prior year; use a negative number for "
        f"a decline. {PERCENT_INSTRUCTION} {BOXED_INSTRUCTION}"
    )
    return Question(Template.YEAR_OVER_YEAR_GROWTH, year, text, rounded(growth, 1))


def _capex_amount(company: Company, rng: random.Random) -> Question:
    year = rng.choice(company.reported_years)
    text = (
        f"How much did {company.name} spend on purchases of property and equipment in {company.period(year)}? "
        f"Report the spending as a positive amount. {MILLIONS_INSTRUCTION} {BOXED_INSTRUCTION}"
    )
    return Question(
        Template.CAPEX_AMOUNT, year, text, rounded(_millions(company, -company.value(LineItem.CAPEX, year)), 2)
    )


def _free_cash_flow(company: Company, rng: random.Random) -> Question | None:
    year = rng.choice(company.reported_years)
    operating = company.value(LineItem.OPERATING_CASH_FLOW, year)
    free = operating + company.value(LineItem.CAPEX, year)
    if abs(free) < 0.05 * abs(operating):
        return None
    text = (
        f"What was {company.name}'s free cash flow in {company.period(year)}? Define free cash flow as net cash "
        "provided by operating activities minus purchases of property and equipment; use a negative number if it "
        f"is negative. {MILLIONS_INSTRUCTION} {BOXED_INSTRUCTION}"
    )
    return Question(Template.FREE_CASH_FLOW, year, text, rounded(_millions(company, free), 2))


def _total_debt(company: Company, rng: random.Random) -> Question | None:
    year = rng.choice(company.balance_sheet_years)
    debt = company.value(LineItem.CURRENT_DEBT, year) + company.value(LineItem.LONG_TERM_DEBT, year)
    if debt == 0:
        return None
    text = (
        f"What was {company.name}'s total debt at the end of {company.period(year)}? Define total debt as the "
        f"current portion of long-term debt plus long-term debt, net of current portion. {MILLIONS_INSTRUCTION} "
        f"{BOXED_INSTRUCTION}"
    )
    return Question(Template.TOTAL_DEBT, year, text, rounded(_millions(company, debt), 2))


def _dividends_paid(company: Company, rng: random.Random) -> Question:
    year = rng.choice(company.reported_years)
    dividends = -company.value(LineItem.DIVIDENDS, year)
    text = (
        f"Did {company.name} pay cash dividends to shareholders in {company.period(year)}? Answer yes or no, then "
        f"give the total dividends paid as a positive amount (0 if none). {MILLIONS_INSTRUCTION} "
        f"{BOXED_INSTRUCTION}"
    )
    return Question(
        Template.DIVIDENDS_PAID, year, text, rounded(_millions(company, dividends), 2), "yes" if dividends else "no"
    )


def _current_ratio_value(company: Company, year: int) -> Fraction:
    return Fraction(company.value(LineItem.CURRENT_ASSETS, year), company.value(LineItem.CURRENT_LIABILITIES, year))


def _current_ratio(company: Company, rng: random.Random) -> Question:
    year = rng.choice(company.balance_sheet_years)
    text = (
        f"What was {company.name}'s current ratio at the end of {company.period(year)}? Define the current ratio as "
        f"total current assets divided by total current liabilities, rounded to two decimal places. "
        f"{BOXED_INSTRUCTION}"
    )
    return Question(Template.CURRENT_RATIO, year, text, rounded(_current_ratio_value(company, year), 2))


def _current_ratio_above(company: Company, rng: random.Random) -> Question | None:
    year = rng.choice(company.balance_sheet_years)
    return _yes_no_question(
        Template.CURRENT_RATIO_ABOVE,
        year,
        _current_ratio_value(company, year),
        2,
        (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0),
        f"A lender requires a current ratio (total current assets / total current liabilities) above {{threshold}}. "
        f"Did {company.name} meet this requirement at the end of {company.period(year)}? Report the current ratio "
        "rounded to two decimal places.",
        rng,
    )


def _quick_ratio_above(company: Company, rng: random.Random) -> Question | None:
    year = rng.choice(company.balance_sheet_years)
    quick_assets = sum(
        company.value(item, year) for item in (LineItem.CASH, LineItem.SHORT_TERM_INVESTMENTS, LineItem.RECEIVABLES)
    )
    return _yes_no_question(
        Template.QUICK_RATIO_ABOVE,
        year,
        Fraction(quick_assets, company.value(LineItem.CURRENT_LIABILITIES, year)),
        2,
        (0.5, 0.75, 1.0, 1.5, 2.0, 3.0),
        f"Did {company.name}'s quick ratio exceed {{threshold}} at the end of {company.period(year)}? Define the "
        "quick ratio as (cash and cash equivalents + short-term investments + accounts receivable, net) divided by "
        "total current liabilities; exclude inventories and prepaid or other current assets. Report the quick "
        "ratio rounded to two decimal places.",
        rng,
    )


def _days_payable_outstanding(company: Company, rng: random.Random) -> Question:
    year = company.reported_years[0]
    payables = _average(company, LineItem.PAYABLES, year)
    cost = company.value(LineItem.COST_OF_REVENUE, year)
    # FinanceBench's definition uses purchases (cost of goods sold plus the change in inventory) as the base.
    if rng.random() < 0.5:
        base = cost
        base_text = "cost of goods sold"
    else:
        base = cost + company.value(LineItem.INVENTORIES, year) - company.value(LineItem.INVENTORIES, year - 1)
        base_text = "(cost of goods sold + ending inventory - beginning inventory)"
    text = (
        f"Calculate {company.name}'s days payable outstanding for {company.period(year)} as "
        f"{DAYS_PER_YEAR} x average accounts payable / {base_text}. {AVERAGE_NOTE} Round to one decimal place. "
        f"{BOXED_INSTRUCTION}"
    )
    return Question(Template.DAYS_PAYABLE_OUTSTANDING, year, text, rounded(DAYS_PER_YEAR * payables / base, 1))


def _inventory_turnover(company: Company, rng: random.Random) -> Question | None:
    year = company.reported_years[0]
    inventory = _average(company, LineItem.INVENTORIES, year)
    if inventory == 0:
        return None
    text = (
        f"What was {company.name}'s inventory turnover in {company.period(year)}? Define inventory turnover as cost "
        f"of goods sold divided by average inventory. {AVERAGE_NOTE} Round to two decimal places. "
        f"{BOXED_INSTRUCTION}"
    )
    value = company.value(LineItem.COST_OF_REVENUE, year) / inventory
    return Question(Template.INVENTORY_TURNOVER, year, text, rounded(value, 2))


def _operating_cash_flow_ratio(company: Company, rng: random.Random) -> Question:
    year = rng.choice(company.balance_sheet_years)
    value = Fraction(
        company.value(LineItem.OPERATING_CASH_FLOW, year), company.value(LineItem.CURRENT_LIABILITIES, year)
    )
    text = (
        f"What was {company.name}'s operating cash flow ratio for {company.period(year)}? Define it as net cash "
        "provided by operating activities divided by total current liabilities at the end of the fiscal year, "
        f"rounded to two decimal places. {BOXED_INSTRUCTION}"
    )
    return Question(Template.OPERATING_CASH_FLOW_RATIO, year, text, rounded(value, 2))


def _cash_conversion_cycle(company: Company, rng: random.Random) -> Question | None:
    year = company.reported_years[0]
    inventory = _average(company, LineItem.INVENTORIES, year)
    if inventory == 0:
        return None
    revenue, cost = company.value(LineItem.REVENUE, year), company.value(LineItem.COST_OF_REVENUE, year)
    receivable_days = DAYS_PER_YEAR * _average(company, LineItem.RECEIVABLES, year) / revenue
    inventory_days = DAYS_PER_YEAR * inventory / cost
    payable_days = DAYS_PER_YEAR * _average(company, LineItem.PAYABLES, year) / cost
    cycle = receivable_days + inventory_days - payable_days
    if abs(cycle) < 10:
        return None
    text = (
        f"Calculate {company.name}'s cash conversion cycle for {company.period(year)} as days sales outstanding + "
        f"days inventory outstanding - days payable outstanding, where DSO = {DAYS_PER_YEAR} x average accounts "
        f"receivable / revenue, DIO = {DAYS_PER_YEAR} x average inventory / cost of goods sold, and DPO = "
        f"{DAYS_PER_YEAR} x average accounts payable / cost of goods sold. {AVERAGE_NOTE} Round the cycle to one "
        f"decimal place. {BOXED_INSTRUCTION}"
    )
    return Question(Template.CASH_CONVERSION_CYCLE, year, text, rounded(cycle, 1))


TEMPLATES: dict[Template, Callable[[Company, random.Random], Question | None]] = {
    Template.OPERATING_MARGIN: _operating_margin,
    Template.OPERATING_MARGIN_ABOVE: _operating_margin_above,
    Template.RETURN_ON_ASSETS: _return_on_assets,
    Template.CAPEX_TO_REVENUE: _capex_to_revenue,
    Template.CAPEX_INTENSITY_ABOVE: _capex_intensity_above,
    Template.YEAR_OVER_YEAR_GROWTH: _year_over_year_growth,
    Template.CAPEX_AMOUNT: _capex_amount,
    Template.FREE_CASH_FLOW: _free_cash_flow,
    Template.TOTAL_DEBT: _total_debt,
    Template.DIVIDENDS_PAID: _dividends_paid,
    Template.CURRENT_RATIO: _current_ratio,
    Template.CURRENT_RATIO_ABOVE: _current_ratio_above,
    Template.QUICK_RATIO_ABOVE: _quick_ratio_above,
    Template.DAYS_PAYABLE_OUTSTANDING: _days_payable_outstanding,
    Template.INVENTORY_TURNOVER: _inventory_turnover,
    Template.OPERATING_CASH_FLOW_RATIO: _operating_cash_flow_ratio,
    Template.CASH_CONVERSION_CYCLE: _cash_conversion_cycle,
}

CAPABILITY_TEMPLATES: dict[str, tuple[Template, ...]] = {
    REPORTING_ANALYSIS: (
        Template.OPERATING_MARGIN,
        Template.OPERATING_MARGIN_ABOVE,
        Template.RETURN_ON_ASSETS,
        Template.CAPEX_TO_REVENUE,
        Template.CAPEX_INTENSITY_ABOVE,
        Template.YEAR_OVER_YEAR_GROWTH,
    ),
    REPORTING_DISCLOSURES: (
        Template.CAPEX_AMOUNT,
        Template.FREE_CASH_FLOW,
        Template.TOTAL_DEBT,
        Template.DIVIDENDS_PAID,
    ),
    WORKING_CAPITAL: (
        Template.CURRENT_RATIO,
        Template.CURRENT_RATIO_ABOVE,
        Template.QUICK_RATIO_ABOVE,
        Template.DAYS_PAYABLE_OUTSTANDING,
        Template.INVENTORY_TURNOVER,
        Template.OPERATING_CASH_FLOW_RATIO,
        Template.CASH_CONVERSION_CYCLE,
    ),
}


def finance_problem(capability_id: str, index: int, seed: int) -> dict[str, Any]:
    """Build problem ``index`` for a capability; templates cycle so every template gets equal coverage."""
    templates = CAPABILITY_TEMPLATES[capability_id]
    template = templates[index % len(templates)]
    rng = random.Random(f"{seed}/{capability_id}/{index}")
    # Some companies cannot support a template (no inventory, near-zero change); draw until one does.
    while True:
        company = generate_company(rng)
        question = TEMPLATES[template](company, rng)
        if question is not None:
            break
    return {
        "request_id": f"problem-{index:05d}",
        "capability_id": capability_id,
        "template": str(template),
        "fiscal_year": question.fiscal_year,
        "units": str(company.units),
        "yes_no": question.yes_no,
        "problem": f"{render_filing(company)}\n\nQuestion: {question.text}",
        "answer": question.answer,
        "accepted": True,
    }


@dataclass(frozen=True)
class FinanceProblemsConfig:
    output_path: str
    capability_id: str
    count: int
    seed: int


def write_finance_problems(config: FinanceProblemsConfig) -> Artifact:
    """Write the problems Parquet and a manifest."""
    rows = [finance_problem(config.capability_id, index, config.seed) for index in range(config.count)]
    output = StoragePath(config.output_path)
    output.mkdirs()
    write_table(output, PROBLEMS_FILENAME, rows, FINANCE_PROBLEM_SCHEMA)
    templates: dict[str, int] = {}
    for row in rows:
        templates[row["template"]] = templates.get(row["template"], 0) + 1
    manifest = {
        "capability_id": config.capability_id,
        "generator": "programmatic",
        "seed": config.seed,
        "problems": len(rows),
        "templates": templates,
        "verified": "reference answers are computed exactly from the rendered statements",
    }
    (output / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2) + "\n")
    logger.info("wrote %s finance problems for %s", len(rows), config.capability_id)
    return Artifact(path=config.output_path)


def generate_finance_problems(capability_id: str, *, version: str, count: int, seed: int) -> ArtifactStep[Artifact]:
    """Build one CPU step that writes ``count`` programmatic finance problems for a capability."""

    def build_config(ctx: StepContext) -> FinanceProblemsConfig:
        return FinanceProblemsConfig(output_path=ctx.output_path, capability_id=capability_id, count=count, seed=seed)

    return ArtifactStep(
        name=user_owned_name(f"documents/curriculum-sft/{capability_id}/finance-problems"),
        version=version,
        artifact_type=Artifact,
        run=write_finance_problems,
        build_config=build_config,
    )
