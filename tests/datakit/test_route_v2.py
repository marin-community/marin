# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of router v2: the draw, the preference label, the priced frontier, and the gates.

Router v2 differs from v1 in four places where a mistake would be silent rather than loud, and each
one is asserted here against a case constructed to have a known answer. The draw has to spread over
domains instead of deepening publishers, because domains are what a domain-disjoint split can spend.
The label table has to record the documents no judge saw, because a route that produced nothing is a
routing decision with a right answer. The frontier has to charge escalation per page and in CPU
core-hours, because pages are what the feed path and the model cost and CPU is what this cluster is
short of. And the legibility gate has to be arithmetic, because a page rendered below the floor
cannot be read by the VLM whatever a score says about it.
"""

import numpy as np
import polars as pl
import pytest

from experiments.datakit.build_pdf_source.ocr_extract.render import DEFAULT_MAX_VISUAL_TOKENS
from experiments.datakit.build_pdf_source.quality import route_v2_features as contract
from experiments.datakit.build_pdf_source.quality.analyze_route_v2 import (
    at_budget,
    clumping,
    confusion,
    frontier,
    knee,
)
from experiments.datakit.build_pdf_source.quality.build_inspector_output_study import measure
from experiments.datakit.build_pdf_source.quality.build_preference_set import (
    MAX_PER_DOMAIN,
    TARGET_DOCUMENTS,
    Outcome,
    informative_pages,
    select,
)
from experiments.datakit.build_pdf_source.quality.judge_preference_set import (
    ESCALATE_COLUMN,
    graded_target,
    label_table,
    prefers_vlm,
)

# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------


def pool(domains: int, per_domain: list[int]) -> pl.DataFrame:
    rows = [
        {
            "source_id": f"{index}:{depth}",
            "domain": f"d{index}",
            "num_pages": 4,
            "trustworthy": depth % 2 == 0,
            "mean_rtl_ratio": 0.0,
            "mean_cjk_ratio": 0.0,
            "inspector_ok": True,
            "docling_ok": True,
            "inspector_docling_bigram_recall_mean": 1.0,
            "mean_fonts_unmappable": 0.0,
            "mean_replacement_ratio": 0.0,
            "garbled_text_ratio": 0.0,
            "mean_math_unicode_ratio": 0.0,
            "mean_math_font_ratio": 0.0,
            "mean_column_count": 1.0,
            "inspector_extract_pages_with_tables": 0,
            "inspector_page_count": 4,
            "inspector_pdf_type": "text_based",
        }
        for index in range(domains)
        for depth in range(per_domain[index])
    ]
    return pl.DataFrame(rows)


def test_draw_takes_every_domain_before_deepening_any_of_them(monkeypatch):
    """A publisher with a thousand documents must not crowd out a publisher with one.

    Near-duplicates cluster by publisher, so a deep domain contributes almost no independent
    evidence past its first few documents, while a domain left out of the draw entirely costs the
    split a whole unit of sample size. The regression this guards is a plain subsample of the capped
    pool, which drops whole domains off its tail.
    """
    monkeypatch.setattr("experiments.datakit.build_pdf_source.quality.build_preference_set.TARGET_DOCUMENTS", 30)
    frame = pool(20, [50] * 5 + [1] * 15)

    drawn = select(frame, seed=7)

    assert drawn.height == 30
    assert drawn["domain"].n_unique() == 20, "every domain must appear before any domain gets a second slot"
    assert drawn.group_by("domain").len()["len"].max() <= MAX_PER_DOMAIN


def test_draw_respects_the_per_domain_cap_even_when_it_cannot_fill_the_target():
    """A pool smaller than the target is taken whole rather than topped up from one deep publisher."""
    frame = pool(4, [400, 400, 400, 400])

    drawn = select(frame, seed=7)

    assert drawn.height == 4 * MAX_PER_DOMAIN < TARGET_DOCUMENTS
    assert drawn.group_by("domain").len()["len"].to_list() == [MAX_PER_DOMAIN] * 4


def test_second_judge_subset_is_a_uniform_slice_of_the_draw():
    """Consistency is measured on a subsample of the draw, not on its head.

    The graded target is only meaningful if the documents carrying it look like the ones that do
    not, so the subset must not correlate with domain depth.
    """
    frame = pool(300, [8] * 300)

    drawn = select(frame, seed=11)
    chosen = drawn.filter(pl.col("second_judge"))

    assert chosen.height > 0
    assert chosen["domain"].n_unique() > 0.5 * drawn["domain"].n_unique()


# ---------------------------------------------------------------------------
# Page selection
# ---------------------------------------------------------------------------


def test_page_selection_leads_with_the_page_the_routes_disagree_on():
    """A judge shown three title pages adjudicates nothing.

    Selection ranks by pdf-inspector's recall against the VLM, so the page where it collapsed is
    first in the packet and a page they agree on comes along as a control.
    """
    vlm = ["Shared opening paragraph about alpha beta gamma", "Second page delta epsilon zeta eta"]
    vlm += ["Totally different content here: kappa lambda mu nu xi omicron"]
    inspector = [vlm[0], vlm[1], "!!!! ???? ####"]

    chosen = informative_pages(inspector, vlm, count=3)

    assert chosen[0].page_index >= 0
    ranked = sorted(chosen, key=lambda choice: choice.inspector_recall)
    assert ranked[0].page_index == 2, "the page pdf-inspector garbled must be in the packet"
    assert any(choice.reason == "control" for choice in chosen)


def test_page_selection_aligns_by_content_when_a_route_drops_a_page():
    """Pairing by index after a dropped page shows every later page beside its neighbour's text.

    That bug invalidated an earlier adjudication pass outright: judges reported one route
    "fabricating" content on documents where nothing had gone wrong.
    """
    vlm = ["blank cover", "alpha beta gamma delta", "epsilon zeta eta theta"]
    inspector = ["alpha beta gamma delta", "epsilon zeta eta theta"]

    chosen = {choice.page_index: choice.source_index["inspector"] for choice in informative_pages(inspector, vlm, 3)}

    assert chosen[1] == 0
    assert chosen[2] == 1


# ---------------------------------------------------------------------------
# The label
# ---------------------------------------------------------------------------


def entry(packet_id: str, outcome: Outcome, labels: dict[str, str] | None = None) -> dict:
    return {
        "packet_id": packet_id,
        "source_id": f"src-{packet_id}",
        "domain": "example.org",
        "stratum": "latin_text_baseline",
        "trustworthy": True,
        "outcome": str(outcome),
        "labels": labels or {"A": "vlm", "B": "inspector"},
    }


def verdict(packet_id: str, ranking: list[str], margin: str = "large") -> dict:
    return {"packet_id": packet_id, "verdict": {"ranking": ranking, "margin": margin}}


def test_blinding_is_read_back_through_the_key_rather_than_position():
    """The routes are shuffled per document, so the verdict means nothing without its own key."""
    vlm_first = entry("p1", Outcome.JUDGED, labels={"A": "vlm", "B": "inspector"})
    inspector_first = entry("p2", Outcome.JUDGED, labels={"A": "inspector", "B": "vlm"})

    assert prefers_vlm(vlm_first, {"ranking": ["A", "B"]}) is True
    assert prefers_vlm(inspector_first, {"ranking": ["A", "B"]}) is False


def test_routes_that_produced_nothing_become_labels_rather_than_gaps():
    """A route that lost the document is a routing decision with a known right answer.

    Dropping these rows would train a router that has never seen either failure, and one of them --
    the VLM's -- is 16.7% of the corpus and the single thing v1 had to keep as a separate gate.
    """
    entries = [
        entry("p1", Outcome.INSPECTOR_FAILED),
        entry("p2", Outcome.VLM_FAILED),
        entry("p3", Outcome.JUDGED),
    ]

    table = label_table(entries, {"p3": verdict("p3", ["A", "B"])}, {})
    by_packet = {row["packet_id"]: row for row in table.to_dicts()}

    assert by_packet["p1"][ESCALATE_COLUMN] is True, "no cheap route survived, so escalation is forced"
    assert by_packet["p2"][ESCALATE_COLUMN] is False, "the VLM lost this document; escalating buys nothing"
    assert by_packet["p3"][ESCALATE_COLUMN] is True
    assert by_packet["p1"]["label_source"] == "inspector_failed"


def test_graded_target_comes_from_two_judges_agreeing_not_from_a_claimed_margin():
    """Self-reported margin agreed with a human 0.22 of the time; cross-judge agreement is external.

    A document both judges call the same way keeps the full-confidence target; one they split on
    lands at 0.5, which is the honest statement that the evidence does not decide it.
    """
    assert graded_target(True, True) == 1.0
    assert graded_target(False, False) == 0.0
    assert graded_target(True, False) == 0.5
    assert graded_target(False, True) == 0.5
    assert graded_target(True, None) == 1.0, "one competent verdict is still the best estimate available"
    assert graded_target(None, None) is None


def test_margin_is_recorded_but_never_becomes_the_target():
    """The miscalibration has to stay visible in the table without leaking into the label."""
    entries = [entry("p1", Outcome.JUDGED)]

    table = label_table(entries, {"p1": verdict("p1", ["A", "B"], margin="none")}, {})

    assert table["margin"].to_list() == ["none"]
    assert table[ESCALATE_COLUMN].to_list() == [True], "a 'none' margin still carries its ranking"


# ---------------------------------------------------------------------------
# The priced frontier
# ---------------------------------------------------------------------------


def test_escalation_is_charged_per_page_not_per_document():
    """Page counts run p50 6, p90 38, p99 207, so a document budget is not a page budget.

    A router that escalates one 100-page report and one 1-page flyer has escalated 50% of documents
    and 99% of pages, and only the second number is money.
    """
    scores = np.array([0.9, 0.1])
    escalate = np.array([True, False])
    pages = np.array([100.0, 1.0])

    point = at_budget(frontier(scores, escalate, pages, router_core_hours=0.0, needs_inspector=True), 0.5)

    assert point.document_budget == pytest.approx(0.5)
    assert point.page_budget == pytest.approx(100 / 101, abs=1e-3)
    assert point.gpu_hours == pytest.approx(contract.VLM_GPU_HOURS * 100 / 101, rel=1e-3)


def test_a_router_that_needs_pdf_inspector_pays_for_it_on_escalated_pages_too():
    """Reading the extraction's own signals means the extraction cannot be skipped when escalating.

    That is the whole difference between a ~15.7 and a ~17.8 core-h marginal escalation cost, and it
    is the price of the "free" feature groups being free.
    """
    scores = np.array([1.0, 0.0])
    escalate = np.array([True, False])
    pages = np.array([1.0, 1.0])

    dependent = at_budget(frontier(scores, escalate, pages, 0.0, needs_inspector=True), 0.5)
    independent = at_budget(frontier(scores, escalate, pages, 0.0, needs_inspector=False), 0.5)

    assert dependent.cpu_core_hours > independent.cpu_core_hours
    assert dependent.cpu_core_hours - independent.cpu_core_hours == pytest.approx(
        0.5 * contract.INSPECTOR_CORE_HOURS, rel=1e-6
    )


def test_a_perfect_score_removes_all_quality_loss_at_its_own_base_rate():
    """The frontier's shape is only meaningful if a perfect ranker reaches zero where it should."""
    escalate = np.array([True] * 30 + [False] * 70)
    scores = escalate.astype(float)
    pages = np.ones(100)

    points = frontier(scores, escalate, pages, 0.0, needs_inspector=True)
    perfect = at_budget(points, 0.30)

    assert perfect.quality_loss_pages == pytest.approx(0.0, abs=1e-9)
    assert perfect.recall_of_escalations == pytest.approx(1.0)
    assert perfect.wasted_escalation == pytest.approx(0.0, abs=1e-9)


def test_cost_rises_and_quality_loss_falls_as_more_is_escalated():
    """A frontier that is not monotone in cost is a bug in the sweep, not a finding."""
    rng = np.random.default_rng(3)
    escalate = rng.random(400) < 0.4
    scores = rng.random(400) + escalate * 0.5
    pages = rng.integers(1, 40, 400).astype(float)

    points = frontier(scores, escalate, pages, contract.ROUTE_FEATURES_CORE_HOURS, needs_inspector=True)
    costs = [point.cpu_core_hours for point in points]
    losses = [point.quality_loss_pages for point in points]

    assert costs == sorted(costs)
    assert losses == sorted(losses, reverse=True)
    assert min(costs) >= contract.INSPECTOR_CORE_HOURS + contract.ROUTE_FEATURES_CORE_HOURS


def test_clumping_reports_a_rule_that_cannot_rank():
    """Two prior scores failed here: 91.3% tied at 0.0, and 17.4% pinned at exactly 1.0.

    A frontier drawn over a clumped score is a fiction inside the clump, so every arm has to declare
    this rather than hide behind an AUC.
    """
    degenerate = np.array([0.0] * 913 + list(np.linspace(0.1, 1.0, 87)))
    spread = np.linspace(0.0, 1.0, 1000)

    assert clumping(degenerate)["largest_clump_share"] == pytest.approx(0.913)
    assert clumping(spread)["largest_clump_share"] == pytest.approx(0.001)
    assert clumping(spread)["tied_share"] == pytest.approx(0.0)


def test_knee_is_read_on_the_cost_axis_the_router_is_bought_on():
    """A curve that bends early in documents can bend late in core-hours, and CPU is what is scarce."""
    rng = np.random.default_rng(5)
    escalate = rng.random(500) < 0.35
    scores = rng.random(500) + escalate * 0.6
    pages = np.ones(500)

    points = frontier(scores, escalate, pages, 0.0, needs_inspector=True)
    bend = knee(points)

    assert 0.0 < bend.document_budget < 1.0
    assert bend.cpu_core_hours > contract.INSPECTOR_CORE_HOURS


def test_confusion_separates_the_recoverable_error_from_the_silent_one():
    """Escalating needlessly costs CPU and is recoverable; keeping a bad document is silent damage."""
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    escalate = np.array([True, False, True, False])
    pages = np.ones(4)

    counts = confusion(frontier(scores, escalate, pages, 0.0, True), scores, escalate, budget=0.5)

    assert counts["escalated_correctly"] == 1
    assert counts["escalated_wastefully"] == 1
    assert counts["kept_and_degraded"] == 1
    assert counts["kept_correctly"] == 1


# ---------------------------------------------------------------------------
# The arithmetic gates and the feature contract
# ---------------------------------------------------------------------------


def test_legibility_gate_tracks_the_budget_it_is_asked_about():
    """DPI scales with the square root of the visual-token budget, so raising it rescues pages.

    The gate exists because a page under the floor cannot be read by the VLM at all; whether to
    raise the budget instead of skipping the page is a cost question, and it needs this arithmetic
    to be answerable at more than one budget.
    """
    frame = pl.DataFrame({"mean_render_dpi": [40.0, 80.0, 146.0]})

    at_default = frame.select(contract.legible_at_budget(DEFAULT_MAX_VISUAL_TOKENS))
    at_four_x = frame.select(contract.legible_at_budget(4 * DEFAULT_MAX_VISUAL_TOKENS))

    assert at_default.to_series().to_list() == [False, False, True]
    assert at_four_x.to_series().to_list() == [False, True, True], "80 DPI doubles to 160 at 4x the budget"


def test_free_and_paid_groups_are_priced_apart():
    """The router pass only survives if it earns 3.4 core-h per million pages the free groups do not.

    Pricing a group pdf-inspector already produced at anything above zero would charge the router
    for a pass it did not cause, and the comparison would stop meaning anything.
    """
    free = contract.cost_of(contract.FREE_GROUPS)
    paid = contract.cost_of(contract.PAID_GROUPS)

    assert free == 0.0
    assert paid == pytest.approx(contract.ROUTE_FEATURES_CORE_HOURS + contract.INSPECTOR_DETECT_CORE_HOURS)
    assert contract.cost_of(("page_signals",)) == contract.ROUTE_FEATURES_CORE_HOURS


def test_feature_groups_do_not_overlap_and_columns_survive_selection():
    """A column claimed by two groups would be double-counted in the gain split that prices them."""
    seen: set[str] = set()
    for group in contract.GROUPS:
        assert not seen & set(group.columns), f"{group.name} re-declares a column another group owns"
        seen.update(group.columns)

    every = contract.columns_for(tuple(group.name for group in contract.GROUPS))
    assert len(every) == len(seen)


# ---------------------------------------------------------------------------
# Output statistics, the signal v1 structurally could not have
# ---------------------------------------------------------------------------


def test_output_statistics_separate_clean_text_from_the_two_failures_they_exist_to_catch():
    """Garbling and repetition are observations on real output, not predictions from font tables."""
    clean = measure(["The quick brown fox jumps over the lazy dog near the river bank."], num_pages=1)
    garbled = measure(["Th� q�ick br�wn f�x"], num_pages=1)
    looping = measure(["Continue reading\n" * 40], num_pages=1)

    assert clean["inspector_output_replacement_ratio"] == 0.0
    assert garbled["inspector_output_replacement_ratio"] > clean["inspector_output_replacement_ratio"]
    assert clean["inspector_output_repeat_line_ratio"] == pytest.approx(0.0)
    assert looping["inspector_output_repeat_line_ratio"] > 0.9
    assert looping["inspector_output_max_line_repeats"] == pytest.approx(40.0)


def test_output_statistics_of_an_empty_extraction_are_defined_rather_than_missing():
    """A document pdf-inspector read nothing from is the strongest signal in the table.

    Returning nulls here would make it indistinguishable from a document the pass never reached.
    """
    empty = measure([], num_pages=5)

    assert empty["inspector_output_empty_page_fraction"] == 1.0
    assert empty["inspector_output_chars_per_source_page"] == 0.0
    assert empty["inspector_output_alpha_ratio"] == 0.0


def test_expected_output_length_is_absolute_because_it_predicts_truncation():
    """Truncation is a completion-budget failure, so the predictor has to be a length not a ratio."""
    short = measure(["brief"], num_pages=1)
    long_page = measure(["word " * 4000], num_pages=1)

    assert long_page["inspector_output_chars_per_source_page"] > 100 * short["inspector_output_chars_per_source_page"]
