# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behaviour of router v2: the two gates, the score, the render policy, and the feature contract.

The booster is replaced by a stand-in whose ranking is known; what is under test is the routing,
not XGBoost.
"""

import json

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.build_pdf_source import classify
from experiments.datakit.build_pdf_source import route_v2_features as contract
from experiments.datakit.build_pdf_source.classify import (
    DECIDED_BY_SCORE,
    ESCALATION_THRESHOLD,
    GATE_NO_TEXT,
    GATE_UNRENDERABLE,
    RouteDecision,
    gate,
    render_budget,
    route_batch,
    router_threshold,
    shard_routing,
)
from experiments.datakit.build_pdf_source.extract_inspector import SIGNAL_COLUMNS
from experiments.datakit.build_pdf_source.ocr_extract.render import (
    DEFAULT_LEGIBILITY_FLOOR_DPI,
    DEFAULT_MAX_VISUAL_TOKENS,
    RAISED_MAX_VISUAL_TOKENS,
)

_WARC = "crawl-data/CC-MAIN-0001/warc/x.warc.gz"
_MODEL_DIR = "s3://bucket/staged/pdf_route_v2"
_OUTPUT_NAMES = tuple(
    name.removeprefix("inspector_output_") for name in SIGNAL_COLUMNS if name.startswith("inspector_output_")
)


def _signals(offset: int, **overrides) -> dict:
    """One routable document's signal row: everything present, nothing remarkable."""
    row = {
        "warc_filename": _WARC,
        "warc_record_offset": offset,
        "content_digest": f"sha1:{offset}",
        "url": f"https://example.org/{offset}.pdf",
        "num_pages": 4,
        "pdf_bytes": 120_000,
        "mean_render_dpi": 149.5,
        "pages_below_legibility_floor": 0,
        "inspector_pdf_type": "text_based",
        "inspector_confidence": 0.9,
        "inspector_page_count": 4,
        "inspector_has_title": True,
        "inspector_ocr_reasons": "{}",
        "inspector_detect_pages_needing_ocr": 0,
        "inspector_extract_is_complex_layout": False,
        "inspector_extract_pages_needing_ocr": 0,
        "inspector_extract_pages_with_tables": 1,
        "inspector_extract_pages_with_columns": 0,
        "inspector_extracted_pages": 4,
        "inspector_markdown_chars": 8_000,
        "inspector_error": None,
        **{f"inspector_output_{name}": 0.1 for name in _OUTPUT_NAMES},
    }
    return row | overrides


def _batch(rows: list[dict]) -> pa.RecordBatch:
    frame = pl.DataFrame([{name: row[name] for name in SIGNAL_COLUMNS} for row in rows])
    return frame.to_arrow().to_batches()[0]


class _ScriptedBooster:
    """A booster that returns a prepared score per row, in order.

    The threshold means nothing on three fixture rows, so the ranking is supplied rather than learned.
    """

    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.matrices: list[np.ndarray] = []
        self.loads = 0

    def inplace_predict(self, matrix, validate_features=True):
        self.matrices.append(matrix)
        return np.asarray(self.scores[: len(matrix)], dtype=np.float32)


@pytest.fixture
def route(monkeypatch):
    """Run ``route_batch`` against a scripted ranking, with the real gates and real feature frame."""

    def run(rows: list[dict], scores: list[float] | None = None) -> tuple[list[dict], _ScriptedBooster]:
        booster = _ScriptedBooster(scores if scores is not None else [])

        def load(_model_dir):
            booster.loads += 1
            return booster, ESCALATION_THRESHOLD

        monkeypatch.setattr(classify, "load_router", load)
        records = list(route_batch(_batch(rows), model_dir=_MODEL_DIR, floor_dpi=DEFAULT_LEGIBILITY_FLOOR_DPI))
        return records, booster

    return run


# --- the feature contract ------------------------------------------------------------------------


def test_the_booster_is_fed_the_forty_three_contract_columns():
    """The booster was fit on 43 columns; ``inspector_detect`` is the only group that costs a call."""
    assert len(contract.ROUTER_FEATURES) == 43
    assert contract.PAID_GROUPS == ("inspector_detect",)


def test_feature_groups_do_not_overlap_and_columns_survive_selection():
    """A column claimed by two groups would be double-counted in the gain split that prices them."""
    seen: set[str] = set()
    for group in contract.GROUPS:
        assert not seen & set(group.columns), f"{group.name} re-declares a column another group owns"
        seen.update(group.columns)

    assert len(contract.columns_for(contract.ALL_GROUPS)) == len(seen)


# --- the gates -------------------------------------------------------------------------------------


def test_a_document_with_no_extracted_text_is_gated_rather_than_scored(route):
    """The model was neither trained nor calibrated on no-text documents, so a score would be an extrapolation."""
    records, booster = route([_signals(1, inspector_markdown_chars=0)])

    (record,) = records
    assert record["needs_ocr"] is True
    assert record["route_reason"] == GATE_NO_TEXT
    assert record["escalation_score"] is None
    assert booster.matrices == [], "the gated row never reached the model"


def test_a_failed_extraction_is_gated_by_the_same_rule_as_an_empty_one():
    """A library refusal is a no-text document too."""
    assert gate(_signals(1, inspector_markdown_chars=None)) == GATE_NO_TEXT


def test_a_document_nothing_can_render_is_kept_rather_than_escalated(route):
    """The VLM route renders through the same library, so escalating a document it cannot open buys
    nothing."""
    records, booster = route([_signals(1, mean_render_dpi=None, pages_below_legibility_floor=None, num_pages=0)])

    (record,) = records
    assert record["needs_ocr"] is False
    assert record["route_reason"] == GATE_UNRENDERABLE
    assert booster.matrices == []


def test_the_no_text_gate_wins_over_the_unrenderable_one():
    """A document that is both has nothing to keep, so it goes to the route that might read it."""
    assert gate(_signals(1, inspector_markdown_chars=0, mean_render_dpi=None)) == GATE_NO_TEXT


def test_the_legibility_floor_is_not_a_gate(route):
    """The score decides a below-floor document like any other; only the render budget changes."""
    records, _ = route([_signals(1, mean_render_dpi=36.5, pages_below_legibility_floor=4)], scores=[0.0])

    (record,) = records
    assert record["route_reason"] == DECIDED_BY_SCORE
    assert record["needs_ocr"] is False


# --- the score --------------------------------------------------------------------------------------


def test_the_threshold_is_inclusive_at_its_own_value(route):
    """The calibration is a quantile, so the document exactly at the cut belongs above it."""
    records, _ = route(
        [_signals(1), _signals(2), _signals(3)],
        scores=[ESCALATION_THRESHOLD - 1e-6, ESCALATION_THRESHOLD, ESCALATION_THRESHOLD + 1e-6],
    )

    assert [record["needs_ocr"] for record in records] == [False, True, True]
    assert [record["route_reason"] for record in records] == [DECIDED_BY_SCORE] * 3


def test_scores_are_matched_back_to_the_rows_they_came_from(route):
    """Gated rows are dropped from the matrix, so the prediction vector is shorter than the batch and
    a positional mistake here would route documents on each other's scores."""
    rows = [_signals(1, inspector_markdown_chars=0), _signals(2), _signals(3, mean_render_dpi=None), _signals(4)]

    records, booster = route(rows, scores=[0.9, 0.1])

    assert [record["warc_record_offset"] for record in records] == [1, 2, 3, 4]
    assert [record["escalation_score"] for record in records] == [None, pytest.approx(0.9), None, pytest.approx(0.1)]
    assert [record["needs_ocr"] for record in records] == [True, True, False, False]
    assert booster.matrices[0].shape == (2, len(contract.ROUTER_FEATURES))


def test_the_matrix_is_built_in_the_contract_order(route):
    """XGBoost scores a bare float matrix by position; a reordered column is silent nonsense."""
    row = _signals(1, pdf_bytes=987_654, num_pages=7)

    _, booster = route([row], scores=[0.5])

    (matrix,) = booster.matrices
    position = contract.ROUTER_FEATURES.index("pdf_bytes")
    assert matrix[0][position] == pytest.approx(987_654.0)
    assert matrix[0][contract.ROUTER_FEATURES.index("num_pages")] == pytest.approx(7.0)
    assert matrix.dtype == np.float32


def test_a_batch_of_nothing_but_gated_documents_never_loads_the_model(route):
    """Loading the booster on a shard that cannot use it is a per-task cost for no decision."""
    records, booster = route([_signals(1, inspector_markdown_chars=0), _signals(2, inspector_markdown_chars=0)])

    assert [record["needs_ocr"] for record in records] == [True, True]
    assert booster.loads == 0
    assert booster.matrices == []


# --- the render policy ------------------------------------------------------------------------------


def test_an_escalated_document_below_the_floor_is_flagged_for_the_raised_budget(route):
    records, _ = route([_signals(1, mean_render_dpi=36.5, pages_below_legibility_floor=4)], scores=[0.99])

    (record,) = records
    assert record["needs_ocr"] is True
    assert record["render_visual_tokens"] == RAISED_MAX_VISUAL_TOKENS


def test_a_kept_document_below_the_floor_is_not_flagged(route):
    """The policy is a render policy: it only means anything for a document that will be rendered."""
    records, _ = route([_signals(1, mean_render_dpi=36.5, pages_below_legibility_floor=4)], scores=[0.0])

    (record,) = records
    assert record["needs_ocr"] is False
    assert record["render_visual_tokens"] == DEFAULT_MAX_VISUAL_TOKENS


def test_a_legible_escalation_keeps_the_default_budget(route):
    records, _ = route([_signals(1)], scores=[0.99])

    assert records[0]["render_visual_tokens"] == DEFAULT_MAX_VISUAL_TOKENS


@pytest.mark.parametrize(
    ("dpi", "expected"),
    [
        (None, DEFAULT_MAX_VISUAL_TOKENS),
        (0.0, RAISED_MAX_VISUAL_TOKENS),
        (DEFAULT_LEGIBILITY_FLOOR_DPI - 0.1, RAISED_MAX_VISUAL_TOKENS),
        (DEFAULT_LEGIBILITY_FLOOR_DPI, DEFAULT_MAX_VISUAL_TOKENS),
    ],
)
def test_the_render_budget_turns_on_the_floor_and_nothing_else(dpi, expected):
    assert render_budget(dpi, DEFAULT_LEGIBILITY_FLOOR_DPI) == expected


# --- the routing table -------------------------------------------------------------------------------


def test_every_routed_record_satisfies_the_declared_schema(route):
    """A record that does not fit fails only at write time, a whole shard later."""
    rows = [_signals(1), _signals(2, inspector_markdown_chars=0), _signals(3, mean_render_dpi=None)]

    records, _ = route(rows, scores=[0.99])

    assert pa.RecordBatch.from_pylist(records, schema=classify.ROUTING_SCHEMA).num_rows == 3


def test_the_routing_table_records_the_gates_own_inputs(route):
    """A routing decision that cannot be re-derived from the table cannot be audited."""
    records, _ = route([_signals(1, inspector_markdown_chars=0, mean_render_dpi=36.5)])

    (record,) = records
    assert record["inspector_markdown_chars"] == 0
    assert record["mean_render_dpi"] == pytest.approx(36.5)
    assert record["num_pages"] == 4


# --- the routing shard a consumer reads -------------------------------------------------------------


def _routing_row(offset: int, needs_ocr: bool, tokens: int) -> dict:
    return {
        "warc_filename": _WARC,
        "warc_record_offset": offset,
        "content_digest": f"sha1:{offset}",
        "url": f"https://example.org/{offset}.pdf",
        "needs_ocr": needs_ocr,
        "route_reason": DECIDED_BY_SCORE,
        "escalation_score": None,
        "render_visual_tokens": tokens,
        "inspector_markdown_chars": None,
        "mean_render_dpi": None,
        "num_pages": None,
    }


def test_shard_routing_reads_the_decisions_of_one_fetched_shard(tmp_path):
    """The consumer's side of the join: one shard, keyed by WARC record, carrying route and budget."""
    rows = [_routing_row(1, True, RAISED_MAX_VISUAL_TOKENS), _routing_row(2, False, DEFAULT_MAX_VISUAL_TOKENS)]
    pq.write_table(pa.Table.from_pylist(rows, schema=classify.ROUTING_SCHEMA), tmp_path / "part-00003-of-00010.parquet")

    routing = shard_routing(str(tmp_path), "part-00003-of-00010.parquet")

    assert routing == {
        (_WARC, 1): RouteDecision(needs_ocr=True, render_visual_tokens=RAISED_MAX_VISUAL_TOKENS),
        (_WARC, 2): RouteDecision(needs_ocr=False, render_visual_tokens=DEFAULT_MAX_VISUAL_TOKENS),
    }


def test_a_fetched_shard_without_a_routing_shard_is_an_error_rather_than_an_empty_route(tmp_path):
    """A missing routing shard is the co-partitioning invariant broken, not a shard with no
    escalations."""
    with pytest.raises(FileNotFoundError, match="part-00004-of-00010"):
        shard_routing(str(tmp_path), "part-00004-of-00010.parquet")


# --- the pinned artifact ------------------------------------------------------------------------------


def _sidecar(features, threshold=ESCALATION_THRESHOLD) -> dict:
    return json.loads(json.dumps({"features": list(features), "escalation_threshold": threshold}))


def test_the_shipped_sidecar_and_this_module_agree():
    """The happy path: the pinned artifact's calibration and feature list against this module's contract."""
    assert router_threshold(contract.ROUTER_FEATURES, _sidecar(contract.ROUTER_FEATURES)) == ESCALATION_THRESHOLD


def test_a_booster_whose_feature_order_moved_is_refused():
    """It would score confident nonsense rather than fail: the matrix is positional."""
    shuffled = list(contract.ROUTER_FEATURES)
    shuffled[0], shuffled[1] = shuffled[1], shuffled[0]

    with pytest.raises(ValueError, match="different order or set"):
        router_threshold(tuple(shuffled), _sidecar(shuffled))


def test_a_booster_with_no_declared_features_is_refused():
    """A model saved without ``feature_names`` carries no contract at all to check against."""
    with pytest.raises(ValueError, match="different order or set"):
        router_threshold((), _sidecar(contract.ROUTER_FEATURES))


def test_a_sidecar_calibrating_a_different_feature_set_is_refused():
    """The threshold is a quantile of one model's output; another model's is a different number."""
    with pytest.raises(ValueError, match="calibrates a different feature set"):
        router_threshold(contract.ROUTER_FEATURES, _sidecar(list(contract.ROUTER_FEATURES)[:-1]))


def test_a_sidecar_threshold_that_disagrees_with_the_step_is_refused():
    """The step hash is keyed on the constant, so a silent sidecar move would route at a budget the
    run's own identity does not describe."""
    with pytest.raises(ValueError, match="but this step is keyed on"):
        router_threshold(contract.ROUTER_FEATURES, _sidecar(contract.ROUTER_FEATURES, threshold=0.5))
