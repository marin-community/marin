"""Regression tests for DROP answer extraction."""

import sys
from pathlib import Path

import datasets
from lm_eval.tasks import TaskManager
from lm_eval.tasks._yaml_loader import load_yaml
from lm_eval.utils import apply_template

# The evalchemy `eval` package is copied into site-packages, not editable-installed, so
# put this worktree first on the path to import the in-tree fix under test.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from lm_eval.tasks.drop.utils import process_results  # noqa: E402

from eval.lm_eval_tasks.drop.utils import (  # noqa: E402
    drop_answer_extraction_filter,
    extract_drop_answer,
    extract_drop_short_answer,
    process_docs,
)
from eval.generation_stops import DROP_STOP_SEQUENCES  # noqa: E402
from eval.regression.lm_eval_task_contracts import mapping_key_target_violation  # noqa: E402
from eval.sample_logging import canonicalize_samples  # noqa: E402

# A realistic no-template completion: the model restates the passage/question and only
# incidentally contains the gold number ("42"). This is what the DROP metric actually
# receives when there is no chat template.
VERBOSE_COMPLETION = (
    "Looking at the passage we can see that the home team and the away team both "
    "played several drives and the question asks how many total points were scored "
    "in the game so we add up all of the touchdowns and field goals mentioned to "
    "arrive at a grand total of 42 points across the entire contest as described. Answer: 42"
)

GOLD_NUMBER_DOC = {"answers": [("42",)]}

RAW_DROP_DOC = {
    "query_id": "fixture",
    "passage": "The team scored 42 points.",
    "question": "How many points did the team score?",
    "answer": {
        "number": "42",
        "date": {"day": "", "month": "", "year": ""},
        "spans": [],
        "worker_id": "worker",
        "hit_id": "hit",
    },
    "validated_answers": {"number": [], "date": [], "spans": []},
}


def _f1(doc, results):
    return process_results(doc, results)["f1"]


def _filter_one(completion):
    return drop_answer_extraction_filter([[completion]], [{}])[0]


# --------------------------------------------------------------------------------------
# The bug (issue #31): raw verbose completions collapse to ~0 f1.
# --------------------------------------------------------------------------------------
def test_verbose_completion_collapses_on_raw_path():
    f1 = _f1(GOLD_NUMBER_DOC, [VERBOSE_COMPLETION])
    assert f1 < 0.1, f"expected collapsed f1 (<0.1) on raw verbose completion, got {f1}"


# --------------------------------------------------------------------------------------
# The fix: the extraction filter recovers the correct answer.
# --------------------------------------------------------------------------------------
def test_filter_recovers_verbose_completion():
    f1 = _f1(GOLD_NUMBER_DOC, _filter_one(VERBOSE_COMPLETION))
    assert f1 == 1.0, f"expected f1==1.0 after extraction, got {f1}"


def test_extract_marker_mid_text():
    doc = {"answers": [("27",)]}
    comp = "After careful analysis, the answer is 27 touchdowns scored in the game."
    assert extract_drop_short_answer(comp) == "27"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_multiple_numbers_takes_final():
    doc = {"answers": [("28",)]}
    comp = "There were 3 field goals and 4 touchdowns, but the final total was 28. Answer: 28"
    assert extract_drop_short_answer(comp) == "28"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_marker_wins_over_earlier_numbers():
    doc = {"answers": [("15",)]}
    comp = "we compute 10 plus 5 which means the answer is 15"
    assert extract_drop_short_answer(comp) == "15"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_unmarked_entity_after_incidental_number():
    doc = {"answers": [("Chaz Schilens",)]}
    comp = (
        "The first touchdown of the game was scored by the Raiders. Quarterback JaMarcus Russell "
        "completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens."
    )

    assert extract_drop_short_answer(comp) == "Chaz Schilens"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_unmarked_entity_extraction_is_classified_in_sample_artifacts():
    response = "Quarterback JaMarcus Russell completed a 20-yard touchdown pass to Chaz Schilens."
    selected = extract_drop_answer(response)
    sample = {
        "resps": [[response]],
        "filtered_resps": [[selected]],
        "metrics": ["f1"],
        "f1": 1.0,
    }
    record = canonicalize_samples("drop", [sample])[0]

    assert str(selected) == "Chaz Schilens"
    assert record["drop_extractions"][0][0]["classification"] == "entity"


def test_extract_explicit_marker_preserves_numeric_date_entity_and_multi_span_answers():
    cases = [
        ("Answer: 27", "27"),
        ("Final answer: 24 May 1993", "24 May 1993"),
        ("Answer: Chaz Schilens", "Chaz Schilens"),
        ("Answer: red; blue", "red; blue"),
    ]

    for completion, expected in cases:
        assert extract_drop_short_answer(completion) == expected


def test_extract_full_date_answer():
    doc = {"answers": [("24 May 1993",)]}
    comp = "The event happened on 24 May 1993 according to the passage. Answer: 24 May 1993"
    assert extract_drop_short_answer(comp) == "24 May 1993"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_month_year_date_answer():
    doc = {"answers": [("December 1941",)]}
    comp = "It occurred in December 1941 during the war, per the text. Answer: December 1941"
    assert extract_drop_short_answer(comp) == "December 1941"
    assert _f1(doc, _filter_one(comp)) == 1.0


def test_extract_decimal_and_negative():
    assert extract_drop_short_answer("the difference came out to -3.5 in the end. Answer: -3.5") == "-3.5"


def test_exact_short_answer_is_idempotent():
    assert extract_drop_short_answer("42") == "42"
    assert _f1(GOLD_NUMBER_DOC, _filter_one("42")) == 1.0


# --------------------------------------------------------------------------------------
# The metric must NOT be weakened: genuinely-wrong completions still score ~0.
# --------------------------------------------------------------------------------------
def test_wrong_numeric_answer_stays_zero():
    comp = "the total was 17 points scored by both teams combined in the game. Answer: 17"
    assert extract_drop_short_answer(comp) == "17"
    assert _f1(GOLD_NUMBER_DOC, _filter_one(comp)) == 0.0


def test_no_answer_present_does_not_fabricate():
    comp = "the home team ultimately won the game after a hard fought contest"
    # Falls back to the (verbose) text -> still collapses; we never fabricate "42".
    assert _f1(GOLD_NUMBER_DOC, _filter_one(comp)) < 0.1


def test_drop_target_renders_processed_answers_not_raw_answer_keys():
    """The override scores the normalized DROP gold, not Jinja's mapping-key join."""
    task_manager = TaskManager(include_path=[str(_REPO_ROOT / "eval" / "lm_eval_tasks")])
    config = task_manager.task_index["drop"].cfg
    processed = process_docs(datasets.Dataset.from_list([RAW_DROP_DOC]))[0]
    rendered = apply_template(config["doc_to_target"], processed)
    malformed = apply_template("{{ answer|join(',')}}", processed)

    assert rendered == "Answer: 42"
    assert mapping_key_target_violation(processed, rendered) is None
    violation = mapping_key_target_violation(processed, malformed)
    assert violation is not None
    assert violation.field_path == "answer"
    assert violation.target == malformed


# --------------------------------------------------------------------------------------
# Filter plumbing + robustness.
# --------------------------------------------------------------------------------------
def test_filter_preserves_shape_over_docs_and_samples():
    resps = [
        ["the answer is 7", "Answer: 7"],
        ["Answer: 12", "the answer is 12"],
    ]
    out = drop_answer_extraction_filter(resps, [{}, {}])
    assert out == [["7", "7"], ["12", "12"]]


def test_filter_handles_non_string_and_empty():
    out = drop_answer_extraction_filter([[None, "", "  "]], [{}])
    assert out == [["", "", ""]]


def test_drop_task_uses_non_newline_stop_sequences_and_explicit_answer_prompt():
    task_manager = TaskManager(include_path=[str(_REPO_ROOT / "eval" / "lm_eval_tasks")])
    entry = task_manager.task_index["drop"]
    assert entry.yaml_path is not None
    config = load_yaml(entry.yaml_path, resolve_func=True)

    assert config["generation_kwargs"]["until"] == DROP_STOP_SEQUENCES
    assert config["doc_to_text"].endswith('Respond with exactly "Answer: <short answer>".')
    assert config["doc_to_target"] == "Answer: {{ answers[0]|join(',')}}"


# --------------------------------------------------------------------------------------
# Wiring: the override YAML actually takes precedence over upstream `drop`.
# --------------------------------------------------------------------------------------
def test_override_yaml_takes_precedence():
    from lm_eval.tasks import TaskManager

    from eval.eval import DEFAULT_LM_EVAL_INCLUDE_DIR

    tm = TaskManager(include_path=[DEFAULT_LM_EVAL_INCLUDE_DIR])
    entry = tm.task_index.get("drop")
    assert entry is not None and entry.yaml_path is not None
    # The resolved `drop` config must be OUR override file, not the packaged one.
    assert Path(entry.yaml_path).resolve() == (Path(DEFAULT_LM_EVAL_INCLUDE_DIR) / "drop" / "drop.yaml").resolve()
    config = load_yaml(entry.yaml_path, resolve_func=True)
    assert config["generation_kwargs"]["until"] == DROP_STOP_SEQUENCES
