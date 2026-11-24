# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from marin.datakit.decon import (
    ATTRIBUTE_SCHEMA,
    ITEM_SCHEMA,
    DeconKeys,
    ScoreWorkers,
    _record_scores,
    build_keys,
    located_questions,
    mark_source,
    scan_source,
    score_pairs,
    shared_holders,
)
from marin.datakit.decon_keys import (
    KeyStream,
    Windows,
    candidate_windows,
    excerpt,
    keys_table,
    record_readings,
    text_stream,
    window_hashes,
)
from marin.datakit.decon_tables import parquet_files, read_table
from marin.datakit.normalize import NormalizedData

PLANETS = "Which planet in our solar system has the largest number of known moons today?"
PLANET_CHOICES = ["Jupiter", "Saturn", "Uranus", "Neptune"]
DUCKS = "Janet's ducks lay sixteen eggs per day and she eats three of them for breakfast every morning."
RESOURCES = ResourceConfig(cpu=1, ram="1g")


@pytest.fixture(autouse=True)
def local_client():
    with set_current_client(LocalClient()):
        yield


@dataclass(frozen=True)
class QuestionInExcerpt:
    """A stand-in for the encoder models: 1.0 when the excerpt holds the item's whole question, else 0.2.
    Each call leaves a file in *calls_dir*."""

    calls_dir: str
    version: str = "question-in-excerpt"

    def load(self, items_dir: str, device: str):
        questions = {
            row["id"]: row["question"]
            for path in parquet_files(items_dir)
            for row in read_table(path, ["id", "question"]).to_pylist()
        }

        def score(pairs: pd.DataFrame) -> np.ndarray:
            (Path(self.calls_dir) / uuid.uuid4().hex).write_text(str(len(pairs)))
            return np.array(
                [
                    1.0 if text_stream(questions[item]) in text_stream(right) else 0.2
                    for item, right in zip(pairs["item"], pairs["right"], strict=True)
                ]
            )

        return score


def _write(path: Path, rows: list[dict], schema: pa.Schema | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), str(path))
    return str(path)


def _items(root: Path) -> str:
    def row(item_id: str, eval_name: str, question: str, choices: list[str]) -> dict:
        return {
            "id": item_id,
            "eval": eval_name,
            "suite": "test",
            "question": question,
            "choices": choices,
            "answer": "",
            "prompt": question,
            "passage": "",
            "choices_mode": "none",
            "reason": "",
        }

    _write(root / "items" / "planets.parquet", [row("planets-0", "planets", PLANETS, PLANET_CHOICES)], ITEM_SCHEMA)
    _write(root / "items" / "ducks.parquet", [row("ducks-0", "ducks", DUCKS, [])], ITEM_SCHEMA)
    return str(root / "items")


def _keys(tmp_path: Path) -> DeconKeys:
    reference = _write(
        tmp_path / "reference" / "part-00000-of-00001.parquet",
        [{"id": f"ref-{n}", "text": f"an unrelated reference document number {n} about gardening"} for n in range(20)],
    )
    return build_keys(
        items_dir=_items(tmp_path),
        reference_shards=[reference],
        output_path=str(tmp_path / "keys"),
        worker_resources=RESOURCES,
        max_workers=1,
    )


def _source(tmp_path: Path) -> NormalizedData:
    filler = "Some text about the weather and the city council meeting. " * 20
    shards = {
        "part-00000-of-00002.parquet": [
            {"id": "a0", "text": filler},
            {"id": "a1", "text": f"{filler}\nQuiz: {PLANETS}\nA) Jupiter B) Saturn C) Uranus D) Neptune\n{filler}"},
            {"id": "a2", "text": filler},
        ],
        "part-00001-of-00002.parquet": [
            {
                "id": "b0",
                "text": f"{filler} Janet's ducks lay sixteen eggs per day and she eats three of them. {filler}",
            },
            {"id": "b1", "text": filler},
        ],
    }
    for name, rows in shards.items():
        _write(tmp_path / "normalized" / name, rows)
    return NormalizedData(main_output_dir=str(tmp_path / "normalized"), dup_output_dir="", counters={})


def _mark(tmp_path: Path, keys: DeconKeys, scorer: QuestionInExcerpt):
    """The scan, score and mark stages of one source, as the pipeline's three steps run them."""
    normalized = _source(tmp_path)
    scan = scan_source(
        normalized=normalized, keys=keys, output_path=str(tmp_path / "scan"), worker_resources=RESOURCES, max_workers=2
    )
    workers = ScoreWorkers(resources=RESOURCES, max_workers=1, pairs_per_worker=100, extras=())
    scores = score_pairs(scan=scan, keys=keys, scorer=scorer, score_workers=workers, output_path=str(tmp_path / "score"))
    return mark_source(
        normalized=normalized,
        scores=scores,
        cutoff=0.5,
        flagged_sample_size=4,
        output_path=str(tmp_path / "marks"),
        worker_resources=RESOURCES,
        max_workers=2,
    )


def test_keys_cover_every_item(tmp_path):
    keys = _keys(tmp_path)
    assert keys.counters["keyed_items"] == 2
    locators = pq.read_table(keys.locators_path).to_pandas()
    assert set(locators["item"]) == {"planets-0", "ducks-0"}


def test_marks_follow_the_input_shards_and_flag_only_the_record_holding_an_item(tmp_path):
    calls = tmp_path / "calls"
    calls.mkdir()
    attributes = _mark(tmp_path, _keys(tmp_path), QuestionInExcerpt(str(calls)))
    main = Path(attributes.main_output_dir)
    assert sorted(path.name for path in main.glob("*.parquet")) == [
        "part-00000-of-00002.parquet",
        "part-00001-of-00002.parquet",
    ]
    first = pq.read_table(main / "part-00000-of-00002.parquet")
    assert first.schema.equals(ATTRIBUTE_SCHEMA)
    rows = first.to_pylist()
    assert [row["id"] for row in rows] == ["a0", "a1", "a2"]
    assert [row["contaminated"] for row in rows] == [False, True, False]
    assert rows[1]["matched_items"] == ["planets-0"] and rows[1]["max_score"] == 1.0
    assert {row["partition_id"] for row in rows} == {0}
    flagged = pq.read_table(Path(attributes.flagged_output_dir) / "part-00000-of-00002.parquet").to_pylist()
    assert [row["id"] for row in flagged] == ["a1"]
    assert PLANETS in flagged[0]["text"] and flagged[0]["matched_evals"] == ["planets"]
    assert attributes.counters["decon/contaminated"] == 1
    assert attributes.counters["decon/clean"] == 4


def test_a_pair_below_the_cutoff_keeps_the_record_with_its_score(tmp_path):
    calls = tmp_path / "calls"
    calls.mkdir()
    attributes = _mark(tmp_path, _keys(tmp_path), QuestionInExcerpt(str(calls)))
    rows = pq.read_table(Path(attributes.main_output_dir) / "part-00001-of-00002.parquet").to_pylist()
    # b0 holds the key of the ducks item but not its whole question, so its pair scores 0.2.
    assert rows[0]["id"] == "b0" and not rows[0]["contaminated"]
    assert rows[0]["max_score"] == pytest.approx(0.2) and rows[0]["matched_items"] == []


def test_a_record_lists_each_matched_item_once(tmp_path):
    # A long record gives one pair per distinct excerpt, so one item can reach the cutoff twice.
    path = tmp_path / "scores.parquet"
    rows = {"record": ["r", "r"], "item": ["i", "i"], "eval": ["e", "e"], "right": ["a", "b"], "score": [0.9, 0.8]}
    pq.write_table(pa.table(rows), str(path))
    best, matched = _record_scores(str(path), cutoff=0.5)
    assert matched == {"r": [("i", "e")]}
    assert best["r"] == (0.9, "a")


def test_a_rerun_scores_no_pair_again(tmp_path):
    calls = tmp_path / "calls"
    calls.mkdir()
    keys = _keys(tmp_path)
    _mark(tmp_path, keys, QuestionInExcerpt(str(calls)))
    first = len(list(calls.iterdir()))
    _mark(tmp_path, keys, QuestionInExcerpt(str(calls)))
    assert first > 0 and len(list(calls.iterdir())) == first


def test_a_shared_tie_keys_each_item_off_the_question_stem_that_other_items_share():
    stem = "Which of the following propositions is an immediate consequence of the given premises?"
    questions = [
        f"{stem} {premise}" for premise in ("Henry rides a red bicycle daily", "Mara bakes seven loaves weekly")
    ]
    streams = [text_stream(q) for q in questions]
    hashes, owners, offsets = candidate_windows(streams)
    ids = np.array(["formal-0", "formal-1"], dtype=object)
    windows = Windows(hashes, owners, offsets, ids, np.array([KeyStream.QUESTION] * 2, dtype=object))
    # No window occurs in the reference sample, so the tie order decides the keys.
    counts = np.zeros(len(hashes), dtype=np.int64)
    stem_windows = set(window_hashes(text_stream(stem)).tolist())
    earliest = keys_table(windows, counts).column("key1_hash").to_pylist()
    shared = keys_table(windows, counts, tie=shared_holders(windows))
    assert all(key in stem_windows for key in earliest)
    assert not any(key in stem_windows for key in shared.column("key1_hash").to_pylist())


def test_two_key_streams_of_one_item_count_as_one_holder_of_a_shared_window():
    streams = [text_stream("Henry made two stops during his sixty mile bike trip")] * 2
    hashes, owners, offsets = candidate_windows(streams)
    ids = np.array(["gsm8k-0", "gsm8k-0"], dtype=object)
    windows = Windows(hashes, owners, offsets, ids, np.array([KeyStream.QUESTION, KeyStream.UNLETTERED], dtype=object))
    assert set(shared_holders(windows).tolist()) == {1}


def test_an_excerpt_lands_on_the_exact_copy_after_a_changed_version_when_the_prompt_text_is_left_out():
    stem = "Jan has three times the number of pets as Marcia. Marcia has two more pets than Cindy."
    questions = {
        "gsm8k-0": f"Question: {stem} If Cindy has four pets, how many total pets do the three have?\nAnswer:",
        "gsm8k-1": "Question: Henry made two stops during his sixty mile bike trip, how far did he go?\nAnswer:",
    }
    question = located_questions(questions, {"gsm8k-0": "gsm8k", "gsm8k-1": "gsm8k"})["gsm8k-0"]
    assert question == text_stream(f"{stem} If Cindy has four pets, how many total pets do the three have?")
    changed = f"{stem} If Cindy has ten pets, how many pets does Jan have?"
    record = (
        f'test_a = "{changed}"\n'
        + "x = 1\n" * 1000
        + f'test_b = "{stem} If Cindy has four pets, how many total pets do the three have?"\n'
    )
    full = text_stream(questions["gsm8k-0"])
    key, offset = full[8:28], 8
    with_prompt, _, _ = excerpt(record_readings(record), len(questions["gsm8k-0"]), full, key, offset)
    without, _, _ = excerpt(record_readings(record), len(questions["gsm8k-0"]), question, key, offset)
    assert "four pets, how many total" not in with_prompt
    assert "four pets, how many total" in without
