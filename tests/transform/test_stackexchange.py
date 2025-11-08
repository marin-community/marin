# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from marin.schemas.web.convert import TrafilaturaConfig
from marin.transform.stackexchange.filter_stackexchange import (
    FilterStackExchangeConfig,
    _process_file_with_filtering,
    filter_stackexchange,
)
from marin.transform.stackexchange.transform_stackexchange import (
    StackExchangeExtractionConfig,
    process_record,
    process_stackexchange_dump,
)

SAMPLE_QUESTION_HTML = """<p>I'm trying to understand how <code>async/await</code> works in Python.</p>
<p>Can someone explain the difference between:</p>
<pre><code>async def foo():
    return 42
</code></pre>
<p>and regular functions?</p>"""

SAMPLE_ANSWER_HTML = """<p>The <code>async</code> keyword defines a coroutine function.</p>
<p>Key differences:</p>
<ul>
<li>Async functions return coroutines</li>
<li>Must be awaited or run in event loop</li>
<li>Can use <code>await</code> inside</li>
</ul>"""

SAMPLE_STACKEXCHANGE_RECORD = {
    "id": "se-python-12345",
    "created": "2023-01-15T10:30:00Z",
    "title": "Understanding async/await in Python",
    "question": SAMPLE_QUESTION_HTML,
    "url": "https://stackoverflow.com/questions/12345",
    "tags": ["python", "async", "coroutines"],
    "metadata": {
        "title": "Understanding async/await in Python",
        "question": SAMPLE_QUESTION_HTML,
        "url": "https://stackoverflow.com/questions/12345",
        "tags": ["python", "async", "coroutines"],
        "votes": 42,
        "id": "12345",
        "answers": [
            {"body": SAMPLE_ANSWER_HTML, "votes": 35},
            {
                "body": "<p>Here's another way to think about it...</p>",
                "votes": 10,
            },
        ],
    },
}


def test_process_record_basic():
    """Test processing a basic StackExchange record."""
    extract_config = TrafilaturaConfig.default_config()

    result = process_record(
        row=SAMPLE_STACKEXCHANGE_RECORD,
        extract_method="trafilatura",
        extract_config=extract_config,
        shuffle_answers_template=False,
        seed=42,
    )

    assert result is not None
    assert result["id"] == "se-python-12345"
    assert result["url"] == "https://stackoverflow.com/questions/12345"
    assert result["title"] == "Understanding async/await in Python"
    assert result["date_created"] == "2023-01-15T10:30:00Z"
    assert "text" in result
    assert "# Question" in result["text"]
    assert "# Answer" in result["text"]


def test_process_stackexchange_dump(tmp_path: Path, write_jsonl_gz, read_all_jsonl_gz):
    """Test full pipeline processing of StackExchange dump."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create test data files
    records = [
        {
            "id": "se-001",
            "created": "2023-01-01T00:00:00Z",
            "metadata": {
                "title": "Question 1",
                "question": "<p>First question</p>",
                "url": "https://stackoverflow.com/q/001",
                "tags": ["python"],
                "votes": 10,
                "id": "001",
                "answers": [{"body": "<p>Answer 1</p>", "votes": 5}],
            },
        },
        {
            "id": "se-002",
            "created": "2023-01-02T00:00:00Z",
            "metadata": {
                "title": "Question 2",
                "question": "<p>Second question</p>",
                "url": "https://stackoverflow.com/q/002",
                "tags": ["java"],
                "votes": 20,
                "id": "002",
                "answers": [{"body": "<p>Answer 2</p>", "votes": 15}],
            },
        },
    ]

    write_jsonl_gz(input_dir / "001.jsonl.gz", [records[0]])
    write_jsonl_gz(input_dir / "002.jsonl.gz", [records[1]])

    config = StackExchangeExtractionConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        extract_method="trafilatura",
        extract_config=TrafilaturaConfig.default_config(),
        shuffle_answers_template=False,
        seed=42,
    )

    process_stackexchange_dump(config)

    # Verify output
    output_records = read_all_jsonl_gz(output_dir)
    assert len(output_records) == 2

    ids = {r["id"] for r in output_records}
    assert ids == {"se-001", "se-002"}

    # Verify structure
    for record in output_records:
        assert "id" in record
        assert "url" in record
        assert "title" in record
        assert "date_created" in record
        assert "text" in record
        assert "# Question" in record["text"]
        assert "# Answer" in record["text"]


def test_process_file_with_filtering_basic(tmp_path: Path, write_jsonl_gz):
    """Test basic vote filtering."""
    input_file = tmp_path / "test.jsonl.gz"

    records = [
        {
            "id": "rec-001",
            "text": "High score",
            "metadata": {"id": "q-001", "votes": 50},
        },
        {
            "id": "rec-002",
            "text": "Low score",
            "metadata": {"id": "q-002", "votes": 5},
        },
        {
            "id": "rec-003",
            "text": "Medium score",
            "metadata": {"id": "q-003", "votes": 15},
        },
    ]

    write_jsonl_gz(input_file, records)

    config = FilterStackExchangeConfig(
        input_path=str(tmp_path),
        output_path=str(tmp_path / "output"),
        min_vote_threshold=10,
        remove_duplicate_questions=False,
    )

    filtered = list(_process_file_with_filtering(str(input_file), config))

    assert len(filtered) == 2
    votes = [r["metadata"]["votes"] for r in filtered]
    assert all(v >= 10 for v in votes)
    assert 5 not in votes


def test_filter_stackexchange_full_pipeline(tmp_path: Path, write_jsonl_gz, read_all_jsonl_gz):
    """Test complete filtering pipeline."""

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Create multiple files with various vote scores and duplicates
    file1_records = [
        {"id": "f1-rec-001", "text": "High score", "metadata": {"id": "q-100", "votes": 50}},
        {"id": "f1-rec-002", "text": "Low score", "metadata": {"id": "q-101", "votes": 3}},
        {"id": "f1-rec-003", "text": "Medium score", "metadata": {"id": "q-102", "votes": 15}},
    ]

    file2_records = [
        {"id": "f2-rec-001", "text": "Another high", "metadata": {"id": "q-200", "votes": 40}},
        {"id": "f2-rec-002", "text": "Duplicate in file", "metadata": {"id": "q-200", "votes": 45}},
        {"id": "f2-rec-003", "text": "Below threshold", "metadata": {"id": "q-201", "votes": 8}},
    ]

    write_jsonl_gz(input_dir / "file1.jsonl.gz", file1_records)
    write_jsonl_gz(input_dir / "file2.jsonl.gz", file2_records)

    config = FilterStackExchangeConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        min_vote_threshold=10,
        remove_duplicate_questions=True,
    )

    filter_stackexchange(config)

    output_records = read_all_jsonl_gz(output_dir)

    # Should have filtered out low scores and duplicates within each file
    # File 1: q-100 (50), q-102 (15) - q-101 filtered (3)
    # File 2: q-200 (40, first occurrence) - q-200 duplicate filtered, q-201 filtered (8)
    assert len(output_records) == 3

    votes = [r["metadata"]["votes"] for r in output_records]
    assert all(v >= 10 for v in votes)

    # Check that low scores are filtered
    assert 3 not in votes
    assert 8 not in votes
