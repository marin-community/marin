# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for HTML conversion transforms."""

from bs4 import BeautifulSoup
from marin.schemas.web.convert import ResiliparseConfig
from marin.transform.simple_html_to_md.process import SimpleHtmlToMdConfig, html_to_md
from marin.transform.wikipedia.transform_wikipedia import (
    clean_wiki_html,
)
from marin.transform.wikipedia.transform_wikipedia import (
    process_record as wiki_process_record,
)
from marin.web.convert import convert_page

SAMPLE_WIKIPEDIA_HTML = """<!DOCTYPE html>
<html>
<head><title>Sample Article</title></head>
<body>
<h1>Sample Article</h1>
<table class="infobox">
  <tr><th>Type</th><td>Example</td></tr>
</table>
<p>Introduction with inline math: <span class="mwe-math-element">
  <math><annotation encoding="application/x-tex">E = mc^2</annotation></math>
</span></p>
<dl>
  <dd>
    <span class="mwe-math-element">
      <math><annotation encoding="application/x-tex">\\int_0^\\infty e^{-x} dx = 1</annotation></math>
    </span>
  </dd>
</dl>
<h2 id="References">References</h2>
<div class="reflist">
  <ol>
    <li>Reference 1</li>
  </ol>
</div>
</body>
</html>"""

SAMPLE_FINEWEB_HTML_RECORD = {
    "id": "test-123",
    "text": (
        """<!DOCTYPE html>
<html>
<head><title>Test Page</title></head>
<body>
<h1>Main Title</h1>
<p>This is content with <a href="/relative">a link</a>.</p>
<img src="/image.png" alt="image">
</body>
</html>"""
    ),
    "source": "fineweb",
    "metadata": {
        "fineweb_metadata": {
            "url": "https://example.com/page",
            "language": "en",
            "timestamp": "2025-01-01",
        }
    },
}


# Integration test of wiki cleaning
def test_clean_wiki_html():
    cleaned = clean_wiki_html(SAMPLE_WIKIPEDIA_HTML, remove_reference_section=True)
    soup = BeautifulSoup(cleaned, "html.parser")

    # References should be removed
    assert soup.find("div", {"class": "reflist"}) is None
    assert soup.find("h2", {"id": "References"}) is None

    # Infobox should be moved
    assert soup.find("h2", string="InfoBox") is not None

    # Equations should be unwrapped
    assert "$E = mc^2$" in str(cleaned)


def test_convert_page_with_resiliparse(sample_html_simple):
    config = ResiliparseConfig()
    result = convert_page(sample_html_simple, url=None, config=config)

    assert "content" in result
    assert "title" in result
    assert "html" in result
    assert result["content"] is not None
    assert len(result["content"]) > 0


# Full wiki processing
def test_wiki_process_record_with_html():
    row = {
        "identifier": "wiki_123",
        "url": "https://en.wikipedia.org/wiki/Test",
        "name": "Test Article",
        "abstract": "This is a test",
        "date_created": "2025-01-01",
        "article_body": {"html": SAMPLE_WIKIPEDIA_HTML},
    }
    config = ResiliparseConfig()

    result = wiki_process_record(
        row,
        extract_config=config,
        remove_reference_section=True,
        digit_threshold=50,
        word_threshold=5,
        special_char_threshold=50,
    )

    assert result is not None
    assert result["id"] == "wiki_123"
    assert result["url"] == "https://en.wikipedia.org/wiki/Test"
    assert result["title"] == "Test Article"
    assert "text" in result
    assert len(result["text"]) > 0


def test_wiki_process_record_defaults_keep_ordinary_prose():
    row = {
        "identifier": "wiki_prose",
        "url": "https://en.wikipedia.org/wiki/Prose",
        "name": "Prose",
        "article_body": {"html": f"<p>{'Article prose ' * 80}</p>"},
    }

    result = wiki_process_record(row, extract_config=ResiliparseConfig())

    assert result is not None
    assert result["id"] == "wiki_prose"
    assert result["text"].count("Article prose") == 80


# Critical filtering
def test_wiki_process_record_filters_bad_content():
    row = {
        "identifier": "wiki_123",
        "url": "https://en.wikipedia.org/wiki/Test",
        "name": "Test Article",
        "article_body": {"html": SAMPLE_WIKIPEDIA_HTML},
    }
    config = ResiliparseConfig()

    # Set unreasonable thresholds to trigger filtering
    result = wiki_process_record(
        row,
        extract_config=config,
        remove_reference_section=True,
        digit_threshold=0,  # Filter if any digits
        word_threshold=1000,  # Require 1000+ words
        special_char_threshold=0,  # Filter if any special chars
    )

    assert result is None


# E2E pipeline
def test_simple_html_to_md_pipeline(tmp_path, write_jsonl_gz, read_all_jsonl_gz):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"

    # Write test data
    write_jsonl_gz(
        input_dir / "test.jsonl.gz",
        [SAMPLE_FINEWEB_HTML_RECORD, SAMPLE_FINEWEB_HTML_RECORD],
    )

    config = SimpleHtmlToMdConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        config=ResiliparseConfig(),
    )

    html_to_md(config)

    # Verify output
    results = read_all_jsonl_gz(output_dir)
    assert len(results) == 2
    for result in results:
        assert result["id"] == "test-123"
        assert result["format"] == "md"
        assert "text" in result
