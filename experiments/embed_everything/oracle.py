# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle labeling of documents using Claude and OpenAI.

Provides quality scoring (0-5 discrete rubric) and topic classification
from a fixed taxonomy, with support for multiple LLM backends.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from enum import StrEnum

from iris.marin_fs import open_url

from marin.execution import THIS_OUTPUT_PATH

logger = logging.getLogger(__name__)

MAX_DOC_CHARS = 8000
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0


class OracleBackend(StrEnum):
    CLAUDE = "claude"
    OPENAI = "openai"


# FineWeb-Edu style quality rubric (discrete 0-5)
QUALITY_PROMPT = """\
You are an expert document quality evaluator. \
Score the following document on a 0-5 scale based on its educational and informational value.

Scoring rubric:
0 - Unintelligible, spam, or purely navigational content with no informational value.
1 - Low quality: mostly ads, SEO content, or very shallow content with minimal useful information.
2 - Below average: some useful information but poorly written, disorganized, or mostly redundant with common knowledge.
3 - Average: reasonably informative content that covers a topic adequately but without depth or insight.
4 - Good: well-written, informative content that provides useful knowledge or analysis on a topic.
5 - Excellent: high-quality, educational content with depth, clarity, and insight. Could serve as reference material.

Respond with ONLY a JSON object: {{"score": <int>, "reasoning": "<brief explanation>"}}

Document:
---
{text}
---"""

# Fixed taxonomy for topic classification
TOPIC_TAXONOMY = [
    "mathematics",
    "computer_science",
    "natural_science",
    "engineering",
    "medicine_health",
    "social_science",
    "humanities",
    "law",
    "business_finance",
    "news_journalism",
    "creative_writing",
    "reference_encyclopedia",
    "web_forum_discussion",
    "code",
    "other",
]

TOPIC_PROMPT = """\
You are an expert document classifier. \
Assign the most appropriate topic label to the following document.

Available topics:
{topics}

Respond with ONLY a JSON object: {{"topic": "<topic_label>", "reasoning": "<brief explanation>"}}

Document:
---
{text}
---"""


def _truncate_text(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    """Truncate document text to fit within context limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


def _call_claude(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Claude API and return the response text."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API and return the response text."""
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _call_llm(prompt: str, backend: OracleBackend) -> str:
    """Call the appropriate LLM backend with retries."""
    call_fn = _call_claude if backend == OracleBackend.CLAUDE else _call_openai
    for attempt in range(MAX_RETRIES):
        try:
            return call_fn(prompt)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.warning("LLM call failed (attempt %d/%d), retrying in %.1fs", attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def _parse_json_response(response: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = response.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


def _read_jsonl(path: str) -> list[dict]:
    """Read a JSONL file and return list of dicts."""
    docs = []
    with open_url(path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def _write_jsonl(docs: list[dict], path: str) -> None:
    """Write list of dicts to a JSONL file."""
    with open_url(path, "w") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")


@dataclass(frozen=True)
class LabelQualityConfig:
    """Config for oracle quality labeling."""

    input_path: str
    output_path: str = THIS_OUTPUT_PATH
    backend: OracleBackend = OracleBackend.CLAUDE
    prompt_version: str = "v1"


def label_quality(config: LabelQualityConfig) -> None:
    """Score each document on a 0-5 quality rubric using an LLM oracle."""
    input_file = os.path.join(config.input_path, "quality_samples.jsonl")
    docs = _read_jsonl(input_file)
    logger.info("Labeling %d documents for quality with backend=%s", len(docs), config.backend)

    labeled: list[dict] = []
    for i, doc in enumerate(docs):
        truncated = _truncate_text(doc["text"])
        prompt = QUALITY_PROMPT.format(text=truncated)

        try:
            response = _call_llm(prompt, config.backend)
            parsed = _parse_json_response(response)
            doc["oracle_quality_score"] = int(parsed["score"])
            doc["oracle_quality_reasoning"] = parsed.get("reasoning", "")
            doc["oracle_backend"] = str(config.backend)
        except Exception:
            logger.exception("Failed to label document %s", doc.get("doc_id", i))
            doc["oracle_quality_score"] = -1
            doc["oracle_quality_reasoning"] = "labeling_failed"
            doc["oracle_backend"] = str(config.backend)

        labeled.append(doc)

        if (i + 1) % 10 == 0:
            logger.info("Quality labeling progress: %d/%d", i + 1, len(docs))

    os.makedirs(config.output_path, exist_ok=True)
    _write_jsonl(labeled, os.path.join(config.output_path, "quality_labeled.jsonl"))

    failed = sum(1 for d in labeled if d["oracle_quality_score"] == -1)
    logger.info("Quality labeling complete: %d labeled, %d failed", len(labeled) - failed, failed)


@dataclass(frozen=True)
class LabelTopicConfig:
    """Config for oracle topic labeling."""

    input_path: str
    output_path: str = THIS_OUTPUT_PATH
    backend: OracleBackend = OracleBackend.CLAUDE
    prompt_version: str = "v1"
    taxonomy: list[str] = field(default_factory=lambda: list(TOPIC_TAXONOMY))


def label_topics(config: LabelTopicConfig) -> None:
    """Assign a topic label from a fixed taxonomy to each document using an LLM oracle."""
    input_file = os.path.join(config.input_path, "topic_samples.jsonl")
    docs = _read_jsonl(input_file)
    topics_str = "\n".join(f"- {t}" for t in config.taxonomy)
    logger.info("Labeling %d documents for topics with backend=%s", len(docs), config.backend)

    labeled: list[dict] = []
    for i, doc in enumerate(docs):
        truncated = _truncate_text(doc["text"])
        prompt = TOPIC_PROMPT.format(text=truncated, topics=topics_str)

        try:
            response = _call_llm(prompt, config.backend)
            parsed = _parse_json_response(response)
            topic = parsed["topic"]
            if topic not in config.taxonomy:
                logger.warning(
                    "LLM returned unknown topic '%s' for doc %s, mapping to 'other'", topic, doc.get("doc_id", i)
                )
                topic = "other"
            doc["oracle_topic"] = topic
            doc["oracle_topic_reasoning"] = parsed.get("reasoning", "")
            doc["oracle_backend"] = str(config.backend)
        except Exception:
            logger.exception("Failed to label document %s", doc.get("doc_id", i))
            doc["oracle_topic"] = "labeling_failed"
            doc["oracle_topic_reasoning"] = "labeling_failed"
            doc["oracle_backend"] = str(config.backend)

        labeled.append(doc)

        if (i + 1) % 10 == 0:
            logger.info("Topic labeling progress: %d/%d", i + 1, len(docs))

    os.makedirs(config.output_path, exist_ok=True)
    _write_jsonl(labeled, os.path.join(config.output_path, "topic_labeled.jsonl"))

    failed = sum(1 for d in labeled if d["oracle_topic"] == "labeling_failed")
    logger.info("Topic labeling complete: %d labeled, %d failed", len(labeled) - failed, failed)
