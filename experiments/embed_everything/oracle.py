# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle labeling of documents using Claude and OpenAI.

Provides quality scoring (0-5 discrete rubric) and topic classification from a
fixed taxonomy. Uses provider-native structured outputs so the response shape
is enforced end-to-end:

- Anthropic: forced tool-use whose ``input_schema`` is the Pydantic model's
  JSON schema. Claude must emit a tool call; the tool input is validated on
  our side via ``schema.model_validate``.
- OpenAI:   ``beta.chat.completions.parse(response_format=<PydanticModel>)``
  which returns an already-parsed Pydantic instance.
"""

import logging
import os
import time
from enum import StrEnum
from typing import Literal, TypeVar, get_args

from pydantic import BaseModel, Field
from zephyr.readers import load_parquet
from zephyr.writers import write_parquet_file

logger = logging.getLogger(__name__)

MAX_DOC_CHARS = 8000
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
# Leaves room for a long `reasoning` field plus tool-call overhead on
# Anthropic. The previous 256 occasionally truncated reasoning strings.
MAX_RESPONSE_TOKENS = 1024


class OracleBackend(StrEnum):
    CLAUDE = "claude"
    OPENAI = "openai"


# ---------------------------------------------------------------------------
# Structured-output schemas
# ---------------------------------------------------------------------------
# Provider-enforced via tool-use / response_format so the LLM cannot return
# an out-of-range score or an off-taxonomy topic. Pydantic validates on our
# side as a second layer, catching any provider-side schema drift.
#
# Note: we deliberately omit numeric bounds (ge/le) on `score` because
# OpenAI's strict structured-output mode does not support
# minimum/maximum. The 0-5 rubric is enforced by the prompt instead.

TopicLiteral = Literal[
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
TOPIC_TAXONOMY: tuple[str, ...] = get_args(TopicLiteral)


class QualityLabel(BaseModel):
    score: int = Field(description="Quality score from 0 (unintelligible) to 5 (excellent).")
    reasoning: str = Field(description="Brief explanation for the score.")


class TopicLabel(BaseModel):
    topic: TopicLiteral = Field(description="Best-fitting topic label from the taxonomy.")
    reasoning: str = Field(description="Brief explanation for the topic assignment.")


LabelSchema = TypeVar("LabelSchema", bound=BaseModel)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
# Structured outputs enforce the response shape, so the prompts describe only
# the task — no "respond in JSON" instruction needed.

QUALITY_PROMPT = """\
You are an expert document quality evaluator. \
Score the following document on a 0-5 scale based on its educational and informational value.

Scoring rubric:
0 - Unintelligible, spam, or purely navigational content with no informational value.
1 - Low quality: mostly ads, SEO content, or very shallow content with minimal useful information.
2 - Below average: some useful information but poorly written, disorganized, or mostly redundant.
3 - Average: reasonably informative content that covers a topic adequately but without depth or insight.
4 - Good: well-written, informative content that provides useful knowledge or analysis on a topic.
5 - Excellent: high-quality, educational content with depth, clarity, and insight.

Document:
---
{text}
---"""

TOPIC_PROMPT = """\
You are an expert document classifier. \
Assign the most appropriate topic label to the following document.

Available topics:
{topics}

Document:
---
{text}
---"""


def _truncate_text(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    """Truncate document text to fit within context limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[... truncated ...]"


# ---------------------------------------------------------------------------
# Structured LLM calls
# ---------------------------------------------------------------------------
# Temperature is intentionally left at provider defaults (Anthropic=1.0,
# OpenAI=1.0). For classification tasks temp=0 is the usual reproducibility
# default, but single-sample non-determinism is an accepted noise source for
# this experiment; revisit with multi-sample majority vote if
# oracle-self-agreement becomes a bottleneck.


def _call_claude_structured(
    prompt: str,
    schema: type[LabelSchema],
    model: str = "claude-sonnet-4-20250514",
) -> LabelSchema:
    """Call Claude with forced tool-use to get a schema-conformant response."""
    import anthropic

    client = anthropic.Anthropic()
    tool_name = f"record_{schema.__name__.lower()}"
    response = client.messages.create(
        model=model,
        max_tokens=MAX_RESPONSE_TOKENS,
        tools=[
            {
                "name": tool_name,
                "description": f"Record the {schema.__name__} for the document.",
                "input_schema": schema.model_json_schema(),
            }
        ],
        tool_choice={"type": "tool", "name": tool_name},
        messages=[{"role": "user", "content": prompt}],
    )
    for block in response.content:
        if block.type == "tool_use":
            return schema.model_validate(block.input)
    raise RuntimeError(f"No tool_use block in Claude response: {response.content!r}")


def _call_openai_structured(
    prompt: str,
    schema: type[LabelSchema],
    model: str = "gpt-4o-mini",
) -> LabelSchema:
    """Call OpenAI with response_format=schema; return a parsed Pydantic instance."""
    import openai

    client = openai.OpenAI()
    response = client.beta.chat.completions.parse(
        model=model,
        max_tokens=MAX_RESPONSE_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        response_format=schema,
    )
    message = response.choices[0].message
    if message.parsed is None:
        raise RuntimeError(f"OpenAI returned no parsed output (refusal={message.refusal!r})")
    return message.parsed


def _call_llm_structured(
    prompt: str,
    schema: type[LabelSchema],
    backend: OracleBackend,
) -> LabelSchema:
    """Dispatch to the right backend with exponential-backoff retries."""
    call_fn = _call_claude_structured if backend == OracleBackend.CLAUDE else _call_openai_structured
    for attempt in range(MAX_RETRIES):
        try:
            return call_fn(prompt, schema)
        except Exception:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = RETRY_BACKOFF_BASE ** (attempt + 1)
            logger.warning("LLM call failed (attempt %d/%d), retrying in %.1fs", attempt + 1, MAX_RETRIES, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def label_quality(
    output_path: str,
    input_path: str,
    backend: OracleBackend = OracleBackend.CLAUDE,
) -> None:
    """Score each document on a 0-5 quality rubric using an LLM oracle."""
    input_file = os.path.join(input_path, "quality_samples.parquet")
    docs = list(load_parquet(input_file))
    logger.info("Labeling %d documents for quality with backend=%s", len(docs), backend)

    labeled: list[dict] = []
    for i, doc in enumerate(docs):
        truncated = _truncate_text(doc["text"])
        prompt = QUALITY_PROMPT.format(text=truncated)

        try:
            label = _call_llm_structured(prompt, QualityLabel, backend)
            doc["oracle_quality_score"] = label.score
            doc["oracle_quality_reasoning"] = label.reasoning
            doc["oracle_backend"] = str(backend)
        except Exception:
            logger.exception("Failed to label document %s", doc.get("doc_id", i))
            doc["oracle_quality_score"] = -1
            doc["oracle_quality_reasoning"] = "labeling_failed"
            doc["oracle_backend"] = str(backend)

        labeled.append(doc)

        if (i + 1) % 10 == 0:
            logger.info("Quality labeling progress: %d/%d", i + 1, len(docs))

    write_parquet_file(labeled, os.path.join(output_path, "quality_labeled.parquet"))

    failed = sum(1 for d in labeled if d["oracle_quality_score"] == -1)
    logger.info("Quality labeling complete: %d labeled, %d failed", len(labeled) - failed, failed)


def label_topics(
    output_path: str,
    input_path: str,
    backend: OracleBackend = OracleBackend.CLAUDE,
) -> None:
    """Assign a topic label from the fixed taxonomy to each document."""
    input_file = os.path.join(input_path, "topic_samples.parquet")
    docs = list(load_parquet(input_file))
    topics_str = "\n".join(f"- {t}" for t in TOPIC_TAXONOMY)
    logger.info("Labeling %d documents for topics with backend=%s", len(docs), backend)

    labeled: list[dict] = []
    for i, doc in enumerate(docs):
        truncated = _truncate_text(doc["text"])
        prompt = TOPIC_PROMPT.format(text=truncated, topics=topics_str)

        try:
            label = _call_llm_structured(prompt, TopicLabel, backend)
            doc["oracle_topic"] = label.topic
            doc["oracle_topic_reasoning"] = label.reasoning
            doc["oracle_backend"] = str(backend)
        except Exception:
            logger.exception("Failed to label document %s", doc.get("doc_id", i))
            doc["oracle_topic"] = "labeling_failed"
            doc["oracle_topic_reasoning"] = "labeling_failed"
            doc["oracle_backend"] = str(backend)

        labeled.append(doc)

        if (i + 1) % 10 == 0:
            logger.info("Topic labeling progress: %d/%d", i + 1, len(docs))

    write_parquet_file(labeled, os.path.join(output_path, "topic_labeled.parquet"))

    failed = sum(1 for d in labeled if d["oracle_topic"] == "labeling_failed")
    logger.info("Topic labeling complete: %d labeled, %d failed", len(labeled) - failed, failed)
