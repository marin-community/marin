# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Oracle labeling via LLM APIs (Claude, OpenAI).

Labels documents with quality scores (discrete 0-5 rubric) or topic classifications
using Claude or OpenAI as the oracle. Supports batch processing with rate limiting.
"""

import json
import logging
import os
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# FineWeb-Edu style quality rubric prompt
QUALITY_PROMPT_V1 = """\
Evaluate the quality of the following document on a scale from 0 to 5.

Score guidelines:
0 - Incoherent, spam, or machine-generated noise with no meaningful content.
1 - Very low quality: poorly written, mostly boilerplate, ads, or navigation text.
2 - Low quality: some meaningful content but poorly structured, many errors, or very shallow.
3 - Medium quality: coherent and readable but not particularly informative or well-written.
4 - High quality: well-written, informative, and well-structured content.
5 - Exceptional quality: expertly written, deeply informative, educational, or reference-grade.

Respond with ONLY a single integer from 0 to 5.

Document:
{text}
"""

# Topic classification prompt
TOPIC_PROMPT_V1 = """\
Classify the following document into exactly one of these topics:

- science (natural sciences, physics, chemistry, biology)
- technology (computing, engineering, software)
- mathematics (pure math, statistics, applied math)
- medicine (health, clinical, biomedical)
- law (legal documents, regulations, case law)
- finance (economics, business, markets)
- humanities (philosophy, history, literature, arts)
- education (teaching materials, textbooks, tutorials)
- news (journalism, current events, reporting)
- social_media (forums, discussions, comments, social posts)
- reference (encyclopedias, wikis, documentation)
- code (source code, programming, scripts)
- creative_writing (fiction, poetry, storytelling)
- government (policy, public administration)
- other (anything that doesn't fit above)

Respond with ONLY the topic label (one of the words before the parentheses).

Document:
{text}
"""

QUALITY_PROMPTS = {"v1": QUALITY_PROMPT_V1}
TOPIC_PROMPTS = {"v1": TOPIC_PROMPT_V1}

VALID_TOPICS = {
    "science",
    "technology",
    "mathematics",
    "medicine",
    "law",
    "finance",
    "humanities",
    "education",
    "news",
    "social_media",
    "reference",
    "code",
    "creative_writing",
    "government",
    "other",
}


@dataclass
class OracleResult:
    """Artifact returned by label_quality / label_topics."""

    path: str
    """Path to the output JSONL with oracle labels."""
    n_labeled: int
    n_failed: int
    backend: str


def _truncate_text(text: str, max_chars: int = 4000) -> str:
    """Truncate text to fit within API context limits."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...truncated]"


def _call_claude(prompt: str, model: str = "claude-sonnet-4-20250514") -> str:
    """Call Claude API and return the text response."""
    import anthropic

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def _call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call OpenAI API and return the text response."""
    import openai

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        max_tokens=16,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()


def _call_backend(prompt: str, backend: str) -> str:
    if backend == "claude":
        return _call_claude(prompt)
    elif backend == "openai":
        return _call_openai(prompt)
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'claude' or 'openai'.")


def label_quality(
    output_path: str,
    input_path: str,
    backend: str = "claude",
    prompt_version: str = "v1",
    max_retries: int = 3,
    rate_limit_delay: float = 0.5,
) -> OracleResult:
    """Score each document with an oracle quality rubric (0-5).

    Args:
        output_path: Directory to write the labeled JSONL.
        input_path: Path to JSONL with sampled documents (must have "id", "text" fields).
        backend: "claude" or "openai".
        prompt_version: Version key for the quality prompt template.
        max_retries: Retries per document on API failure.
        rate_limit_delay: Seconds to wait between API calls.

    Returns:
        OracleResult with path to labeled JSONL.
    """
    os.makedirs(output_path, exist_ok=True)
    prompt_template = QUALITY_PROMPTS[prompt_version]
    out_file = os.path.join(output_path, "oracle_quality.jsonl")

    docs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    n_labeled = 0
    n_failed = 0

    with open(out_file, "w") as f:
        for doc in docs:
            text = _truncate_text(doc.get("text", ""))
            prompt = prompt_template.format(text=text)

            score = None
            for attempt in range(max_retries):
                try:
                    response = _call_backend(prompt, backend)
                    parsed = int(response)
                    if 0 <= parsed <= 5:
                        score = parsed
                        break
                except (ValueError, TypeError):
                    logger.warning(f"Failed to parse quality score for doc {doc['id']}: {response!r}")
                except Exception as e:
                    logger.warning(f"API error for doc {doc['id']} (attempt {attempt + 1}): {e}")
                    time.sleep(rate_limit_delay * (attempt + 1))

            labeled_doc = {**doc, "oracle_quality_score": score}
            f.write(json.dumps(labeled_doc) + "\n")

            if score is not None:
                n_labeled += 1
            else:
                n_failed += 1
                logger.warning(f"Failed to label doc {doc['id']} after {max_retries} attempts")

            time.sleep(rate_limit_delay)

    logger.info(f"Quality labeling complete: {n_labeled} labeled, {n_failed} failed")
    return OracleResult(path=out_file, n_labeled=n_labeled, n_failed=n_failed, backend=backend)


def label_topics(
    output_path: str,
    input_path: str,
    backend: str = "claude",
    prompt_version: str = "v1",
    max_retries: int = 3,
    rate_limit_delay: float = 0.5,
) -> OracleResult:
    """Assign a topic label to each document using an LLM oracle.

    Args:
        output_path: Directory to write the labeled JSONL.
        input_path: Path to JSONL with sampled documents.
        backend: "claude" or "openai".
        prompt_version: Version key for the topic prompt template.
        max_retries: Retries per document on API failure.
        rate_limit_delay: Seconds to wait between API calls.

    Returns:
        OracleResult with path to labeled JSONL.
    """
    os.makedirs(output_path, exist_ok=True)
    prompt_template = TOPIC_PROMPTS[prompt_version]
    out_file = os.path.join(output_path, "oracle_topics.jsonl")

    docs = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))

    n_labeled = 0
    n_failed = 0

    with open(out_file, "w") as f:
        for doc in docs:
            text = _truncate_text(doc.get("text", ""))
            prompt = prompt_template.format(text=text)

            topic = None
            for attempt in range(max_retries):
                try:
                    response = _call_backend(prompt, backend)
                    response_lower = response.lower().strip()
                    if response_lower in VALID_TOPICS:
                        topic = response_lower
                        break
                    else:
                        logger.warning(f"Invalid topic {response!r} for doc {doc['id']}")
                except Exception as e:
                    logger.warning(f"API error for doc {doc['id']} (attempt {attempt + 1}): {e}")
                    time.sleep(rate_limit_delay * (attempt + 1))

            labeled_doc = {**doc, "oracle_topic_label": topic}
            f.write(json.dumps(labeled_doc) + "\n")

            if topic is not None:
                n_labeled += 1
            else:
                n_failed += 1
                logger.warning(f"Failed to label doc {doc['id']} after {max_retries} attempts")

            time.sleep(rate_limit_delay)

    logger.info(f"Topic labeling complete: {n_labeled} labeled, {n_failed} failed")
    return OracleResult(path=out_file, n_labeled=n_labeled, n_failed=n_failed, backend=backend)
