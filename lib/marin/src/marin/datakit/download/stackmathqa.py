# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""math-ai/StackMathQA download and transform.

StackMathQA is a Math StackExchange-style Q&A corpus with one canonical
``stackmathqafull-qalist`` view containing one row per question and an
``A_list`` of answers. This transform keeps that source-native grouping and
renders one LM document per question for math midtraining.
"""

import hashlib

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, counters, load_jsonl

from marin.datakit.download.huggingface import download_hf_step
from marin.datakit.normalize import normalize_step
from marin.execution.step_spec import StepSpec

HF_DATASET_ID = "math-ai/StackMathQA"
HF_REVISION = "5a3e18f6fa122652e1959c3cc3e714758778fdad"

FULL_QALIST_CONFIG = "stackmathqafull-qalist"
FULL_QALIST_GLOB = "preprocessed/stackexchange-math/*.jsonl"

RAW_STEP_NAME = "raw/stackmathqa/full-qalist"
PROCESSED_STEP_NAME = "processed/stackmathqa/full-qalist"
NORMALIZED_STEP_NAME = "normalized/stackmathqa/full-qalist"

# Provisional until the normalized source is materialized and measured with the
# canonical Marin tokenizer.
STACKMATHQA_FULL_QALIST_ROUGH_TOKENS_B = 0.85


def _clean_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    return text


def _optional_text(metadata: dict, key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str):
        return ""

    return value.strip()


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value

    if not isinstance(value, str):
        return None

    text = value.strip()
    if text.lstrip("-").isdigit():
        return int(text)

    return None


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _render_text(question: str, answers: list[str]) -> str:
    parts = [f"<question>\n{question}\n</question>"]
    for index, answer in enumerate(answers):
        parts.append(f'<answer index="{index}">\n{answer}\n</answer>')

    return "\n\n".join(parts)


def _metadata(row: dict) -> dict:
    value = row.get("meta")
    if isinstance(value, dict):
        return value

    return {}


def row_to_doc(row: dict) -> list[dict]:
    question = _clean_text(row.get("Q"))
    if question is None:
        counters.increment("stackmathqa/full_qalist/dropped_empty_question")
        return []

    answer_values = row.get("A_list")
    if not isinstance(answer_values, list):
        counters.increment("stackmathqa/full_qalist/dropped_missing_answer_list")
        return []

    answers = [answer for value in answer_values if (answer := _clean_text(value)) is not None]
    if not answers:
        counters.increment("stackmathqa/full_qalist/dropped_empty_answer_list")
        return []

    text = _render_text(question, answers)
    metadata = _metadata(row)
    url = _optional_text(metadata, "url")
    question_score_raw = _optional_text(metadata, "question_score")

    counters.increment("stackmathqa/full_qalist/kept")
    counters.increment("stackmathqa/full_qalist/answers_kept", len(answers))

    return [
        {
            "id": _hash_text(url or text),
            "text": text,
            "source": HF_DATASET_ID,
            "stackmathqa_config": FULL_QALIST_CONFIG,
            "url": url,
            "language": _optional_text(metadata, "language"),
            "timestamp": _optional_text(metadata, "timestamp"),
            "stackmathqa_source": _optional_text(metadata, "source"),
            "question_hash": _hash_text(question),
            "num_answers_rendered": len(answers),
            "answer_count": _optional_int(metadata.get("answer_count")),
            "question_score": _optional_int(metadata.get("question_score")),
            "question_score_raw": question_score_raw,
        }
    ]


def transform(input_path: str, output_path: str) -> None:
    pipeline = (
        Dataset.from_files(f"{input_path}/{FULL_QALIST_GLOB}")
        .flat_map(load_jsonl)
        .flat_map(row_to_doc)
        .write_parquet(f"{output_path}/data-{{shard:05d}}-of-{{total:05d}}.parquet", skip_existing=True)
    )
    ctx = ZephyrContext(name="stackmathqa-full-qalist-transform", resources=ResourceConfig(cpu=2, ram="16g"))
    ctx.execute(pipeline)


def download_stackmathqa_full_qalist_step() -> StepSpec:
    """Download and transform StackMathQA full qlist rows into question-grouped documents."""
    dl = download_hf_step(
        RAW_STEP_NAME,
        hf_dataset_id=HF_DATASET_ID,
        revision=HF_REVISION,
        hf_urls_glob=[FULL_QALIST_GLOB],
    )

    return StepSpec(
        name=PROCESSED_STEP_NAME,
        deps=[dl],
        fn=lambda output_path: transform(
            input_path=dl.output_path,
            output_path=output_path,
        ),
        hash_attrs={"version": "v1", "config": FULL_QALIST_CONFIG},
    )


def stackmathqa_full_qalist_normalize_steps() -> tuple[StepSpec, ...]:
    """Return the ``(download+transform, normalize)`` chain for StackMathQA qlist."""
    processed = download_stackmathqa_full_qalist_step()
    return (
        processed,
        normalize_step(name=NORMALIZED_STEP_NAME, download=processed),
    )
