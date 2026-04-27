# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json
from pathlib import Path
from unittest.mock import patch

from datasets import Dataset

from marin.datakit.ingestion_manifest import (
    IdentityTreatment,
    IngestionPolicy,
    IngestionSourceManifest,
    SampleCapConfig,
    SecretRedaction,
    StagingMetadata,
    UsagePolicy,
)
from marin.transform.evaluation.raw_lm_eval import (
    LmEvalRawRenderer,
    LmEvalRawStagingConfig,
    stage_lm_eval_source,
)


def _read_staged_records(output_path: Path, filename: str = "staged.jsonl.gz") -> list[dict]:
    with gzip.open(output_path / filename, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mmlu_fixture() -> dict:
    return {
        "question": "What is 2 + 2?",
        "subject": "elementary_mathematics",
        "choices": ["1", "2", "4", "8"],
        "answer": 2,
    }


def _gsm8k_fixture() -> dict:
    return {
        "question": "Natalia sold 48 clips in April and half as many in May. How many in total?",
        "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n48 + 24 = <<48+24=72>>72.\n#### 72",
    }


def _manifest(
    renderer: LmEvalRawRenderer, *, source_label: str, split: str, subset: str | None
) -> IngestionSourceManifest:
    dataset_key = "cais/mmlu" if renderer is LmEvalRawRenderer.MMLU else "openai/gsm8k"
    return IngestionSourceManifest(
        dataset_key=dataset_key,
        slice_key=f"test/{renderer.value}/{split}",
        source_label=source_label,
        source_urls=(f"https://huggingface.co/datasets/{dataset_key}",),
        source_license="test-only",
        source_format="huggingface_parquet_eval_probe",
        surface_form="prompt_plus_correct_answer",
        policy=IngestionPolicy(
            usage_policy=UsagePolicy.EVAL_ONLY,
            use_policy="Eval-only probe.",
            requires_sanitization=False,
            identity_treatment=IdentityTreatment.PRESERVE,
            secret_redaction=SecretRedaction.NONE,
            contamination_risk="high: held-out eval contamination if reused in training",
            provenance_notes="Pinned revision.",
        ),
        staging=StagingMetadata(
            transform_name="stage_lm_eval_source",
            serializer_name=renderer.value,
            split=split,
            subset=subset,
        ),
        sample_caps=SampleCapConfig(max_examples=10),
    )


def test_stage_lm_eval_source_renders_mmlu_with_choices_and_answer(tmp_path: Path):
    manifest = _manifest(LmEvalRawRenderer.MMLU, source_label="mmlu:test", split="dev", subset="all")
    cfg = LmEvalRawStagingConfig(
        input_path="fake://mmlu",
        output_path=str(tmp_path),
        source_label="mmlu:test",
        renderer_name=LmEvalRawRenderer.MMLU,
        split="dev",
        subset="all",
        source_manifest=manifest,
        content_fingerprint=manifest.fingerprint(),
    )

    with patch(
        "marin.transform.evaluation.raw_lm_eval._load_hf_iterable",
        return_value=iter([_mmlu_fixture()]),
    ):
        result = stage_lm_eval_source(cfg)

    assert result["record_count"] == 1
    records = _read_staged_records(tmp_path)
    assert "Subject: elementary_mathematics" in records[0]["text"]
    assert "A. 1" in records[0]["text"]
    assert "C. 4" in records[0]["text"]
    assert records[0]["provenance"]["renderer"] == LmEvalRawRenderer.MMLU.value


def test_stage_lm_eval_source_renders_gsm8k_answer_verbatim(tmp_path: Path):
    manifest = _manifest(LmEvalRawRenderer.GSM8K, source_label="gsm8k:test", split="train", subset="main")
    cfg = LmEvalRawStagingConfig(
        input_path="fake://gsm8k",
        output_path=str(tmp_path),
        source_label="gsm8k:test",
        renderer_name=LmEvalRawRenderer.GSM8K,
        split="train",
        subset="main",
        source_manifest=manifest,
        content_fingerprint=manifest.fingerprint(),
    )

    with patch(
        "marin.transform.evaluation.raw_lm_eval._load_hf_iterable",
        return_value=iter([_gsm8k_fixture()]),
    ):
        result = stage_lm_eval_source(cfg)

    assert result["record_count"] == 1
    records = _read_staged_records(tmp_path)
    assert "Question: Natalia sold 48 clips" in records[0]["text"]
    assert "#### 72" in records[0]["text"]
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["source_manifest"]["source_label"] == "gsm8k:test"
    assert metadata["materialized_output"]["record_count"] == 1


def test_stage_lm_eval_source_loads_downloaded_parquet_split(tmp_path: Path):
    input_dir = tmp_path / "raw"
    output_dir = tmp_path / "staged"
    input_dir.mkdir()
    Dataset.from_list([_mmlu_fixture()]).to_parquet(input_dir / "dev-00000-of-00001.parquet")
    Dataset.from_list([_mmlu_fixture()]).to_parquet(input_dir / "validation-00000-of-00001.parquet")

    cfg = LmEvalRawStagingConfig(
        input_path=str(input_dir),
        output_path=str(output_dir),
        source_label="mmlu:test",
        renderer_name=LmEvalRawRenderer.MMLU,
        split="dev",
    )
    result = stage_lm_eval_source(cfg)

    assert result["record_count"] == 1
    records = _read_staged_records(output_dir)
    assert records[0]["id"] == "mmlu:test:dev:00000000"
