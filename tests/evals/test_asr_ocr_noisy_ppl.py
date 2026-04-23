# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import gzip
import json

from experiments.evals.asr_ocr_noisy_ppl import (
    ASR_OCR_NOISY_DATASET_ROOT,
    ASR_OCR_NOISY_SLICES,
    NoisyAsrOcrRawConfig,
    NoisyTextFamily,
    NoisyTextSlice,
    linearize_noisy_clean_row,
    materialize_noisy_asr_ocr_raw,
    noisy_asr_ocr_raw_validation_sets,
)
from marin.processing.tokenize import HfDatasetSpec


def test_linearize_noisy_clean_row_uses_first_hypothesis_and_preserves_reference():
    row = {"hyps": ["THE CAT SAT", "THE CATS AT"], "ref": "the cat sat"}

    linearized = linearize_noisy_clean_row(row, noisy_key="hyps", clean_key="ref")

    assert linearized == {"noisy_text": "THE CAT SAT", "clean_text": "the cat sat"}


def test_noisy_asr_ocr_raw_validation_sets_registers_clean_and_noisy_slices():
    class _SyntheticRawStep:
        def cd(self, path: str) -> str:
            return f"gs://synthetic/{path}"

    datasets = noisy_asr_ocr_raw_validation_sets(noisy_asr_ocr_raw=_SyntheticRawStep())

    first_slice = ASR_OCR_NOISY_SLICES[0]
    noisy_key = f"{ASR_OCR_NOISY_DATASET_ROOT}/{first_slice.registry_name}/noisy"
    clean_key = f"{ASR_OCR_NOISY_DATASET_ROOT}/{first_slice.registry_name}/clean"

    assert datasets[noisy_key].text_key == "noisy_text"
    assert datasets[clean_key].text_key == "clean_text"
    assert isinstance(datasets[noisy_key].input_path, str)
    assert f"/{first_slice.registry_name}/data-*.jsonl.gz" in datasets[noisy_key].input_path
    assert datasets[noisy_key].tags[-1] == "variant:noisy"
    assert datasets[clean_key].tags[-1] == "variant:clean"


def test_materialize_noisy_asr_ocr_raw_respects_per_slice_cap(tmp_path, monkeypatch):
    from experiments.evals import asr_ocr_noisy_ppl

    rows = [
        {"hyps": ["NOISY ONE"], "ref": "clean one"},
        {"hyps": ["NOISY TWO"], "ref": "clean two"},
        {"hyps": ["NOISY THREE"], "ref": "clean three"},
    ]

    def _fake_load_dataset(*args, **kwargs):
        del args, kwargs
        return rows

    monkeypatch.setattr(asr_ocr_noisy_ppl, "load_dataset", _fake_load_dataset)
    slice_ = NoisyTextSlice(
        registry_name="synthetic",
        family=NoisyTextFamily.ASR,
        source_url="https://example.com",
        hf_dataset=HfDatasetSpec(id="synthetic/dataset"),
        split="test",
        noisy_key="hyps",
        clean_key="ref",
        max_rows=2,
    )

    materialize_noisy_asr_ocr_raw(NoisyAsrOcrRawConfig(output_path=str(tmp_path), slices=(slice_,)))

    with gzip.open(tmp_path / "synthetic" / "data-00000-of-00001.jsonl.gz", "rt") as handle:
        materialized = [json.loads(line) for line in handle]

    assert materialized == [
        {"noisy_text": "NOISY ONE", "clean_text": "clean one"},
        {"noisy_text": "NOISY TWO", "clean_text": "clean two"},
    ]
