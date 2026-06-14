# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.datakit.download.automathtext_v2 import automathtext_v2_normalize_steps

EXPECTED_HF_DATASET_ID = "OpenSQZ/AutoMathText-V2"
EXPECTED_HF_REVISION = "2a8d19c8edff7eeab35ffaa36f7845e86e2b3417"

EXPECTED_SOURCES: dict[str, tuple[str, tuple[str, ...], str]] = {
    "automathtext_v2/math_web": ("math_web", ("math_web/*/*.parquet",), "math_web"),
    "automathtext_v2/reasoning_qa": ("reasoning_qa", ("reasoning_qa/*/*.parquet",), "reasoning_qa"),
}


def test_automathtext_v2_steps_download_only_selected_domains():
    steps_by_name = automathtext_v2_normalize_steps()

    assert set(steps_by_name) == set(EXPECTED_SOURCES)

    for marin_name, (subset_name, hf_urls_glob, data_subdir) in EXPECTED_SOURCES.items():
        steps = steps_by_name[marin_name]
        assert len(steps) == 2
        download = steps[0]
        normalized = steps[1]

        assert download.override_output_path == f"raw/automathtext_v2/{subset_name}-{EXPECTED_HF_REVISION[:7]}"
        assert download.hash_attrs["hf_dataset_id"] == EXPECTED_HF_DATASET_ID
        assert download.hash_attrs["revision"] == EXPECTED_HF_REVISION
        assert download.hash_attrs["hf_urls_glob"] == list(hf_urls_glob)
        assert normalized.name == f"normalized/{marin_name}"
        assert normalized.deps == [download]
        assert normalized.hash_attrs["relative_input_path"] == data_subdir
