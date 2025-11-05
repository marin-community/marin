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

import pytest

from marin.execution import InputName
from marin.processing.tokenize.tokenize import TokenizeConfig

# Dummy values for other required TokenizeConfig fields
DUMMY_CACHE_PATH = "/dummy/cache"
DUMMY_TOKENIZER = "dummy_tokenizer"
DUMMY_VALIDATION_PATHS = []


@pytest.mark.parametrize(
    "train_paths, should_error, expected_error_path",
    [
        (["gs://bucket/data/train/file.jsonl"], False, None),
        (["gs://bucket/data/test/file.jsonl"], True, "gs://bucket/data/test/file.jsonl"),
        (["gs://bucket/data/validation/file.jsonl"], True, "gs://bucket/data/validation/file.jsonl"),
        (["gs://bucket/data/latest_updates/file.jsonl"], False, None),
        (
            [
                "gs://bucket/data/train/file1.jsonl",
                "gs://bucket/data/test/file2.jsonl",
                "gs://bucket/data/train/file3.jsonl",
            ],
            True,
            "gs://bucket/data/test/file2.jsonl",
        ),
        ([], False, None),
    ],
)
def test_train_paths_variants(train_paths, should_error, expected_error_path):
    if should_error:
        with pytest.raises(ValueError) as excinfo:
            TokenizeConfig(
                train_paths=train_paths,
                validation_paths=DUMMY_VALIDATION_PATHS,
                cache_path=DUMMY_CACHE_PATH,
                tokenizer=DUMMY_TOKENIZER,
            )
        assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
        if expected_error_path:
            assert expected_error_path in str(excinfo.value)
    else:
        try:
            TokenizeConfig(
                train_paths=train_paths,
                validation_paths=DUMMY_VALIDATION_PATHS,
                cache_path=DUMMY_CACHE_PATH,
                tokenizer=DUMMY_TOKENIZER,
            )
        except ValueError as e:
            if "contains a forbidden pattern" in str(e):
                pytest.fail("Unexpected ValueError for valid path")


@pytest.mark.parametrize(
    "input_name, should_error",
    [
        (InputName.hardcoded("gs://bucket/data/train/file.jsonl"), False),
        (InputName.hardcoded("gs://bucket/data/test/file.jsonl"), True),
        (InputName.hardcoded("gs://bucket/data/validation/file.jsonl"), True),
        (InputName.hardcoded("gs://bucket/data/latest_updates/file.jsonl"), False),
        (InputName.hardcoded("gs://bucket/data/train/file_test.jsonl"), True),
        (InputName.hardcoded("gs://bucket/data/train/file_validation.jsonl"), True),
    ],
)
def test_inputname_variants(input_name, should_error):
    if should_error:
        with pytest.raises(ValueError) as excinfo:
            TokenizeConfig(
                train_paths=[input_name],
                validation_paths=DUMMY_VALIDATION_PATHS,
                cache_path=DUMMY_CACHE_PATH,
                tokenizer=DUMMY_TOKENIZER,
            )
        assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
        assert input_name.name in str(excinfo.value)
    else:
        try:
            TokenizeConfig(
                train_paths=[input_name],
                validation_paths=DUMMY_VALIDATION_PATHS,
                cache_path=DUMMY_CACHE_PATH,
                tokenizer=DUMMY_TOKENIZER,
            )
        except ValueError as e:
            if "contains a forbidden pattern" in str(e):
                pytest.fail("Unexpected ValueError for valid InputName")


def test_mixed_paths_one_invalid_inputname():
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=[
                "gs://bucket/data/train/file1.jsonl",
                InputName.hardcoded("gs://bucket/data/test/file2.jsonl"),
                "gs://bucket/data/train/file3.jsonl",
            ],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/test/file2.jsonl" in str(excinfo.value)
