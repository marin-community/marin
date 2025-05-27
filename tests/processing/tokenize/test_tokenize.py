import pytest

from marin.processing.tokenize.tokenize import TokenizeConfig


class MockInputName:
    def __init__(self, name: str):
        self.name = name


# Dummy values for other required TokenizeConfig fields
DUMMY_CACHE_PATH = "/dummy/cache"
DUMMY_TOKENIZER = "dummy_tokenizer"
DUMMY_VALIDATION_PATHS = []


def test_valid_train_url():
    """Tests that a valid train URL does not raise an error."""
    try:
        TokenizeConfig(
            train_paths=["gs://bucket/data/train/file.jsonl"],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    except ValueError:
        pytest.fail("ValueError raised for a valid train URL")


def test_train_url_with_test_whole_word():
    """Tests that a train URL with '\\btest\\b' raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=["gs://bucket/data/test/file.jsonl"],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/test/file.jsonl" in str(excinfo.value)


def test_train_url_with_validation():
    """Tests that a train URL with 'validation' raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=["gs://bucket/data/validation/file.jsonl"],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/validation/file.jsonl" in str(excinfo.value)


def test_train_url_with_test_substring_not_whole_word():
    """Tests that 'test' as a substring (not whole word) does not raise an error."""
    try:
        TokenizeConfig(
            train_paths=["gs://bucket/data/latest_updates/file.jsonl"],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    except ValueError:
        pytest.fail("ValueError raised for 'test' as a substring, not a whole word")


def test_multiple_train_urls_one_invalid():
    """Tests that if one of multiple URLs is invalid, a ValueError is raised."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=[
                "gs://bucket/data/train/file1.jsonl",
                "gs://bucket/data/test/file2.jsonl",
                "gs://bucket/data/train/file3.jsonl",
            ],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/test/file2.jsonl" in str(excinfo.value)


def test_empty_train_paths_no_error():
    """Tests that if train_paths is empty, no validation error occurs (other validation might fail)."""
    try:
        TokenizeConfig(
            train_paths=[],
            validation_paths=DUMMY_VALIDATION_PATHS,  # Must provide validation_paths if train_paths is empty
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    except ValueError as e:
        # We only care about the custom URL validation, not the check for empty paths
        if "contains a forbidden pattern" in str(e):
            pytest.fail("ValueError for forbidden pattern raised with empty train_paths")


def test_train_url_with_test_in_filename():
    """Tests that a train URL with 'test' in the filename raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=["gs://bucket/data/train/file_test.jsonl"],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/train/file_test.jsonl" in str(excinfo.value)


def test_train_url_with_validation_in_filename():
    """Tests that a train URL with 'validation' in the filename raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=["gs://bucket/data/train/file_validation.jsonl"],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/train/file_validation.jsonl" in str(excinfo.value)


def test_valid_train_inputname():
    """Tests that a valid train InputName.name does not raise an error."""
    try:
        TokenizeConfig(
            train_paths=[MockInputName("gs://bucket/data/train/file.jsonl")],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    except ValueError:
        pytest.fail("ValueError raised for a valid train InputName.name")


def test_train_inputname_with_test_whole_word():
    """Tests that a train InputName.name with '\\btest\\b' raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=[MockInputName("gs://bucket/data/test/file.jsonl")],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/test/file.jsonl" in str(excinfo.value)


def test_train_inputname_with_validation():
    """Tests that a train InputName.name with 'validation' raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=[MockInputName("gs://bucket/data/validation/file.jsonl")],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/validation/file.jsonl" in str(excinfo.value)


def test_mixed_paths_one_invalid_inputname():
    """Tests mixed paths with one invalid InputName.name, expecting ValueError."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=[
                "gs://bucket/data/train/file1.jsonl",
                MockInputName("gs://bucket/data/test/file2.jsonl"),
                "gs://bucket/data/train/file3.jsonl",
            ],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/test/file2.jsonl" in str(excinfo.value)


def test_inputname_with_test_substring_not_whole_word():
    """Tests that 'test' as a substring in InputName.name (not whole word) does not raise an error."""
    try:
        TokenizeConfig(
            train_paths=[MockInputName("gs://bucket/data/latest_updates/file.jsonl")],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    except ValueError:
        pytest.fail("ValueError raised for 'test' as a substring in InputName.name")


def test_inputname_with_test_in_filename():
    """Tests that an InputName.name with 'test' in the filename raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=[MockInputName("gs://bucket/data/train/file_test.jsonl")],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/train/file_test.jsonl" in str(excinfo.value)


def test_inputname_with_validation_in_filename():
    """Tests that an InputName.name with 'validation' in the filename raises a ValueError."""
    with pytest.raises(ValueError) as excinfo:
        TokenizeConfig(
            train_paths=[MockInputName("gs://bucket/data/train/file_validation.jsonl")],
            validation_paths=DUMMY_VALIDATION_PATHS,
            cache_path=DUMMY_CACHE_PATH,
            tokenizer=DUMMY_TOKENIZER,
        )
    assert "contains a forbidden pattern ('test' or 'validation')" in str(excinfo.value)
    assert "gs://bucket/data/train/file_validation.jsonl" in str(excinfo.value)
