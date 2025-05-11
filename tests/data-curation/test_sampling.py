import fsspec

from marin.classifiers.utils import create_dataset_shard, reservoir_sample
from marin.utils import fsspec_rm

TEST_OUTPUT_PATH = "gs://marin-us-east5/documents/test-sampling.jsonl.gz"


def test_sample_document_matches_keeping_all_examples(test_file_path: str):
    # Remove the success file if it exists
    fsspec_rm(f"{TEST_OUTPUT_PATH}.SUCCESS")

    num_examples = 0
    with fsspec.open(test_file_path, "r", compression="gzip") as f:
        for _ in f:
            num_examples += 1

    create_dataset_shard(
        test_file_path,
        TEST_OUTPUT_PATH,
        label_func=None,
        input_attr_file_paths=[],
        sampling_rate=1.0,
        seed=42,
        columns_to_keep=["text"],
    )

    num_examples_sampled = 0
    with fsspec.open(TEST_OUTPUT_PATH, "r", compression="gzip") as f:
        for _ in f:
            num_examples_sampled += 1

    assert num_examples_sampled == num_examples, f"Got {num_examples_sampled} examples, expected {num_examples}"


def test_reservoir_sample_matches_expected_number_of_examples(test_file_path: str):
    fsspec_rm(f"{TEST_OUTPUT_PATH}.SUCCESS")

    num_examples = 0
    with fsspec.open(test_file_path, "r", compression="gzip") as f:
        for _ in f:
            num_examples += 1

    expected_num_examples_sampled = int(num_examples * 0.5)
    reservoir_sample(
        [test_file_path],
        TEST_OUTPUT_PATH,
        sample_size=expected_num_examples_sampled,
        seed=42,
    )

    num_examples_sampled = 0
    with fsspec.open(TEST_OUTPUT_PATH, "r", compression="gzip") as f:
        for _ in f:
            num_examples_sampled += 1

    assert (
        num_examples_sampled == expected_num_examples_sampled
    ), f"Got {num_examples_sampled} examples, expected {expected_num_examples_sampled}"


def test_multiple_file_reservoir_sample_matches_expected_number_of_examples(test_file_path: str):
    fsspec_rm(f"{TEST_OUTPUT_PATH}.SUCCESS")

    num_examples = 0
    with fsspec.open(test_file_path, "r", compression="gzip") as f:
        for _ in f:
            num_examples += 1

    expected_num_examples_sampled = 2 * num_examples
    reservoir_sample(
        [test_file_path] * 3,
        TEST_OUTPUT_PATH,
        sample_size=expected_num_examples_sampled,
        seed=42,
    )

    num_examples_sampled = 0
    with fsspec.open(TEST_OUTPUT_PATH, "r", compression="gzip") as f:
        for _ in f:
            num_examples_sampled += 1

    assert (
        num_examples_sampled == expected_num_examples_sampled
    ), f"Got {num_examples_sampled} examples, expected {expected_num_examples_sampled}"
