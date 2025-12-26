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

import json
import logging
import random
import time
from collections.abc import Callable

import fsspec
from zephyr import Backend, Dataset

logger = logging.getLogger("ray")


def sample_file(input_file_path: str, label_weights: dict[int, float]):
    """Sample records from a file based on label weights.

    Reads an input file with a "label" field and yields sampled examples
    based on the percentage weights for each label.

    Args:
        input_file_path: The path to the input file (must be .jsonl.gz).
        label_weights: A dictionary mapping each label to the sampling percentage (0.0-1.0).

    Yields:
        Sampled example dictionaries.

    Example:
        >>> from zephyr import Dataset, create_backend
        >>> backend = create_backend("ray", max_parallelism=10)
        >>> ds = (Dataset
        ...     .from_files("/input/**/*.jsonl.gz")
        ...     .map(lambda path: sample_file(path, {0: 0.5, 1: 1.0}))
        ...     .write_jsonl("/output/sampled-{shard:05d}.jsonl.gz")
        ... )
        >>> list(backend.execute(ds))
    """
    # Add exponential backoff retry logic
    max_retries = 5
    base_delay = 1
    for attempt in range(max_retries):
        try:
            assert input_file_path.endswith("jsonl.gz"), "Only jsonl.gz files are supported for now."

            # Read all examples and group by label
            label_groups = {label: [] for label in label_weights.keys()}

            with fsspec.open(input_file_path, "r", compression="gzip") as f:
                for line in f:
                    example = json.loads(line)
                    label = example["label"]
                    if label in label_weights:
                        label_groups[label].append(example)

            # Sample from each group according to weights
            for label, examples in label_groups.items():
                if examples:
                    weight = label_weights[label]
                    num_to_sample = int(len(examples) * weight)
                    if num_to_sample > 0:
                        sampled = random.sample(examples, min(num_to_sample, len(examples)))
                        for example in sampled:
                            yield example
            break  # Success, exit retry loop

        except Exception:
            if attempt == max_retries - 1:  # Last attempt
                raise  # Re-raise the last exception
            delay = base_delay * (2**attempt)  # Exponential backoff
            time.sleep(delay)


def convert_labeled_document_to_score(example: dict, extract_score_fn: Callable[[str], int]):
    """Convert a single labeled document into a record with parsed score.

    Args:
        example: Record with "text" and "generated_text" fields
        extract_score_fn: Function to extract the score from generated text.

    Returns:
        Record with "text" and "label" fields if valid, None otherwise.

    Note:
        Examples with unparseable scores (score == -1) or empty text return None.
    """
    text = example["text"]
    generated_text = example["generated_text"]
    score = extract_score_fn(generated_text)

    if score != -1 and text != "":
        return {"text": text, "label": score}
    elif score == -1:
        if "id" in example:
            logger.warning(f"Failed to parse score for example {example['id']}: {generated_text}")
        else:
            logger.warning(f"Failed to parse score for example: {generated_text}")

    return None


class DatasetSampler:
    """Sample a dataset by label weights."""

    def __init__(self, input_path: str, output_path: str, label_weights: dict[int, float]):
        self.input_path = input_path
        self.output_path = output_path
        self.label_weights = label_weights

    def sample_dataset(self):
        """Sample the dataset and write output files.

        Returns:
            Sequence of output file paths written.
        """
        pipeline = (
            Dataset.from_files(f"{self.input_path}/**/*.jsonl.gz")
            .map(lambda path: sample_file(path, label_weights=self.label_weights))
            .write_jsonl(f"{self.output_path}/sampled-{{shard:05d}}-of-{{total:05d}}.jsonl.gz")
        )
        return Backend.execute(pipeline)
