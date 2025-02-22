import json
import random
import time
from collections.abc import Callable
from typing import ClassVar

import fsspec
import ray

from marin.core.runtime import TaskConfig, map_files_in_directory


@ray.remote
def sample_file(input_file_path: str, output_file_path: str, label_weights: dict[int, float]):
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
            sampled_examples = []
            for label, examples in label_groups.items():
                if examples:
                    weight = label_weights[label]
                    num_to_sample = int(len(examples) * weight)
                    if num_to_sample > 0:
                        sampled = random.sample(examples, min(num_to_sample, len(examples)))
                        sampled_examples.extend(sampled)

            # Write sampled examples
            with fsspec.open(output_file_path, "w", compression="gzip") as f:
                for example in sampled_examples:
                    f.write(json.dumps(example) + "\n")
            break  # Success, exit retry loop

        except Exception:
            if attempt == max_retries - 1:  # Last attempt
                raise  # Re-raise the last exception
            delay = base_delay * (2**attempt)  # Exponential backoff
            time.sleep(delay)


# Stage 4: Convert labeled documents into scores
@ray.remote
def convert_labeled_documents_to_scores(
    input_file_path: str, output_file_path: str, score_values: list[int], extract_score_fn: Callable[[str], int]
):
    score_distribution = {k: 0 for k in score_values}
    with fsspec.open(input_file_path, "r", compression="gzip") as input_file:
        with fsspec.open(output_file_path, "w", compression="gzip") as output_file:
            for line in input_file:
                example = json.loads(line)
                text = example["text"]
                generated_text = example["generated_text"]
                score = extract_score_fn(generated_text)

                if score != -1 and text != "":
                    output_file.write(json.dumps({"text": text, "label": score}) + "\n")
                    score_distribution[score] += 1

    return score_distribution


class DatasetOutputProcessor:
    def __init__(self, input_path: str, output_path: str, score_values: list[int]):
        self.input_path = input_path
        self.output_path = output_path
        self.score_values = score_values
        self.score_distribution = {}

    @staticmethod
    def extract_score(text: str) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    def convert_dataset(self):
        responses = map_files_in_directory(
            convert_labeled_documents_to_scores.remote,
            self.input_path,
            "**/*.jsonl.gz",
            self.output_path,
            TaskConfig(),
            False,
            self.score_values,
            self.extract_score,
        )

        for response in ray.get(responses):
            self.score_distribution.update(response)

        return self.score_distribution


class MeduDatasetOutputProcessor(DatasetOutputProcessor):
    SCORE_OPTIONS_DICT: ClassVar[dict[str, int]] = {
        "Great": 5,
        "Good": 4,
        "Okay": 3,
        "Poor": 2,
        "Useless": 1,
    }

    def __init__(self, input_path: str, output_path: str):
        super().__init__(input_path, output_path, list(MeduDatasetOutputProcessor.SCORE_OPTIONS_DICT.values()))

    @staticmethod
    def extract_score(text: str) -> int:
        for key in MeduDatasetOutputProcessor.SCORE_OPTIONS_DICT:
            if key in text:
                return MeduDatasetOutputProcessor.SCORE_OPTIONS_DICT[key]
        return -1


class DatasetSampler:
    def __init__(self, input_path: str, output_path: str, label_weights: dict[int, float]):
        self.input_path = input_path
        self.output_path = output_path
        self.label_weights = label_weights

    def extract_score(self, text: str) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    def sample_dataset(self):
        responses = map_files_in_directory(
            sample_file.remote,
            self.input_path,
            "**/*.jsonl.gz",
            self.output_path,
            TaskConfig(),
            False,
            self.label_weights,
        )
        return ray.get(responses)
