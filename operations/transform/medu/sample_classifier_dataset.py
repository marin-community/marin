import json
import os
import random
import time
from dataclasses import dataclass

import draccus
import fsspec
import ray

from marin.utils import fsspec_glob, fsspec_mkdirs, rebase_file_path


@dataclass
class SampleClassifierDatasetConfig:
    input_path: str
    output_path: str
    label_weights: dict[str, float]  # Maps label value (1-5) to desired sampling weight
    input_filetype: str = "jsonl.gz"
    output_filetype: str = "jsonl.gz"


@ray.remote(memory=64 * 1024 * 1024 * 1024)
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
                    label = str(example["label"])
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


@draccus.wrap()
def sample_classifier_dataset_func(config: SampleClassifierDatasetConfig):
    # Get all input files
    files = fsspec_glob(os.path.join(config.input_path, f"**/*.{config.input_filetype}"))
    if not files:
        raise FileNotFoundError(f"No {config.input_filetype} files found in {config.input_path}")

    # Process each file
    tasks = []
    for input_file in files:
        output_file = rebase_file_path(config.input_path, input_file, config.output_path)
        fsspec_mkdirs(os.path.dirname(output_file))
        tasks.append(sample_file.remote(input_file, output_file, config.label_weights))

    # Wait for all tasks to complete
    ray.get(tasks)


if __name__ == "__main__":
    sample_classifier_dataset_func()
