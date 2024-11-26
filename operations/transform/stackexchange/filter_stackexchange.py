import dataclasses

import draccus
import pandas as pd
import ray

from marin.core.runtime import cached_or_construct_output, map_files_in_directory


@dataclasses.dataclass
class FilterStackExchangeConfig:
    input_path: str
    output_path: str
    min_vote_threshold: int = 10
    remove_duplicate_questions: bool = True


@ray.remote
@cached_or_construct_output
def _process_file(input_file_path: str, output_file_path: str, config: FilterStackExchangeConfig):
    """Filters the stackexchange dataset by votes and removes duplicate questions.

    Accepts and outputs a dolma formatted file.
    """

    df = pd.read_json(input_file_path, lines=True)
    df["votes"] = df["metadata"].apply(lambda x: x["votes"])
    df["question_id"] = df["metadata"].apply(lambda x: x["id"])

    if config.remove_duplicate_questions:
        df = df.drop_duplicates(subset=["question_id"], keep="first")

    df = df[df["votes"] >= config.min_vote_threshold]

    # Add orient='records' parameter when writing JSONL
    df.to_json(output_file_path, orient="records", lines=True, compression="gzip")


@ray.remote
def _process_dataset(config: FilterStackExchangeConfig):
    responses = map_files_in_directory(
        _process_file.remote, config.input_path, "*.jsonl.gz", config.output_path, config=config
    )
    ray.get(responses)


@draccus.wrap()
def filter_stackexchange(config: FilterStackExchangeConfig):
    ray.get(_process_dataset.remote(config))


if __name__ == "__main__":
    filter_stackexchange()
