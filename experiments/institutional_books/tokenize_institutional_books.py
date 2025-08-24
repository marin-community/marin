#!/usr/bin/env python3
"""Download, tokenize, and train on the Institutional Books dataset."""

import dataclasses
import logging

from experiments.defaults import default_download, default_tokenize
from experiments.speedrun.llama_75m.llama_75m import speedrun_config as llama75m_speedrun
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import default_speedrun

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")

# Step 1: download the raw dataset
institutional_books_raw = default_download(
    name="raw/institutional-books-1.0",
    hf_dataset_id="institutional/institutional-books-1.0",
    revision="d2f504a",
    override_output_path="raw/institutional-books-d2f504a",
)

# Step 2: tokenize the dataset using the Llama 3 tokenizer
institutional_books_tokenized = default_tokenize(
    name="institutional-books",
    dataset=institutional_books_raw,
    tokenizer="meta-llama/Meta-Llama-3.1-8B",
)

# Step 3: reuse the 75M Llama speedrun config but swap in our tokenized dataset
institutional_books_speedrun_config = dataclasses.replace(
    llama75m_speedrun,
    description="75M Llama trained on Institutional Books",
    tokenized_dataset=institutional_books_tokenized,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            institutional_books_raw,
            institutional_books_tokenized,
            *default_speedrun("institutional-books-75m", institutional_books_speedrun_config),
        ]
    )
