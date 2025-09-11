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

from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.transform.books.transform_books_window import (
    SingleBookTokenWindowConfig,
    transform_single_book_to_token_window_sft,
)

sliding_window_step = ExecutorStep(
    name="documents/books_token_window/great_gatsby_llama3",
    fn=transform_single_book_to_token_window_sft,
    config=SingleBookTokenWindowConfig(
        input_path="gs://marin-us-central2/documents/books/great_gatsby-b71c3c/matches.jsonl.gz",
        output_path=this_output_path(),
        tokenizer_name="meta-llama/Llama-3.1-8B",
        prompt_tokens=50,
        response_tokens=50,
        slice_length=2000,
        cursor_inc=10,
        row_index=0,
        shard_size=10000,
    ),
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(steps=[sliding_window_step])
