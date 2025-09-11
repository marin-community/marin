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
from marin.transform.books.transform_books import FilterBooksByTextConfig, filter_books_by_text

# -----------------------------------------------------------------------------
# Editable parameters
# -----------------------------------------------------------------------------

# Substring to search for (can be title phrase, etc.)
SUBSTRING = "They were careless people, Tom and Daisy – they smashed up things and creatures and then retreated back into their money or their vast carelessness, or whatever it was that kept them together, and let other people clean up the mess"  # noqa: E501, RUF001

# Whether the substring match should be case sensitive
CASE_SENSITIVE = False

# -----------------------------------------------------------------------------
# Step definition
# -----------------------------------------------------------------------------

filter_step = ExecutorStep(
    name="documents/books/great_gatsby",
    fn=filter_books_by_text,  # uses the new parallel implementation under the hood
    config=FilterBooksByTextConfig(
        # Directory (or GCS prefix) that contains many book shards
        input_path="gs://marin-us-central2/raw/books3/",  # <-- change to your directory or prefix
        # The executor will construct an experiment-scoped output file like `<experiment>/filtered.jsonl.gz`
        output_path=this_output_path(),
        substring=SUBSTRING,
        case_sensitive=CASE_SENSITIVE,
    ),
)

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    executor_main(steps=[filter_step])
