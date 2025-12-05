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

from dupekit.text_cleaning import clean_text


def test_clean_text():
    assert clean_text("") == ""
    assert clean_text("Hello, World!") == "hello world"
    assert clean_text("  Multiple   spaces\tand\nnewlines. ") == "multiple spaces and newlines"
    assert clean_text("Punctuation!!! Should be removed...") == "punctuation should be removed"
