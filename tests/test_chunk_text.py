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

import pytest

from marin.generation.inference import ChunkStrategy, chunk_text


def test_chunk_text_char_strategy():
    example = {"id": "doc", "text": "abcdefghijklmnopqrstuvwxyz"}
    chunks = chunk_text(example, ChunkStrategy.CHAR, chunk_size=10)
    assert [c["text"] for c in chunks] == ["abcdefghij", "klmnopqrst", "uvwxyz"]
    assert [c["id"] for c in chunks] == ["doc_0", "doc_1", "doc_2"]
    assert [c["metadata"]["source_document_id"] for c in chunks] == ["doc", "doc", "doc"]


def test_chunk_text_paragraph_strategy():
    example = {"id": "doc", "text": "para1\npara2\npara3"}
    chunks = chunk_text(example, ChunkStrategy.PARAGRAPH)
    assert [c["text"] for c in chunks] == ["para1", "para2", "para3"]
    assert [c["id"] for c in chunks] == ["doc_0", "doc_1", "doc_2"]
    assert [c["metadata"]["source_document_id"] for c in chunks] == ["doc", "doc", "doc"]


@pytest.mark.parametrize("strategy", [ChunkStrategy.CHAR, ChunkStrategy.PARAGRAPH])
def test_chunk_text_requires_text_field(strategy):
    example = {}
    if strategy is ChunkStrategy.CHAR:
        chunks = chunk_text(example, strategy, chunk_size=5)
    else:
        chunks = chunk_text(example, strategy)
    assert all("text" in c for c in chunks)
