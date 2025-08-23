import pytest

from marin.generation.inference import ChunkStrategy, chunk_text


def test_chunk_text_char_strategy():
    example = {"id": "doc", "text": "abcdefghijklmnopqrstuvwxyz"}
    chunks = chunk_text(example, ChunkStrategy.CHAR, chunk_size=10)
    assert [c["text"] for c in chunks] == ["abcdefghij", "klmnopqrst", "uvwxyz"]
    assert [c["id"] for c in chunks] == ["doc_0", "doc_1", "doc_2"]


def test_chunk_text_paragraph_strategy():
    example = {"id": "doc", "text": "para1\npara2\npara3"}
    chunks = chunk_text(example, ChunkStrategy.PARAGRAPH)
    assert [c["text"] for c in chunks] == ["para1", "para2", "para3"]
    assert [c["id"] for c in chunks] == ["doc_0", "doc_1", "doc_2"]


@pytest.mark.parametrize("strategy", [ChunkStrategy.CHAR, ChunkStrategy.PARAGRAPH])
def test_chunk_text_requires_text_field(strategy):
    example = {}
    if strategy is ChunkStrategy.CHAR:
        chunks = chunk_text(example, strategy, chunk_size=5)
    else:
        chunks = chunk_text(example, strategy)
    assert all("text" in c for c in chunks)
