import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest
from levanter.data import ShardedDataSource
from levanter.data.text import LmDatasetSourceConfigBase, TextLmDatasetFormat
from levanter.store import TreeCache, SerialCacheWriter
from transformers import AutoTokenizer
from marin.speedrun.slice_cache import _do_slice_cache
from jax.random import PRNGKey


@dataclass
class MockDatasetSource(LmDatasetSourceConfigBase):
    # def __init__(self, cache_dir, num_docs: int, tokens_per_doc: int, tags):
    #     self.num_docs = num_docs
    #     self.tokens_per_doc = tokens_per_doc
    #     self.format = TextLmDatasetFormat()
    #     self.cache_dir = cache_dir
    #     self.tags = tags
    num_docs: int = 100
    tokens_per_doc: int = 500

    def get_shard_source(self, split) -> Optional[ShardedDataSource[dict]]:
        raise NotImplementedError

    def load_cache(self, tokenizer):  # noqa
        # Create a mock dataset with the specified number of docs and tokens per doc
        exemplar = {"input_ids": np.array([0] * self.tokens_per_doc, dtype=np.int32)}

        with SerialCacheWriter(self.cache_dir, exemplar) as writer:
            for i in range(self.num_docs):
                # Create a document with unique token IDs for easy verification
                doc = {"input_ids": np.array([i] * self.tokens_per_doc, dtype=np.int32)}
                writer.write_batch([doc])

        out = TreeCache.load(self.cache_dir, exemplar)
        return out


@pytest.mark.parametrize("num_docs,tokens_per_doc,requested_tokens", [
    (10, 100, 500),  # Request half the tokens
    (5, 200, 1000),  # Request all tokens
    (20, 50, 300),   # Request a third of the tokens
])
def test_slice_cache(num_docs: int, tokens_per_doc: int, requested_tokens: int):
    # Create a mock dataset source

    with tempfile.TemporaryDirectory() as tmpdir:
        source = MockDatasetSource(
            cache_dir=f"{tmpdir}/base", num_docs=num_docs, tokens_per_doc=tokens_per_doc, tags=["test"]
        )
        # First run: create the sliced cache
        output_path = f"{tmpdir}/sliced_cache"
        key = PRNGKey(0)
        
        sliced_config = _do_slice_cache(source, "gpt2", output_path, requested_tokens, key)
        
        # Verify the sliced cache
        cache = TreeCache.load(output_path, {"input_ids": np.array([0], dtype=np.int32)})
        
        # Count total tokens in the sliced cache
        total_tokens = 0
        unique_docs = set()
        for doc in cache:
            total_tokens += len(doc["input_ids"])
            # Verify all tokens in a doc are the same (as per our mock dataset)
            assert len(set(doc["input_ids"])) == 1
            unique_docs.add(doc["input_ids"][0])
        
        assert requested_tokens <= total_tokens <= requested_tokens + tokens_per_doc
        
        # Verify we got a reasonable number of documents
        # We should have at least one document, and at most all documents
        assert 1 <= len(unique_docs) <= num_docs
        
        # Second run: should reuse existing cache
        key2 = PRNGKey(1)  # Different key shouldn't matter
        sliced_config2 = _do_slice_cache(source, "gpt2", output_path, requested_tokens, key2)
        
        # Verify we got the same config back
        assert sliced_config.cache_dir == sliced_config2.cache_dir
        assert sliced_config.tags == sliced_config2.tags


def test_slice_cache_too_small():
    """Test that _do_slice_cache raises ValueError when requested token budget is larger than available tokens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a small cache with 100 docs of 10 tokens each = 1000 total tokens
        source = MockDatasetSource(
            cache_dir=f"{tmpdir}/base",
            num_docs=100,
            tokens_per_doc=10,
            tags=["test"]
        )
        
        # Request more tokens than available (2000 > 1000)
        with pytest.raises(ValueError, match="Cache does not seem to be big enough"):
            _do_slice_cache(source, "gpt2", f"{tmpdir}/sliced_cache", 2000, PRNGKey(0))


def test_slice_cache_tags():
    """Test that _do_slice_cache adds the correct tags to the output config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a source with some initial tags
        source = MockDatasetSource(
            cache_dir=f"{tmpdir}/base",
            num_docs=100,
            tokens_per_doc=10,
            tags=["original", "test"]
        )
        
        # Request 500 tokens (50K tokens)
        sliced_config = _do_slice_cache(source, "gpt2", f"{tmpdir}/sliced_cache", 500, PRNGKey(0))
        
        # Verify tags
        assert "original" in sliced_config.tags
        assert "test" in sliced_config.tags
        assert "subsampled" in sliced_config.tags
        assert "subsampled-500" in sliced_config.tags  # human-friendly token count
        assert len(sliced_config.tags) == 4  # exactly these 4 tags
        
        # Test with a different token count to verify the token count tag changes
        sliced_config2 = _do_slice_cache(source, "gpt2", f"{tmpdir}/sliced_cache2", 1000, PRNGKey(0))
        assert "subsampled-1K" in sliced_config2.tags
        assert "subsampled-500" not in sliced_config2.tags  # old tag shouldn't be there
