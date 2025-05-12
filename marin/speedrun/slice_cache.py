"""
Step to read a Levanter cache and produce a subsample of that cache.
"""
import dataclasses
import logging
from dataclasses import dataclass

import humanfriendly
from jax.random import PRNGKey
from levanter.data.text import LmDatasetSourceConfigBase
from levanter.store import SerialCacheWriter, TreeCache
from transformers import AutoTokenizer

from marin.execution import THIS_OUTPUT_PATH

logger = logging.getLogger(__name__)


def _do_slice_cache(input_config: LmDatasetSourceConfigBase,
                    tokenizer_spec: str,
                    output_path: str,
                    num_tokens: int,
                    key: PRNGKey) -> LmDatasetSourceConfigBase:
    """
    Read a Levanter cache and produce a subsample of that cache.

    This only works for datasets with input ids right now. Luckily this is all the datasets we care about atm.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
    # gonna be lazy with async stuff
    train_set = input_config.load_cache(tokenizer)
    # we want a random sample of docs (token seqs would probably be better but much more work)

    exemplar = train_set.as_sync_dataset()[0]

    assert "input_ids" in exemplar, "This only works for datasets with input ids right now"

    human_friendly_tokens = humanfriendly.format_size(num_tokens)[0:-1].replace(" ", "").replace("byte", "")
    subsampled_source_spec = _patch_source_config(input_config, output_path, extra_tags=
                                                  ["subsampled", f"subsampled-{human_friendly_tokens}"])

    try:
        TreeCache.load(output_path, exemplar)
        return subsampled_source_spec
    except FileNotFoundError:
        pass

    expected_input_ids = train_set.store.tree["input_ids"].data_size

    if expected_input_ids < num_tokens:
        raise ValueError(f"Cache does not seem to be big enough: {expected_input_ids} in cache but ${num_tokens} requested")

    train_set_shuffled = train_set.shuffle(key).as_sync_dataset()
    num_docs = len(train_set_shuffled)


    with SerialCacheWriter(output_path, exemplar) as output_writer:
        # TensorStore has high latency so we load biggish batches
        BS_TO_LOAD = 1024

        loaded_tokens = 0
        first_doc = 0
        while loaded_tokens < num_tokens and first_doc < num_docs:
            end_doc = min(first_doc + BS_TO_LOAD, num_docs)
            batch = train_set_shuffled.get_batch(range(first_doc, end_doc))
            first_doc = end_doc

            # decide how many docs to take from this batch
            batch_to_write = []
            for ex in batch:
                if loaded_tokens + len(ex["input_ids"]) > num_tokens:
                    break
                batch_to_write.append(ex)
                loaded_tokens += len(ex["input_ids"])

            output_writer.write_batch(batch_to_write)

    if loaded_tokens < num_tokens:
        raise ValueError("Provided cache doesn't have enough tokens")

    return subsampled_source_spec


def _patch_source_config(input_config: LmDatasetSourceConfigBase, output_path: str, extra_tags: list[str] = []) -> LmDatasetSourceConfigBase:
    """
    Patch the source config to point to the new cache.

    TODO: would be better to make this more explicit somehow...
    """
    return dataclasses.replace(input_config, cache_dir=output_path, tags=input_config.tags + extra_tags)


@dataclass
class SliceCacheConfig:
    """Configuration for slicing a Levanter cache."""
    input_config: LmDatasetSourceConfigBase
    num_tokens: int
    seed: int = 42
    output_path: str = THIS_OUTPUT_PATH
    tokenizer_spec: str = "stanford-crfm/marin_tokenizer"


def slice_cache(cfg: SliceCacheConfig) -> LmDatasetSourceConfigBase:
    """High-level function to slice a Levanter cache.
    
    This is the main entry point for slicing a cache. It creates and runs a SliceCacheStep.
    
    Args:
        cfg: Configuration for the slice operation
        
    Returns:
        The configuration for the sliced cache
    """


