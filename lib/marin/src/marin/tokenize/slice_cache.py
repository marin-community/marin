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

"""
Library code for reading a Levanter cache and producing a subsample of that cache.

We mostly use this in Speedrun to create smaller caches that can be downloaded directly from Hugging Face.

The slice_cache operation will take a Levanter cache and produce a subsample of that cache by sampling
documents randomly from the cache (without replacement) until the desired number of tokens is reached.

For step wrappers that integrate with the executor, see experiments.steps.slice_cache.
"""

# TODO: this is painfully slow. Should figure out what's going on

import dataclasses
import logging
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import fsspec
import humanfriendly
from jax.random import PRNGKey
from levanter.data.text import (
    HfDatasetSourceConfig,
    LMDatasetSourceConfig,
    LmDatasetSourceConfigBase,
    UrlDatasetSourceConfig,
)
from levanter.store import SerialCacheWriter, TreeCache
from tqdm_loggable.auto import tqdm
from transformers import AutoTokenizer

from marin.processing.tokenize.tokenize import TokenizeConfigBase

if TYPE_CHECKING:
    from marin.execution import StepRef

logger = logging.getLogger(__name__)


@dataclass
class SliceCacheConfig(TokenizeConfigBase):
    """Configuration for slicing a Levanter cache."""

    input_config: LmDatasetSourceConfigBase
    num_tokens: int
    cache_path: str
    tokenizer: str = "stanford-crfm/marin-tokenizer"
    seed: int = 42

    def as_lm_dataset_source_config(
        self, actual_output_path: str | StepRef | None, *, include_raw_paths=True
    ) -> LMDatasetSourceConfig:
        humanfriendly_tokens = humanfriendly.format_size(self.num_tokens)[0:-1].replace(" ", "").replace("byte", "")
        out = _patch_source_config(
            self.input_config, self.cache_path, extra_tags=["subsampled", f"subsampled-{humanfriendly_tokens}"]
        )

        return out  # type: ignore


def _do_slice_cache(
    cfg: SliceCacheConfig,
) -> LmDatasetSourceConfigBase:
    """
    Read a Levanter cache and produce a subsample of that cache.

    This only works for datasets with input ids right now. Luckily this is all the datasets we care about atm.
    """
    key = PRNGKey(cfg.seed)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    split = "train"
    train_set = cfg.input_config.load_cache(split, tokenizer)

    # we want a random sample of docs (token seqs would probably be better but much more work)
    exemplar = train_set.as_sync_dataset()[0]

    assert "input_ids" in exemplar, "This only works for datasets with input ids right now"

    subsampled_source_spec = cfg.as_lm_dataset_source_config(cfg.cache_path)

    try:
        TreeCache.load(os.path.join(cfg.cache_path, split), exemplar)
        return subsampled_source_spec
    except FileNotFoundError:
        pass

    expected_input_ids = train_set.store.tree["input_ids"].data_size

    if expected_input_ids < cfg.num_tokens:
        raise ValueError(
            f"Cache does not seem to be big enough: {expected_input_ids} in cache but ${cfg.num_tokens} requested"
        )

    train_set_shuffled = train_set.shuffle(key).as_sync_dataset()
    num_docs = len(train_set_shuffled)

    logger.info(f"Subsampling {cfg.num_tokens} tokens from {num_docs} docs to {cfg.cache_path}")
    with SerialCacheWriter(os.path.join(cfg.cache_path, split), exemplar) as output_writer:
        # TensorStore has high latency so we load biggish batches
        # Fineweb averages about 1000 tokens per doc.
        BS_TO_LOAD = 4096

        loaded_tokens = 0
        first_doc = 0
        pbar = tqdm(total=cfg.num_tokens, desc="Sampling docs", unit="token")
        while loaded_tokens < cfg.num_tokens and first_doc < num_docs:
            end_doc = min(first_doc + BS_TO_LOAD, num_docs)
            batch = train_set_shuffled.get_batch(range(first_doc, end_doc))
            first_doc = end_doc
            pbar.set_postfix({"docs": first_doc})

            # decide how many docs to take from this batch
            batch_to_write = []
            for ex in batch:
                batch_to_write.append(ex)
                loaded_tokens += len(ex["input_ids"])
                pbar.update(len(ex["input_ids"]))
                if loaded_tokens > cfg.num_tokens:
                    break

            if batch_to_write:
                time_in = time.time()
                output_writer.write_batch(batch_to_write)
                time_out = time.time()
                logger.info(f"Wrote {len(batch_to_write)} docs in {time_out - time_in:.2f} seconds")

    if loaded_tokens < cfg.num_tokens:
        raise ValueError("Provided cache doesn't have enough tokens")

    out = TreeCache.load(os.path.join(cfg.cache_path, split), exemplar)
    # ensure it's the right size
    if out.store.tree["input_ids"].data_size != loaded_tokens:
        raise ValueError(
            f"Cache size mismatch: {out.store.tree['input_ids'].data_size} != {loaded_tokens}"
            f" (expected at least {cfg.num_tokens} tokens)"
        )

    # These are usually uploaded to HF, so we write a README
    _create_readme(cfg.cache_path, cfg.input_config, loaded_tokens, cfg.tokenizer, cfg.seed)

    return subsampled_source_spec


def _create_readme(
    output_path: str, input_config: LmDatasetSourceConfigBase, num_tokens: int, tokenizer_spec: str, seed: int
):
    """
    Create a README file for the cache.
    """
    readme_path = f"{output_path}/README.md"
    with fsspec.open(readme_path, "w") as f:
        f.write("# Marin/Levanter Subsampled Pretokenized Dataset\n\n")
        f.write("## Dataset\n\n")
        f.write(_short_desc_from_lm_config(input_config))
        f.write("\n\n## Factsheet\n\n")
        f.write(f"* Original cache: {input_config.cache_dir}\n")
        f.write(f"* Tokenizer: [{tokenizer_spec}](https://huggingface.co/{tokenizer_spec})\n")
        f.write(f"* Seed {seed}\n")
        f.write(f"* Number of tokens: {humanfriendly.format_number(num_tokens)}\n")
        f.write("\n\n(This readme is automatically generated by Marin.)\n")


def _short_desc_from_lm_config(input_config: LmDatasetSourceConfigBase) -> str:
    """
    Get a short description of the dataset from the config.
    """
    if isinstance(input_config, HfDatasetSourceConfig):
        ds_id = input_config.id
        url = f"[{ds_id}](https://huggingface.co/datasets/{ds_id})"
        if input_config.name:
            url += f" (name: {input_config.name})"
        return url
    elif isinstance(input_config, UrlDatasetSourceConfig):
        out = ""
        if input_config.train_urls:
            out = "Train Urls: \n"
            for url in input_config.train_urls:
                out += f"- {url}\n"

        if input_config.validation_urls:
            out = "Validation Urls: \n"
            for url in input_config.validation_urls:
                out += f"- {url}\n"

        if not out:
            out = "{missing urls}"

        return out
    else:
        return ""


def _patch_source_config(
    input_config: LmDatasetSourceConfigBase, output_path: str, extra_tags: list[str]
) -> LmDatasetSourceConfigBase:
    """
    Patch the source config to point to the new cache.

    TODO: would be better to make this more explicit somehow...
    """
    base_tags = input_config.tags or []
    return dataclasses.replace(input_config, cache_dir=output_path, tags=base_tags + extra_tags)


def _slice_cache_in_ray(cfg: SliceCacheConfig):
    """Entry point for Ray execution of slice_cache.

    This function sets up logging and calls the main implementation.
    """
    logging.basicConfig(level=logging.INFO)
    logger.info(f"Starting slice cache with config: {cfg}")
    return _do_slice_cache(cfg)
