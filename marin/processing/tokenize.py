"""
Main for running Levanter's tokenizer infrastructure on a dataset using an existing Ray cluster.

Usage:
    ray job submit --working-dir . --no-wait -- python -m marin.processing.tokenize \
        --input_path <input-dir> --cache_path <cache-path> --dataset_name <dataset-name> --tokenizer <tokenizer_name>

    input_path: The input directory containing the jsonl files or the name of a hf dataset
    cache_path: The base directory to save the tokenized files
    dataset_name: The name of the dataset for the cache dir. This must be the same as the dataset name used
                  in the Levanter training run
    tokenizer: The name of the tokenizer to use. This must be the same as the tokenizer used in the Levanter
               training run

    The data will be tokenized to $cache_path/$dataset_name/train
"""

import dataclasses
import logging
import os

import draccus
import ray
import transformers

from marin.utils import fsspec_glob, fsspec_isdir

logger = logging.getLogger(__name__)


def _get_jsonls(input_path):
    if fsspec_isdir(input_path) or input_path.endswith("/"):
        logger.info(f"Getting all jsonl files in {input_path}")
        logger.info(f"Using glob: {os.path.join(input_path, '**/*.jsonl.gz')}")
        return fsspec_glob(os.path.join(input_path, "**/*.jsonl.gz"))
    else:
        return fsspec_glob(input_path)


def is_hf_dataset(path):
    # see if looks like an hf dataset or not.
    import fsspec

    protocol, _ = fsspec.core.split_protocol(path)

    if protocol is not None:
        return False

    if fsspec_isdir(path):
        return False

    return True


@ray.remote
def levanter_tokenize(input_path: str, tokenizer_name: str, output_path: str):
    import levanter
    from levanter.data.metrics_monitor import LoggerMetricsMonitor
    from levanter.data.text import BatchTokenizer
    from levanter.store.cache import build_or_load_cache

    logging.basicConfig(level=logging.INFO)

    print(f"Caching {input_path} to {output_path}.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    # connect or start the actor
    batch_tokenizer = BatchTokenizer(tokenizer, enforce_eos=True)

    if is_hf_dataset(input_path):
        source = levanter.data.datasource_from_hf(input_path, split="train")
    else:
        jsonls = _get_jsonls(input_path)
        source = levanter.data.datasource_from_jsonl(jsonls)

    source = source.map(lambda d: d["text"])

    cache = build_or_load_cache(
        cache_dir=output_path,
        input_shards=source,
        processor=batch_tokenizer,
        await_finished=False,
        monitors=[LoggerMetricsMonitor()],
    )

    cache.await_finished()
    print(f"Finished caching {input_path} to {output_path}.")


@dataclasses.dataclass
class TokenizeConfig:
    input_path: str  # input dir containing jsonl files, or hf dataset
    cache_path: str  # base path to save the tokenized files
    dataset_name: str  # dataset name. Must be the same as you intend to use in the dataset spec for the training run
    tokenizer: str  # tokenizer name. Should be the same as you intend to use in the tokenizer spec for the training run


def tokenize(config: TokenizeConfig):
    output_path = os.path.join(config.cache_path, config.dataset_name, "train")
    response = levanter_tokenize.remote(config.input_path, config.tokenizer, output_path)
    ray.get(response)


@draccus.wrap()
def main(config: TokenizeConfig):
    ray.init()
    tokenize(config)


if __name__ == "__main__":
    main()
