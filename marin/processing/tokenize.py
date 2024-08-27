"""
Main for running Levanter's tokenizer infrastructure on a dataset using an existing Ray cluster.

Usage:
    ray job submit --working-dir . --no-wait -- python -m marin.processing.tokenize --input_dir <input-dir> --cache_dir <cache-dir> --dataset_name <dataset-name> --tokenizer <tokenizer_name>

    input_dir: The input directory containing the jsonl files or the name of a hf dataset
    cache_dir: The base directory to save the tokenized files
    dataset_name: The name of the dataset for the cache dir. This must be the same as the dataset name used in the Levanter training run
    tokenizer: The name of the tokenizer to use. This must be the same as the tokenizer used in the Levanter training run

    The data will be tokenized to $cache_dir/$dataset_name/train
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
    if fsspec_isdir(input_path):
        return fsspec_glob(os.path.join(input_path, "/**/*.jsonl*"))
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
    from levanter.data.text import BatchTokenizer
    from levanter.data.shard_cache import build_or_load_cache  # noqa
    from levanter.data.metrics_monitor import LoggerMetricsMonitor

    logging.basicConfig(level=logging.INFO)

    print(f"Caching {input_path} to {output_path}.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    # connect or start the actor
    batch_tokenizer = BatchTokenizer(tokenizer, enforce_eos=True)

    if is_hf_dataset(input_path):
        hf_source = levanter.data.sharded_dataset.WrappedHFDataset(input_path, split="train")
        source = hf_source.map(lambda x: x["text"])
    else:
        jsonls = _get_jsonls(input_path)
        source = levanter.data.sharded_dataset.TextUrlDataset(jsonls)

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
    input_dir: str  # input dir containing jsonl files, or hf dataset
    cache_dir: str  # base path to save the tokenized files
    dataset_name: str  # dataset name. Must be the same as you intend to use in the dataset spec for the training run
    tokenizer: str  # tokenizer name. Should be the same as you intend to use in the tokenizer spec for the training run


@draccus.wrap()
def main(config: TokenizeConfig):
    output_dir = os.path.join(config.cache_dir, config.dataset_name, "train")
    response = levanter_tokenize.remote(config.input_dir, config.tokenizer, output_dir)
    ray.get(response)


if __name__ == "__main__":
    ray.init()
    main()
    ray.shutdown()
