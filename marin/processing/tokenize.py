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
from levanter.data.sharded_datasource import TextUrlDataSource
from levanter.data.text import LMDatasetConfig, LMDatasetSourceConfig

from marin.utils import fsspec_glob, fsspec_isdir

logger = logging.getLogger(__name__)


@ray.remote
def levanter_tokenize(input_path: list[str] | str, tokenizer_name: str, output_path: str):
    import levanter
    from levanter.data.metrics_monitor import LoggerMetricsMonitor
    from levanter.data.text import BatchTokenizer
    from levanter.store.cache import build_or_load_cache

    logging.basicConfig(level=logging.INFO)

    logger.info(f"Caching {input_path} to {output_path}.")
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)
    # connect or start the actor
    batch_tokenizer = BatchTokenizer(tokenizer, enforce_eos=True)

    if isinstance(input_path, str) and is_hf_dataset(input_path):
        source = levanter.data.datasource_from_hf(input_path, split="train")
        source = source.map(lambda d: d["text"])
    else:
        jsonls = _get_jsonls(input_path)
        logger.info(f"Found {len(jsonls)} jsonl files.")
        if len(jsonls) == 0:
            raise ValueError(f"No jsonl files found in {input_path}")
        source = TextUrlDataSource(jsonls)

    cache = build_or_load_cache(
        cache_dir=output_path,
        input_shards=source,
        processor=batch_tokenizer,
        await_finished=False,
        monitors=[LoggerMetricsMonitor("ray")],
    )

    cache.await_finished()
    logger.info(f"Finished caching {input_path} to {output_path}.")


@dataclasses.dataclass
class TokenizeConfig:
    input_path: list[str] | str  # input dir containing jsonl files, or hf dataset
    cache_path: str  # base path to save the tokenized files
    dataset_name: str  # dataset name. Must be the same as you intend to use in the dataset spec for the training run
    tokenizer: str  # tokenizer name. Should be the same as you intend to use in the tokenizer spec for the training run

    def as_lm_dataset_source_config(self) -> LMDatasetSourceConfig:
        """
        For use in Levanter trainign runs with mixtures of datasets.
        """
        return LMDatasetSourceConfig(
            tags=[self.dataset_name],
            train_urls=[self.input_path] if isinstance(self.input_path, str) else self.input_path,
        )

    def as_lm_dataset_task_config(self) -> LMDatasetConfig:
        """
        For use in Levanter training runs with a single dataset.
        """
        return LMDatasetConfig(
            cache_dir=self.cache_path,
            train_urls=[self.input_path] if isinstance(self.input_path, str) else self.input_path,
            tags=[self.dataset_name],
            tokenizer=self.tokenizer,
        )


def tokenize(config: TokenizeConfig):
    output_path = os.path.join(config.cache_path, config.dataset_name, "train")
    response = levanter_tokenize.remote(config.input_path, config.tokenizer, output_path)
    out = ray.get(response)
    print(out)
    return out


def _get_jsonls(input_path: str | list[str]):
    if not isinstance(input_path, list):
        input_path = [input_path]

    output_paths = []
    for path in input_path:
        if fsspec_isdir(path) or path.endswith("/"):
            logger.info(f"Getting all jsonl files in {path}")
            logger.info(f"Using glob: {os.path.join(path, '**/*.jsonl.{gz,zst,zstd}')}")
            output_paths.extend(fsspec_glob(os.path.join(path, "**/*.jsonl.{gz,zst,zstd}")))
        else:
            output_paths.extend(fsspec_glob(path))

    return output_paths


def is_hf_dataset(path):
    # see if looks like an hf dataset or not.
    import fsspec

    protocol, _ = fsspec.core.split_protocol(path)

    if protocol is not None:
        return False

    if fsspec_isdir(path):
        return False

    return True


@draccus.wrap()
def main(config: TokenizeConfig):
    tokenize(config)


if __name__ == "__main__":
    main()
