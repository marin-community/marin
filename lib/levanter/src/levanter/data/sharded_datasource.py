# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import io
import json
import os
import warnings
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Sized,
    Tuple,
    TypeVar,
)

import datasets
import fsspec
import numpy as np
import pyarrow.parquet as pq

from levanter.utils import fsspec_utils

from ..data import AsyncDataset
from ..utils.fsspec_utils import expand_glob
from ._preprocessor import (
    BatchResult,
    _BatchMapTransform,
    _construct_composite_batch_processor,
    _DatasetTransform,
    _MapTransform,
)
from .utils import batched


if TYPE_CHECKING:
    from .metrics_monitor import MetricsMonitor

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)
T_co = TypeVar("T_co", covariant=True)
U = TypeVar("U")


class ShardedDataSource(Generic[T_co]):
    """
    A ShardedDataset is the main interface for reading data. It's basically a mapping from shard names to iterators,
    with the extra feature that it exposes the ability to skip to a particular row in a shard.

    The difference between a [ShardedDataset][] and a [ShardableDataset][] is that a ShardedDataset
    has a fixed number of shards, and a ShardableDataset `shard` method that can be used to
    split the dataset into multiple shards.
    """

    @property
    def shard_names(self) -> Sequence[str]:
        raise NotImplementedError

    @property
    def num_shards(self) -> int:
        return len(self.shard_names)

    def open_shard(self, shard_name: str) -> Iterator[T_co]:
        return self.open_shard_at_row(shard_name, 0)

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T_co]:
        raise NotImplementedError

    def __iter__(self):
        """
        Iterate over all data in the dataset, in order.
        """
        for shard_name in self.shard_names:
            for doc in self.open_shard(shard_name):
                yield doc

    def build_or_load_cache(
        self,
        path: str,
        *,
        await_finished: bool = True,
        monitors: Optional[Sequence["MetricsMonitor"]] = None,
    ) -> AsyncDataset[T]:
        """
        Constructs a shard cache version of this dataset using Ray.

        Levanter's preprocessing pipeline offers the following features/guarantees:
        * distributed, sharded preprocessing using Ray
        * deterministic ordering of data
        * interruptible and resumable
        * streaming results (no need to wait for everything to finish)

        Note that this is an experimental API and is subject to change.

        Returns:
            A new AsyncDataset that is backed by the cache.
        """

        source, processor = _construct_composite_batch_processor(self)
        from ..store.cache import build_or_load_cache

        cache = build_or_load_cache(
            path,
            source,
            processor,
            await_finished=await_finished,
            monitors=monitors,
        )
        return cache

    def map(self, fn: Callable[[T_co], U]) -> "ShardedDataSource[U]":
        return _MappedShardedDataSource(self, fn)

    def map_batches(
        self,
        fn: Callable[[list[T_co]], BatchResult],
        batch_size,
        *,
        num_cpus=1,
        num_gpus=0,
        output_exemplar=None,
        **resources,
    ) -> "ShardedDataSource[dict]":
        """
        **Lazily** map a function over batches of data. This is useful for doing things like batching data for a model,
        or for batched preprocessing.

        This function is **lazy**.

        Args:
            fn:  A function that takes a list of data and returns an iterable of results
            batch_size: The batch size to use
            num_cpus: passed to ray
            num_gpus: passed to ray
            **resources: Resources to pass to Ray

        Returns:
            A new ShardedDataset.
        """
        return _BatchMappedShardedDataSource(
            self, fn, batch_size, num_cpus=num_cpus, num_gpus=num_gpus, output_exemplar=output_exemplar, **resources
        )


class UrlBackedShardedDataSource(ShardedDataSource[T_co], abc.ABC):
    """
    A base class ShardedDataset that is backed by a list of URLs. This is useful for datasets that are stored in a cloud storage
    system, such as S3 or GCS.
    """

    urls: Sequence[str]

    def __init__(self, urls):
        self.urls = urls

    @cached_property
    def _shard_name_to_url_mapping(self):
        return _mk_shard_name_mapping(self.urls)

    @property
    def shard_names(self) -> Sequence[str]:
        return list(self._shard_name_to_url_mapping.keys())


def datasource_from_hf(id: str, *, split, **kwargs) -> ShardedDataSource[dict]:
    """
    Create a ShardedDataset from a HuggingFace dataset. Arguments are passed to load_dataset.
    """
    return WrappedHFDataSource(id, split=split, **kwargs)


def datasource_from_jsonl(urls_or_paths: Sequence[str]) -> ShardedDataSource[dict]:
    return JsonlDataSource(urls_or_paths)


def datasource_from_json(urls_or_paths: Sequence[str]) -> ShardedDataSource[dict]:
    return JsonDataSource(urls_or_paths)


def datasource_from_parquet(urls_or_paths: Sequence[str]) -> ShardedDataSource[dict]:
    return ParquetDataSource(urls_or_paths)


class WrappedHFDataSource(ShardedDataSource[dict]):
    """
    This class is responsible for loading a dataset from HuggingFace Datasets and returning the shards.
    Only (some) IterableDatasets are actually sharded in any meaningful way, so we just return a single shard
    for all other datasets.

    kwargs are passed to load_dataset
    """

    def __init__(self, id, *, split, streaming: bool = True, **kwargs):
        self.id = id
        self.split = split
        self.streaming = streaming
        self.kwargs = kwargs
        self._shard_names = self._compute_shard_names()

    @property
    def shard_names(self) -> Sequence[str]:
        return self._shard_names

    def _compute_shard_names(self):
        dataset = self._load_dataset()
        if isinstance(dataset, datasets.IterableDataset):
            try:
                return [str(i) for i in range(dataset.n_shards)]
            except NotImplementedError:
                return ["data"]
        else:
            return ["data"]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        dataset = self._load_dataset()
        if isinstance(dataset, datasets.IterableDataset) and shard_name != "data":
            # ex_iterable has a key that gets discarded typically
            shard = map(
                lambda t: t[1],
                dataset._ex_iterable.shard_data_sources(index=int(shard_name), num_shards=dataset.n_shards),
            )
        else:
            shard = dataset

        idx = 0
        for doc in shard:
            if idx >= row:
                yield doc
            idx += 1

    def _load_dataset(self):
        # obnoxiously, the dataset loading stuff doesn't work with ray because of multiprocessing
        # so we have to do this hacky thing where we load the dataset in the worker
        return datasets.load_dataset(self.id, split=self.split, streaming=self.streaming, **self.kwargs)


class TextUrlDataSource(ShardedDataSource[str]):
    """
    Dataset for various text formats.
    """

    def __init__(self, urls, text_key="text"):
        self.text_key = text_key
        self.base_ds = UrlDataSource(urls, columns=[text_key])

    @property
    def shard_names(self) -> Sequence[str]:
        return self.base_ds.shard_names

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[str]:
        url = self.base_ds._shard_name_to_url_mapping[shard_name]
        i = 0
        compression = "infer"
        if url.endswith(".zstd"):  # hacky way to detect zstd
            compression = "zstd"

        format = _sniff_format_for_dataset(url)

        # special case for txt files
        if format == ".txt":
            with fsspec.open(url, "r", compression=compression) as f:
                for line in f:
                    if i >= row:
                        yield line
                    i += 1
        else:
            for doc in self.base_ds.open_shard_at_row(shard_name, row):
                yield doc[self.text_key]


class UrlDataSource(UrlBackedShardedDataSource[dict]):
    """
    Dataset for various dict-like formats.
    """

    def __init__(self, urls, columns=None):
        super().__init__(urls)
        self.columns = columns

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        compression = "infer"
        if url.endswith(".zstd"):  # hacky way to detect zstd
            compression = "zstd"

        format = _sniff_format_for_dataset(url)
        match format:
            case ".jsonl":
                with fsspec.open(url, "r", compression=compression) as f:
                    # TODO: would be nice if we could seek faster than this. Right now, all we do is skip json parsing
                    # which is not nothing, but not ideal.
                    for line in f:
                        if i >= row:
                            obj = json.loads(line)
                            if self.columns:
                                yield {col: obj[col] for col in self.columns}
                            else:
                                yield obj
                        i += 1
            case ".json":
                with fsspec.open(url, "r", compression=compression) as f:
                    data = json.load(f)
                    for doc in data[row:]:
                        if self.columns:
                            yield {col: doc[col] for col in self.columns}
                        else:
                            yield doc
            case ".parquet":
                # TODO: fix this duplication
                with fsspec.open(url, "rb", compression=compression) as f:
                    parquet_file = pq.ParquetFile(f)
                    total_rows = parquet_file.metadata.num_rows
                    if row >= total_rows:
                        return

                    num_row_groups = parquet_file.metadata.num_row_groups

                    # Compute cumulative row counts
                    row_counts = [parquet_file.metadata.row_group(i).num_rows for i in range(num_row_groups)]
                    cumulative_rows = [0]
                    for count in row_counts:
                        cumulative_rows.append(cumulative_rows[-1] + count)

                    # Find the starting row group and row within it
                    for idx, cum_row in enumerate(cumulative_rows):
                        if cum_row > row:
                            row_group_index = idx - 1
                            start_row_in_group = row - cumulative_rows[row_group_index]
                            break

                    # Read from the starting row group onwards
                    for rg_idx in range(row_group_index, parquet_file.num_row_groups):
                        table = parquet_file.read_row_group(rg_idx, columns=self.columns)
                        if rg_idx == row_group_index:
                            table = table.slice(start_row_in_group)
                        yield from table.to_pylist()
            case _:
                raise ValueError(f"Unknown format {format}")


class AudioTextUrlDataSource(UrlBackedShardedDataSource[Tuple[np.ndarray, int, str]]):
    """
    Dataset for various audio and text formats.
    """

    def __init__(self, urls, text_key="sentence", audio_key="audio", sampling_rate=16000):
        super().__init__(urls)
        self.text_key = text_key
        self.audio_key = audio_key
        self.sampling_rate = sampling_rate

    @staticmethod
    def resolve_audio_pointer(audio_pointer, sampling_rate) -> dict[str, Any]:
        import librosa  # noqa F401

        def _load_audio_file(file_name, sampling_rate):
            with fsspec.open(audio_pointer, "rb", compression="infer") as f:
                array, sr = librosa.load(f, sr=sampling_rate)
            return {"array": array, "sampling_rate": sr}

        if isinstance(audio_pointer, dict):
            # These are the 3 ways HuggingFace stores audio in it's Audio type
            # https://huggingface.co/docs/datasets/v2.5.1/en/about_dataset_features#the-audio-type
            if "array" in audio_pointer and "sampling_rate" in audio_pointer:
                audio = audio_pointer
            elif "bytes" in audio_pointer:
                array, sr = librosa.load(io.BytesIO(audio_pointer["bytes"]), sr=sampling_rate)
                audio = {"array": array, "sampling_rate": sr}
            elif "path" in audio_pointer:
                audio = _load_audio_file(audio_pointer["path"], sampling_rate)
            else:
                raise ValueError(f"Unsupported audio format {audio_pointer}")
        elif isinstance(audio_pointer, str):
            # This supports filename pointers to arbitrary audio types
            audio = _load_audio_file(audio_pointer, sampling_rate)
        else:
            raise ValueError(f"Unsupported audio format {audio_pointer}")
        return audio

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[Tuple[np.ndarray, int, str]]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        with fsspec.open(url, "r", compression="infer") as f:
            format = _sniff_format_for_dataset(url)
            match format:
                case ".jsonl":
                    # TODO: would be nice if we could seek faster than this. Right now, all we do is skip json parsing
                    # which is not nothing, but not ideal.
                    for line in f:
                        if i >= row:
                            mat_json = json.loads(line)
                            audio_pointer = mat_json[self.audio_key]
                            audio = AudioTextUrlDataSource.resolve_audio_pointer(audio_pointer, self.sampling_rate)
                            yield (audio["array"], audio["sampling_rate"], mat_json[self.text_key])
                        i += 1
                case ".json":
                    data = json.load(f)
                    for doc in data[row:]:
                        audio_pointer = doc[self.audio_key]
                        audio = AudioTextUrlDataSource.resolve_audio_pointer(audio_pointer, self.sampling_rate)
                        yield (audio["array"], audio["sampling_rate"], doc[self.text_key])
                case _:
                    raise ValueError(f"Unknown format {format}")


def _sniff_format_for_dataset(url):
    good_formats = [".jsonl", ".txt", ".json", ".parquet"]
    format_from_url = None
    # try both with and without compression (could be gz, bz2, etc, so look at the "first" extension)
    extensions = [os.path.splitext(url)[1], os.path.splitext(os.path.splitext(url)[0])[1]]
    for format in good_formats:
        if any(ext == format for ext in extensions):
            format_from_url = format
            break

    if format_from_url is None:
        raise ValueError(f"Unknown format for {url}")

    if format_from_url == ".json":
        # unfortunately, HF (and others) will use "json" for jsonl files,
        # so we have to do some extra work to distinguish them.
        # Choices are
        # 1. look at the first 2 chars, if the first is "[", then it's probably json.
        #    If it's "{\n", also json. If it's { something else", then it's probably jsonl
        # 2. look at the first line. If it's valid json, then it's probably jsonl, unless there's only one line.
        #
        # (You can't actually distinguish between jsonl and json in a file with one line,
        #  which we'll just declare to be json and not jsonl, since that seems more likely)
        # (1) is cheating a bit, but it's fast and works in most cases we care about. (2) is more robust, but slower.
        with fsspec.open(url, "r", compression="infer") as f:
            first_two = f.read(2)

            if first_two[0] == "[" or first_two == "{\n" or first_two == "{\r":
                return ".json"
            elif first_two[0] == "{":
                return ".jsonl"

            # this is (much) heavier. This is particularly slow if we're dealing with packed/non-prettified json
            # since we're parsing the whole file.
            first_line = first_two + f.readline()
            try:
                json.loads(first_line)
                format_from_url = ".jsonl"
            except json.JSONDecodeError:
                return format_from_url

            if not f.readline():
                # only one line
                format_from_url = ".json"

    return format_from_url


class JsonlDataSource(UrlBackedShardedDataSource[dict]):
    def __init__(self, urls):
        super().__init__(urls)

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        i = 0
        with fsspec.open(url, "r", compression="infer") as f:
            # TODO: would be nice if we could seek faster than this. Right now, all we do is skip json parsing
            # which is not nothing, but not ideal.
            for line in f:
                if i >= row:
                    yield json.loads(line)
                i += 1


class JsonDataSource(UrlBackedShardedDataSource[dict]):
    def __init__(self, urls):
        super().__init__(urls)

    @property
    def shard_names(self) -> Sequence[str]:
        return list(self._shard_name_to_url_mapping.keys())

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        with fsspec.open(url, "r", compression="infer") as f:
            # TODO: would be nice if we could seek faster than this. Can't even skip json parsing
            data = json.load(f)
            return iter(data[row:])


class ParquetDataSource(UrlBackedShardedDataSource[dict]):
    def __init__(self, urls, columns=None):
        super().__init__(urls)
        self.columns = columns

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        with fsspec.open(url, "rb", compression="infer") as f:
            parquet_file = pq.ParquetFile(f)
            total_rows = parquet_file.metadata.num_rows
            if row >= total_rows:
                return

            num_row_groups = parquet_file.metadata.num_row_groups

            # Compute cumulative row counts
            row_counts = [parquet_file.metadata.row_group(i).num_rows for i in range(num_row_groups)]
            cumulative_rows = [0]
            for count in row_counts:
                cumulative_rows.append(cumulative_rows[-1] + count)

            # find starting row group and also find the row within it
            for idx, cum_row in enumerate(cumulative_rows):
                if cum_row > row:
                    row_group_index = idx - 1
                    start_row_in_group = row - cumulative_rows[row_group_index]
                    break

            # read from the starting row group onwards
            for rg_idx in range(row_group_index, parquet_file.num_row_groups):
                table = parquet_file.read_row_group(rg_idx, columns=self.columns)

                # if we're in the row group we want, slice the table at/from the row we want
                if rg_idx == row_group_index:
                    table = table.slice(start_row_in_group)

                yield from table.to_pylist()


def _mk_shard_name_mapping(urls):
    _shard_name_to_url_mapping = {}
    # remove common prefix
    if len(urls) == 1:
        common_prefix = os.path.dirname(urls[0])
    else:
        common_prefix = os.path.commonprefix(urls)

    missing_urls: List[str] = []

    old_urls = urls
    urls = [globbed for url in urls for globbed in expand_glob(url)]

    if old_urls and not urls:
        raise ValueError(f"Could not expand any of the urls: {old_urls}")

    for url in urls:
        if not fsspec_utils.exists(url):
            missing_urls.append(url)
            continue
        # escape the url for the shard name
        shard_name = url
        if common_prefix:
            shard_name = url[len(common_prefix) :]
            if shard_name.startswith("/"):
                shard_name = shard_name[1:]

        shard_name = shard_name.replace(".", "_")
        if shard_name in _shard_name_to_url_mapping:
            raise ValueError(f"Duplicate shard name {shard_name}")
        _shard_name_to_url_mapping[shard_name] = url

    if missing_urls:
        # format nicely
        missing_urls_str = "\n  - ".join(missing_urls)
        raise FileNotFoundError(f"Could not find the following urls:\n  - {missing_urls_str}")

    return _shard_name_to_url_mapping


class _TransformedDataset:
    source: ShardedDataSource
    _transform: _DatasetTransform


class _MappedShardedDataSource(ShardedDataSource[T], _TransformedDataset):
    def __init__(self, source: ShardedDataSource[T_co], fn: Callable[[T_co], T]):
        self.source = source
        self.fn = fn
        self._transform = _MapTransform(fn)

    @property
    def shard_names(self) -> Sequence[str]:
        return self.source.shard_names

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T]:
        for doc in self.source.open_shard_at_row(shard_name, row):
            yield self.fn(doc)


class _BatchMappedShardedDataSource(ShardedDataSource[T], _TransformedDataset):
    def __init__(
        self,
        source: ShardedDataSource[T_co],
        fn: Callable[[list[T_co]], Iterable[U]],
        batch_size,
        num_cpus=1,
        num_gpus=0,
        output_exemplar=None,
        **resources,
    ):
        self.source = source
        self._transform = _BatchMapTransform(
            fn, batch_size, num_cpus, num_gpus, resources, output_exemplar=output_exemplar
        )

    @property
    def shard_names(self) -> Sequence[str]:
        return self.source.shard_names

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T]:
        warnings.warn("This is not the best way to use batched preprocessing. Use build_cache instead.")
        # this one is tricky because we have to do batching ourselves and there's no guarantee that input and output
        # batch sizes are the same
        i = 0
        shard_iter = self.source.open_shard_at_row(shard_name, row)
        for batch in batched(shard_iter, self._transform.batch_size):  # type: ignore
            result = self._transform.fn(batch)  # type: ignore
            if isinstance(result, Sized) and len(result) + i < row:
                i += len(result)
                continue

            for doc in result:
                if i >= row:
                    yield doc
                i += 1
