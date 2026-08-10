# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import abc
import io
import json
import logging
import os
import warnings
from functools import cached_property
from typing import Any, Callable, Generic, Iterable, Iterator, List, Sequence, Sized, Tuple, TypeVar

import datasets
import numpy as np
import pyarrow.parquet as pq
from rigging.filesystem import StoragePath, open_url

from ._preprocessor import (
    BatchResult,
    _BatchMapTransform,
    _MapTransform,
    _TransformedDataset,
)
from .utils import batched

logger = logging.getLogger(__name__)

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
            num_cpus: CPU resources to request for each batch-map worker
            num_gpus: GPU resources to request for each batch-map worker
            **resources: Extra resource hints forwarded to the preprocessing executor

        Returns:
            A new ShardedDataset.
        """
        return _BatchMappedShardedDataSource(
            self, fn, batch_size, num_cpus=num_cpus, num_gpus=num_gpus, output_exemplar=output_exemplar, **resources
        )


class FirstRowsShardedDataSource(ShardedDataSource[T]):
    """A single-shard view over the first rows of another sharded source."""

    def __init__(self, source: ShardedDataSource[T], max_rows: int):
        if max_rows <= 0:
            raise ValueError("max_rows must be positive")
        self.source = source
        self.max_rows = max_rows

    @property
    def shard_names(self) -> Sequence[str]:
        return ["data"]

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[T]:
        if shard_name != "data":
            raise ValueError(f"Unknown shard {shard_name!r}")
        if row >= self.max_rows:
            return

        emitted = 0
        for item in self.source:
            if emitted >= row:
                emitted += 1
                yield item
                if emitted >= self.max_rows:
                    return
            else:
                emitted += 1


class UrlBackedShardedDataSource(ShardedDataSource[T_co], abc.ABC):
    """
    A base class ShardedDataset that is backed by a list of URLs. This is useful for datasets that are stored in a cloud storage
    system, such as S3 or GCS.
    """

    urls: Sequence[str]

    def __init__(self, urls):
        self.urls = urls
        # Force materialization early so duplicate shard names surface immediately.
        _ = self._shard_name_to_url_mapping

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


def datasource_from_hf_or_none(id: str, *, split, **kwargs) -> ShardedDataSource[dict] | None:
    """
    Like `datasource_from_hf`, but returns None when the requested split is missing or empty.

    HuggingFace raises a ``ValueError`` whose message starts with "Bad split" when the split does
    not exist; we treat that (and a source with no shards) as an absent dataset rather than an error.
    """
    try:
        source = datasource_from_hf(id, split=split, **kwargs)
    except ValueError as e:
        if str(e).startswith("Bad split"):
            logger.warning("Split %s not found for HF dataset %s %s", split, id, kwargs.get("name"))
            return None
        raise

    if len(source.shard_names) == 0:
        return None
    return source


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
        # HF dataset loading has historically not been multiprocessing-safe, so we load
        # lazily in the worker rather than sharing a dataset handle across processes.
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
            with open_url(url, "r", compression=compression) as f:
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
        compression = "infer"
        if url.endswith(".zstd"):  # hacky way to detect zstd
            compression = "zstd"

        format = _sniff_format_for_dataset(url)
        match format:
            case ".jsonl":
                with open_url(url, "r", compression=compression) as f:
                    for obj in _iter_jsonl_from_row(f, row):
                        if self.columns:
                            yield {col: obj[col] for col in self.columns}
                        else:
                            yield obj
            case ".json":
                with open_url(url, "r", compression=compression) as f:
                    data = json.load(f)
                    for doc in data[row:]:
                        if self.columns:
                            yield {col: doc[col] for col in self.columns}
                        else:
                            yield doc
            case ".parquet":
                with open_url(url, "rb", compression=compression) as f:
                    parquet_file = pq.ParquetFile(f)
                    yield from _iter_parquet_from_row(parquet_file, row, columns=self.columns)
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
            with open_url(audio_pointer, "rb", compression="infer") as f:
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
        with open_url(url, "r", compression="infer") as f:
            format = _sniff_format_for_dataset(url)
            match format:
                case ".jsonl":
                    for mat_json in _iter_jsonl_from_row(f, row):
                        audio_pointer = mat_json[self.audio_key]
                        audio = AudioTextUrlDataSource.resolve_audio_pointer(audio_pointer, self.sampling_rate)
                        yield (audio["array"], audio["sampling_rate"], mat_json[self.text_key])
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
        with open_url(url, "r", compression="infer") as f:
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


def _iter_jsonl_from_row(f: Iterable[str], row: int) -> Iterator[Any]:
    """Yield parsed JSON objects from a JSONL stream, skipping the first ``row`` lines.

    TODO: would be nice if we could seek faster than this. Right now, all we do is skip json parsing
    which is not nothing, but not ideal.
    """
    for i, line in enumerate(f):
        if i >= row:
            yield json.loads(line)


def _iter_parquet_from_row(parquet_file: pq.ParquetFile, row: int, columns=None) -> Iterator[dict]:
    """Iterate over rows in a ParquetFile starting from a given row offset.

    Seeks to the correct row group and yields dicts for each row from ``row`` onward.
    """
    total_rows = parquet_file.metadata.num_rows
    if row >= total_rows:
        return

    num_row_groups = parquet_file.metadata.num_row_groups

    # Compute cumulative row counts to find the starting row group
    row_counts = [parquet_file.metadata.row_group(i).num_rows for i in range(num_row_groups)]
    cumulative_rows = [0]
    for count in row_counts:
        cumulative_rows.append(cumulative_rows[-1] + count)

    row_group_index = 0
    start_row_in_group = row
    for idx, cum_row in enumerate(cumulative_rows):
        if cum_row > row:
            row_group_index = idx - 1
            start_row_in_group = row - cumulative_rows[row_group_index]
            break

    for rg_idx in range(row_group_index, num_row_groups):
        table = parquet_file.read_row_group(rg_idx, columns=columns)
        if rg_idx == row_group_index:
            table = table.slice(start_row_in_group)
        yield from table.to_pylist()


class ParquetDataSource(UrlBackedShardedDataSource[dict]):
    def __init__(self, urls, columns=None):
        super().__init__(urls)
        self.columns = columns

    def open_shard_at_row(self, shard_name: str, row: int) -> Iterator[dict]:
        url = self._shard_name_to_url_mapping[shard_name]
        with open_url(url, "rb", compression="infer") as f:
            parquet_file = pq.ParquetFile(f)
            yield from _iter_parquet_from_row(parquet_file, row, columns=self.columns)


def _mk_shard_name_mapping(urls):
    missing_urls: List[str] = []

    def _expand_or_placeholder(url):
        # expand_glob keeps a named-but-absent literal (so it warns/fails below rather
        # than vanishing); the fallback keeps an all-glob spec that matched nothing.
        expanded = [str(m) for m in StoragePath(url).expand_glob()]
        return expanded if expanded else [url]

    urls = [globbed for url in urls for globbed in _expand_or_placeholder(url)]

    _shard_name_to_url_mapping = {}

    # remove common prefix, computed on expanded urls
    if len(urls) == 1:
        common_prefix = os.path.dirname(urls[0])
    else:
        common_prefix = os.path.commonprefix(urls)

    for url in urls:
        exists = StoragePath(url).exists()
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

        if not exists:
            missing_urls.append(url)

    if missing_urls:
        missing_urls_str = "\n  - ".join(missing_urls)
        warnings.warn("Some shard URLs do not exist yet; they will fail when accessed:\n  - " + missing_urls_str)

    return _shard_name_to_url_mapping


class _MappedShardedDataSource(ShardedDataSource[T], _TransformedDataset):
    def __init__(self, source: ShardedDataSource[T_co], fn: Callable[[T_co], T]):
        self.source = source
        self.fn: Callable[..., T] = fn
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
