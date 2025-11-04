# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, Iterable, Mapping, Sequence, TypeVar, Union

import numpy as np
import pyarrow as pa


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
T_contra = TypeVar("T_contra", contravariant=True)
U = TypeVar("U")
U_co = TypeVar("U_co", covariant=True)


BatchResult = Union[pa.RecordBatch, Sequence[Mapping[str, Any]], Mapping[str, Sequence]]
"""
The result of a batched function. This can be a RecordBatch, a list of dicts, or a dict of lists.
"""


class BatchProcessor(Generic[T_contra, U_co], ABC):
    """
    A BatchProcessor is the main interface for preprocessing data. It takes a batch of data and returns a batch of
    processed data. It can be used to tokenize data, convert it to a RecordBatch, or do any other kind of preprocessing.
    The number of output examples can be different from the number of input examples.
    """

    @abstractmethod
    def __call__(
        self, batch: Sequence[T_contra]
    ) -> Sequence[U_co] | U_co:  # U can be batched "structure of arrays" form
        """
        Process a batch of data. You should return a sequence of dicts (one per output
        example), or a dict of sequences (one per output field).
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def output_exemplar(self):
        """
        An exemplar of what this processor returns. This is used to determine the output schema of a dataset.
        """
        raise NotImplementedError

    @property
    def resources(self) -> Dict[str, float]:
        """Any resources that this processor needs to run. Ray uses this to schedule tasks."""
        return {}

    @property
    @abstractmethod
    def num_cpus(self) -> float | int:
        """The number of CPUs this processor needs to run."""
        raise NotImplementedError

    @property
    def num_gpus(self) -> int:
        return 0

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Any metadata that changes the behavior of this processor."""
        raise NotImplementedError


class _DatasetTransform(ABC):
    pass


class _MapTransform(_DatasetTransform):
    fn: Callable[[T_co], T]

    def __init__(self, fn):
        self.fn = fn


class _BatchMapTransform(_DatasetTransform):
    fn: Callable[[list[T_co]], Iterable[U]]
    batch_size: int
    num_cpus: int
    num_gpus: int
    resources: dict
    output_exemplar: Any

    def __init__(self, fn, batch_size, num_cpus, num_gpus, resources, output_exemplar=None):
        self.fn = fn
        self.batch_size = batch_size
        self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.resources = resources
        self.output_exemplar = output_exemplar


def as_record_batch(doc: BatchResult) -> pa.RecordBatch:
    """Converts a document to an arrow-compatible record batch."""

    if isinstance(doc, pa.RecordBatch):
        return doc

    if isinstance(doc, Mapping):
        # structure of arrays
        def _as_array(x):
            # for dumb reasons, pa.array doesn't support ndarrays with ndim > 1
            if isinstance(x, np.ndarray) and x.ndim > 1:
                return [_as_array(y) for y in x]
            elif isinstance(x, np.ndarray):
                return list(x)
            else:
                return pa.array(x)

        names, columns = zip(*[(k, _as_array(v)) for k, v in doc.items()])

        return pa.RecordBatch.from_arrays(list(columns), names)
    elif isinstance(doc, Sequence):
        return pa.RecordBatch.from_pylist(doc)
    else:
        raise ValueError(f"Cannot convert {type(doc)} to record batch")


def _construct_composite_batch_processor(dataset):
    """
    Construct a batch processor from a dataset which has some chain of transforms applied to it. Also returns
    the source dataset and the batch size.
    """

    def rec(dataset):
        from levanter.data.sharded_datasource import _TransformedDataset

        if isinstance(dataset, _TransformedDataset):
            source, transforms, batch_transform = rec(dataset.source)
            match dataset._transform:
                case _MapTransform():
                    transforms = transforms + [dataset._transform]
                case _BatchMapTransform():
                    if batch_transform is not None:
                        raise ValueError("Only one batch transform is allowed right now. sorry!")
                    batch_transform = dataset._transform
                    transforms = transforms + [dataset._transform]
                case _DatasetTransform():
                    raise ValueError(f"Unknown transform {dataset._transform}")
            return source, transforms, batch_transform
        else:
            return dataset, [], None

    source, transforms, batch_transform = rec(dataset)

    # batch_size = batch_transform.batch_size if batch_transform is not None else 1024
    cpus = batch_transform.num_cpus if batch_transform is not None else 1
    gpus = batch_transform.num_gpus if batch_transform is not None else 0
    resources = batch_transform.resources if batch_transform is not None else {}

    return source, _CompositeBatchProcessor(transforms, cpus, gpus, resources)


class _CompositeBatchProcessor(BatchProcessor):
    def __init__(self, transforms, num_cpus, num_gpus, resources):
        self.transforms = transforms
        self._num_cpus = num_cpus
        self._num_gpus = num_gpus
        self._resources = resources

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_cpus(self):
        return self._num_cpus

    @property
    def num_gpus(self):
        return self._num_gpus

    @property
    def resources(self):
        return self._resources

    @property
    def output_exemplar(self):
        return self.transforms[-1].output_exemplar

    def __call__(self, batch):
        # batch is initially a list of elements, but after a BatchMapTransform
        # it can be a recordbatch, dict of lists, or list of dicts
        # if it's a dict of lists or record batch, we'll convert it to a list of dicts
        # before applying the next transform
        is_soa_form = False
        for transform in self.transforms:
            if is_soa_form:
                batch = as_record_batch(batch)
                batch = batch.to_pylist()
                is_soa_form = False

            match transform:
                case _MapTransform(fn=fn):
                    batch = [fn(x) for x in batch]
                case _BatchMapTransform(fn=fn):
                    batch = fn(batch)
                    is_soa_form = isinstance(batch, dict) or isinstance(batch, pa.RecordBatch)
                case _DatasetTransform():
                    raise ValueError(f"Unknown transform {transform}")

        if is_soa_form:
            return batch

        # mostly this is for map objects
        if isinstance(batch, Iterable) and not isinstance(batch, Sequence):
            batch = list(batch)

        return batch

    @property
    def metadata(self):
        return {}


def dict_from_record_batch(b) -> dict:
    # we follow the convention from hf batchencoding where homogeneous-lengthed arrays are turned into nd arrays
    # while heterogeneous lists are left as lists of arrays

    def to_hf_batched(x):
        if len(x) == 0:
            return list(x)
        elif isinstance(x[0], Sequence) or isinstance(x[0], np.ndarray):
            if all(len(y) == len(x[0]) for y in x):
                return np.stack(x)
            else:
                return list(x)
        else:
            return x

    return {b.field(i).name: to_hf_batched(b.column(i).to_numpy(zero_copy_only=False)) for i in range(b.num_columns)}


def _canonicalize_batch(batch: Union[dict, list[dict]]) -> list[dict]:
    if isinstance(batch, pa.RecordBatch):
        batch = dict_from_record_batch(batch)

    if isinstance(batch, dict):
        return _to_list_of_dicts(batch)
    else:
        return batch


def _to_list_of_dicts(batch: dict) -> list[dict]:
    """
    Convert a batch of dictionaries to a list of dictionaries, suitable for writing to a cache.
    """
    keys = list(batch.keys())
    values = list(batch.values())
    num_rows = len(values[0])
    return [{key: values[i][j] for i, key in enumerate(keys)} for j in range(num_rows)]


class IdentityProcessor(BatchProcessor[T, T]):
    def __init__(self, exemplar):
        self.exemplar = exemplar

    def __call__(self, batch: Sequence[T]) -> Sequence[T]:
        return batch

    @property
    def output_exemplar(self):
        return self.exemplar

    @property
    def num_cpus(self) -> float:
        return 0.1

    @property
    def metadata(self) -> Dict[str, Any]:
        return {}
