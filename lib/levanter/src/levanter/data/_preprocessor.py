# Copyright The Levanter Authors
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
    def output_exemplar(self) -> Any:
        """
        An exemplar of what this processor returns. This is used to determine the output schema of a dataset.
        """
        raise NotImplementedError

    @property
    def resources(self) -> Dict[str, float]:
        """Any resources that this processor needs to run."""
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


class _TransformedDataset:
    """Marker mixin for datasets carrying a lazy ``_DatasetTransform``.

    ``source`` is a ``ShardedDataSource`` at runtime but left untyped here.
    """

    source: Any
    _transform: _DatasetTransform


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


def canonicalize_batch(batch: Union[dict, list[dict], pa.RecordBatch]) -> list[dict]:
    """Normalize a batch into a list of row dicts suitable for writing to a cache.

    Accepts a column-oriented dict, a PyArrow ``RecordBatch``, or an already
    row-oriented list of dicts.
    """
    if isinstance(batch, pa.RecordBatch):
        batch = dict_from_record_batch(batch)

    if isinstance(batch, dict):
        keys = list(batch.keys())
        values = list(batch.values())
        num_rows = len(values[0]) if values else 0
        return [{key: values[i][j] for i, key in enumerate(keys)} for j in range(num_rows)]

    return list(batch)


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
