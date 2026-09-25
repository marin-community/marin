# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded verification of an ordered CSR H5AD normalization artifact.

The profile has cells on rows, genes on columns, positive integer layers/counts
and floating X with identical canonical CSR support. Both sparse groups use
AnnData's csr_matrix 0.1.0 encoding. Other metadata may be present; required
obs/var columns are checked by value, including categoricals. External links,
soft links, virtual datasets and external raw storage are rejected.
"""

import hashlib
import math
import struct
from collections.abc import Iterator, Sequence
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np

from experiments.post_training.bio_tasks.h5ad_contract import (
    HASH_PREFIX,
    MAX_INTEGER,
    H5adNormalizationContract,
    validate_identifiers,
)

CHUNK_ENTRIES = 65536
MAX_HDF_CHUNK_BYTES = 8 * 1024 * 1024
MAX_HDF_OBJECTS = 10000
MAX_HDF_DEPTH = 64
SparseArray = np.ndarray | h5py.Dataset


def count_digest_header(cell_ids: Sequence[str], feature_ids: Sequence[str], nonzeros: int):
    digest = hashlib.sha256(HASH_PREFIX)
    digest.update(struct.pack("<QQQ", len(cell_ids), len(feature_ids), nonzeros))
    for identifiers in (cell_ids, feature_ids):
        for identifier in identifiers:
            encoded = identifier.encode("utf-8")
            digest.update(struct.pack("<Q", len(encoded)))
            digest.update(encoded)
    return digest


def integer_array(array: SparseArray, length: int) -> None:
    if array.shape != (length,) or array.dtype.kind not in "iu" or array.dtype.itemsize > 8:
        raise ValueError("h5ad_integer_array")


def csr_indptr(indptr: SparseArray, rows: int, nonzeros: int) -> np.ndarray:
    integer_array(indptr, rows + 1)
    values = np.asarray(indptr[:])
    if values[0] != 0 or values[-1] != nonzeros or np.any(values[1:] < values[:-1]) or np.any(values > nonzeros):
        raise ValueError("h5ad_csr_indptr")
    return values.astype(np.int64)


def count_chunks(
    indptr: np.ndarray, indices: SparseArray, data: SparseArray, columns: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield bounded canonical CSR segments, checking every count and coordinate."""
    nonzeros = int(indptr[-1])
    integer_array(indices, nonzeros)
    integer_array(data, nonzeros)
    for row, (left, right) in enumerate(pairwise(indptr)):
        previous = -1
        for start in range(int(left), int(right), CHUNK_ENTRIES):
            stop = min(start + CHUNK_ENTRIES, int(right))
            coordinates = np.asarray(indices[start:stop])
            counts = np.asarray(data[start:stop])
            if (
                np.any(coordinates >= columns)
                or np.any(coordinates < 0)
                or int(coordinates[0]) <= previous
                or np.any(coordinates[1:] <= coordinates[:-1])
            ):
                raise ValueError("h5ad_csr_coordinates")
            if np.any(counts <= 0) or np.any(counts > MAX_INTEGER):
                raise ValueError("h5ad_positive_integer_counts")
            previous = int(coordinates[-1])
            yield row, start, coordinates.astype(np.int64), counts.astype(np.int64)


def count_triples(row: int, coordinates: np.ndarray, counts: np.ndarray) -> bytes:
    triples = np.empty((len(counts), 3), dtype="<u8")
    triples[:, 0] = row
    triples[:, 1] = coordinates
    triples[:, 2] = counts
    return triples.tobytes(order="C")


def canonical_count_sha256(
    cell_ids: Sequence[str], feature_ids: Sequence[str], indptr: np.ndarray, indices: np.ndarray, data: np.ndarray
) -> str:
    """Hash positive CSR counts independent of dtype, compression and HDF5 bytes.

    SHA256 receives ``BIOH5AD_COUNTS_V1\\0``, then three little-endian uint64s
    (cells, features, nonzeros), then each cell ID and each feature ID as a
    uint64 UTF-8 byte length followed by those bytes. Finally it receives
    row-major zero-based (row, column, count) little-endian uint64 triples.
    """
    validate_identifiers(cell_ids)
    validate_identifiers(feature_ids)
    pointers = csr_indptr(indptr, len(cell_ids), len(data))
    digest = count_digest_header(cell_ids, feature_ids, len(data))
    for row, _, coordinates, counts in count_chunks(pointers, indices, data, len(feature_ids)):
        digest.update(count_triples(row, coordinates, counts))
    return digest.hexdigest()


def check_storage(root: h5py.File, max_decoded_bytes: int) -> None:
    """Inspect links before dereferencing them and bound numeric decoding."""
    pending = [(root, 0)]
    visited = set()
    objects = 0
    decoded = 0
    while pending:
        group, depth = pending.pop()
        address = h5py.h5o.get_info(group.id).addr
        if address in visited:
            continue
        visited.add(address)
        if depth > MAX_HDF_DEPTH:
            raise ValueError("h5ad_storage_depth")
        for name in group:
            objects += 1
            if objects > MAX_HDF_OBJECTS:
                raise ValueError("h5ad_storage_objects")
            if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                raise ValueError("h5ad_external_or_soft_link")
            item = group[name]
            if isinstance(item, h5py.Group):
                pending.append((item, depth + 1))
                continue
            if not isinstance(item, h5py.Dataset):
                raise ValueError("h5ad_storage_object")
            if item.is_virtual or item.external:
                raise ValueError("h5ad_external_dataset")
            # AnnData represents None metadata with an HDF5 null dataspace.
            elements = 0 if item.shape is None else item.size
            decoded += elements * item.dtype.itemsize
            if decoded > max_decoded_bytes:
                raise ValueError("h5ad_decoded_size")
            if item.chunks and math.prod(item.chunks) * item.dtype.itemsize > MAX_HDF_CHUNK_BYTES:
                raise ValueError("h5ad_storage_chunk")
            filters = item.id.get_create_plist()
            if any(filters.get_filter(index)[0] not in {1, 2, 3, 5, 6} for index in range(filters.get_nfilters())):
                raise ValueError("h5ad_unsupported_filter")


def encoding(item: h5py.Group, kind: str, version: str) -> None:
    if item.attrs.get("encoding-type") != kind or item.attrs.get("encoding-version") != version:
        raise ValueError("h5ad_encoding")


def child_group(parent: h5py.Group, name: str) -> h5py.Group:
    if name not in parent or not isinstance(parent[name], h5py.Group):
        raise ValueError("h5ad_missing_group")
    return parent[name]


def child_dataset(parent: h5py.Group, name: str) -> h5py.Dataset:
    if name not in parent or not isinstance(parent[name], h5py.Dataset):
        raise ValueError("h5ad_missing_dataset")
    return parent[name]


def string_values(dataset: h5py.Dataset, expected: Sequence[str]) -> None:
    if dataset.shape != (len(expected),) or h5py.check_string_dtype(dataset.dtype) is None:
        raise ValueError("h5ad_string_column")
    # Fixed-width conversion also bounds returned values for variable-length strings.
    width = max((len(value.encode()) for value in expected), default=0) + 1
    for start in range(0, len(expected), 256):
        actual = dataset.astype(f"S{width}")[start : start + 256]
        if list(actual) != [value.encode() for value in expected[start : start + 256]]:
            raise ValueError("h5ad_metadata_values")


def nullable_string_values(group: h5py.Group, expected: Sequence[str]) -> None:
    """Check AnnData's nullable string encoding without accepting masked identities."""
    encoding(group, "nullable-string-array", "0.1.0")
    mask = child_dataset(group, "mask")
    if mask.shape != (len(expected),) or mask.dtype.kind != "b":
        raise ValueError("h5ad_nullable_string_mask")
    for start in range(0, len(expected), CHUNK_ENTRIES):
        if np.any(mask[start : start + CHUNK_ENTRIES]):
            raise ValueError("h5ad_masked_string")
    string_values(child_dataset(group, "values"), expected)


def metadata_column(group: h5py.Group, name: str, expected: list[str | int]) -> None:
    if name not in group:
        raise ValueError("h5ad_missing_metadata")
    item = group[name]
    if isinstance(item, h5py.Dataset):
        if isinstance(expected[0], str):
            string_values(item, expected)
        else:
            integer_array(item, len(expected))
            for start in range(0, len(expected), CHUNK_ENTRIES):
                if not np.array_equal(item[start : start + CHUNK_ENTRIES], expected[start : start + CHUNK_ENTRIES]):
                    raise ValueError("h5ad_metadata_values")
        return
    if not isinstance(item, h5py.Group) or not isinstance(expected[0], str):
        raise ValueError("h5ad_metadata_type")
    if item.attrs.get("encoding-type") == "nullable-string-array":
        nullable_string_values(item, expected)
        return
    encoding(item, "categorical", "0.2.0")
    codes = child_dataset(item, "codes")
    categories = child_dataset(item, "categories")
    integer_array(codes, len(expected))
    if categories.ndim != 1 or categories.size > len(expected):
        raise ValueError("h5ad_categories")
    # Infer category labels from their uses; unobserved categories need not affect grading.
    expected_categories = {}
    for start in range(0, len(expected), CHUNK_ENTRIES):
        actual = codes[start : start + CHUNK_ENTRIES]
        if np.any(actual < 0) or np.any(actual >= categories.size):
            raise ValueError("h5ad_category_codes")
        for code, value in zip(actual, expected[start : start + CHUNK_ENTRIES], strict=True):
            key = int(code)
            if key in expected_categories and expected_categories[key] != value:
                raise ValueError("h5ad_metadata_values")
            expected_categories[key] = value
    if h5py.check_string_dtype(categories.dtype) is None:
        raise ValueError("h5ad_category_type")
    for code, value in expected_categories.items():
        if categories.astype(f"S{len(value.encode()) + 1}")[code] != value.encode():
            raise ValueError("h5ad_metadata_values")


def check_axis(root: h5py.File, name: str, identifiers: list[str], metadata: dict[str, list[str | int]]) -> None:
    group = child_group(root, name)
    encoding(group, "dataframe", "0.2.0")
    index_name = group.attrs.get("_index")
    if not isinstance(index_name, str) or "/" in index_name:
        raise ValueError("h5ad_dataframe_index")
    columns = group.attrs.get("column-order")
    if columns is None or np.ndim(columns) != 1 or len(columns) > MAX_HDF_OBJECTS:
        raise ValueError("h5ad_dataframe_columns")
    if any(not isinstance(column, str) or "/" in column or column not in group for column in columns):
        raise ValueError("h5ad_dataframe_columns")
    if len(set(columns)) != len(columns) or not set(metadata) <= set(columns):
        raise ValueError("h5ad_dataframe_columns")
    index = group.get(index_name)
    if isinstance(index, h5py.Dataset):
        string_values(index, identifiers)
    elif isinstance(index, h5py.Group):
        nullable_string_values(index, identifiers)
    else:
        raise ValueError("h5ad_missing_index")
    for column, expected in metadata.items():
        metadata_column(group, column, expected)


def sparse_group(parent: h5py.Group, name: str, target: H5adNormalizationContract) -> h5py.Group:
    group = child_group(parent, name)
    encoding(group, "csr_matrix", "0.1.0")
    if not np.array_equal(group.attrs.get("shape"), [len(target.cell_ids), len(target.feature_ids)]):
        raise ValueError("h5ad_matrix_shape")
    return group


def check_h5ad(path: Path, target: H5adNormalizationContract) -> dict:
    """Verify identities, every preserved count and every normalized nonzero."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > target.max_bytes:
        raise ValueError("h5ad_file_size_or_type")
    with h5py.File(path, "r", rdcc_nbytes=1024 * 1024) as root:
        check_storage(root, target.max_decoded_bytes)
        encoding(root, "anndata", "0.1.0")
        check_axis(root, "obs", target.cell_ids, target.obs_metadata)
        check_axis(root, "var", target.feature_ids, target.var_metadata)
        layers = child_group(root, "layers")
        encoding(layers, "dict", "0.1.0")
        counts = sparse_group(layers, "counts", target)
        normalized = sparse_group(root, "X", target)
        pointers = csr_indptr(child_dataset(counts, "indptr"), len(target.cell_ids), target.nonzeros)
        normalized_pointers = csr_indptr(child_dataset(normalized, "indptr"), len(target.cell_ids), target.nonzeros)
        if not np.array_equal(pointers, normalized_pointers):
            raise ValueError("h5ad_normalized_support")
        normalized_indices = child_dataset(normalized, "indices")
        integer_array(normalized_indices, target.nonzeros)
        normalized_data = child_dataset(normalized, "data")
        if normalized_data.shape != (target.nonzeros,) or normalized_data.dtype.kind != "f":
            raise ValueError("h5ad_normalized_dtype")
        digest = count_digest_header(target.cell_ids, target.feature_ids, target.nonzeros)
        totals = [0] * len(target.cell_ids)
        maximum_error = 0.0
        for row, start, coordinates, values in count_chunks(
            pointers, child_dataset(counts, "indices"), child_dataset(counts, "data"), len(target.feature_ids)
        ):
            stop = start + len(values)
            if not np.array_equal(normalized_indices[start:stop], coordinates):
                raise ValueError("h5ad_normalized_support")
            if int(values.max()) > MAX_INTEGER // len(values):
                subtotal = sum(map(int, values))
            else:
                subtotal = int(values.sum(dtype=np.int64))
            totals[row] += subtotal
            digest.update(count_triples(row, coordinates, values))
            observed = normalized_data[start:stop].astype(np.float64)
            expected = np.log1p(values.astype(np.float64) * (target.target_sum / target.eligible_gene_reads[row]))
            errors = np.abs(observed - expected)
            if not np.all(np.isfinite(observed)) or np.any(observed <= 0):
                raise ValueError("h5ad_normalized_nonfinite_or_nonpositive")
            if np.any(errors > target.atol + target.rtol * np.abs(expected)):
                raise ValueError("h5ad_normalized_values")
            maximum_error = max(maximum_error, float(errors.max()))
        if totals != target.eligible_gene_reads:
            raise ValueError("h5ad_cell_totals")
        observed_hash = digest.hexdigest()
        if observed_hash != target.counts_sha256:
            raise ValueError("h5ad_changed_counts")
    return {
        "passed": True,
        "cells": len(target.cell_ids),
        "features": len(target.feature_ids),
        "nonzeros": target.nonzeros,
        "counts_sha256": observed_hash,
        "maximum_absolute_error": maximum_error,
    }
