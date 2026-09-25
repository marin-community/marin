# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import math
import struct

import h5py
import numpy as np
import pytest

from experiments.post_training.bio_tasks.contract import Column, Contract, grade_files
from experiments.post_training.bio_tasks.h5ad_contract import H5adNormalizationContract
from experiments.post_training.bio_tasks.h5ad_verifier import canonical_count_sha256, check_h5ad


def tiny_h5ad(path, compression=None, integer_dtype="int32", categorical=True, nullable_strings=False):
    with h5py.File(path, "w") as handle:
        handle.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        for axis, identifiers in (("obs", ["cell-a", "cell-b"]), ("var", ["ENSG1", "ENSG2", "ENSG3"])):
            group = handle.create_group(axis)
            group.attrs.update({"encoding-type": "dataframe", "encoding-version": "0.2.0", "_index": "id"})
            group.attrs["column-order"] = np.array(
                ["original_endogenous_reads", "sorting_gate"] if axis == "obs" else [],
                dtype=h5py.string_dtype("utf-8"),
            )
            if nullable_strings:
                index = group.create_group("id")
                index.attrs.update({"encoding-type": "nullable-string-array", "encoding-version": "0.1.0"})
                index.create_dataset("mask", data=np.zeros(len(identifiers), dtype=bool))
                index.create_dataset("values", data=identifiers, dtype=h5py.string_dtype("utf-8"))
            else:
                group.create_dataset("id", data=identifiers, dtype=h5py.string_dtype("utf-8"))
        handle["obs"].create_dataset("original_endogenous_reads", data=[10, 5])
        if nullable_strings:
            gate = handle["obs"].create_group("sorting_gate")
            gate.attrs.update({"encoding-type": "nullable-string-array", "encoding-version": "0.1.0"})
            gate.create_dataset("mask", data=np.zeros(2, dtype=bool))
            gate.create_dataset("values", data=["HSPC", "Prog"], dtype=h5py.string_dtype("utf-8"))
        elif categorical:
            gate = handle["obs"].create_group("sorting_gate")
            gate.attrs.update({"encoding-type": "categorical", "encoding-version": "0.2.0"})
            gate.create_dataset("codes", data=[1, 0])
            gate.create_dataset("categories", data=["Prog", "HSPC"], dtype=h5py.string_dtype("utf-8"))
        else:
            handle["obs"].create_dataset("sorting_gate", data=["HSPC", "Prog"], dtype=h5py.string_dtype("utf-8"))
        for name, values in (
            ("layers/counts", np.array([2, 6, 3, 1], dtype=integer_dtype)),
            ("X", np.array([math.log1p(2500), math.log1p(7500), math.log1p(7500), math.log1p(2500)])),
        ):
            group = handle.create_group(name)
            group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": [2, 3]})
            group.create_dataset("indptr", data=np.array([0, 2, 4], dtype=integer_dtype), compression=compression)
            group.create_dataset("indices", data=np.array([0, 2, 1, 2], dtype=integer_dtype), compression=compression)
            group.create_dataset("data", data=values, compression=compression)
        handle.create_group("uns").create_dataset("unrelated_annotation", data="retained metadata")
        handle["uns"].create_dataset("log1p/base", data=h5py.Empty("float64"))
        handle["layers"].attrs.update({"encoding-type": "dict", "encoding-version": "0.1.0"})


@pytest.fixture
def h5ad_target():
    # Independently spelled out canonical content for two rows, including zeros
    # omitted by CSR. Neither the expected counts nor normalization use the checker.
    content = b"BIOH5AD_COUNTS_V1\0" + struct.pack("<QQQ", 2, 3, 4)
    for identifier in (b"cell-a", b"cell-b", b"ENSG1", b"ENSG2", b"ENSG3"):
        content += struct.pack("<Q", len(identifier)) + identifier
    content += struct.pack("<12Q", 0, 0, 2, 0, 2, 6, 1, 1, 3, 1, 2, 1)
    return H5adNormalizationContract(
        cell_ids=["cell-a", "cell-b"],
        feature_ids=["ENSG1", "ENSG2", "ENSG3"],
        nonzeros=4,
        counts_sha256=hashlib.sha256(content).hexdigest(),
        eligible_gene_reads=[8, 4],
        target_sum=10000.0,
        atol=1e-10,
        rtol=1e-8,
        max_bytes=1024 * 1024,
        max_decoded_bytes=1024 * 1024,
        obs_metadata={"sorting_gate": ["HSPC", "Prog"], "original_endogenous_reads": [10, 5]},
    )


@pytest.mark.parametrize("compression,integer_dtype,categorical", [(None, "int32", True), ("gzip", "int64", False)])
def test_h5ad_storage_changes_preserve_scientific_content(
    tmp_path, h5ad_target, compression, integer_dtype, categorical
):
    artifact = tmp_path / "normalized.h5ad"
    tiny_h5ad(artifact, compression, integer_dtype, categorical)
    result = check_h5ad(artifact, h5ad_target)
    assert result["passed"] and result["nonzeros"] == 4
    assert result["counts_sha256"] == h5ad_target.counts_sha256
    assert result["maximum_absolute_error"] < 1e-12
    with h5py.File(artifact, "r") as handle:
        counts = handle["layers/counts"]
        digest = canonical_count_sha256(
            h5ad_target.cell_ids,
            h5ad_target.feature_ids,
            counts["indptr"][:],
            counts["indices"][:],
            counts["data"][:],
        )
    assert digest == h5ad_target.counts_sha256


def test_h5ad_nullable_string_identity_and_metadata_require_unmasked_values(tmp_path, h5ad_target):
    artifact = tmp_path / "normalized.h5ad"
    tiny_h5ad(artifact, nullable_strings=True)
    assert check_h5ad(artifact, h5ad_target)["passed"]
    with h5py.File(artifact, "r+") as handle:
        handle["obs/id/mask"][0] = True
    with pytest.raises(ValueError, match="h5ad_masked_string"):
        check_h5ad(artifact, h5ad_target)


def test_h5ad_is_required_even_with_a_correct_summary(tmp_path, h5ad_target):
    contract = Contract(
        columns={"cells": Column(kind="integer", description="observed cells", unit="cells")},
        expected={"summary": {"cells": 2}},
        h5ad={"normalized.h5ad": h5ad_target},
    )
    reference, answer = tmp_path / "reference.json", tmp_path / "answer.json"
    artifact = tmp_path / "normalized.h5ad"
    reference.write_text(contract.model_dump_json())
    answer.write_text('[{"id":"summary","cells":2}]')
    assert grade_files(reference, answer).reward == 0
    tiny_h5ad(artifact)
    assert grade_files(reference, answer).reward == 1
    with h5py.File(artifact, "r+") as handle:
        handle["X/data"][0] = 0.5
    verdict = grade_files(reference, answer)
    assert verdict.reward == 0
    assert all(check["passed"] for check in verdict.detail["checks"])
    assert not verdict.detail["artifact_checks"]["normalized.h5ad"]["passed"]
    artifact.write_bytes(b"not an HDF5 file")
    assert grade_files(reference, answer).reward == 0


@pytest.mark.parametrize(
    "mistake", ["double_log", "original_depth", "one_value", "nan", "changed_counts", "changed_support"]
)
def test_h5ad_wrong_matrix_fails_with_unchanged_summaries(tmp_path, h5ad_target, mistake):
    artifact = tmp_path / "normalized.h5ad"
    tiny_h5ad(artifact)
    with h5py.File(artifact, "r+") as handle:
        if mistake == "double_log":
            handle["X/data"][:] = np.log1p(handle["X/data"][:])
        elif mistake == "original_depth":
            handle["X/data"][:] = [math.log1p(2000), math.log1p(6000), math.log1p(6000), math.log1p(2000)]
        elif mistake == "changed_counts":
            # Preserve library totals and make X correct for these altered counts.
            handle["layers/counts/data"][:2] = [3, 5]
            handle["X/data"][:2] = [math.log1p(3750), math.log1p(6250)]
        elif mistake == "changed_support":
            handle["X/indices"][0] = 1
        else:
            handle["X/data"][0] = float("nan") if mistake == "nan" else 0.5
    with pytest.raises(ValueError):
        check_h5ad(artifact, h5ad_target)


@pytest.mark.parametrize(
    "mistake", ["cell_id", "feature_id", "gate", "depth", "duplicate_coordinate", "floating_counts"]
)
def test_h5ad_preserves_ids_metadata_and_integer_counts(tmp_path, h5ad_target, mistake):
    artifact = tmp_path / "normalized.h5ad"
    tiny_h5ad(artifact)
    with h5py.File(artifact, "r+") as handle:
        if mistake == "cell_id":
            handle["obs/id"][0] = "cell-b"
        elif mistake == "feature_id":
            handle["var/id"][0] = "ENSG2"
        elif mistake == "gate":
            handle["obs/sorting_gate/codes"][0] = 0
        elif mistake == "depth":
            handle["obs/original_endogenous_reads"][0] = 8
        elif mistake == "duplicate_coordinate":
            handle["layers/counts/indices"][1] = 0
        else:
            del handle["layers/counts/data"]
            handle["layers/counts"].create_dataset("data", data=[2.0, 6.0, 3.0, 1.0])
    with pytest.raises(ValueError):
        check_h5ad(artifact, h5ad_target)


@pytest.mark.parametrize("link_type", ["soft", "external", "virtual", "external_raw"])
def test_h5ad_rejects_indirection_even_in_unscored_metadata(tmp_path, h5ad_target, link_type):
    artifact = tmp_path / "normalized.h5ad"
    tiny_h5ad(artifact)
    with h5py.File(artifact, "r+") as handle:
        if link_type == "soft":
            handle["uns/indirection"] = h5py.SoftLink("/obs/id")
        elif link_type == "external":
            handle["uns/indirection"] = h5py.ExternalLink("absent.h5", "/private")
        elif link_type == "virtual":
            layout = h5py.VirtualLayout(shape=(2,), dtype="int64")
            layout[:] = h5py.VirtualSource("absent.h5", "private", shape=(2,))
            handle["uns"].create_virtual_dataset("indirection", layout)
        else:
            handle["uns"].create_dataset(
                "indirection", shape=(2,), dtype="int64", external=[(str(tmp_path / "external.bin"), 0, 16)]
            )
    with pytest.raises(ValueError):
        check_h5ad(artifact, h5ad_target)
