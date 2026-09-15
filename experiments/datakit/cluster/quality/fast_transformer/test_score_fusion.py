# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The fusion scoring step's pure pieces: shard pairing, id packing, and the pin checks."""

import hashlib
from dataclasses import replace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from experiments.datakit.cluster.quality.fast_transformer.data import PAD_ID, UNK_ID
from experiments.datakit.cluster.quality.fast_transformer.quality_model import (
    CALIBRATION_FILE,
    QualityPin,
    calibration_sha256,
    model_sha256,
    require_pinned_calibration,
    require_pinned_model,
)
from experiments.datakit.cluster.quality.fast_transformer.score_fusion import (
    pad_ids,
    paired_basenames,
    rebatch,
    verify_remap,
)

VOCAB = 10


def test_pad_ids_truncates_remaps_and_marks_unknown_ids():
    ids = pad_ids([[0, 1, 2, 3], [7], [], [9, 0]], max_tokens=3, vocab_size=VOCAB)

    assert ids.dtype == np.int32
    # Raw ids shift past the two reserved slots; a raw id at or above the vocab is UNK.
    assert ids.tolist() == [[2, 3, 4], [9, PAD_ID, PAD_ID], [PAD_ID] * 3, [UNK_ID, 2, PAD_ID]]


def test_rebatch_yields_full_batches_and_one_tail():
    batches = [
        pa.RecordBatch.from_pydict({"id": [str(i) for i in range(start, start + n)]})
        for start, n in ((0, 3), (3, 5), (8, 1))
    ]

    out = list(rebatch(iter(batches), 4))

    assert [b.num_rows for b in out] == [4, 4, 1]
    assert [i for b in out for i in b.column("id").to_pylist()] == [str(i) for i in range(9)]


def test_verify_remap_rejects_a_compacted_remap():
    assert verify_remap({0: 2, 1: 3, 2: 4}) == 3
    with pytest.raises(ValueError, match="identity offset"):
        verify_remap({0: 2, 5: 3})


def write_parquet(directory, names):
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        pq.write_table(pa.table({"id": pa.array([], pa.string())}), directory / name)


def test_paired_basenames_refuses_an_asymmetric_leaf(tmp_path):
    """A basename one side lacks is a document set that would leave no trace in the output."""
    names = ["part-00000-of-00002.parquet", "part-00001-of-00002.parquet"]
    write_parquet(tmp_path / "text", names)
    write_parquet(tmp_path / "embed", names)
    write_parquet(tmp_path / "other", [names[0], "part-00002-of-00003.parquet"])

    assert paired_basenames(str(tmp_path / "text"), str(tmp_path / "embed")) == names
    with pytest.raises(ValueError, match="not co-partitioned"):
        paired_basenames(str(tmp_path / "text"), str(tmp_path / "other"))
    with pytest.raises(FileNotFoundError):
        paired_basenames(str(tmp_path / "empty"), str(tmp_path / "embed"))


def write_model_dir(root, calibration=b"knots"):
    root.mkdir(parents=True, exist_ok=True)
    (root / "m.eqx").write_bytes(b"weights")
    (root / "m_remap.json").write_bytes(b"{}")
    (root / "m_meta.json").write_bytes(b'{"k": 1}')
    (root / CALIBRATION_FILE).write_bytes(calibration)
    return root


def test_model_digest_covers_the_scorer_artifacts_and_not_the_calibration(tmp_path):
    """The two digests move independently: a refit calibration must not rescore the corpus."""
    root = write_model_dir(tmp_path / "model")
    model = model_sha256(str(root))
    calibration = calibration_sha256(str(root))

    (root / CALIBRATION_FILE).write_bytes(b"other knots")
    assert model_sha256(str(root)) == model
    assert calibration_sha256(str(root)) != calibration

    (root / "m_meta.json").write_bytes(b'{"k": 2}')
    assert model_sha256(str(root)) != model, "changed bytes are a different model"

    assert (
        calibration == hashlib.sha256(CALIBRATION_FILE.encode() + b"\0" + hashlib.sha256(b"knots").digest()).hexdigest()
    )


def test_digests_survive_copying_the_directory(tmp_path):
    root = write_model_dir(tmp_path / "model")
    copy = write_model_dir(tmp_path / "elsewhere" / "model")
    assert model_sha256(str(root)) == model_sha256(str(copy))
    assert calibration_sha256(str(root)) == calibration_sha256(str(copy))


def test_scoring_refuses_bytes_that_are_not_the_pinned_ones(tmp_path):
    """The write side of the collision the output path exists to close."""
    root = write_model_dir(tmp_path / "model")
    pin = QualityPin(
        name="pin",
        model_path="model",
        model_sha256=model_sha256(str(root)),
        calibration_sha256=calibration_sha256(str(root)),
        tokenizer="tok",
    )

    assert require_pinned_model(pin, str(root)) == pin.model_sha256
    assert require_pinned_calibration(pin, str(root)) == pin.calibration_sha256
    with pytest.raises(ValueError, match="claims"):
        require_pinned_model(replace(pin, model_sha256="0" * 64), str(root))
    with pytest.raises(ValueError, match="different calibration"):
        require_pinned_calibration(replace(pin, calibration_sha256="0" * 64), str(root))


def test_a_model_dir_without_an_artifact_does_not_digest(tmp_path):
    with pytest.raises(ValueError, match=r"no \.eqx artifact"):
        model_sha256(str(tmp_path))
