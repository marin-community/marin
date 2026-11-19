# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from fray.current_client import set_current_client
from fray.local_backend import LocalClient
from fray.types import ResourceConfig
from levanter.testing.tokenizer import stage_gpt2_tokenizer
from marin.datakit.low_value.features import FEATURE_NAMES
from marin.datakit.normalize import NormalizedData
from zephyr.context import ZephyrContext

from experiments.datakit.cluster.quality.low_value.pipeline import LowValuePin, score_source


@pytest.fixture
def gpt2_tokenizer_path(tmp_path):
    source = Path(__file__).resolve().parents[5] / "lib" / "levanter" / "tests"
    destination = tmp_path / "tokenizer"
    destination.mkdir()
    return str(stage_gpt2_tokenizer(source, destination))


@pytest.fixture(autouse=True)
def local_client():
    client = LocalClient()
    with set_current_client(client):
        yield client
    client.shutdown()


def test_quality_leaf_keeps_partition_order_and_rejects_changed_model_before_output(
    tmp_path, monkeypatch, gpt2_tokenizer_path
):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    model_dir = tmp_path / "models" / "test"
    model_dir.mkdir(parents=True)
    shutil.copytree(gpt2_tokenizer_path, model_dir / "tokenizer")
    tokenizer_hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (model_dir / "tokenizer").iterdir()
        if path.is_file()
    }
    payload = {
        "features_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "embedding_dimensions": 0,
        "window_tokens": 256,
        "tokenizer": "tokenizer",
        "tokenizer_files_sha256": tokenizer_hashes,
        "kind": "gradient_boosting",
        "intercept": 0,
        "learning_rate": 1,
        "trees": [
            {
                "left": [1, -1, -1],
                "right": [2, -1, -1],
                "feature": [0, -2, -2],
                "threshold": [5, -2, -2],
                "value": [0, -3, 3],
            }
        ],
    }
    model_path = model_dir / "model.json"
    model_path.write_text(json.dumps(payload))
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    calibration = {
        "model_sha256": model_sha,
        "thresholds": {"0.05": {"cutoff": 0.5, "calibration": {"documents": 100, "false_drop_rate": 0.04}}},
    }
    calibration_path = model_dir / "calibration.json"
    calibration_path.write_text(json.dumps(calibration))
    pin = LowValuePin("models/test", model_sha, hashlib.sha256(calibration_path.read_bytes()).hexdigest())
    normalized_dir = tmp_path / "normalized"
    normalized_dir.mkdir()
    basenames = ["part-00000.parquet", "part-00001.parquet"]
    pq.write_table(
        pa.table({"id": ["z", "a"], "text": ["short document", "history and science " * 300]}),
        normalized_dir / basenames[0],
    )
    pq.write_table(pa.table({"id": ["m"], "text": ["another short document"]}), normalized_dir / basenames[1])
    normalized = NormalizedData(main_output_dir=str(normalized_dir), dup_output_dir=str(tmp_path / "dups"), counters={})
    context = ZephyrContext(name="low-value-test", max_workers=1, resources=ResourceConfig(cpu=1, ram="4g"))
    result = score_source(
        str(tmp_path / "out"),
        normalized=normalized,
        source="example",
        pin=pin,
        target_false_drop=0.05,
        zephyr_context=context,
    )
    first = pq.read_table(tmp_path / "out" / basenames[0]).to_pydict()
    assert first["id"] == ["z", "a"]
    assert first["is_low_value"] == [False, True]
    assert pq.read_table(tmp_path / "out" / basenames[1]).column("id").to_pylist() == ["m"]
    assert result.documents_scored_this_run == 3
    again = score_source(
        str(tmp_path / "out"),
        normalized=normalized,
        source="example",
        pin=pin,
        target_false_drop=0.05,
        zephyr_context=context,
    )
    assert again.documents_scored_this_run == 0
    original_model = model_path.read_bytes()
    model_path.write_text(model_path.read_text() + "\n")
    with pytest.raises(ValueError, match="differs from pinned"):
        score_source(
            str(tmp_path / "wrong-pin"),
            normalized=normalized,
            source="example",
            pin=pin,
            target_false_drop=0.05,
            zephyr_context=context,
        )
    assert not (tmp_path / "wrong-pin").exists()

    model_path.write_bytes(original_model)
    tokenizer_path = model_dir / "tokenizer" / "tokenizer.json"
    tokenizer_path.write_text(tokenizer_path.read_text() + "\n")
    with pytest.raises(ValueError, match="differs from pinned"):
        score_source(
            str(tmp_path / "wrong-tokenizer"),
            normalized=normalized,
            source="example",
            pin=pin,
            target_false_drop=0.05,
            zephyr_context=context,
        )
    assert not (tmp_path / "wrong-tokenizer").exists()
