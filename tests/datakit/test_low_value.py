# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from levanter.data.text.formats import TextLmDatasetFormat
from levanter.store.jagged_array import JaggedArrayStore
from marin.datakit.low_value.features import FEATURE_NAMES, document_features
from marin.datakit.low_value.model import load_model, load_model_tokenizer, load_operating_point
from marin.datakit.low_value.parquet import LowValueScores, score_batches
from marin.datakit.low_value.scoring import (
    cache_views,
    load_scorer,
    score_texts,
    score_token_cache,
    score_views,
    text_views,
)
from marin.execution.artifact import read_artifact, write_artifact
from marin.processing.tokenize._core import text_preprocessor
from sklearn.ensemble import GradientBoostingClassifier
from tokenizers import Tokenizer


def write_model(path: Path, payload: dict) -> str:
    path.write_text(json.dumps(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_contract() -> dict:
    return {
        "features_version": 1,
        "feature_names": list(FEATURE_NAMES),
        "embedding_dimensions": 0,
        "window_tokens": 2048,
        "tokenizer": "tokenizer",
        "tokenizer_files_sha256": {"tokenizer.json": "test-tokenizer"},
    }


def encoded_model(tmp_path, tokenizer_dir):
    shutil.copytree(tokenizer_dir, tmp_path / "tokenizer")
    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (tmp_path / "tokenizer").iterdir()
        if path.is_file()
    }
    feature = FEATURE_NAMES.index("max_encoded_share")
    payload = {
        **model_contract(),
        "tokenizer_files_sha256": hashes,
        "kind": "gradient_boosting",
        "intercept": 0,
        "learning_rate": 1,
        "trees": [
            {
                "left": [1, -1, -1],
                "right": [2, -1, -1],
                "feature": [feature, -2, -2],
                "threshold": [0.5, -2, -2],
                "value": [0, -3, 3],
            }
        ],
    }
    path = tmp_path / "model.json"
    digest = write_model(path, payload)
    return load_model(str(path), digest)


def test_exported_tree_head_matches_sklearn_probabilities_and_float32_boundaries(tmp_path):
    rng = np.random.default_rng(100)
    values = rng.normal(size=(200, len(FEATURE_NAMES)))
    labels = (values[:, 0] * values[:, 1] + values[:, 2]) > 0
    reference = GradientBoostingClassifier(n_estimators=12, max_depth=2, random_state=8).fit(values, labels)
    prior = reference.init_.predict_proba(values[:1])[0, 1]
    trees = []
    for estimator in reference.estimators_[:, 0]:
        tree = estimator.tree_
        trees.append(
            {
                "left": tree.children_left.tolist(),
                "right": tree.children_right.tolist(),
                "feature": tree.feature.tolist(),
                "threshold": tree.threshold.tolist(),
                "value": tree.value[:, 0, 0].tolist(),
            }
        )
    payload = {
        **model_contract(),
        "kind": "gradient_boosting",
        "intercept": float(np.log(prior / (1 - prior))),
        "learning_rate": reference.learning_rate,
        "trees": trees,
    }
    path = tmp_path / "model.json"
    model = load_model(str(path), write_model(path, payload))
    # Values immediately around a learned threshold exercise sklearn's float32 cast.
    root = reference.estimators_[0, 0].tree_
    boundary = values[:3].copy()
    boundary[:, root.feature[0]] = root.threshold[0] + np.array([-1e-12, 0, 1e-12])
    values = np.concatenate((values, boundary))
    np.testing.assert_allclose(model.predict(values), reference.predict_proba(values)[:, 1], rtol=1e-12, atol=1e-14)


def test_cache_scores_include_middle_windows_and_preserve_requested_indices(
    tmp_path, gpt2_tokenizer, gpt2_tokenizer_path
):
    natural = "A detailed explanation of history, observations and their interpretation. " * 400
    text = natural + "\n" + "ABCD" * 4000 + "\n" + natural
    texts = [natural, text]
    producer = text_preprocessor(TextLmDatasetFormat(), gpt2_tokenizer)
    rows = [np.asarray(row["input_ids"], dtype=np.int32) for row in producer([{"text": text} for text in texts])]
    cache = tmp_path / "cache"
    writer = JaggedArrayStore.open(str(cache / "input_ids"), mode="w", item_rank=1, dtype=np.int32)
    writer.extend(rows)
    model = encoded_model(tmp_path, gpt2_tokenizer_path)
    calibration_path = tmp_path / "calibration.json"
    calibration = {
        "model_sha256": model.sha256,
        "thresholds": {"0.05": {"cutoff": 0.5, "calibration": {"documents": 100, "false_drop_rate": 0.04}}},
    }
    digest = write_model(calibration_path, calibration)
    point = load_operating_point(str(calibration_path), digest, model.sha256, 0.05)
    scores = list(
        score_token_cache(
            str(cache),
            scorer=load_scorer(model),
            operating_point=point,
            indices=[1, 0],
            batch_size=1,
        )
    )
    assert [score.id for score in scores] == ["1", "0"]
    assert [score.is_low_value for score in scores] == [True, False]
    reader = JaggedArrayStore.open(str(cache / "input_ids"), mode="r", dtype=np.int32)
    views = cache_views(reader, 1, gpt2_tokenizer, model.window_tokens)
    assert len(views.intervals) == 3
    assert sum(end - start for start, end in views.intervals) == 3 * model.window_tokens
    assert views.windows == text_views(text, gpt2_tokenizer, model.window_tokens).windows
    np.testing.assert_allclose([scores[0].low_value_score], score_views([views], model), rtol=0, atol=0)
    # A recalibration referring to another model cannot be used for these scores.
    with pytest.raises(ValueError, match="different model bytes"):
        load_operating_point(str(calibration_path), digest, "different", 0.05)


def test_parquet_batches_keep_ids_and_exact_cutoff_decisions(tmp_path, gpt2_tokenizer, gpt2_tokenizer_path):
    model = encoded_model(tmp_path, gpt2_tokenizer_path)
    texts = ["Readable content about historical events. " * 100, "ABCD" * 400]
    documents = [text_views(text, gpt2_tokenizer, model.window_tokens) for text in texts]
    expected = score_views(documents, model)
    from_point = {
        "model_sha256": model.sha256,
        "thresholds": {
            "0.02": {"cutoff": float(expected[1]), "calibration": {"documents": 200, "false_drop_rate": 0.01}}
        },
    }
    path = tmp_path / "calibration.json"
    point = load_operating_point(str(path), write_model(path, from_point), model.sha256, 0.02)
    ids = [f"doc-{index:03d}" for index in range(300)]
    batches = iter([pa.record_batch({"id": ids, "text": texts * 150})])
    output = list(score_batches(batches, source="example", scorer=load_scorer(model), operating_point=point))
    destination = tmp_path / "scores.parquet"
    pq.write_table(pa.Table.from_batches(output), destination)
    result = pq.read_table(destination).to_pydict()
    assert result["id"] == ids
    assert result["is_low_value"] == [False, True] * 150
    np.testing.assert_array_equal(result["low_value_score"], np.tile(expected, 150))
    np.testing.assert_array_equal(score_texts(texts, load_scorer(model)), expected)


def test_tokenizer_wrappers_do_not_change_structural_evidence():
    text = "An explanation of a historical event with dates and sources. " * 20
    np.testing.assert_array_equal(
        document_features([text], 1000),
        document_features(["<|begin_of_text|>" + text + "<|end_of_text|>"], 1000),
    )


def test_score_artifact_relocates_to_consumer_region(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-02a/marin")
    artifact = LowValueScores(
        output_dir="s3://marin-us-east-02a/marin/datakit/quality/example",
        source="example",
        model_sha256="model",
        calibration_sha256="calibration",
        target_false_drop=0.02,
        documents_scored_this_run=10,
    )
    write_artifact(artifact, str(tmp_path))
    monkeypatch.setenv("MARIN_PREFIX", "s3://marin-us-east-08a/marin")
    restored = read_artifact(str(tmp_path), LowValueScores)
    assert restored.output_dir == "s3://marin-us-east-08a/marin/datakit/quality/example"


def test_tokenizer_pins_at_same_source_path_load_distinct_verified_backends(tmp_path, gpt2_tokenizer_path):
    source = tmp_path / "tokenizer-source"
    shutil.copytree(gpt2_tokenizer_path, source)
    model = encoded_model(tmp_path, gpt2_tokenizer_path)

    def hashes():
        return tuple(
            sorted(
                (path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in source.iterdir() if path.is_file()
            )
        )

    first = replace(model, tokenizer_path=str(source), tokenizer_files_sha256=hashes())
    tokenizer_a = load_model_tokenizer(first)
    sentinel = "a_unique_pinned_tokenizer_sentinel_52611"
    old_ids = tokenizer_a.encode(sentinel)
    backend = Tokenizer.from_file(str(source / "tokenizer.json"))
    backend.add_tokens([sentinel])
    backend.save(str(source / "tokenizer.json"))
    second = replace(model, tokenizer_path=str(source), tokenizer_files_sha256=hashes())
    tokenizer_b = load_model_tokenizer(second)
    assert tokenizer_b.encode(sentinel) == [backend.token_to_id(sentinel)]
    assert tokenizer_b.encode(sentinel) != old_ids
    assert tokenizer_a.encode(sentinel) == old_ids
