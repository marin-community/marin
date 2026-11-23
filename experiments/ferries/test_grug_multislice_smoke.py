# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import materialized_config

from experiments.ferries import grug_multislice_smoke as smoke
from experiments.marin_tokenizer import marin_tokenizer


def test_build_trains_on_the_prebuilt_cache_through_mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("MARIN_PREFIX", str(tmp_path))
    monkeypatch.delenv("GRUG_MULTISLICE_OUTPUT_PATH", raising=False)

    checkpoint = smoke.build()
    (train,) = checkpoint.deps
    write_record(ArtifactRecord(output_path=train.path(str(tmp_path)), config={"tokenizer": marin_tokenizer}))
    data = materialized_config(checkpoint, str(tmp_path)).data

    assert train.name == "fineweb-edu-10M"
    assert {name: component.cache_dir for name, component in data.components.items()} == {
        "fineweb-edu-10M": train.path("mirror://")
    }
    assert data.train_weights == {"fineweb-edu-10M": 1.0}
    assert data.tokenizer == marin_tokenizer
