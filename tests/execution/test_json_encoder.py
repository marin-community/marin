# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import dataclasses
import json
import logging
from pathlib import Path

from jax.sharding import PartitionSpec

from marin.utilities.json_encoder import CustomJsonEncoder


@dataclasses.dataclass(frozen=True)
class NestedConfig:
    axis: PartitionSpec
    output_dir: Path


@dataclasses.dataclass(frozen=True)
class ExampleConfig:
    name: str
    nested: NestedConfig


def test_custom_json_encoder_serializes_dataclasses_and_partition_specs_without_warning(caplog):
    payload = ExampleConfig(
        name="jpeg-tokenizer-k16-smoke",
        nested=NestedConfig(axis=PartitionSpec("data"), output_dir=Path("/tmp/out")),
    )

    with caplog.at_level(logging.WARNING, logger="marin.utilities.json_encoder"):
        encoded = json.dumps(payload, cls=CustomJsonEncoder)

    assert json.loads(encoded) == {
        "name": "jpeg-tokenizer-k16-smoke",
        "nested": {
            "axis": "PartitionSpec('data',)",
            "output_dir": "/tmp/out",
        },
    }
    assert "Could not serialize object" not in caplog.text
