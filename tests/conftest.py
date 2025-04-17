import hashlib
import time

import pytest

from marin.evaluation.evaluators.evaluator import ModelConfig

model_config = ModelConfig(
    name="test-llama-200m",
    path="gs://marin-us-east5/gcsfuse_mount/perplexity-models/llama-200m",
    engine_kwargs={"enforce_eager": True, "max_model_len": 2048},
)


@pytest.fixture
def unique_hash():
    timestamp = str(time.time_ns()).encode()

    # Hash it using SHA256 (or any other algo)
    digest = hashlib.sha256(timestamp).hexdigest()

    return digest
