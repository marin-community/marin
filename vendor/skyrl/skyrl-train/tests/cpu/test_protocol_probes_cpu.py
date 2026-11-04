"""The protocol probes run through the production trainer on CPU, and each attempt's assessment reaches the dump.

``PROTOCOL_PROBES_MODEL`` names a local Hugging Face model directory to evaluate instead of the pinned
CatCount policy, for example a Qwen3-0.6B snapshot; a chat model that renders tools shows what the
labels report for real tool calls or narrated ones.
"""

import json
import os
from collections import Counter
from multiprocessing.context import ForkServerContext
from pathlib import Path

import pytest
from skyrl_train.evaluate import evaluation_dump_dir
from skyrl_train.utils.trainer_utils import sanitize_data_source
from taskcompendium.examples.protocol_probes import PROBE_DATASET

from tests.cpu.tiny_training.protocol_probes import (
    MACHINE_FREE_CHECKS,
    RUN_TIMEOUT_SECONDS,
    protocol_probes_config,
    run_protocol_probes,
)

pytestmark = pytest.mark.slow
VARIANTS = 2
MODEL_ENV = "PROTOCOL_PROBES_MODEL"


@pytest.fixture
def probe_model(request) -> Path:
    model = os.environ.get(MODEL_ENV)
    return Path(model) if model else request.getfixturevalue("cat_count_policy")


def evaluate_probes(forkserver_context: ForkServerContext, root: Path, model: Path) -> list[dict]:
    cfg = protocol_probes_config(root, model, variants=VARIANTS)
    process = forkserver_context.Process(target=run_protocol_probes, args=(cfg,))
    process.start()
    process.join(RUN_TIMEOUT_SECONDS)
    if process.exitcode is None:
        process.kill()
        process.join()
        pytest.fail(f"the run did not finish within {RUN_TIMEOUT_SECONDS} seconds")
    assert process.exitcode == 0
    dump = Path(evaluation_dump_dir(str(cfg.trainer.export_path), 0)) / f"{sanitize_data_source(PROBE_DATASET)}.jsonl"
    return [json.loads(line) for line in dump.read_text().splitlines()]


def test_each_machine_free_check_is_assessed_on_every_variant_and_labeled_in_the_dump(
    tmp_path, probe_model, forkserver_context
):
    rows = evaluate_probes(forkserver_context, tmp_path / "probes", probe_model)

    assessments = [
        (check, assessment)
        for row in rows
        if row["verification"] is not None
        for check, assessment in row["verification"]["diagnostics"].get("protocol", {}).items()
    ]
    assert Counter(check for check, _ in assessments) == dict.fromkeys(MACHINE_FREE_CHECKS, VARIANTS)
    assert all("label" in assessment for _, assessment in assessments)
    print("PROTOCOL_PROBES", json.dumps(Counter(f"{check}:{a['label']}" for check, a in assessments), sort_keys=True))
