# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from marin.execution.lazy import StepContext

from experiments.post_training.bfcl_rl.launch import recovered_model


def test_recovery_handoff_requires_export_and_resolves_saved_policy(tmp_path):
    # The first live DPO update saved a native checkpoint but skipped its HF hook.
    # The RL learner must fail rather than silently reverting to the starting model.
    model = recovered_model("2026.10.04.17")
    root = model.step.path(str(tmp_path))
    ctx = StepContext.for_run(str(tmp_path / "rl"), str(tmp_path), deps=model.deps())
    with pytest.raises(ValueError, match="has no HF export"):
        model.resolve(ctx)

    export = Path(root) / "hf" / "step-1"
    export.mkdir(parents=True)
    (export / "config.json").write_text("{}")
    (export / "tokenizer_config.json").write_text("{}")
    (export / "model.safetensors").write_bytes(b"saved-policy")
    resolved = model.resolve(ctx)
    assert resolved.uri == str(export)
    assert Path(resolved.uri, "model.safetensors").read_bytes() == b"saved-policy"
