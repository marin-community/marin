# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the evalchemy HTTP-client child's durable upload."""

import json
import tempfile
from pathlib import Path

import fsspec

from marin.evaluation.evalchemy.client import upload_task_output
from marin.evaluation.lm_eval_samples import is_scratch_artifact


def _write_task_tree(scratch_root: Path, acc: float) -> str:
    """A fresh ``tmp<random>/`` scratch dir holding one lm-eval ``<model>/results_*.json`` tree."""
    local_out = tempfile.mkdtemp(dir=scratch_root)
    model_dir = Path(local_out) / "model"
    model_dir.mkdir()
    (model_dir / "results_x.json").write_text(json.dumps({"results": {"mmlu": {"acc": acc}}}))
    return local_out


def test_retried_upload_leaves_one_task_tree(tmp_path):
    """A retry reusing the same durable dest replaces the prior tree instead of nesting a second one.

    Without removal, fsspec ``put`` nests the second attempt's scratch tempdir under the existing
    dest as ``mmlu_5shot/tmp<random>/model/results_*.json``, so a real MMLU record carries the panel
    twice.
    """
    scratch_root = tmp_path / "scratch"
    scratch_root.mkdir()
    dest = str(tmp_path / "out" / "mmlu_5shot")
    out_fs, _ = fsspec.core.url_to_fs(dest)

    upload_task_output(out_fs, _write_task_tree(scratch_root, 0.63502), dest)
    upload_task_output(out_fs, _write_task_tree(scratch_root, 0.63488), dest)

    results = sorted(p.relative_to(dest).as_posix() for p in Path(dest).rglob("results_*.json"))
    assert results == ["model/results_x.json"]
    assert not any(is_scratch_artifact(r) for r in results)
