# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import io
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import zstandard
from levanter.store.cache import CacheMetadata
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import materialized_config
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import launch_starcoder_tpp10 as original
from experiments.domain_phase_mix import launch_tpp10_instruction_sweep as launcher
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as old_data
from experiments.domain_phase_mix import prepare_tpp10_instruction_sweep as preparation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment


def _zstd(records: list[dict]) -> bytes:
    return zstandard.ZstdCompressor().compress("".join(json.dumps(r) + "\n" for r in records).encode())


def test_zstd_lines_decodes_bounded_records_and_requires_text():
    payload = _zstd([{"text": "alpha"}, {"text": "beta", "id": 2}])
    assert [r["text"] for r in preparation.zstd_lines(io.BytesIO(payload), len(payload), "gs://x")] == ["alpha", "beta"]
    with pytest.raises(ValueError, match="Missing text field"):
        list(preparation.zstd_lines(io.BytesIO(_zstd([{"id": 1}])), 10**6, "gs://x"))
    with pytest.raises(ValueError, match="source-read budget"):
        list(preparation.zstd_lines(io.BytesIO(payload), len(payload) // 2, "gs://x"))


def test_instruction_training_step_keeps_frozen_streams_and_adds_qa_evaluations(tmp_path):
    design = experiment.load_design()
    old = old_data.data_steps(design)
    new = preparation.data_steps(design)
    metadata = CacheMetadata(old_data.FORMAT.build_preprocessor(experiment.verified_tokenizer()).metadata)
    for i, step in enumerate([*old.values(), *new.values()]):
        path = step.path(str(tmp_path))
        write_record(
            ArtifactRecord(
                output_path=path,
                fingerprint=step.fingerprint(),
                config={"tokenizer": experiment.TOKENIZER, "format": {"text_key": "text"}},
            )
        )
        if step is old["evaluation"]:
            continue
        ids = np.repeat(np.arange(1000 * (i + 1), 1000 * (i + 1) + 512, dtype=np.int32), experiment.SEQ_LEN)
        old_data.write_part(
            [{"input_ids": ids}], path + "/train", metadata=metadata, identity={"n": step.name}, expected_tokens=len(ids)
        )
    caches = {
        **old,
        "starcoder": new[f"{preparation.DOMAIN}/parent"],
        f"subset_{experiment.SUBSET_SEEDS[0]}": new[f"{preparation.DOMAIN}/matched"],
    }
    request = experiment.RunSpec(
        "flan-50", experiment.Arm.MATCHED, 50, experiment.TRAINER_SEEDS[0], experiment.SUBSET_SEEDS[0], 2532, 32
    )
    extended = materialized_config(launcher.training_step(design, request, caches), str(tmp_path)).training
    frozen = materialized_config(original.training_step(design, request, caches), str(tmp_path))
    data = extended.pod.train_config.data
    expected = {
        experiment.PRIMARY_METRIC.removeprefix("eval/").removesuffix("/bpb"),
        *preparation.evaluation_paths(design),
    }
    assert {f"qa/{name}-tpp10" for name in preparation.QA_EVALS} | {preparation.HELDOUT} <= expected
    assert all(data.train_weights[name] == 0.0 for name in preparation.evaluation_paths(design))
    trained = {name: weight for name, weight in data.train_weights.items() if weight > 0}
    assert trained == {name: weight for name, weight in frozen.pod.train_config.data.train_weights.items() if weight > 0}
    assert extended.pod.train_config.trainer.tracker.group == launcher.TRACKER_GROUP
    components = dict(data.components)
    for name in expected:
        cache = str(tmp_path / "eval" / name)
        components[name] = replace(components[name], cache_dir=cache)
        old_data.write_part(
            [{"input_ids": np.arange(4 * experiment.SEQ_LEN, dtype=np.int32)}],
            cache + "/validation",
            metadata=metadata,
            identity={"eval": name},
            expected_tokens=4 * experiment.SEQ_LEN,
        )
    assert launcher.validate_evaluation_tags(replace(data, components=components), design) == {
        name: 4 for name in expected
    }


def test_final_metrics_require_every_qa_evaluation(tmp_path):
    request = {"output_path": str(tmp_path), "run_name": "example", "total_steps": 10}
    path = Path(LevanterCheckpoint(path=str(tmp_path)).checkpoint_dir) / "eval_metrics.jsonl"
    path.parent.mkdir()
    row = {"step": 9, experiment.PRIMARY_METRIC: 1.0}
    row.update({f"eval/uncheatable_eval/{name}/bpb": 2.0 for name in launcher.uncheatable.COMPONENTS})
    row.update({metric: 3.0 for metric in preparation.ON_TARGET_METRICS[:-1]})
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Missing final evaluations"):
        launcher.final_metrics(request)
    row[preparation.ON_TARGET_METRICS[-1]] = 4.0
    path.write_text(json.dumps(row) + "\n")
    assert launcher.final_metrics(request)[preparation.ON_TARGET_METRICS[-1]] == 4.0
    assert launcher.metric_column("eval/qa/sciq-tpp10/bpb") == "sciq_bpb"
    assert launcher.metric_column(f"eval/{preparation.HELDOUT}/bpb") == "dolmino_flan_heldout_bpb"
