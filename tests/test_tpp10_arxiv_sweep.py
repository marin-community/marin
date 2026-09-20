# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import haliax as hax
import jax
import numpy as np
import pytest
from fray.types import JobRequest
from levanter.store.cache import CacheMetadata
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import materialized_config
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import analyze_tpp10_arxiv_sweep as analysis
from experiments.domain_phase_mix import launch_starcoder_tpp10 as original
from experiments.domain_phase_mix import launch_tpp10_arxiv_sweep as launcher
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as old_data
from experiments.domain_phase_mix import prepare_tpp10_arxiv_sweep as preparation
from experiments.domain_phase_mix import resume_tpp10_arxiv_sweep as recovery
from experiments.domain_phase_mix import starcoder_tpp10 as experiment


@pytest.fixture(scope="module")
def design():
    return experiment.load_design()


def test_arxiv_training_step_keeps_frozen_streams_and_adds_s2orc_evaluation(design, tmp_path):
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
    _, key = jax.random.split(jax.random.PRNGKey(experiment.DATA_SEED))
    streams = {}
    for percent in (0, 5, 100):
        request = experiment.RunSpec(
            f"arxiv-{percent}",
            experiment.Arm.MATCHED,
            percent,
            experiment.TRAINER_SEEDS[0],
            experiment.SUBSET_SEEDS[0],
            2532,
            32,
        )
        extended = materialized_config(launcher.training_step(design, request, caches), str(tmp_path)).training
        frozen = materialized_config(original.training_step(design, request, caches), str(tmp_path))
        assert extended.pod.train_config.trainer.tracker.group == launcher.TRACKER_GROUP
        data = extended.pod.train_config.data
        assert preparation.S2ORC in data.components and data.train_weights[preparation.S2ORC] == 0.0
        if percent == 0:
            components = dict(data.components)
            expected = {
                experiment.PRIMARY_METRIC.removeprefix("eval/").removesuffix("/bpb"),
                *preparation.evaluation_paths(design),
            }
            assert preparation.S2ORC in expected
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
        observed = []
        for config in (data, frozen.pod.train_config.data):
            sets = config.train_sets(hax.Axis("position", experiment.SEQ_LEN), key=key, initial_batch_size=32)
            observed.append(
                {
                    name: [int(np.asarray(item.tokens)[0]) for item in asyncio.run(s.get_batch(range(32)))]
                    for name, s in sets.items()
                }
            )
        assert observed[0] == observed[1]
        assert not any(name.startswith(("uncheatable", "paloma/m2d2")) for name in observed[0])
        streams[percent] = observed[0]
    assert set(streams[0]) == set(experiment.WEB_COUNTS)
    assert {name: streams[5][name] for name in experiment.WEB_COUNTS} == streams[0]
    assert streams[100] == {"starcoder": streams[5]["starcoder"]}


def test_final_metrics_require_s2orc_and_every_uncheatable_component(tmp_path):
    request = {"output_path": str(tmp_path), "run_name": "example", "total_steps": 10}
    path = Path(LevanterCheckpoint(path=str(tmp_path)).checkpoint_dir) / "eval_metrics.jsonl"
    path.parent.mkdir()
    row = {"step": 9, experiment.PRIMARY_METRIC: 1.0}
    row.update({f"eval/uncheatable_eval/{name}/bpb": 2.0 for name in launcher.uncheatable.COMPONENTS})
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Missing final evaluations"):
        launcher.final_metrics(request)
    row[preparation.S2ORC_METRIC] = 3.0
    path.write_text(json.dumps(row) + "\n")
    assert launcher.final_metrics(request)[preparation.S2ORC_METRIC] == 3.0


def test_analysis_reports_interior_minimum_and_neighbor_excess():
    grid = [0, 5, 50, 100]
    rows = [
        {"run_name": f"r{p}", "domain": "arxiv_papers", "arm": "matched", "percent": p, "s2orc_bpb": v, "paloma_bpb": w}
        for p, v, w in zip(grid, [1.0, 0.8, 0.9, 1.2], [0.5, 0.4, 0.3, 0.2], strict=True)
    ]
    allocations = {p: {"matched_epochs": p / 10, "target_epochs": p / 10} for p in grid}
    result = analysis.analyze(rows, allocations, grid, "matched")["metrics"]
    assert result["s2orc_bpb"]["minimum_percent"] == 5 and not result["s2orc_bpb"]["boundary_minimum"]
    assert result["s2orc_bpb"]["neighbor_excess_percent"] == {"0": pytest.approx(25.0), "50": pytest.approx(12.5)}
    assert result["paloma_bpb"]["minimum_percent"] == 100 and result["paloma_bpb"]["boundary_minimum"]
    with pytest.raises(ValueError, match="Missing or duplicate"):
        analysis.analyze(rows[:-1], allocations, grid, "matched")


def test_recovery_lowers_only_frozen_preparation_requests():
    raw = JobRequest(name="prepare_raw-1234", entrypoint=object(), resources=preparation.CPU)
    assert recovery.smaller_preparation_request(raw).resources == replace(preparation.CPU, ram="4g")
    tpu = JobRequest(name="tpp10_arxiv_papers_matched_p100_s20260910", entrypoint=object(), resources=launcher.TPU)
    assert recovery.smaller_preparation_request(tpu) is tpu
    with pytest.raises(ValueError, match="only the frozen preparation CPU tasks"):
        recovery.smaller_preparation_request(replace(raw, name="evaluate-1234"))
    with pytest.raises(ValueError, match="only the frozen preparation CPU tasks"):
        recovery.smaller_preparation_request(replace(raw, resources=replace(preparation.CPU, cpu=4)))
