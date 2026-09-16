# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import io
import json
from dataclasses import replace
from pathlib import Path

import haliax as hax
import jax
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from levanter.store.cache import CacheMetadata
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import materialized_config
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import analyze_tpp10_domain_sweeps as analysis
from experiments.domain_phase_mix import launch_starcoder_tpp10 as original
from experiments.domain_phase_mix import launch_tpp10_domain_sweeps as launcher
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as old_data
from experiments.domain_phase_mix import prepare_tpp10_domain_sweeps as preparation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment


def test_parquet_reader_supports_real_arrow_and_bounds_repeated_range_reads():
    output = io.BytesIO()
    pq.write_table(pa.table({"text": ["first document", "second document"], "unused": [1, 2]}), output)
    payload = output.getvalue()
    reader = preparation.BoundedParquetReader(io.BytesIO(payload), 3 * len(payload))
    with pq.ParquetFile(reader) as parquet:
        preparation.check_parquet_memory(parquet, 1024)
        with pytest.raises(ValueError, match="Parquet row group"):
            preparation.check_parquet_memory(parquet, 1)
        batch = next(parquet.iter_batches(columns=["text"], batch_size=1))
        assert batch.to_pydict() == {"text": ["first document"]}
    reader = preparation.BoundedParquetReader(io.BytesIO(payload), 10)
    assert reader.read(6) == payload[:6]
    reader.seek(0)
    with pytest.raises(ValueError, match="source-read budget"):
        reader.read(6)


@pytest.fixture(scope="module")
def design():
    return experiment.load_design()


def test_added_eval_and_domain_handles_preserve_real_source_streams(design, tmp_path):
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
            [{"input_ids": ids}],
            path + "/train",
            metadata=metadata,
            identity={"component": step.name},
            expected_tokens=len(ids),
        )
    _, key = jax.random.split(jax.random.PRNGKey(experiment.DATA_SEED))
    baseline = None
    domain_focus = {}
    for domain in preparation.DOMAINS:
        caches = {
            **old,
            "starcoder": new[f"{domain}/parent"],
            f"subset_{experiment.SUBSET_SEEDS[0]}": new[f"{domain}/matched"],
        }
        for percent in (0, 5, 100):
            request = experiment.RunSpec(
                f"{domain}-{percent}",
                experiment.Arm.MATCHED,
                percent,
                experiment.TRAINER_SEEDS[0],
                experiment.SUBSET_SEEDS[0],
                2532,
                32,
            )
            extended = launcher.training_step(design, domain, request, caches)
            frozen = original.training_step(design, request, caches)
            configs = [materialized_config(s, str(tmp_path)) for s in (extended, frozen)]
            data = configs[0].training.pod.train_config.data
            frozen_data = configs[1].pod.train_config.data
            if domain == preparation.DOMAINS[0] and percent == 0:
                components = dict(data.components)
                expected = {
                    experiment.PRIMARY_METRIC.removeprefix("eval/").removesuffix("/bpb"),
                    *preparation.evaluation_paths(),
                }
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
                assert launcher.validate_evaluation_tags(replace(data, components=components)) == {
                    name: 4 for name in expected
                }
            observed = []
            for config in (data, frozen_data):
                streams = config.train_sets(hax.Axis("position", experiment.SEQ_LEN), key=key, initial_batch_size=32)
                observed.append(
                    {
                        name: [int(np.asarray(item.tokens)[0]) for item in asyncio.run(stream.get_batch(range(32)))]
                        for name, stream in streams.items()
                    }
                )
            assert observed[0] == observed[1]
            assert not any(name.startswith("uncheatable") for name in observed[0])
            if percent == 0:
                if baseline is None:
                    baseline = observed[0]
                assert observed[0] == baseline
            elif percent == 5:
                assert {name: observed[0][name] for name in experiment.WEB_COUNTS} == baseline
                domain_focus[domain] = observed[0]["starcoder"]
            else:
                assert observed[0] == {"starcoder": domain_focus[domain]}
    assert domain_focus["wikipedia"] != domain_focus["finemath_3plus"]


def test_missing_or_conflicting_endpoint_component_prevents_collection(tmp_path):
    request = {"output_path": str(tmp_path), "run_name": "example", "total_steps": 10}
    path = Path(LevanterCheckpoint(path=str(tmp_path)).checkpoint_dir) / "eval_metrics.jsonl"
    path.parent.mkdir()
    row = {"step": 9, experiment.PRIMARY_METRIC: 1.0}
    row.update({f"eval/uncheatable_eval/{name}/bpb": 2.0 for name in launcher.uncheatable.COMPONENTS[:-1]})
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="Missing final evaluations"):
        launcher.final_metrics(request)
    row["eval/uncheatable_eval/ao3_english/bpb"] = 3.0
    path.write_text(json.dumps(row) + "\n")
    assert launcher.final_metrics(request)["eval/uncheatable_eval/ao3_english/bpb"] == 3.0
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, experiment.PRIMARY_METRIC: 5.0}) + "\n")
    with pytest.raises(ValueError, match="conflicting final metric"):
        launcher.final_metrics(request)


def test_selection_uses_measured_target_loss_and_shared_zero_weight_control():
    grid = [0, 5, 50, 100]
    rows = [
        {"domain": "example", "arm": arm, "percent": p, "macro_bpb": value}
        for arm, values in [("matched", [0.9, 0.8, 0.8]), ("target", [0.7, 0.6, 0.9])]
        for p, value in zip(grid[1:], values, strict=True)
    ]
    allocations = {p: {"matched_epochs": p / 10, "target_epochs": p / 10} for p in grid}
    result = analysis.analyze(rows, {"matched": 0.95, "target": 0.5}, allocations, grid)["domains"]["example"]
    assert result["curves"]["matched"]["minimum_percent"] == 50
    assert result["curves"]["target"]["minimum_percent"] == 0
    assert result["curves"]["target"]["boundary_minimum"]
    assert result["target_loss_at_proxy_choice"] == 0.6
    assert result["target_grid_regret"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="Missing or duplicate"):
        analysis.analyze(rows[:-1], {"matched": 0.95, "target": 0.5}, allocations, grid)
