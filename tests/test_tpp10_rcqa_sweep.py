# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import numpy as np
import pytest
from levanter.store.cache import CacheMetadata
from marin.execution.artifact import ArtifactRecord, write_record
from marin.execution.lazy import materialized_config
from marin.training.training import LevanterCheckpoint

from experiments.domain_phase_mix import launch_starcoder_tpp10 as original
from experiments.domain_phase_mix import launch_tpp10_rcqa_sweep as launcher
from experiments.domain_phase_mix import prepare_starcoder_tpp10 as old_data
from experiments.domain_phase_mix import prepare_tpp10_rcqa_sweep as preparation
from experiments.domain_phase_mix import starcoder_tpp10 as experiment


def test_rcqa_pool_recipe_groups_pinned_files_into_whole_parts(tmp_path):
    design = experiment.load_design()
    steps = preparation.data_steps(design)
    recipe = materialized_config(steps[f"{preparation.DOMAIN}/raw"], str(tmp_path))
    assert recipe.group_size == 350 and len(recipe.raw.sources) == 1400
    assert recipe.raw.tokens == experiment.PARENT_SEQUENCES * experiment.SEQ_LEN and recipe.raw.split == "train"
    assert all(s.uri.startswith("gs://marin-us-central1/raw/dolma3_dolmino_pool") for s in recipe.raw.sources)
    heldout = materialized_config(preparation.evaluation_steps(design)[preparation.HELDOUT], str(tmp_path))
    assert heldout.raw.tokens == preparation.HELDOUT_TOKENS and heldout.group_size == len(heldout.raw.sources) == 32


def test_rcqa_training_step_keeps_frozen_streams_and_adds_evaluations(tmp_path):
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
        "rcqa-30", experiment.Arm.MATCHED, 30, experiment.TRAINER_SEEDS[0], experiment.SUBSET_SEEDS[0], 2532, 32
    )
    extended = materialized_config(launcher.training_step(design, request, caches), str(tmp_path)).training
    frozen = materialized_config(original.training_step(design, request, caches), str(tmp_path))
    data = extended.pod.train_config.data
    expected = {
        experiment.PRIMARY_METRIC.removeprefix("eval/").removesuffix("/bpb"),
        *preparation.evaluation_paths(design),
    }
    assert {f"qa/{name}-tpp10" for name in preparation.QA_EVALS} | {preparation.HELDOUT} <= expected
    assert "dolmino_flan/heldout-tpp10" not in expected
    trained = {name: weight for name, weight in data.train_weights.items() if weight > 0}
    assert trained == {name: weight for name, weight in frozen.pod.train_config.data.train_weights.items() if weight > 0}
    assert extended.pod.train_config.trainer.tracker.group == launcher.TRACKER_GROUP


def test_final_metrics_require_every_rcqa_evaluation(tmp_path):
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
    assert launcher.metric_column(f"eval/{preparation.HELDOUT}/bpb") == "wiki_to_rcqa_heldout_bpb"
