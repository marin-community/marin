# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import pytest

from experiments.domain_phase_mix.release import naming, paths
from experiments.domain_phase_mix.release.configs import partition_tasks

pytestmark = pytest.mark.skipif(not paths.MIXSEL.exists(), reason="mixture-selection checkout absent")


def test_partition_has_39_buckets_with_consistent_epochs():
    out = partition_tasks.export(None)["partition"]
    assert len(out) == 39 and out.bucket_id.is_unique
    assert list(out.bucket_id) == naming.BUCKETS
    expected = paths.D_TGT / out.pool_tokens
    assert (abs(out.epochs_per_unit_weight - expected) / expected < 1e-9).all()
    assert out.pool.isin(["dolma3", "dolmino"]).all()
    assert set(out.quality_tier.dropna()) == {"high", "low"}


def test_partition_names_each_pools_license_and_citation():
    out = partition_tasks.export(None)["partition"]
    assert out.pool_license.notna().all() and out.pool_citation_key.notna().all()
    # The licenses that the pools' Hugging Face cards, allenai/dolma3_pool and allenai/dolma3_dolmino_pool, declared on
    # 2026-10-15; the paper cites Olmo 3 for both pools.
    by_pool = out.groupby("pool")[["pool_license", "pool_citation_key"]].agg(lambda c: sorted(set(c)))
    assert by_pool.to_dict("index") == {
        "dolma3": {"pool_license": ["odc-by"], "pool_citation_key": ["olmoOlmo32025"]},
        "dolmino": {"pool_license": ["odc-by"], "pool_citation_key": ["olmoOlmo32025"]},
    }


def test_tasks_have_58_rows_and_objective_weights_sum_to_one():
    out = partition_tasks.export(None)["tasks"]
    assert len(out) == 58 and out.task_id.is_unique
    assert list(out.task_id) == naming.TASKS
    weights = naming.task_weights()
    # The paper's fixed Uncheatable rule applies the stored byte shares unchanged; they sum to 1 + 2.1e-8.
    assert sum(weights["uncheatable_fixed"].values()) == pytest.approx(1.0, abs=1e-6)
    objectives = pd.read_csv(paths.MIXSEL / "data/objectives.csv")
    stored_shares = objectives[objectives.objective == "uncheatable"].weight.tolist()
    assert list(weights["uncheatable_fixed"].values()) == stored_shares
    assert sum(weights["obe_easy"].values()) == pytest.approx(1.0, abs=1e-12)
    assert len(weights["uncheatable_fixed"]) == 7 and len(weights["obe_easy"]) == 51


def test_task_public_accepts_every_internal_spelling():
    assert naming.task_public("olmo_base_eval/easy_bpb/arc_challenge/bpb") == "arc_challenge"
    assert naming.task_public("olmo_base_easy/table9/arc_challenge/bpb") == "arc_challenge"
    assert naming.task_public("table9/arc_challenge/bpb") == "arc_challenge"
    assert naming.task_public("eval/uncheatable_eval/ao3_english/bpb") == "uncheatable/ao3_english"
    assert naming.task_public("uncheatable_ao3_english_bpb") == "uncheatable/ao3_english"


def test_objective_public_maps_every_source_spelling():
    assert naming.objective_public("table9") == "obe_easy"
    assert naming.objective_public("table9_qa15") == "obe_easy_qa15"
    assert naming.objective_public("table9_macro_bpb") == "obe_easy"
    assert naming.objective_public("uncheatable_bpb") == "uncheatable"
    assert naming.objective_public("uncheatable_and_table9") == "uncheatable_and_obe_easy"
    assert naming.objective_public("baseline") == "none"  # the held-out registry's label of its baselines
    assert all(naming.objective_public(name) == name for name in naming.OBJECTIVES)
    with pytest.raises(ValueError):  # an unknown spelling must never reach a public table
        naming.objective_public("table9_macro")


def test_paper_names_appear_verbatim_in_the_appendix():
    appendix = (paths.DRIVE / "sections/appendix.tex").read_text()
    out = partition_tasks.export(None)
    missing = [n for n in out["partition"].paper_name if n not in appendix]
    assert missing == []
