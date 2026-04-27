# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.exp5053_lm_eval_bridge import lm_eval_bridge_raw_validation_sets


def test_lm_eval_bridge_raw_validation_sets_cover_selected_first_pass_slices() -> None:
    datasets = lm_eval_bridge_raw_validation_sets()

    assert set(datasets) == {
        "lm_eval/gsm8k_train",
        "lm_eval/mmlu_auxiliary_train",
        "lm_eval/mmlu_dev",
    }
    assert datasets["lm_eval/mmlu_dev"].text_key == "text"
    assert "issue:5053" in datasets["lm_eval/gsm8k_train"].tags
