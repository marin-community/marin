# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from experiments.evals.synthetic_reasoning_ppl import (
    SYNTHETIC_REASONING_HF_DATASET_ID,
    SYNTHETIC_REASONING_ICL_SHOTS,
    SYNTHETIC_REASONING_PROMPT_FORMAT,
    synthetic_reasoning_raw_validation_sets,
)

EXPECTED_SYNTHETIC_REASONING_KEYS = {
    "synthetic_reasoning_ppl/clrs_style/clrs_bfs",
    "synthetic_reasoning_ppl/clrs_style/clrs_binary_search",
    "synthetic_reasoning_ppl/clrs_style/clrs_connected_components",
    "synthetic_reasoning_ppl/clrs_style/clrs_dijkstra",
    "synthetic_reasoning_ppl/clrs_style/clrs_floyd_warshall",
    "synthetic_reasoning_ppl/clrs_style/clrs_insertion_sort",
    "synthetic_reasoning_ppl/clrs_style/clrs_lcs_length",
    "synthetic_reasoning_ppl/clrs_style/clrs_lis",
    "synthetic_reasoning_ppl/clrs_style/clrs_mst_prim",
    "synthetic_reasoning_ppl/clrs_style/clrs_topological_sort",
    "synthetic_reasoning_ppl/native/bfs_shortest_path",
    "synthetic_reasoning_ppl/native/binary_search",
    "synthetic_reasoning_ppl/native/coin_change_dp",
    "synthetic_reasoning_ppl/native/connected_components",
    "synthetic_reasoning_ppl/native/dijkstra_shortest_path",
    "synthetic_reasoning_ppl/native/edit_distance_dp",
    "synthetic_reasoning_ppl/native/euclid_gcd",
    "synthetic_reasoning_ppl/native/floyd_warshall_apsp",
    "synthetic_reasoning_ppl/native/insertion_sort",
    "synthetic_reasoning_ppl/native/interval_scheduling",
    "synthetic_reasoning_ppl/native/kmp_string_search",
    "synthetic_reasoning_ppl/native/knapsack_01_dp",
    "synthetic_reasoning_ppl/native/lcs_dp",
    "synthetic_reasoning_ppl/native/lis_dp",
    "synthetic_reasoning_ppl/native/n_queens_backtracking",
    "synthetic_reasoning_ppl/native/parentheses_balance",
    "synthetic_reasoning_ppl/native/prim_mst",
    "synthetic_reasoning_ppl/native/propositional_entailment",
    "synthetic_reasoning_ppl/native/topological_sort",
    "synthetic_reasoning_ppl/native/union_find_connectivity",
    "synthetic_reasoning_ppl/stepmath/algebra_linear_equation",
    "synthetic_reasoning_ppl/stepmath/arithmetic",
}

REMOVED_LEGACY_ALIAS_KEYS = {
    "synthetic_reasoning_ppl/native/binary_search_5shot_icl",
    "synthetic_reasoning_ppl/native/edit_distance_dp_1shot_icl",
    "synthetic_reasoning_ppl/native/euclid_gcd_5shot_icl",
    "synthetic_reasoning_ppl/native/interval_scheduling_1shot_icl",
    "synthetic_reasoning_ppl/native/kmp_string_search_1shot_icl",
    "synthetic_reasoning_ppl/native/knapsack_01_dp_1shot_icl",
    "synthetic_reasoning_ppl/native/lcs_dp_1shot_icl",
    "synthetic_reasoning_ppl/native/n_queens_backtracking_1shot_icl",
    "synthetic_reasoning_ppl/native/parentheses_balance_5shot_icl",
    "synthetic_reasoning_ppl/native/union_find_connectivity_1shot_icl",
    "synthetic_reasoning_ppl/stepmath/algebra_linear_equation_3shot_icl",
    "synthetic_reasoning_ppl/stepmath/arithmetic_5shot_icl",
}


def test_synthetic_reasoning_registry_uses_pinned_arrow_icl_target_only_configs():
    datasets = synthetic_reasoning_raw_validation_sets()

    assert set(datasets) == EXPECTED_SYNTHETIC_REASONING_KEYS

    for key in (
        "synthetic_reasoning_ppl/native/euclid_gcd",
        "synthetic_reasoning_ppl/native/connected_components",
        "synthetic_reasoning_ppl/clrs_style/clrs_bfs",
        "synthetic_reasoning_ppl/stepmath/arithmetic",
    ):
        dataset = datasets[key]
        assert dataset.hf_dataset_id == SYNTHETIC_REASONING_HF_DATASET_ID
        assert dataset.hf_dataset_revision == "0cedc373025ae08edf24e24d73eeda75980ddf8c"
        assert dataset.hf_dataset_name is not None
        assert dataset.hf_dataset_name.endswith("_arrow_5shot_icl")
        assert dataset.input_key == "input"
        assert dataset.target_key == "target"
        assert "loss:target_only" in dataset.tags
        assert f"icl_shots:{SYNTHETIC_REASONING_ICL_SHOTS}" in dataset.tags
        assert f"prompt_format:{SYNTHETIC_REASONING_PROMPT_FORMAT}" in dataset.tags

    assert datasets["synthetic_reasoning_ppl/native/euclid_gcd"].hf_dataset_name == ("native_euclid_gcd_arrow_5shot_icl")
    assert datasets["synthetic_reasoning_ppl/clrs_style/clrs_bfs"].hf_dataset_name == (
        "clrs_style_clrs_bfs_arrow_5shot_icl"
    )


def test_legacy_icl_slice_aliases_are_not_registered_twice():
    datasets = synthetic_reasoning_raw_validation_sets()

    assert datasets.keys().isdisjoint(REMOVED_LEGACY_ALIAS_KEYS)

    assert datasets["synthetic_reasoning_ppl/stepmath/arithmetic"].hf_dataset_name == (
        "stepmath_arithmetic_arrow_5shot_icl"
    )
    assert datasets["synthetic_reasoning_ppl/native/euclid_gcd"].hf_dataset_name == ("native_euclid_gcd_arrow_5shot_icl")
