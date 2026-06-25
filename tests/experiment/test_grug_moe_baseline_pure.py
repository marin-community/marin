# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The full grug-moe baseline lowers to a pure ``StepSpec`` graph: the Nemotron CC
splits plus starcoder and proofpile are pinned ``Dataset`` handles (existing
tokenized data), the paloma/uncheatable validation suites tokenize fresh, the
mixture resolves every component to its path, and the Executor is out of the path.
"""

from marin.execution.executor import executor_context
from marin.execution.lazy import lower, materialized_config

from experiments.evals.uncheatable_lazy import UNCHEATABLE_SUBSETS
from experiments.grug.moe.launch_lazy import grug_moe_baseline_pure
from experiments.paloma import PALOMA_DATASETS_TO_DIR
from experiments.pretraining_datasets.nemotron import NEMOTRON_DATASETS, NEMOTRON_LLAMA3_OVERRIDES
from experiments.pretraining_datasets.simple_lazy import _PROOFPILE_LLAMA3_PIN, _STARCODER_LLAMA3_PIN

# Pinned training components: mixture key -> (pinned tokenized path, weight).
_PINNED_TRAIN = {
    **{f"nemotron_cc/{split}": (NEMOTRON_LLAMA3_OVERRIDES[split], None) for split in NEMOTRON_DATASETS},
    "starcoderdata": (_STARCODER_LLAMA3_PIN, 0.25),
    "proofpile_2": (_PROOFPILE_LLAMA3_PIN, 0.055),
}


def test_baseline_lowers_to_pure_graph():
    ckpt = grug_moe_baseline_pure()
    with executor_context():
        spec = lower(ckpt)

    n_train = len(NEMOTRON_DATASETS) + 2  # nemotron splits + starcoder + proofpile
    n_validation = len(PALOMA_DATASETS_TO_DIR) + len(UNCHEATABLE_SUBSETS)
    # Direct deps of the train step: every training + validation component.
    assert len(spec.deps) == n_train + n_validation

    # Pinned training caches appear verbatim as dependency output paths.
    dep_paths = {dep.override_output_path for dep in spec.deps}
    for pin, _ in _PINNED_TRAIN.values():
        assert pin in dep_paths

    # Each uncheatable subset depends transitively on the one shared raw download.
    uncheatable = [d for d in spec.deps if d.name.startswith("uncheatable_eval/")]
    assert len(uncheatable) == len(UNCHEATABLE_SUBSETS)
    assert all([dep.name for dep in d.deps] == ["raw/uncheatable_eval"] for d in uncheatable)


def test_mixture_resolves_paths_and_weights():
    prefix = "gs://marin-golden"
    config = materialized_config(grug_moe_baseline_pure(), prefix)

    for key, (pin, weight) in _PINNED_TRAIN.items():
        assert config.data.components[key].cache_dir == f"{prefix}/{pin}"
        if weight is not None:
            assert config.data.train_weights[key] == weight

    # Validation suites are present at weight 0, tokenized into fresh explicit caches.
    for subset in PALOMA_DATASETS_TO_DIR:
        assert config.data.train_weights[f"paloma/{subset}"] == 0.0
        assert config.data.components[f"paloma/{subset}"].cache_dir == f"{prefix}/paloma/{subset}/llama3"
    for subset in UNCHEATABLE_SUBSETS:
        assert config.data.train_weights[f"uncheatable_eval/{subset}"] == 0.0

    assert config.output_path == f"{prefix}/grug/4_10_baseline_moe_pure/v1"
