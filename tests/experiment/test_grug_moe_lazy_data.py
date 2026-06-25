# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The grug-moe run whose data is a ``Dataset`` handle must lower through the pure
``lower()`` path — a ``StepSpec`` graph the ``StepRunner`` runs with the
``Executor`` (content-addressing, ``InputName``, ``instantiate_config``) out of the
path entirely.
"""

from marin.execution.executor import executor_context
from marin.execution.lazy import Dataset, lower, materialized_config

from experiments.grug.moe.launch_lazy import grug_moe_slimpajama


def test_slimpajama_lowers_to_pure_stepspec_graph():
    ckpt = grug_moe_slimpajama()
    with executor_context():
        spec = lower(ckpt)

    # train step + one tokenize dependency, both addressed by explicit name@version
    assert spec.name == "grug/4_10_baseline_moe_slim"
    assert spec.override_output_path == "grug/4_10_baseline_moe_slim/v1"
    assert [dep.name for dep in spec.deps] == ["slimpajama-6b"]

    tok = spec.deps[0]
    assert tok.override_output_path == "slimpajama-6b/v1"
    # tokenization runs as its own Fray job; the launcher step runs inline.
    assert tok.resources is not None
    assert spec.resources is None


def test_mixture_resolves_dependency_path_without_executor():
    prefix = "gs://marin-golden"
    ckpt = grug_moe_slimpajama()

    config = materialized_config(ckpt, prefix)

    # The mixture component's cache_dir is a concrete string resolved via
    # ctx.path(dataset) — no InputName, no executor-computed content hash.
    component = config.data.components["slimpajama-6b"]
    assert component.cache_dir == f"{prefix}/slimpajama-6b/v1"
    assert config.output_path == f"{prefix}/grug/4_10_baseline_moe_slim/v1"

    # The tokenize step itself writes its cache at the same explicit path.
    slim: Dataset = ckpt.recipe.deps[0]
    assert materialized_config(slim, prefix).cache_path == f"{prefix}/slimpajama-6b/v1"
