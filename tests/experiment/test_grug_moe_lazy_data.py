# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The grug-moe run whose data is a ``Dataset`` handle must lower through the pure
``lower()`` path — a ``StepSpec`` graph the ``StepRunner`` runs with the
``Executor`` (content-addressing, ``InputName``, ``instantiate_config``) out of the
path entirely.
"""

import pytest
from marin.execution.lazy import Dataset, RunContext, lower, materialized_config
from marin.execution.remote import RemoteCallable
from marin.experiment.data import mixture, tokenized

from experiments.grug.moe.launch_lazy import grug_moe_slimpajama


def test_slimpajama_lowers_to_pure_stepspec_graph():
    ckpt = grug_moe_slimpajama()
    spec = lower(ckpt)

    # train step + one tokenize dependency, both addressed by explicit name@version
    assert spec.name == "grug/4_10_baseline_moe_slim"
    assert spec.override_output_path == "grug/4_10_baseline_moe_slim/v1"
    assert [dep.name for dep in spec.deps] == ["slimpajama-6b"]

    tok = spec.deps[0]
    assert tok.override_output_path == "slimpajama-6b/v1"

    # Resources ride with the fn, never on the graph node: the tokenize handle's fn is
    # wrapped in remote() (its own Fray job), while the grug launcher fn is plain (it
    # dispatches its own training job). lower() puts no resources on the StepSpec.
    slim = ckpt.recipe.deps[0]
    assert isinstance(slim.recipe.fn, RemoteCallable)
    assert not isinstance(ckpt.recipe.fn, RemoteCallable)


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


def test_mixture_rejects_components_with_colliding_names():
    # Two distinct handles share a name -> keyed-by-name component dict would drop one.
    ctx = RunContext.for_run(out="gs://m/out", prefix="gs://m")
    a = tokenized("dupe", tokenizer="t", source="org/a")
    b = tokenized("dupe", tokenizer="t", source="org/b")
    with pytest.raises(ValueError, match="collide"):
        mixture(ctx, {a: 1.0, b: 1.0})
