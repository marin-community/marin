# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

from experiments.domain_phase_mix.release import context, naming, paths
from experiments.domain_phase_mix.release.configs import (
    heldout,
    ladder,
    partition_tasks,
    starcoder,
    swarm_llama,
    swarm_qwen3,
    validation,
)

# Each exporter with the exporters it runs itself, whose files are theirs to list. `fits` and `domain_sweeps` test
# their own lists. The StarCoder export needs no context.
EXPORTERS = {
    partition_tasks: (),
    swarm_qwen3: (),
    swarm_llama: (),
    heldout: (),
    validation: (),
    ladder: (partition_tasks,),
    starcoder: (),
}


@pytest.fixture(scope="module")
def ctx():
    return context.build()


def _recording(monkeypatch) -> set:
    read = set()
    require = paths.require
    monkeypatch.setattr(paths, "require", lambda path: read.add(path) or require(path))
    return read


@pytest.mark.parametrize("module", EXPORTERS, ids=lambda module: module.__name__.rsplit(".", 1)[-1])
def test_required_sources_are_the_files_the_export_reads(ctx, monkeypatch, module):
    # The manifest's provenance and the test modules' skip guards rely on these lists.
    read = _recording(monkeypatch)
    for upstream in EXPORTERS[module]:
        upstream.export(ctx)
    upstream_reads = set(read)
    read.clear()
    if module is starcoder:
        module.export()
    else:
        module.export(ctx)
    required = set(module.required_sources())
    assert required <= read and read - upstream_reads <= required


def test_context_sources_are_the_files_its_names_and_weights_read(monkeypatch):
    read = _recording(monkeypatch)
    for cached in (naming.bucket_table, naming.task_table, naming.task_weights):
        cached.cache_clear()
    naming.bucket_table()
    naming.task_table()
    context.build()
    assert read == set(context.required_sources())
