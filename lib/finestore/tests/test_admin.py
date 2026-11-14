# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import shutil

import pytest
from finestore.admin import fork_relocated_archive
from finestore.commit import CommitCoordinator, CommitDelta
from finestore.layout import FineStoreLayout
from finestore.reader import ReadView
from finestore.store import DataStore


@pytest.mark.parametrize("separator", ["", "/"])
def test_copied_cache_fork_reads_locally_and_accepts_new_vectors(tmp_path, separator):
    original = tmp_path / "original"
    with DataStore.open(str(original)) as store:
        store.write_object("existing", b"first")
        store.flush()
        store.write_object("existing", b"latest")
        store.write_object("second", b"preserved")
    intermediate = tmp_path / "intermediate"
    shutil.copytree(original, intermediate)
    CommitCoordinator(FineStoreLayout(str(intermediate))).commit(CommitDelta())
    copied = tmp_path / "copied"
    shutil.copytree(intermediate, copied)
    original.rename(tmp_path / "historical")
    intermediate.rename(tmp_path / "historical-intermediate")
    output = tmp_path / "fork"
    fork_relocated_archive(
        str(original) + separator,
        str(copied) + separator,
        str(output) + separator,
        reference_roots=(str(intermediate) + separator,),
    )
    with DataStore.open(str(output)) as store:
        store.write_object("new", b"GPU vector")
    assert ReadView(str(output)).read_blobs(["existing", "second", "new"]) == {
        "existing": b"latest",
        "second": b"preserved",
        "new": b"GPU vector",
    }
    with pytest.raises(ValueError):
        fork_relocated_archive(str(original), str(copied), str(output))
