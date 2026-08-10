# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from ducky.catalog import DATAKIT_SCHEMA, FINELOG_SCHEMA, build_catalog
from ducky.config import DuckyConfig


def _config(**overrides) -> DuckyConfig:
    return DuckyConfig(scratch_bucket="/tmp/ducky", **overrides)


def test_build_catalog_finelog_views_glob_lsm_segments():
    catalog = build_catalog(_config(finelog_root="gs://marin-us-central2/finelog/marin", datakit_root=None))

    finelog = {v.name: v for v in catalog.views if v.schema == FINELOG_SCHEMA}
    assert "log" in finelog and "iris.task" in finelog and "zephyr.stage" in finelog
    assert not any(v.schema == DATAKIT_SCHEMA for v in catalog.views)  # datakit disabled
    # dotted namespaces must be quoted in the qualified identifier
    assert finelog["iris.task"].qualified_name == 'finelog."iris.task"'
    assert (
        finelog["iris.task"].definition_sql
        == "SELECT * FROM read_parquet('gs://marin-us-central2/finelog/marin/iris.task/seg_L*.parquet')"
    )


def test_build_catalog_datakit_views_glob_hashed_dirs():
    catalog = build_catalog(_config(finelog_root=None, datakit_root="gs://marin-us-east5/normalized"))

    datakit = {v.name: v for v in catalog.views if v.schema == DATAKIT_SCHEMA}
    assert "finetranslations" in datakit
    # the unguessable recipe hash is globbed with `<name>_*`, and the nested family/subset
    # path segment is preserved
    assert (
        datakit["nemotron_cc_v2_high_quality"].definition_sql
        == "SELECT * FROM read_parquet('gs://marin-us-east5/normalized/nemotron_cc_v2/high_quality_*/outputs/main/*.parquet')"
    )


def test_build_catalog_trailing_slash_on_root_is_normalized():
    catalog = build_catalog(_config(finelog_root="gs://b/finelog/", datakit_root=None))
    log = next(v for v in catalog.views if v.name == "log")
    assert "finelog//log" not in log.definition_sql
    assert "read_parquet('gs://b/finelog/log/seg_L*.parquet')" in log.definition_sql


def test_build_catalog_examples_reference_existing_views():
    catalog = build_catalog(_config(finelog_root="gs://b/finelog", datakit_root="gs://b/normalized"))
    view_idents = {v.qualified_name for v in catalog.views}
    finelog_examples = [e for e in catalog.examples if "finelog." in e.sql]
    assert finelog_examples  # finelog contributed examples
    # every finelog example query targets a view we actually built
    for example in finelog_examples:
        assert any(ident in example.sql for ident in view_idents)
    # the datakit examples reference the configured root directly (browse glob + read_parquet)
    assert any("gs://b/normalized" in e.sql for e in catalog.examples)


def test_example_queries_avoid_unbounded_scans():
    """Every example is bounded — a LIMIT, an aggregate, or a metadata-only glob — never a
    bulk `SELECT *` over a view, which would pull the multi-GB text/data column."""
    catalog = build_catalog(_config(finelog_root="gs://b/finelog", datakit_root="gs://b/normalized"))
    for example in catalog.examples:
        assert "SELECT *" not in example.sql, f"{example.title!r} bulk-selects all columns"
        bounded = any(token in example.sql for token in ("LIMIT", "count(", "GROUP BY", "glob("))
        assert bounded, f"{example.title!r} is an unbounded row scan"


def test_build_catalog_empty_without_roots():
    catalog = build_catalog(_config(finelog_root=None, datakit_root=None))
    assert catalog.views == ()
    assert catalog.examples == ()


def test_build_catalog_available_filters_views_and_examples():
    # only the `log` view is available; the catalog must drop the other views and any example
    # that references a view we didn't create.
    catalog = build_catalog(_config(finelog_root="gs://b/finelog", datakit_root=None), available={'finelog."log"'})
    assert {v.qualified_name for v in catalog.views} == {'finelog."log"'}
    assert catalog.examples  # log-only examples survive
    for example in catalog.examples:
        assert 'finelog."iris.task"' not in example.sql
        assert 'finelog."iris.worker"' not in example.sql


def test_build_catalog_available_empty_drops_source_entirely():
    catalog = build_catalog(_config(finelog_root="gs://b/finelog", datakit_root="gs://b/normalized"), available=set())
    assert catalog.views == ()
    assert catalog.examples == ()  # no examples advertised for a source with no created views
