# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for the autoscaler dashboard tab with a 2-group TPU config.

Boots a local cluster with two scale groups (v5litepod-16 and v5litepod-32)
and validates that the autoscaler tab correctly renders scale group info,
slice state, recent actions, and logs.
"""

import time

import pytest
from iris.client.client import IrisClient
from iris.cluster.config import load_config, make_local_config
from iris.cluster.manager import connect_cluster
from iris.rpc import config_pb2
from iris.rpc.cluster_connect import ControllerServiceClientSync

from .conftest import (
    IRIS_ROOT,
    TestCluster,
    _is_noop_page,
    assert_visible,
    dashboard_click,
    wait_for_dashboard_ready,
)

DEFAULT_CONFIG = IRIS_ROOT / "examples" / "demo.yaml"

pytestmark = pytest.mark.e2e


def _add_scale_group(
    config: config_pb2.IrisClusterConfig,
    name: str,
    variant: str,
    num_vms: int,
) -> None:
    sg = config.scale_groups[name]
    sg.name = name
    sg.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.accelerator_variant = variant
    sg.num_vms = num_vms
    sg.min_slices = 1
    sg.max_slices = 2
    sg.resources.cpu = 128
    sg.resources.memory_bytes = 128 * 1024**3
    sg.resources.disk_bytes = 1024 * 1024**3
    sg.slice_template.preemptible = True
    sg.slice_template.num_vms = num_vms
    sg.slice_template.accelerator_type = config_pb2.ACCELERATOR_TYPE_TPU
    sg.slice_template.accelerator_variant = variant
    sg.slice_template.local.SetInParent()


def _make_two_group_config() -> config_pb2.IrisClusterConfig:
    config = load_config(DEFAULT_CONFIG)
    config.scale_groups.clear()
    _add_scale_group(config, "tpu_v5e_16", "v5litepod-16", num_vms=2)
    _add_scale_group(config, "tpu_v5e_32", "v5litepod-32", num_vms=4)
    return make_local_config(config)


@pytest.fixture(scope="module")
def cluster():
    """Boots a local cluster with two TPU scale groups for autoscaler dashboard tests."""
    config = _make_two_group_config()
    with connect_cluster(config) as url:
        client = IrisClient.remote(url, workspace=IRIS_ROOT)
        controller_client = ControllerServiceClientSync(address=url, timeout_ms=30000)
        tc = TestCluster(url=url, client=client, controller_client=controller_client)
        # min_slices=1 each: 2 VMs for v5e_16 + 4 VMs for v5e_32 = 6 workers
        tc.wait_for_workers(6, timeout=30)
        yield tc
        controller_client.close()


def _click_autoscaler_tab(page, cluster):
    """Navigate to dashboard root and open the Autoscaler tab."""
    if not _is_noop_page(page):
        page.goto(f"{cluster.url}/")
    wait_for_dashboard_ready(page)
    dashboard_click(page, 'button.tab-btn:has-text("Autoscaler")')
    if not _is_noop_page(page):
        time.sleep(0.5)


def test_autoscaler_tab_shows_scale_groups(cluster, page, screenshot):
    """Both scale groups appear in the Autoscaler tab's Scale Groups table."""
    _click_autoscaler_tab(page, cluster)

    assert_visible(page, "text=tpu_v5e_16")
    assert_visible(page, "text=tpu_v5e_32")

    screenshot("autoscaler-scale-groups")


def test_autoscaler_tab_shows_slices(cluster, page, screenshot):
    """Slice sub-rows with VM state indicators and VM counts are rendered."""
    _click_autoscaler_tab(page, cluster)

    if not _is_noop_page(page):
        page.wait_for_selector(".vm-state-indicator", timeout=10000)
        indicators = page.locator(".vm-state-indicator")
        assert indicators.count() >= 2, f"Expected at least 2 vm-state-indicator elements, got {indicators.count()}"

        vm_cells = page.locator("text=/\\d+ VMs/")
        assert vm_cells.count() >= 2, f"Expected at least 2 VM count cells, got {vm_cells.count()}"

    screenshot("autoscaler-slices")


def test_autoscaler_tab_recent_actions(cluster, page, screenshot):
    """Recent Actions section shows scale_up entries from initial boot."""
    _click_autoscaler_tab(page, cluster)

    if not _is_noop_page(page):
        assert_visible(page, "text=scale up", timeout=10000)

    screenshot("autoscaler-recent-actions")


def test_autoscaler_tab_logs_section_rendered(cluster, page, screenshot):
    """Autoscaler logs section is rendered with a heading and pre element.

    The local platform does not configure a LogBuffer, so the logs section
    will show either real logs or the empty-state message depending on
    environment. We just verify the section itself renders correctly.
    """
    _click_autoscaler_tab(page, cluster)

    if not _is_noop_page(page):
        page.wait_for_selector("h3:has-text('Autoscaler Logs')", timeout=10000)
        logs_pre = page.locator("h3:has-text('Autoscaler Logs') + pre")
        logs_pre.wait_for(timeout=10000)
        content = logs_pre.text_content(timeout=5000)
        assert content is not None and len(content) > 0, "Autoscaler logs pre element is empty"
        assert content != "Loading logs...", "Logs section still showing loading state"

    screenshot("autoscaler-logs")
