# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Frozen original WARC replay evidence shared by planning and completion."""

from experiments.datakit.scripts.metadata_cluster_selection import _digest

FOCUS_SOURCE = "common-crawl-focus-2026-22"
JUSTEXT_REVISION = "20d27c00ebfbe927f86281933da687d3e636cba3"
MODEL_SETTINGS = {"JUSTEXT_MODEL": "sklearn", "JUSTEXT_NO_DOWNLOAD": "1"}
RUNTIME_DEPENDENCIES = {
    "jusText": "4.2.0",
    "scikit-learn": "1.8.0",
    "numpy": "2.3.5",
    "resiliparse": "0.17.2",
    "lxml": "6.1.3",
    "warcio": "1.7.5",
}


def focus_provider_evidence(targets: dict[str, list[dict]]) -> dict:
    """Describe the selected original WARC inventory and verified replay settings."""
    return {
        "kind": "common_crawl_original_warc_targets",
        "targets_sha256": _digest(targets),
        "justext_revision": JUSTEXT_REVISION,
        "model_settings": MODEL_SETTINGS.copy(),
        "runtime_dependencies": RUNTIME_DEPENDENCIES.copy(),
        "archived_extractor_commit": "20b7003fea217ec031c0cfe77643854641abbffc",
        "scope": "Selected UUIDs from original HTTP WARC ranges; cached columns supply locators only",
    }
