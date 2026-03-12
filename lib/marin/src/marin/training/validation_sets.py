# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from levanter.data.text.datasets import DatasetComponent, LmDataConfig, UrlDatasetSourceConfig


def default_validation_components(base_data: LmDataConfig) -> dict[str, DatasetComponent]:
    components: dict[str, DatasetComponent] = {}
    for name, component in base_data.components.items():
        if not isinstance(component, DatasetComponent):
            continue
        source = component.source
        if source is None:
            continue
        if isinstance(source, UrlDatasetSourceConfig) and len(source.validation_urls) == 0:
            continue
        components[name] = replace(component, cache_dir=None)
    return components
