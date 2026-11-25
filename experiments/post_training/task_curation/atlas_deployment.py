# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Atlas deployment identity: the storage regions and served endpoints releases read from.

This module holds deployment identity — infrastructure state that changes only when the
Atlas deployment itself moves. No per-release choice lives here: a release config names a
cluster and the tables below resolve everything else, so release policy stays in the
release config and stage sources carry no deployment literals (the release_stages guard
enforces that; this module deliberately lives outside that package).
"""

import dataclasses
from dataclasses import dataclass


@dataclasses.dataclass(frozen=True)
class RegionDeployment:
    """One cluster's storage deployment: the prefixes a release reads and writes."""

    cluster: str
    region_label: str
    region_prefix: str
    catalog_root: str

    def validate_shape(self) -> None:
        if not self.cluster or not self.region_label:
            raise ValueError("cluster and region_label are required")
        if not self.region_prefix or not self.region_prefix.endswith("/"):
            raise ValueError(f"region_prefix must be a non-empty '/'-suffixed prefix, got {self.region_prefix!r}")
        if not self.catalog_root or self.catalog_root.endswith("/"):
            raise ValueError(f"catalog_root must be a non-empty root path, got {self.catalog_root!r}")


REGION_DEPLOYMENTS: dict[str, RegionDeployment] = {
    "cw-us-east-02a": RegionDeployment(
        cluster="cw-us-east-02a",
        region_label="us-east-02a",
        region_prefix="s3://marin-us-east-02a/marin/rl-data/atlas/",
        catalog_root="s3://marin-us-east-02a/marin/rl-data/catalog/production",
    ),
}


@dataclass(frozen=True)
class AtlasService:
    """The served Atlas endpoints a release verifies against (deployment identity)."""

    service_url: str
    applet_id: str
    applet_origin: str
    public_root: str

    def validate_shape(self) -> None:
        for name in ("service_url", "applet_id", "applet_origin", "public_root"):
            if not getattr(self, name):
                raise ValueError(f"{name} is required deployment identity")


ATLAS_SERVICE = AtlasService(
    service_url="https://marina.oa.dev",
    applet_id="c9742a15-fb46-4d56-bc88-aa916351ebd3",
    applet_origin="https://applets.marina.oa.dev/a/c9742a15-fb46-4d56-bc88-aa916351ebd3",
    public_root="https://atlas.oa.dev",
)


for _deployment in REGION_DEPLOYMENTS.values():
    _deployment.validate_shape()
    for _name in ("region_prefix", "catalog_root"):
        if not getattr(_deployment, _name).startswith("s3://"):
            raise ValueError(f"the {_deployment.cluster} deployment's {_name} must be an s3 path")


def region_deployment(cluster: str) -> RegionDeployment:
    """Resolve one cluster's storage deployment; unknown clusters refuse."""
    deployment = REGION_DEPLOYMENTS.get(cluster)
    if deployment is None:
        raise ValueError(f"unknown cluster {cluster!r}; known: {sorted(REGION_DEPLOYMENTS)}")
    deployment.validate_shape()
    return deployment
