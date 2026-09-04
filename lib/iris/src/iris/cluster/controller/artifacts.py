# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller reads for stored job artifacts."""

from dataclasses import dataclass

from iris.cluster.bundle import BundleStore


@dataclass(frozen=True, slots=True)
class ArtifactDependencies:
    bundles: BundleStore


def bundle_zip(dependencies: ArtifactDependencies, bundle_id: str) -> bytes:
    return dependencies.bundles.get(bundle_id)


def blob_data(dependencies: ArtifactDependencies, blob_id: str) -> bytes:
    return dependencies.bundles.get(blob_id)
