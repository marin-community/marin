# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the worker image and resolve public storage owned by the buckets stack."""

from urllib.parse import urljoin

import pulumi
from infrastructure import deploy
from rigging.filesystem.storage_path import StoragePath

PUBLICATION_DIRECTORY = "shellsim/ports"

config = pulumi.Config()
buckets = pulumi.StackReference(config.require("buckets_stack"))
pulumi.export(
    "url", buckets.require_output("publicUrl").apply(lambda base: urljoin(base.rstrip("/") + "/", PUBLICATION_DIRECTORY))
)
pulumi.export(
    "publicationPrefix",
    buckets.require_output("publicPrefix").apply(lambda base: str(StoragePath(base) / PUBLICATION_DIRECTORY)),
)
pulumi.export("workerImage", deploy(config.require("source_revision")))
