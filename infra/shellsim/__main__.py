# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Publish the worker image and resolve public storage owned by the buckets stack."""

import pulumi
from infrastructure import deploy

config = pulumi.Config()
buckets = pulumi.StackReference(config.require("buckets_stack"))
pulumi.export("url", buckets.require_output("publicUrl").apply(lambda base: f"{base}/shellsim/ports"))
pulumi.export("publicationPrefix", buckets.require_output("publicPrefix").apply(lambda base: f"{base}/shellsim/ports"))
pulumi.export("workerImage", deploy(config.require("source_revision")))
