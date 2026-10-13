# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build ports images and declare the proxy using ambient provider credentials."""

import pulumi
from infrastructure import deploy, stack_config

deployment = deploy(stack_config())
pulumi.export("url", deployment.url)
pulumi.export("workerImage", deployment.worker_image)
