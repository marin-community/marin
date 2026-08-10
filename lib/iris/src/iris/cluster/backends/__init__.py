# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TaskBackend implementations.

Each subpackage implements the `TaskBackend` contract from
`iris.cluster.controller.backend`: `rpc/` fans work out to Iris worker
daemons; `k8s/` places tasks directly on a Kubernetes cluster via Kueue.
Machine-lifecycle providers live in `iris.cluster.platforms`.
"""
