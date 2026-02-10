# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Import order matters for proto descriptor pool: each module must be loaded
# after its dependencies. Chain: time → config → vm → cluster.
from iris.rpc import time_pb2 as time_pb2
from iris.rpc import config_pb2 as config_pb2
from iris.rpc import vm_pb2 as vm_pb2
from iris.rpc import cluster_pb2 as cluster_pb2
