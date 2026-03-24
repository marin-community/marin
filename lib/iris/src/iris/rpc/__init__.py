# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Import order matters for proto descriptor pool: each module must be loaded
# after its dependencies. Chain: time → config → vm → query → cluster.
from importlib import import_module

time_pb2 = import_module(".time_pb2", __name__)
config_pb2 = import_module(".config_pb2", __name__)
vm_pb2 = import_module(".vm_pb2", __name__)
query_pb2 = import_module(".query_pb2", __name__)
cluster_pb2 = import_module(".cluster_pb2", __name__)
