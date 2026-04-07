# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Import order matters for proto descriptor pool: each module must be loaded
# after its dependencies. Chain: time → config → vm → query → job → controller/worker.
from iris.rpc import time_pb2 as time_pb2
from iris.rpc import config_pb2 as config_pb2
from iris.rpc import vm_pb2 as vm_pb2
from iris.rpc import query_pb2 as query_pb2
from iris.rpc import job_pb2 as job_pb2
from iris.rpc import controller_pb2 as controller_pb2
from iris.rpc import worker_pb2 as worker_pb2
