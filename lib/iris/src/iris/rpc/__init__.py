# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Import order matters for proto descriptor pool: each module must be loaded
# after its dependencies. Chain: time → config → vm → cluster.
from iris.rpc import time_pb2 as time_pb2
from iris.rpc import config_pb2 as config_pb2
from iris.rpc import vm_pb2 as vm_pb2
from iris.rpc import cluster_pb2 as cluster_pb2
