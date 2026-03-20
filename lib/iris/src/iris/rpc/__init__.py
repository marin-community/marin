# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Proto submodules (time_pb2, config_pb2, vm_pb2, query_pb2, cluster_pb2)
# are imported directly by their consumers rather than re-exported here.
# Re-exporting them causes circular imports because iris.time_utils
# imports time_pb2, which triggers this __init__, which tries to import
# more submodules from the partially-initialized package.
