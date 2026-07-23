# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from rigging.connect import proxy_path

ENDPOINT_NAME = "/xprof"
PORT_NAME = "xprof"
HEALTH_PATH = "/healthz"
PUBLIC_PATH = proxy_path(ENDPOINT_NAME)
XPROF_PACKAGE = "xprof==2.22.3"
