# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from rigging.connect import proxy_path

ENDPOINT_NAME = "/xprof"
PORT_NAME = "xprof"
HEALTH_PATH = "/healthz"
PUBLIC_PATH = proxy_path(ENDPOINT_NAME)
XPROF_PACKAGE = "xprof==2.22.3"

# XProf overview_page requests over ALL_HOSTS on large profiles routinely take
# longer than the controller proxy's 120s default, so raise the per-endpoint
# upstream timeout well above it.
PROXY_TIMEOUT_SECONDS = 600
