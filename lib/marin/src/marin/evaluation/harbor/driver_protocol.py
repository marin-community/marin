# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Constants shared by the isolated Harbor driver and its parent process."""

FULL_GIT_COMMIT_LENGTH = 40
FULL_GIT_COMMIT_PATTERN = rf"^[0-9a-f]{{{FULL_GIT_COMMIT_LENGTH}}}$"
