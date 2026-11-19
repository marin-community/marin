# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright Prime Intellect. Licensed under Apache-2.0.
# Source: research-environments b41d8759070715dc102000e06fba0d73ebce9955.

ENV = {
    "PATH": (
        "/root/.local/bin:/root/.cargo/bin:/go/bin:/usr/local/go/bin:/usr/local/cargo:"
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    ),
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

REMOVE_GRADING_ARTIFACTS = (
    "\n"
    "set -euo pipefail\n"
    "find /home -maxdepth 1 -type f \\( -name '*.sh' -o -name 'test.patch' \\)   ! -name "
    "'fix.patch'   ! -name 'create_fix_patch.sh'   ! -name 'extract_multiswe_fix_patch.sh'"
    "   -delete\n"
    "rm -f /tmp/multiswe_grading_artifacts.tar.gz\n"
)
