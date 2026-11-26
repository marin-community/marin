# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The uniform task workspace (ruling 2026-11-20, the 10-snapshot consolidation)."""

import re

# Every task uses ONE uniform workspace, the SWE-bench/SWE-smith convention. The shared-source
# image layer's workspace mkdir line is then constant, each group's build context collapses to
# one, and the declared Daytona snapshot identities collapse to the ten base images. A row's
# original per-repo path is rewritten to WORKSPACE wherever the spec wires it (the clone, the
# runtime cwd, the contract config, the install/test commands, and the oracle solve script);
# rows fail their oracle under /testbed as typed exclusions.
WORKSPACE = "/testbed"


def to_workspace(text: str, workdir: str) -> str:
    """Rewrite a row's per-repo workspace path to the uniform /testbed workspace.

    Path-token boundaries keep sibling names (/sympy2, /sympy-bench) and nested paths
    (/x/sympy) intact while rewriting exact occurrences (/sympy, /sympy/tests).
    """
    if not workdir or workdir == WORKSPACE:
        return text
    return re.sub(
        rf"(?<![0-9A-Za-z_/-]){re.escape(workdir)}(?![0-9A-Za-z_-])",
        WORKSPACE,
        text,
    )
