# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Spawn-safe census doubles: a source registry importable from worker processes.

The census fan-out runs :func:`measure_source` in spawned child processes, so an
injected ``sources_fn`` must pickle by qualified name. This module holds that
registry; the declarations travel through the environment, which spawned children
inherit from the test process.
"""

import json
import os
from types import SimpleNamespace

from finestore.rl.curation import QualifiedSourceDeclaration

DECLARATIONS_ENV = "CENSUS_SPAWN_TEST_DECLARATIONS"


def spawned_registry() -> dict:
    """Build the source registry from the test process's exported declarations."""
    declarations = json.loads(os.environ[DECLARATIONS_ENV])
    return {
        source_id: SimpleNamespace(
            info=SimpleNamespace(id=source_id, oracle_filtration=None),
            declaration=QualifiedSourceDeclaration.model_validate(declaration),
        )
        for source_id, declaration in declarations.items()
    }
