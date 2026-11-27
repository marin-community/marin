# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Every task_curation population publishes through the licensed publication chokepoint.

A publisher that calls the Hugging Face Hub write API directly (``create_repo``,
``upload_folder``, ``upload_file``, ``create_commit``, ...) skips the chokepoint's
carry-forward: the licensed card, the byte-frozen readback, ``check_publication`` at
the new pin, and the receipt registration. A re-seal through such a module loses its
licence evidence, so adding one fails CI. Non-population artifacts — images and
reference/registry documents — stay on the explicit allowlist below.
"""

import re
from pathlib import Path

HF_WRITE_CALLS = re.compile(
    r"\b(?:create_repo|upload_folder|upload_file|create_commit|delete_file|delete_folder"
    r"|preupload_lfs|move_repo|CommitOperationAdd)\b"
)

ALLOWED_WRITERS = frozenset(
    {
        # The licensed publication chokepoint and its own upload path.
        "license_publication.py",
        # The licensed raw provenance archive: helper-built card, nested raw/ layout.
        "raw_source_publish.py",
        # The legacy manual rebind for pre-chokepoint releases (no parent receipt exists
        # for it by definition, which is why it runs by hand).
        "release_license_rebind.py",
        # Reference and registry documents, not task populations.
        "reference_data.py",
        "registry.py",
        "derived_contract_census.py",
        "datasets/mimo/binding.py",
        # Actor image registry maps.
        "image_registry.py",
    }
)


def test_task_curation_populations_publish_through_the_licensed_chokepoint():
    root = Path(__file__).parents[1]
    offenders = []
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        if relative.startswith("tests/"):
            continue
        if HF_WRITE_CALLS.search(path.read_text()) and relative not in ALLOWED_WRITERS:
            offenders.append(relative)
    assert not offenders, (
        "These task_curation modules write to Hugging Face directly; route them through "
        "license_publication.publish_licensed_population (or, for non-population artifacts, "
        f"add them to the allowlist with a justification): {offenders}"
    )
