# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The open-ended bundle wire format keeps public inputs and private evidence hydrated and distinct."""

from taskcompendium.models import Source
from taskcompendium.runtime.resources import inline_resource, resource_bytes

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedTaskBrief, OpenEndedTaskBundle


def bundle(private_evidence: bytes = b'{"expected": "MAIT enrichment"}\n') -> OpenEndedTaskBundle:
    return OpenEndedTaskBundle(
        bundle_id="covid-mait-gene-usage",
        split_group="covid-immune-repertoire",
        brief=OpenEndedTaskBrief(
            goal="Assess whether receptor usage supports a transcriptomic annotation.",
            expected_deliverable="A quantitative comparison with a calibrated interpretation.",
            question_family="annotation_validation",
            avoid=("Reveal the expected enrichment in the prompt.",),
        ),
        solver_resources=(
            inline_resource(
                "workspace/data/v_gene_counts.csv",
                b"donor_id,cell_type,n_cells,n_trav1_2\ndonor_001,MAIT,53,43\n",
            ),
        ),
        author_evidence=(inline_resource("reference/expected.json", private_evidence),),
        source=Source(
            dataset="theislab/single-cell-best-practices",
            revision="dd6eb86b30385cc3c5b5ddb2d8b63945b18dc1a5",
            row="covid-mait-gene-usage",
            importer_revision="extract-covid-mait-v0",
        ),
        tags=("biology", "immune-repertoire"),
    )


def test_open_ended_bundle_round_trip_preserves_resource_visibility_and_identity():
    original = bundle()
    loaded = OpenEndedTaskBundle.model_validate_json(original.model_dump_json())

    assert loaded == original
    assert resource_bytes(loaded.solver_resources[0]).startswith(b"donor_id,")
    assert resource_bytes(loaded.author_evidence[0]).startswith(b'{"expected":')
    assert loaded.content_digest == original.content_digest
    assert bundle(b'{"expected": "no enrichment"}\n').content_digest != original.content_digest

    build_input = loaded.build_input()
    assert build_input.solver_resources == loaded.solver_resources
    assert build_input.source == loaded.source
    assert "author_evidence" not in build_input.model_dump()
