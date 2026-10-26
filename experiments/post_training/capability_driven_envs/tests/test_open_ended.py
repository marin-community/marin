# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""The open-ended bundle keeps solver resources and private author evidence distinct."""

from taskcompendium.models import Source
from taskcompendium.runtime.resources import inline_resource, resource_bytes
from taskforge.spec.draft import requirements

from experiments.post_training.capability_driven_envs.open_ended import OpenEndedTaskBrief, OpenEndedTaskBundle
from experiments.post_training.capability_driven_envs.open_ended_packet_schema import OpenEndedReleaseStatus


def bundle(private_evidence: bytes = b"A=80%, B=20%; descriptive only.\n") -> OpenEndedTaskBundle:
    return OpenEndedTaskBundle(
        bundle_id="synthetic-two-group-rates",
        split_group="synthetic-open-ended",
        brief=OpenEndedTaskBrief(
            goal="Compare success rates in two groups.",
            expected_deliverable="A quantitative comparison with a calibrated interpretation.",
            question_family="descriptive_comparison",
            avoid=("Infer causality from descriptive counts.",),
        ),
        solver_resources=(
            inline_resource(
                "workspace/data/counts.csv",
                b"group,success,total\nA,8,10\nB,2,10\n",
            ),
        ),
        author_evidence=(inline_resource("private/expected.txt", private_evidence),),
        source=Source(
            dataset="synthetic/open-ended",
            revision="1",
            row="two-group-rates",
            importer_revision="1",
        ),
        environment_requirements=requirements(image=None),
        release_status=OpenEndedReleaseStatus.UNREVIEWED_DO_NOT_PUBLISH,
        tags=("synthetic", "descriptive-analysis"),
    )


def test_open_ended_bundle_round_trip_preserves_resource_visibility_and_identity():
    original = bundle()
    loaded = OpenEndedTaskBundle.model_validate_json(original.model_dump_json())

    assert loaded == original
    assert resource_bytes(loaded.solver_resources[0]).startswith(b"group,")
    assert resource_bytes(loaded.author_evidence[0]).startswith(b"A=80%")
    assert loaded.content_digest == original.content_digest
    assert bundle(b"No supported difference.\n").content_digest != original.content_digest

    build_input = loaded.build_input()
    assert build_input.solver_resources == loaded.solver_resources
    assert build_input.source == loaded.source
    assert build_input.brief == loaded.brief
    assert build_input.author_evidence == loaded.author_evidence
