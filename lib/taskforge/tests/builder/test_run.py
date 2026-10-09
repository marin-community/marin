# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from shellbox.machine import Backend
from taskcompendium.models import VerifyitGrader

from taskforge.builder.run import DRAFT_DIR, load_draft, run_build
from taskforge.builder.sdk import BuildFailure
from taskforge.builder.step import CacheStatus
from taskforge.builder.template import standard
from taskforge.loop.program import template_program


async def test_the_standard_template_builds_a_shellsim_draft_and_a_rebuild_reuses_every_step(
    tmp_path, proposal, build_services, template_client
):
    program = template_program(standard)
    draft = await run_build(program, proposal, tmp_path / "item", tmp_path / "cache", build_services)

    assert isinstance(draft.task.grader, VerifyitGrader)
    assert draft.lowered.runtime.task_machine is not None
    assert draft.lowered.runtime.task_machine.backend == Backend.SHELLSIM.value
    assert [resource.path for resource in draft.task.resources.worker] == ["workspace/numbers.txt"]
    assert {control.id for control in draft.controls} == {"reference", "sum", "first", "empty"}
    assert load_draft(tmp_path / "item" / DRAFT_DIR) == draft
    calls = list(template_client.calls)

    again = await run_build(program, proposal, tmp_path / "item", tmp_path / "cache", build_services)

    assert again.task == draft.task
    assert {record.status for record in again.provenance.steps} == {CacheStatus.HIT}
    assert template_client.calls == calls


async def test_controls_without_a_shortcut_fail_the_build_at_the_controls_step(tmp_path, proposal, build_services):
    controls = build_services.client.answers["submit_controls"]["controls"]
    build_services.client.answers["submit_controls"] = {"controls": [c for c in controls if c["id"] != "first"]}

    with pytest.raises(BuildFailure) as failure:
        await run_build(template_program(standard), proposal, tmp_path / "item", tmp_path / "cache", build_services)

    assert failure.value.step == "controls"
