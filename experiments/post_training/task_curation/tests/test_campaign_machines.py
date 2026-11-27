# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Campaign routing retains the configured machine provider's identity."""

from shellbox.backends.iris.machine import IrisMachineFactory
from shellbox.context import MachineRoute
from shellbox.machine import Backend, CommandSemantics
from taskcompendium.models import EnvironmentRequirements
from taskcompendium.runtime.local import LocalGraderMachines

from experiments.post_training.task_curation.campaign_machines import CampaignMachines


class NoLocalMachines(LocalGraderMachines):
    def routes(self, _environment: EnvironmentRequirements, _memory_mb: int) -> tuple[MachineRoute, ...]:
        return ()


def test_campaign_records_the_configured_gvisor_image_backend():
    machines = CampaignMachines(
        image_factory=IrisMachineFactory(controller_url="http://controller.invalid"), local=NoLocalMachines()
    )
    requirements = EnvironmentRequirements(
        command_semantics=CommandSemantics.LINUX_PROCESS,
        docker_image="fixture@sha256:" + "0" * 64,
    )
    routes = machines.routes(requirements, 2048)
    assert [route.tier for route in routes] == [Backend.GVISOR]
