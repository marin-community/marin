"""Bind the LOCAL backend to a ready run-owned CPU machine service."""

from iris.actor.resolver import FixedResolver
from shellbox.backends.iris.machine_service import MachineServiceBinding, RemoteMachineFactory
from shellbox.machine import Backend


def local_service_factory(settings: MachineServiceBinding) -> RemoteMachineFactory:
    factory = RemoteMachineFactory(
        FixedResolver({settings.actor_name: settings.address}), settings.actor_name, rpc_timeout=settings.rpc_timeout
    )
    contract = factory.contract
    if contract.backend is not Backend.LOCAL or contract.architecture != "x86_64":
        raise ValueError("LOCAL service must advertise the native x86 CPU backend")
    if contract.region.casefold() != settings.region.casefold() or contract.images != settings.images:
        raise ValueError("LOCAL service region or prepared images differ from the run binding")
    return factory
