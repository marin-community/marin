# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check a task's declared integrity contract before any machine is acquired.

Delivery contracts (``root-delivery``, ``multi-swe-delivery``) are enforced by the
task's own grader: trusted root ``collect`` commands quiesce the actor machine,
deliver the private verifier material and run the source's verifier verbatim, and
copied artifacts carry the reward to the separate grading machine. The consumer
verifies that the task actually carries that delivery path and that the private
material is absent from every actor-visible resource. The SHARED enforcement
classes need protected-root runtime support this consumer does not have, so
those tasks fail closed.
"""

from collections.abc import Sequence
from hashlib import sha256

from taskcompendium.integrity import MultiSweDeliveryContract, RootDeliveryContract, SharedIntegrityContract
from taskcompendium.models import ScriptGrader, TaskResource, TaskSpec
from taskcompendium.runtime.resources import resource_bytes


def _digests(resources: Sequence[TaskResource]) -> set[tuple[str, int]]:
    return {
        (sha256(resource_bytes(resource)).hexdigest(), 0o644 if resource.mode is None else int(resource.mode, 8))
        for resource in resources
    }


def require_integrity_contract(task: TaskSpec) -> None:
    """Raise unless the consumer can honor ``task.integrity_contract`` exactly as declared."""
    contract = task.integrity_contract
    if contract is None:
        return
    if isinstance(contract, SharedIntegrityContract):
        raise NotImplementedError(
            f"Task {task.id} declares a {contract.id} v{contract.version} integrity contract; the consumer "
            "runtime does not enforce shared grading protections, so it refuses the task"
        )
    if contract.agent_user is not None:
        raise NotImplementedError(
            f"Task {task.id} demotes its actor to {contract.agent_user!r}; the consumer keeps the image user"
        )
    grader = task.grader
    if not isinstance(grader, ScriptGrader) or not grader.collect or not grader.artifacts:
        raise ValueError(
            f"Task {task.id} declares {contract.id} but its grader lacks the trusted collect and artifact delivery"
        )
    if contract.undeclared_deadline != (grader.timeout is None):
        raise ValueError(f"Task {task.id} grader deadline disagrees with its {contract.id} contract")
    actor = {digest for digest, _ in _digests(task.resources.all + task.resources.worker)}
    verifier = _digests(task.resources.verifier)
    if isinstance(contract, RootDeliveryContract):
        declared = {(file.sha256, file.mode) for file in contract.verifier_files}
        missing = declared - verifier
    else:
        assert isinstance(contract, MultiSweDeliveryContract)
        declared = {(contract.archive_sha256, mode) for digest, mode in verifier if digest == contract.archive_sha256}
        missing = set() if declared else {(contract.archive_sha256, 0)}
    if missing:
        raise ValueError(
            f"Task {task.id} {contract.id} contract pins {len(missing)} verifier file(s) its private "
            "verifier resources do not carry with the pinned content and mode"
        )
    leaked = {digest for digest, _ in declared} & actor
    if leaked:
        raise ValueError(
            f"Task {task.id} exposes {len(leaked)} contract-delivered verifier file(s) in actor-visible resources"
        )
