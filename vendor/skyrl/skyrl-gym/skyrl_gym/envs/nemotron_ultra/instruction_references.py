"""Resolve source-generated instruction parameters before candidate generation."""

import json
import random
import subprocess
import sys
from importlib.metadata import distribution

from skyrl_gym.envs.nemotron_ultra.instruction_following import build_instruction


def freeze_instruction_references(record: dict, seed: str) -> dict:
    """Run registry construction in a private process with a reproducible seed."""
    result = subprocess.run(
        [sys.executable, "-m", __name__],
        input=json.dumps({"record": record, "seed": seed}, allow_nan=False),
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    return json.loads(result.stdout)


def _resolve(record, seed):
    provenance = json.loads(distribution("verifiable-instructions").read_text("direct_url.json") or "{}")
    revision = "f46a5ac87b1400a4f8973039844b6be9b56e3faf"
    if provenance.get("vcs_info", {}).get("commit_id") != revision:
        raise ValueError("Instruction preparation requires the pinned registry")
    random.seed(seed)
    identities, parameters = record.get("instruction_id_list"), record.get("kwargs")
    if (
        not isinstance(identities, list)
        or not identities
        or not isinstance(parameters, list)
        or len(identities) != len(parameters)
    ):
        raise ValueError("Instruction references must have aligned IDs and kwargs")
    resolved = []
    for identity, arguments in zip(identities, parameters):
        if not isinstance(identity, str) or not isinstance(arguments, dict):
            raise ValueError("Instruction reference is malformed")
        original = build_instruction(identity, {k: v for k, v in arguments.items() if v is not None})
        frozen = original.get_instruction_args()
        if frozen is None:
            frozen = {}
        if not isinstance(frozen, dict):
            raise ValueError("Instruction does not expose resolved arguments")
        json.dumps(frozen, allow_nan=False)
        before = random.getstate()
        restored = build_instruction(identity, frozen)
        if random.getstate() != before or vars(restored) != vars(original):
            raise ValueError("Instruction arguments do not preserve its resolved reference")
        resolved.append(frozen)
    return {**record, "kwargs": resolved, "instruction_reference_revision": revision}


if __name__ == "__main__":
    data = json.load(sys.stdin)
    print(json.dumps(_resolve(data["record"], data["seed"]), allow_nan=False))
