# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backport upstream JAXPP send-lifetime event polling, without changing its pin."""

import importlib.util
from pathlib import Path

UPSTREAM_COMMIT = "78458f8b258f0fd467be13b4b8e7056a853e320b"
# https://github.com/NVIDIA/jaxpp/blob/78458f8b258f0fd467be13b4b8e7056a853e320b/src/jaxpp/dime2.py
EVENT_POLLING = """class PendingSend(NamedTuple):
    event: Event
    capsules: list[Any]


pending_sends: list[PendingSend] = []


# Keep send capsules alive until their post-send CUDA events complete.
# Poll and release on the execution thread, serialized with its PJRT calls:
# capsule destruction releases a PJRT external reference, and doing that from
# a CUDA callback thread previously caused crashes consistent with a PJRT race.
#
# Even a callback that only queues capsules can deadlock: entering Python
# requires the GIL, while DLPack import can hold the GIL during lazy creation
# of a PJRT callback stream. On the tested driver, cuStreamCreate waited for
# the outstanding host callback to return. Event polling removes that cycle.
def drain_completed_send_capsules() -> None:
    remaining = []
    for pending in pending_sends:
        with cuda_device(pending.event.device):
            if pending.event.is_done:
                pending.capsules.clear()
            else:
                remaining.append(pending)
    pending_sends[:] = remaining


"""


def patch_source(source: str) -> str:
    if EVENT_POLLING in source:
        assert "cudaLaunchHostFunc" not in source
        assert "pending_sends.append(PendingSend(stream.record(), capsules))" in source
        return source
    start = source.index("pending_send_callbacks: dict")
    end = source.index("@dataclass(slots=True)\nclass Transfer:", start)
    source = source[:start] + EVENT_POLLING + source[end:]
    for line in (
        "import ctypes\n",
        "import itertools\n",
        "import threading\n",
        "from cuda.bindings import runtime as cuda_runtime\n",
        "completed_send_capsules_lock = threading.Lock()\n",
        "completed_send_capsules: list[list[Any]] = []\n",
    ):
        assert source.count(line) == 1, line
        source = source.replace(line, "")
    old = "launch_send_capsules_callback(stream, capsules)"
    assert source.count(old) == 1
    source = source.replace(old, "pending_sends.append(PendingSend(stream.record(), capsules))")
    old_comment = """    # NOTE: communicators are blocking, so after group end all sends/recvs have
    # been enqueued onto their streams. We can therefore release send capsules
    # after a stream callback marks them complete, and record recv completion
    # events on the streams.
"""
    new_comment = """    # NOTE: communicators are blocking, so after group end all sends/recvs have
    # been enqueued onto their streams, so completion events recorded here cover
    # all sends/recvs in the group.
"""
    assert source.count(old_comment) == 1
    return source.replace(old_comment, new_comment)


def main() -> None:
    spec = importlib.util.find_spec("jaxpp")
    path = Path(next(iter(spec.submodule_search_locations))) / "dime2.py"
    path.write_text(patch_source(path.read_text()))
    print(f"Patched JAXPP send lifetime with upstream {UPSTREAM_COMMIT}: {path}")


if __name__ == "__main__":
    main()
