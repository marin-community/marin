import os
import select

fd = os.open(os.environ["FIXTURE_IMPORT_BARRIER"], os.O_RDONLY | os.O_NONBLOCK)
with open(fd, "rb", buffering=0) as gate:
    if not select.select([gate], [], [], 8)[0] or gate.read(1) != b"r":
        raise RuntimeError("concurrent import never started")
raise RuntimeError("megatron import failed")
