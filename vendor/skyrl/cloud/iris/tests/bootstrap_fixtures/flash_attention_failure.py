import os

with open(os.environ["FIXTURE_IMPORT_BARRIER"], "wb", buffering=0) as gate:
    gate.write(b"r")
raise RuntimeError("flash-attention import failed")
