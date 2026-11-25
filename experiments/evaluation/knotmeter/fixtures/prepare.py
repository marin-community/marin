# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""External preparation fixture for cache publication and expiry tests."""

import json
import os
import sys
from pathlib import Path

args = sys.argv[1:]
output = Path(args[args.index("--output") + 1])
series = json.loads(Path(args[args.index("--series") + 1]).read_text())
events = Path(os.environ["AGENTX_FIXTURE_EVENTS"])
if "--check" in args:
    assert (output / "native" / "dataset.dat").is_file()
    event = "check"
else:
    assert not os.environ.get("HF_HUB_OFFLINE")
    (output / "native").mkdir()
    (output / "native" / "dataset.dat").write_bytes(str(series["random_seed"]).encode() * 100)
    (output / "lineage.json").write_text(json.dumps({"fixture": True, "seed": series["random_seed"]}))
    event = "build"
with events.open("a") as stream:
    stream.write(event + "\n")
