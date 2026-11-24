# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
import hashlib
import json
from pathlib import Path
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.buckets import filesystem_for

configure_coreweave_s3()
source = Path("/cache/benfeuer-facet-repin-20261124/census/census-measurement.json")
data = source.read_bytes()
root = "s3://marin-us-east-02a/marin/rl-data/atlas/facet-quiesce-repair/20261124"
fs, key = filesystem_for(root)
target = fs.open(f"{key}/census-measurement.json", "wb", auto_mkdir=True)
target.write(data)
target.close()
print("uploaded census-measurement.json sha256", hashlib.sha256(data).hexdigest())
print(json.dumps(json.loads(data)["population"], indent=1)[:600])
