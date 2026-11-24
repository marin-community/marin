# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from pathlib import Path

from cloud.iris.launch_config import load_launch_config

rows = []
for path in sys.argv[1:]:
    config = load_launch_config(Path(path))
    placement = config.skyrl.trainer.placement
    allocation = config.iris.allocation
    rows.append(
        {
            "profile": config.runtime.profile,
            "strategy": config.skyrl.trainer.strategy,
            "allocation": {
                "policy_gpus": placement.policy_num_nodes * placement.policy_num_gpus_per_node,
                "reference_gpus": placement.ref_num_nodes * placement.ref_num_gpus_per_node,
                "allocated_gpus": allocation.num_nodes * allocation.gpus_per_node,
                "colocate_policy_ref": placement.colocate_policy_ref,
            },
        }
    )
print(json.dumps(rows))
