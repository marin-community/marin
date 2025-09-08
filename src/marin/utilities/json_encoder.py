# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import logging
from datetime import timedelta
from pathlib import Path

# Todo(Percy, dlwh): Can we remove this jax dependency?
from jax.numpy import bfloat16, float32

logger = logging.getLogger("ray")


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, timedelta):
            return {"days": obj.days, "seconds": obj.seconds, "microseconds": obj.microseconds}
        if isinstance(obj, Path):
            return str(obj)
        if obj in (float32, bfloat16):
            return str(obj)
        try:
            return super().default(obj)
        except TypeError:
            logger.warning(f"Could not serialize object of type {type(obj)}: {obj}")
            return str(obj)
