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

from levanter.inference.engine import InferenceEngineConfig


def f():
    def g(config):
        x = InferenceEngineConfig(
            max_seq_len=1024,
            max_seqs=1,
            max_seqs_in_prefill=1,
            max_prefill_size=1,
            max_queued_tokens=1,
            max_stop_seqs=1,
            max_stop_tokens=1,
        )
        return x

    return g
