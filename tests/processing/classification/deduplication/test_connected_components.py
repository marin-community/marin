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

from marin.processing.classification.deduplication.connected_components import connected_components
from marin.processing.classification.deduplication.minhash_lsh import minhash_lsh
from zephyr.backends import defaultdict
from zephyr.dataset import Dataset
from zephyr.readers import load_file


def test_connected_components_happy_path(sync_backend, docs, tmp_path):
    input_data = [{"text": text, "id": doc_id} for doc_id, text in docs.items()]

    ds = Dataset.from_list(input_data)

    lsh_result = minhash_lsh(ds)

    converged, output_path = connected_components(
        lsh_result, backend=sync_backend, output_dir=tmp_path.as_posix(), max_iterations=5
    )
    assert converged
    results = sync_backend.execute(Dataset.from_list(output_path).flat_map(load_file))
    assert len(results) == len(docs)

    components = defaultdict(list)
    for r in results:
        components[r["component_id"]].append(r["node_id"]["record_id"])

    assert sorted(components.values()) == [["doc_1_diff_header", "doc_1"], ["doc_2"]]
