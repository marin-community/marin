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

"""
Run with:
    - AustralianLegalCorpus: [Local] python marin/download/huggingface/download.py \
        --gcs_output_path="gs://marin-us-central2/raw/legal-open-australian-legal-corpus/66e7085/" \
        --hf_dataset_id="umarbutler/open-australian-legal-corpus" \
    - EDGAR: [Local] python marin/download/huggingface/download.py \
        --gcs_output_path="gs://marin-us-central2/raw/legal-edgar/f7d3ba7/" \
        --hf_dataset_id="eloukas/edgar-corpus" \
    - HUPD: [Local] python marin/download/huggingface/download.py \
        --gcs_output_path="gs://marin-us-central2/raw/legal-hupd/f570a84/" \
        --hf_dataset_id="HUPD/hupd" \
        --hf_url_glob="data/20*.tar.gz"
    - MultiLegalPile [Local] python marin/download/huggingface/download.py \
        --gcs_output_path="gs://marin-us-central2/raw/legal-multi-legal-wikipedia-filtered/483f6c8/" \
        --hf_dataset_id="joelniklaus/MultiLegalPileWikipediaFiltered" \
        --hf_url_glob="data/en_*_train.*.jsonl.xz"
"""
