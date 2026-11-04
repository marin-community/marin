# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build the 13-gram Bloom filter and eval_id index for a target corpus.

Usage: build_bloom.py TARGETS_DIR OUT_DIR ESTIMATED_FEATURES
"""

import sys

from marin.datakit.decon import NGramConfig, build_eval_bloom

# Same matching policy as the hero decontamination marks.
NGRAM = NGramConfig(ngram_length=13, overlap_threshold=0.5, min_matched_features=2)

if __name__ == "__main__":
    # ESTIMATED_FEATURES sizes the Bloom: about twice the targets' n-gram count
    # (10M for the core targets, 25M for targets_v5).
    targets, out, estimated_features = sys.argv[1], sys.argv[2], int(sys.argv[3])
    print(
        build_eval_bloom(
            eval_data_sources=targets,
            output_path=out,
            ngram=NGRAM,
            estimated_doc_count=estimated_features,
            max_workers=4,
        )
    )
