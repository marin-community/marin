"""Build the 13-gram Bloom filter and eval_id index for the target corpus (Gate 2)."""

import sys

from marin.datakit.decon import NGramConfig, build_eval_bloom

# Same matching policy as the hero decontamination marks.
# The Bloom is sized by feature count: the targets yield ~4.6M n-grams.
ESTIMATED_FEATURES = 10_000_000
NGRAM = NGramConfig(ngram_length=13, overlap_threshold=0.5, min_matched_features=2)

if __name__ == "__main__":
    targets, out = sys.argv[1], sys.argv[2]
    print(build_eval_bloom(eval_data_sources=targets, output_path=out, ngram=NGRAM, estimated_doc_count=ESTIMATED_FEATURES, max_workers=4))
