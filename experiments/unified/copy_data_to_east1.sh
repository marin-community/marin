#!/usr/bin/env bash
# Copy all data required by unified_pretrain_demo to gs://marin-us-east1.
# Source: gs://marin-us-central1 (known to have complete data).
#
# Usage:
#   bash experiments/unified/copy_data_to_east1.sh
#
# NOTE: The Nemotron CC training data is large (~10 TiB total). Consider running
# this script in a tmux/screen session or on a GCE VM in us-central1 to avoid
# cross-region egress charges.

set -euo pipefail

SRC="gs://marin-us-central1"
DST="gs://marin-us-east1"

# ---------- Tokenizers ----------
echo "=== Copying tokenizers ==="
gcloud storage cp -r "${SRC}/tokenizers/llama-3.1-8b" "${DST}/tokenizers/"
gcloud storage cp -r "${SRC}/tokenizers/llama3-unified-144k" "${DST}/tokenizers/"

# ---------- Nemotron CC (training data, 7 splits) ----------
NEMOTRON_SPLITS=(
    "hq_actual-5af4cc"
    "hq_synth-3525e2"
    "medium_high-d21701"
    "medium-d86506"
    "medium_low-0fdb07"
    "low_actual-cb3f2c"
    "low_synth-3c57b3"
)

echo "=== Copying Nemotron CC training data (7 splits) ==="
for split in "${NEMOTRON_SPLITS[@]}"; do
    echo "  -> nemotron_cc/${split}"
    gcloud storage cp -r "${SRC}/tokenized/nemotron_cc/${split}" "${DST}/tokenized/nemotron_cc/"
done

# ---------- Paloma validation sets (16 datasets) ----------
PALOMA_SETS=(
    "4chan-496ad5"
    "c4_100_domains-2b6db7"
    "c4_en-cf1f79"
    "dolma-v1_5-d3bed7"
    "dolma_100_programing_languages-369132"
    "dolma_100_subreddits-f25f70"
    "falcon-refinedweb-75d43b"
    "gab-ccaced"
    "m2d2_s2orc_unsplit-7dbcc1"
    "m2d2_wikipedia_unsplit-b33d23"
    "manosphere_meta_sep-a07891"
    "mc4-ea36a2"
    "ptb-628036"
    "redpajama-9d4ddd"
    "twitterAAE_HELM_fixed-2e17c1"
    "wikitext_103-1f5636"
)

echo "=== Copying Paloma validation sets (16 datasets) ==="
for ds in "${PALOMA_SETS[@]}"; do
    echo "  -> paloma/${ds}"
    gcloud storage cp -r "${SRC}/tokenized/paloma/${ds}" "${DST}/tokenized/paloma/"
done

# ---------- Uncheatable eval validation sets (7 datasets) ----------
UNCHEATABLE_SETS=(
    "ao3_english-55e735"
    "arxiv_computer_science-9760a5"
    "arxiv_physics-713363"
    "bbc_news-2ff739"
    "github_cpp-d0da6b"
    "github_python-00e7de"
    "wikipedia_english-ba27aa"
)

echo "=== Copying Uncheatable eval validation sets (7 datasets) ==="
for ds in "${UNCHEATABLE_SETS[@]}"; do
    echo "  -> uncheatable_eval/${ds}"
    gcloud storage cp -r "${SRC}/tokenized/uncheatable_eval/${ds}" "${DST}/tokenized/uncheatable_eval/"
done

# ---------- Raw data dependencies (for executor status tracking) ----------
echo "=== Copying raw data references ==="
gcloud storage cp -r "${SRC}/raw/paloma-fc6827" "${DST}/raw/"
gcloud storage cp -r "${SRC}/raw/uncheatable-eval" "${DST}/raw/"

echo ""
echo "=== Done! Verify with: ==="
echo "  gcloud storage ls ${DST}/tokenizers/"
echo "  gcloud storage ls ${DST}/tokenized/nemotron_cc/"
echo "  gcloud storage ls ${DST}/tokenized/paloma/"
echo "  gcloud storage ls ${DST}/tokenized/uncheatable_eval/"
