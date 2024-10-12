"""
Run with:
    - AustralianLegalCorpus: [Local] python operations/download/huggingface/download.py \
        --gcs_output_path="gs://marin-data/raw/law/australianlegalcorpus" \
        --hf_dataset_id="umarbutler/open-australian-legal-corpus" \
    - EDGAR: [Local] python operations/download/huggingface/download.py \
        --gcs_output_path="gs://marin-data/raw/law/edgar" \
        --hf_dataset_id="eloukas/edgar-corpus" \
    - HUPD: [Local] python operations/download/huggingface/download.py \
        --gcs_output_path="gs://marin-data/raw/law/hupd" \
        --hf_dataset_id="HUPD/hupd" \
        --hf_url_glob="data/20*.tar.gz"
    - MultiLegalPile [Local] python operations/download/huggingface/download.py \
        --gcs_output_path="gs://marin-data/raw/law/multilegalpile" \
        --hf_dataset_id="joelniklaus/MultiLegalPileWikipediaFiltered" \
        --hf_url_glob="data/en_*_train.*.jsonl.xz"
"""
