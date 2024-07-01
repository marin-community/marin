# how to download hf dataset to gcs

python scripts/copy_hf_dataset_to_gcs.py --dataset_name allenai/tulu-v2-sft-mixture --destination_path gs://marin-data/raw/instruct --urls_dir hf_dataset_transfer_bucket

# i think this is just the path to the data transfer bucket
https://storage.googleapis.com/hf_dataset_transfer_bucket/eloukas-edgar-corpus.tsv

# the above is broken the transfer job thing fails for some reason
# says i need to install some transfer_service_default agent
# gcp says i can't change permissions to public since bucket level access is uniform

# FIXED WITH PUBLIC BUCKET

# how to process html into markdown
the script from david doesn't work properly?

# below works now! now i can do local stuff
python scripts/instruct/process.py --input_dir gs://marin-data/raw/instruct/collated_conversations.jsonl.gz 

# next steps: get the ray example working with parquet!

follow up with Abhi and David.