# Use ray remote environment
# since we only need dolma + rust crap for generating the attribute files we should use remote env
# on workers
https://docs.ray.io/en/latest/cluster/running-applications/job-submission/quickstart.html#jobs-remote-cluster


ray job submit --address http://127.0.0.1:8265 --working-dir . --no-wait -- python marin/processing/classification/dummy.py --input_dir gs://marin-us-central2/documents/marin_instructv1/      
