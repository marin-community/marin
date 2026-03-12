# Session Context

## User Prompts

### Prompt 1

take a look at @lib/zephyr/tests/benchmark_threaded_writer.py update it to run data from @/tmp/nemotron_sample/ and write to GCS bucket in gs://marin-tmp-us-west4/ttl=1d/rav/

### Prompt 2

run it with REDACTED on small amount of data

### Prompt 3

ah ok, let's update the benchmark, the tokanization and write should be happening at the same time in specified batches, such that we see the value of freeing the tokenizer to do more work without waiting for write, does that make sense?

### Prompt 4

do we need to delete the GCS data when we rerun the test?

