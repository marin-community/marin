# Debugging log for Stack v3 normalization OOMs

Keep Stack v3 normalization workers within their memory limit.

## Initial status

CoreWeave normalization reached 6,999 of 15,999 reduce shards before 11
workers exited with code 137. The failures occurred during external-sort pass
2 with 32 GiB worker limits. Iris killed the remaining workers after the job
exceeded its failure budget.

## Worker memory

The repository-level records are substantially larger than ordinary normalized
documents. External-sort pass 2, decoded records, and the Parquet writer share
the worker memory limit.

## Changes to make

Request 64 GiB for Stack v3 normalization workers.

## Results

Pending production validation on CoreWeave.
