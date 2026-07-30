# Debugging log for Stack v3 metadata schemas

Keep Stack v3 repository metadata in Parquet without schema drift between
writer batches.

## Initial status

The CoreWeave repository transform failed after its download completed. PyArrow
inferred `github_metadata.forked_from` as `null` in one batch and `string` in a
later batch, then rejected the later table because the Parquet file schema was
already fixed. After the transform fix, normalization failed for the same
reason because its final writers inferred the schema again.

## Nullable metadata schema

The transformed documents preserve nested repository and file metadata. Several
source fields are nullable, so inference from an arbitrary first batch cannot
define a stable output schema.

## Changes to make

Pass explicit PyArrow schemas to the Stack v3 transform and normalization
writers. Keep nullable values as nulls while fixing their logical types across
every batch.

## Results

A local transform and normalization containing a null
`github_metadata.forked_from` now write that field as nullable string and read
the null value back unchanged.

## Production validation

Relaunch the GCP and CoreWeave transforms and confirm the affected shards
complete before considering the incident resolved.
