# Debugging log for Stack v3 metadata schemas

Keep Stack v3 repository metadata in Parquet without schema drift between
writer batches.

## Initial status

The CoreWeave repository transform failed after its download completed. PyArrow
inferred `github_metadata.forked_from` as `null` in one batch and `string` in a
later batch, then rejected the later table because the Parquet file schema was
already fixed.

## Nullable metadata schema

The transformed documents preserve nested repository and file metadata. Several
source fields are nullable, so inference from an arbitrary first batch cannot
define a stable output schema.

## Changes to make

Pass an explicit PyArrow schema to the Stack v3 Parquet writer. Keep nullable
values as nulls while fixing their logical types across every batch.

## Results

A local transform containing a null `github_metadata.forked_from` now writes
that field as nullable string and reads the null value back unchanged.

## Future work

- [ ] Confirm both production transforms pass the affected shards.
