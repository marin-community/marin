# Debugging log for Stack v3 empty file paths

Allow the Stack v3 repository transform to serialize source records whose file
path is empty.

## Initial status

The GCP and CoreWeave transforms failed after `PurePosixPath("").parts`
returned no path components and `_directory_block_dfs` tried to unpack a file
name from the empty tuple. CoreWeave failed on shard 7566 after three attempts;
GCP failed with the same exception. Neither run reached normalization.

## Hypothesis 1

An empty source path can remain in `file_metadata` while the serialized text
uses an explicit unknown-path label. Treating an empty path as a root-level file
also preserves the directory traversal order.

## Changes to make

Handle an empty path when computing its directory and use `(unknown path)` in
the file header while preserving the source path in metadata. Add a regression
test for the serialized document.

## Results

The regression test failed with the production `ValueError` before the fix.
After treating the path components before the final component as the directory,
the Stack v3 tests pass and the empty source path remains in `file_metadata`.
The repository transform now uses `map` because each source repository produces
exactly one document.
