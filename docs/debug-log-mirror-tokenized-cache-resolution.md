# Debugging log for mirror tokenized cache resolution

Fix `mirror://` tokenized-cache paths reaching Levanter tensorstore unchanged in region-agnostic replay jobs.

## Initial status

The 300M `power_family_penalty` replay child failed with:

`ValueError: Unsupported URI scheme for tensorstore: 'mirror'`

The failing path was a tokenized cache object under:

`mirror://tokenized/merged/dolma3_dolmino_top_level/dolma3_stack_edu-a7297b/train/input_ids/offsets`

## Hypothesis 1

`mirror://` support exists only in the fsspec layer, but tokenized caches are opened by tensorstore. Region-agnostic replay correctly switched some cache roots to `mirror://`, but Levanter's tensorstore path builder still only accepts `gs://`, `s3://`, and local file paths.

## Changes to make

- Add a public mirror-resolution helper in `lib/rigging/src/rigging/filesystem.py`.
- Resolve `mirror://...` to a concrete local URL in `lib/levanter/src/levanter/tensorstore_serialization.py` before creating the tensorstore kvstore spec.
- Add focused regression tests in:
  - `lib/rigging/tests/test_mirror_fs.py`
  - `lib/levanter/tests/test_tensorstore_serialization.py`

## Future Work

- [ ] Audit other tensorstore-backed call sites that may now receive `mirror://`.
- [ ] Consider whether executor `MirroredValue` docs should be reconciled with current behavior.

## Results

Implemented a narrow fix:

- `lib/rigging/src/rigging/filesystem.py`
  - added public `resolve_mirror_url()`
  - added `MirrorFileSystem.resolve_url()`
- `lib/levanter/src/levanter/tensorstore_serialization.py`
  - `build_kvstore_spec()` now resolves `mirror://...` to a concrete local URL before selecting the tensorstore driver

Targeted regressions passed:

- `uv run --with pytest python -m pytest lib/rigging/tests/test_mirror_fs.py -k 'resolve_url_copies_from_remote or resolve_mirror_url_uses_registered_filesystem'`
- `uv run --with pytest --with pytest-xdist python -m pytest lib/levanter/tests/test_tensorstore_serialization.py -k 'build_kvstore_spec_resolves_mirror_urls'`

Interpretation:

- Region-agnostic tokenized caches can continue to use `mirror://...` at the config layer.
- Tensorstore-backed readers now see a concrete local `gs://marin-.../...` or local file path.
- The on-demand copy still goes through `MirrorFileSystem`, so transfer-budget enforcement remains centralized.
