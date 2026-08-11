# H100 v12 persistent-cache contract

## Failure

The exact v12 job `/dlwh/shuttle-h100-contract-map-evidence-41cd851-v12`
failed on the first ordinary-XLA cache protocol before later cases or an
accepted bundle. Nine cache records formed six whole-root byte classes. Each
record contained ten files. Compile records 1 and 2 had the same final-HLO
SHA-256 but different root hashes and a three-byte total-size difference.
Each cold/hit pair was byte-identical.

The sealed negative artifact is
`h100_contract_map_evidence_twelfth_launch_failure_41cd85_v0`. It contains the
complete 17-line task log, controller state, submission identity, source
capsule identity, and the bounded nine-record diagnostic. It does not contain
cache paths, file names, cache bytes, or final HLO text.

## Pinned JAX 0.10.1 audit

The audited Python sources and SHA-256 identities are:

- `jax/_src/cache_key.py`: `8638227e42a906e4d1faefb8fc9423c9ffda65a7aea14e72a3715ad6e46616fb`
- `jax/_src/compilation_cache.py`: `be022f1bb563eb9b9abe352ac30482de6f8dc3384b38f5ac1c3fa72dce6d6410`
- `jax/_src/lru_cache.py`: `1a310ad6ff5884c7dc4aabc4106c4395ec91c1b7f5506de9819c0ccd65cf5825`

`cache_key.py:75-149` hashes canonical computation bytecode, JAX library and
backend versions, filtered XLA flags, serialized compile options, accelerator
topology, compression, and the custom hook. It returns the module name plus a
64-hex SHA-256 digest. `lru_cache.py:35,96,141` appends `-cache`.

The default auxiliary XLA cache is root-dependent. `compiler.py:264-281`
places `<JAX_COMPILATION_CACHE_DIR>/xla_gpu_per_fusion_autotune_cache_dir` in
compile options. `cache_key.py:280-334` clears the autotune mode but not that
directory before hashing compile options. Fresh isolated roots therefore
cannot share a key by construction. The evidence runner now pins
`JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none` rather than normalizing the path.

`compilation_cache.py:328-355,427-456` stores
`compress(four-byte compile time || serialized executable)`.
`compiler.py:788-829` converts measured compile seconds to `int` before the
write. JAX's own verification path also excludes this compile-time prefix.
The H100 evidence image's frozen 23-package Python runtime has no `zstandard`
distribution, so pinned JAX selects zlib. The worker rejects another Python
version, a present `zstandard` distribution, malformed or trailing zlib data,
or an inflated payload beyond the reviewed bound.

With `jax_compilation_cache_max_size=-1`, `lru_cache.py:69-82,96-119,141-161`
creates no lock or `-atime` files and does not mutate entry bytes on reads.
The runner pins this setting and rejects nested directories, metadata files,
links, non-regular files, unknown names, missing or multiple `jit_step`
targets, and file-count or byte bounds. Root equality means only the flat path
and byte content; filesystem inode access times are not evidence.

`compiler.py:421-445` emits public cache-request and hit events.
`compilation_cache.py:357-381` emits the cache-miss event when an eligible
compiled entry is written. The listener is registered only around the single
`jax.jit(step).lower(...).compile()` call. Compile and cold workers must report
one request, zero hits, and one write. Hit workers must report one request, one
hit, and zero writes.

## Resolution boundary

Acceptance compares the exact `jit_step` cache key and the SHA-256 of the
serialized executable across all nine records. A key difference still rejects
different compiler inputs. An executable difference still rejects compiler
output nondeterminism. Only the source-proven four-byte compile-time field is
excluded. Compressed-entry hashes, cached compile times, whole-root hashes,
file counts, byte totals, and final-HLO hashes remain bounded diagnostics.
Each cold/hit pair must also preserve its full path-and-byte root hash.

V12 did not retain target keys, executable bytes, or raw HLO text. Its HLO-hash
differences therefore cannot be classified as semantic or serialization-only.
No HLO normalization was added. The existing later exact equality gate across
cache, timing, and profile worker HLO remains unchanged.

No GPU job, image build, retry, or relaunch was performed for this source
repair.
