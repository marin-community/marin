# H100 v14 cache snapshot contract

## Failure boundary

The fourteenth evidence job used source `fbe098a4e342cfa6c8795459c01d5399fbd7524b`
and the previously verified immutable H100 evidence image. It completed source
authentication, generated-candidate compilation, all first-case numerical
gates, and the Nsight Systems timing trace. The first-case ordinary-XLA cache
protocol then rejected six serialized-executable equality classes. All nine
records had the same public target key. Each cold/hit pair had an identical
full cache-root identity, but the three clean compile roots and three paired
roots carried six different serialized-executable hashes. The task stopped
before Nsight Compute, later backends/cases, or accepted bundle publication and
had no retry.

The exact negative artifact is retained separately under
`benchmarks/artifacts/h100_contract_map_evidence_fourteenth_launch_failure_fbe098_v0`.

## Pinned-source conclusion

JAX 0.10.1 public monitoring reports cache request, hit, and miss events without
the cache key as metadata. The persistent-cache key is a public flat filename,
while the entry contains a four-byte compile-time prefix followed by the
serialized executable. The v14 evidence proves that a stable key is not a
promise of fresh serialized-executable determinism. It also proves that final
HLO equality cannot substitute for executable-byte identity: two compile roots
had the same HLO hash and different executable hashes.

## Revised execution contract

The three clean-root processes remain fresh compile-time and first-execution
samples. Their target keys must agree; their executable and HLO hashes are
retained individually. Compile sample zero supplies the canonical populated
root for each backend. The case-level snapshot keeps the full ordinary-XLA
canonical root, including setup/helper entries, and overlays only the two other
validated backend target entries. Duplicate backend keys or different bytes at
one target name reject the run.

Each cache-retrieval, timing, and profile worker receives a fresh byte copy of
the sealed snapshot. Before process launch and after process exit, the
coordinator requires exact file count, total bytes, and path-and-byte root
identity. Inside the worker, each backend compilation is wrapped in a scoped
public listener that requires one request, one hit, and no miss; the expected
target key, compressed-entry hash, serialized-executable hash, and canonical
HLO hash come from the closed cache contract. This is a closed inference from
public events and filesystem evidence; the monitoring event itself does not
identify a key.

No GPU job or image build is authorized by this source checkpoint.
