# Debugging log for H100 contract-map v13 cache diagnostics

The v13 evidence runner must retain a bounded rejection diagnostic when the
nine persistent-cache workers do not converge to one target key and serialized
executable identity.

## Initial status

Job `/dlwh/shuttle-h100-contract-map-evidence-fcbe3a3-v13` authenticated source
commit `fcbe3a3c8ab430dd0fbc6e1e7f789b1342c71387`, completed the first case's
profile, and ran all nine ordinary-XLA cache workers. The validator detected a
non-singleton semantic identity partition, but the verbose JSON rejection
exceeded its 4,096-character bound. The retained output contained only
`cache identity diagnostic exceeds its reviewed bound`.

The negative artifact is sealed at
`lib/tile_lifetime/benchmarks/artifacts/h100_contract_map_evidence_thirteenth_launch_failure_fcbe3a_v0/`.

## Compact diagnostic schema

Keep the semantic gate unchanged. Encode the fixed class and root records as
source-ordered arrays with explicit field-name arrays. Each class retains its
partition label, exact 64-hex target-key digest, and exact serialized-executable
SHA-256. Each of the nine roots retains phase, index, partition label, file
count, byte total, and final-HLO SHA-256. Case, backend, expected partition
count, and observed partition count remain top-level fields.

The maximum valid diagnostic has nine distinct classes, 128-character case and
64-character backend identifiers, maximum 63-bit counts, and private HLO text
containing control and Unicode characters. Its serialized exception is 3,107
characters. Raw HLO, paths, cache filenames, and cache contents are absent.

## Results

The pre-fix regression failed with the same bound assertion as v13. The compact
schema emits the complete nine-class diagnostic below 4,096 characters. The
production `_run_cache_protocol` boundary also emits all nine root records for
the largest reachable six-class partition after the three cold/hit byte pairs
pass unchanged.

No GPU image build, evidence launch, retry, or relaunch was performed for this
source repair.
