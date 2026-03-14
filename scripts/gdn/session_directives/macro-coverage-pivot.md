# Session Directive: Coverage Pivot To D2

Coverage status:
- `S3` complete
- `A3` complete and rejected
- outward `P3` block-boundary family complete and rejected
- broad `G1` branch-wrapper family complete and rejected
- `D1` complete as a partial diagnostic lead

Next required optimization slot:
- `D2`

Diagnostic allowance:
- `S3` only when attribution tooling changes
- `A2` only after a positive `D2` on the same cut
- `G2` only when the chosen lower branch-core cut is const-clean enough and a prior `D2` proved the sharding cut helps
- CE work only when fresh attribution re-implicates CE
