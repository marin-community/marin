# Round-3 propagation run report (R1+R2+R3 cumulative)

Each judge regenerated rubrics against a spec forked with R1+R2+R3 edits applied.
Note: glm51 skipped (no R3 review; R1+R2 hit max_tokens issue).
**FIX in this run**: edit traceability stored in `_origin` metadata field
instead of bracketed description prefix (per gpt51 R3 finding of path leakage).

## Results

| judge | r1+r2+r3 edits | rows | schema_ok | elapsed |
|---|---:|---:|---|---:|
| flash | 21 | 22 | 22/22 | 40.3s |
| gpt51 | 20 | 22 | 22/22 | 123.6s |
| pro | 19 | 22 | 22/22 | 213.1s |
