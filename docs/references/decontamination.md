# Datakit evaluation references

The reference pipeline builds its lexical Bloom filter from three manifests:
the mandatory AA Intelligence Index v4.1.1 corpus, its sealed LMH corpus, and
the mandatory `atlas-eval-coverage-v2` supplement. Outputs resolve beneath the
active regional `MARIN_PREFIX`.

The supplement pins every source file by dataset commit and SHA-256 in
[`reference_catalog.json`](https://github.com/marin-community/marin/blob/main/experiments/datakit/decontam/reference_catalog.json).
It contains all 210 tasks in the pinned Micro210 release, the SWE-bench full,
Verified and Lite **test** splits, and the 120 long-loop validation holdouts.
Micro210 has 21 benchmark labels; excluding tau3 gives 200 tasks across 20 labels.
The catalog also records a replacement source and the previous failure cause
for each of the 121 failed LMH extractions. These replacements read pinned
parquet files or upstream archives directly, without initializing scoring metrics
or executing dataset scripts.

TaskSpec references include the supplied system and user messages. They exclude
grader code, environment setup and solutions. Other sources select explicit
problem, question, choice and answer fields. Image bytes are not indexed.
The Micro references protect the selected tasks at that release's exact pin;
they do not claim full upstream benchmark coverage.

## Build the supplement

Submit a CPU job in the output bucket's region. For CoreWeave US-EAST-02A:

```bash
uv run iris --cluster=marin job run --no-wait \
  --target-cluster=cw-us-east-02a --priority batch --timeout 259200 \
  --cpu 8 --memory 32GB --disk 32GB --enable-extra-resources \
  --extra cpu --extra dedup --sync-package marin-core \
  -e MARIN_PREFIX s3://marin-us-east-02a/marin \
  -- python -m experiments.datakit.decontam.pinned_references
```

The builder writes each family's parquet and publishes `_manifest.json` last.
It fails on a checksum mismatch, missing text fields, or a family with zero
matchable records. The manifest records source counts, indexed counts, output
SHA-256 values and the identifiers omitted because they cannot match an exact
copy under the existing 13-gram, 0.5-overlap, minimum-two-feature policy.
These omissions are coverage limits, not successful detections.

A sealed release cannot be overwritten. Change the coverage version and source
pins for a new release. Preserve the existing AA and LMH release.

## Build the combined Bloom

After the supplement is complete, run the existing reference pipeline with
`--target decon-bloom`. For example, use the same region and CPU-only job
settings above with this command:

```bash
python -m experiments.datakit.reference_pipeline --mode full \
  --target decon-bloom --sources all --pool-workers 8 \
  --pool-cpu 4 --pool-ram 16g --pool-disk 32g
```

The supplement's version and required family names enter the Bloom artifact
identity, including a digest of the source catalog. Its manifest is mandatory: a missing family, wrong count, changed
artifact SHA-256, or an indexed record that cannot match itself stops the build.
The resulting Bloom uses a new hashed output directory. Existing corpus marks
still refer to their original Bloom; rebuilding references alone does not
recompute those marks.
