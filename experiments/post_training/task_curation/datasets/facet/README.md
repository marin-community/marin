# FACET pristine image preservation

`images` stages the 747 pinned Apptainer SIF images used by the separate FACET v2
and v3 task maps (780 and 728 kept tasks). Run it on an Iris CPU worker in
`cw-us-east-02a`. Images and extracted files stay in-region. It never pushes a
registry image or qualifies a source for execution.

The image source is `FWeindel/validated-tasks` at
`815988416b9a982932b5531eee9853d4c0e3b55a`. The frozen index is
`s3://marin-us-east-02a/marin/rl-data/atlas/ingest-b/fweindel-facet/2026.11.17.1/inputs/verified-sif-image-index-v1.json`
(SHA-256 `91a1825aa7f94b0e56f590aaa1f8e293d1fed736966cbbd051547e6efdd9e6d1`).
The adjacent `lineage-receipts-v2.json` contains the checked per-folder licenses.

The input index records each upstream image path, SHA-256, size and task binding.
Its `tasks` map contains separate `facet_v2` and `facet_v3` maps, while `bundles`
and `files` bind every referenced image. The producer checks the counts and
equality of those image sets. A bundle ID is a key in `bundles`.
Supply the index's immutable URI and SHA, the checked
per-folder license receipts and a tool manifest whose entries contain `path`
and `binary_sha256` for `siftool`, `unsquashfs`, `bwrap`, `tar` and `umoci`.
The producer verifies these bindings before downloading each SIF.

```bash
python -m experiments.post_training.task_curation.datasets.facet.images \
  --index INDEX-URI --index-sha256 INDEX-SHA256 \
  --licenses LICENSES-URI --licenses-sha256 LICENSES-SHA256 \
  --tools TOOLS-URI --tools-sha256 TOOLS-SHA256 \
  --workdir /regional/facet-work \
  --artifact-root s3://marin-us-east-02a/marin/rl-data/atlas/facet-images \
  --artifact-version VERSION --max-concurrent 8 --output MANIFEST-URI
```

Without `--run`, this prints every artifact path and fingerprint. Add `--run` to
execute, or repeat `--bundle BUNDLE-ID` to select a subset. Each independently
sealed artifact retains the original SIF, OCI layout, resolved runtime metadata
and a full filesystem equality proof. `preservation.json` binds every object's
URI, size and SHA. The filesystem comparison includes paths, kinds, modes,
owners, timestamps, contents, links, devices and extended attributes.
`MANIFEST-URI` receives the list of `PreservedImage` results for the selected
bundles; each result points to its own `preservation.json` and SHA.

A failed conversion retains its partial regional outputs. Use a new artifact
version for that retry. No failed image falls back to a Dockerfile rebuild.
Registry publication is a separate step: require verified internal or private visibility,
preserve the OCI manifest digest, then bind that preserved digest into each
native task package before original-oracle and no-operation controls. Follow
the [environment repair procedure](../../../../../docs/references/task-curation.md#environment-fidelity-and-repair)
for that publication and qualification.

`registry.publish_image` checks package visibility before transferring an image
and again afterward, then verifies the pushed manifest against the preserved
digest. It refuses public packages. Record the actual internal or private
visibility in the publication receipt. Use one GHCR package for FACET; do not
share it with other sources.
