# EndlessTerminals pristine runtimes

The separate kept v2 and v3 populations contain 1,958 and 121 tasks at
`FWeindel/validated-tasks@317659ac88c8b706d318020c5fc3813a931fb6ec`.
The files `endless_v2/tasks.parquet` and `endless_v3/tasks.parquet` contain the
upstream's selected tasks. Their native Parquet metadata binds all 2,079 tasks to
1,777 original Apptainer SIF bundles in the same repository at revision
`0e3d341e05545d2e997602cfa95f4f7d46d87898`.
The files total 218,815,377,408 bytes. The image index was checked against every
upstream file's size and SHA-256; its task maps retain the original content hashes.

Run the `images` module on an Iris CPU worker in `cw-us-east-02a`. It uses the
maintained [SIF conversion functions](../../../../../lib/shellbox/src/shellbox/sif.py). Each artifact
preserves the original SIF, its OCI layout, runtime metadata and a complete
filesystem equality proof. It never rebuilds the source Dockerfiles or pushes
a registry image. All bulk input and output stay in-region.

The frozen index is
`s3://marin-us-east-02a/marin/rl-data/atlas/ingest-a/endless/2026.11.19.1/inputs/verified-sif-image-index-v1.json`
(SHA-256 `81424898d51e7db35d16c93aaecef5e2a44ecb0287933fe034434f2fce5394f6`).
Its `files` map records upstream readbacks; keys in `bundles` are the `BUNDLE-ID`
values. The adjacent `folder-lineage-receipts-v1.json` provides the typed license
receipts (SHA-256 `665c6a3cda15181e80d7cde5874ab011f0077f15aedbe0d776b306cb34428020`).
The regional launcher records the tool manifest before graph execution.

```bash
python -m experiments.post_training.task_curation.datasets.endless.images \
  --index INDEX-URI --index-sha256 INDEX-SHA \
  --licenses LICENSES-URI --licenses-sha256 LICENSES-SHA \
  --tools TOOLS-URI --tools-sha256 TOOLS-SHA \
  --workdir /regional/endless-work \
  --artifact-root s3://marin-us-east-02a/marin/rl-data/atlas/endless-images \
  --artifact-version VERSION --max-concurrent 8 --output MANIFEST-URI
```

Without `--run`, the entrypoint prints artifact paths and fingerprints. Add
`--run` to execute, or repeat `--bundle BUNDLE-ID` to select images. The tool
manifest maps each tool name to `{"path": "/absolute/executable", "binary_sha256": "SHA256"}`
for `siftool`, `unsquashfs`,
`bwrap`, `tar` and `umoci`. Each step checks its tool binaries and the typed
per-folder license receipts before downloading an image. The recorded user MIT
exception for the exact upstream Endless pin remains part of those receipts;
the raw automated license checks remain unchanged. The receipt decisions name
`obiwan96/endless-terminals@26ecf78458e7f756e4d06780d5fbf3dd78e91815`,
quote its body declaration `license: mit`, and cite the recorded authorization.

The filesystem comparison includes paths, contents, modes, owners, timestamps,
symbolic and hard links, devices and extended attributes. Preservation does not
establish executable qualification or Atlas admission. Publish images through
the maintained [registry publisher](../facet/registry.py) into a source-specific
internal package, then bind its
verified digests to the original task, oracle, verifier and runtime declarations.
The [curation pipeline](../../../../../docs/references/task-curation.md) defines
original-oracle and NoP controls, final-pin SAMPLE and admission.
Size Daytona work from the account-owner file
`/Users/benfeuer/Documents/experiments/active/DAYTONA-QUOTAS.md` and a fresh,
complete account inventory covering every lane's sandboxes. Retain about 20%
headroom and pace organization-wide create requests.

## Native task conversion

`endless.tasks.sources(images_reference, census_reference)` builds intake recipes
for the kept Endless v2 and v3 tasks. The arguments are SHA-bound
`ContentEvidenceReference` values for the image publication manifest and the
native task/file/runtime census report. Each pipeline run records these inputs.

The reader checks each task archive checksum and every regular file's content,
mode and nanosecond timestamp against that census. It binds the task to the
published OCI manifest digest of its preserved SIF image. Conversion keeps the
instruction and oracle bytes, the original image working directory, and the
source's resource and deadline declarations. Public `setup_files` are delivered
unchanged; the source oracle invokes its own setup where present. The converter
adds no setup command beyond the native declaration.

After the actor running as Unix root stops, the
[native reward-file grader](../../../../../lib/taskcompendium/src/taskcompendium/harbor/root_delivery.py)
recreates `/tests/test.sh` and `/tests/test_final_state.py` with their original
bytes and metadata, runs the source verifier, and reads its reward file. The
actor can modify the image's installed dependencies; the integrity contract
records that shared state. The archived `task.toml` supplies resource and network
limits for the separate native runtime binding, whose applied allocations must
be checked during controls.

This factory supplies converted inputs. Original controls, retained releases,
SAMPLE review and admission follow the
[curation pipeline](../../../../../docs/references/task-curation.md).
