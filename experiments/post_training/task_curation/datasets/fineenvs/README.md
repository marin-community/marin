# FineEnvs public-input images

The native FineEnvs tasks stage packaged data from `/setup_files/inputs` into
`/home/user/input` with their original `stage_inputs.py` checksum checks. Some
input sets exceed the converter’s inline-file byte budget. The image builder stores
those files in a derived image based on the digest-pinned native image.

Each input set lives at `/opt/fineenvs-inputs/<input-signature>`. The signature
covers every input file's path, size, content hash, mode and modification time.
The builder verifies the native input archive and adds only public input files.
It compares the filesystem before OCI export with the filesystem recovered from
the OCI layout. Base file contents and metadata are preserved, except that
adding inputs can update their parent directories' modification times. Original
image layers and runtime configuration are preserved.

Daytona counts each combination of image and resource allocation as a separate
snapshot identity. The approved FineEnvs plan uses nine images built from grouped
input sets and ten snapshot identities; one image serves both native profiles
listed below. Keep each task's original allocation unchanged when grouping inputs.
Grouping exposes other tasks' public input sets outside the workdir to the root user.
The image provenance records this deviation. Conversion must bind only the
task's own set at `/setup_files/inputs` before running the unchanged staging
script. Original prompts, graders and resource allocations remain unchanged,
including any allocation that is too small for its inputs.

Run `python -m experiments.post_training.task_curation.datasets.fineenvs.images
--help` for the artifact graph's pinned input arguments. A dry run prints the
artifact paths and fingerprints. Building requires a regional Iris CPU job,
verified tool hashes and passing Daytona CPU, memory, disk, network ALLOW, file
metadata and setup-command probes for both native profiles: 1 CPU / 1024 MiB memory / 5120 MiB disk / network ALLOW, and
2 CPU / 4096 MiB memory / 10240 MiB disk / network ALLOW.

Each artifact contains the OCI layout, complete filesystem inventory and
checksum-bound provenance, including the source's typed restriction tier and
checked license-receipt binding. Publish images to an internal or private registry
package and verify their digests before conversion. After publication, convert the tasks, run original-oracle and no-op trials,
complete SAMPLE review, and obtain Atlas admission.

The converter's `sources(images_reference)` requires the immutable image-publication
URI and its SHA-256 digest. It reads one native row at a time, checks the complete archive and
every file against the sealed native census, and streams the large input files
through checksum validation. Those files remain in the image identified by that publication; every other
native file retains its bytes, mode and modification time in the TaskSpec.

Public `/setup_files/setup.sh` executes before the actor starts, after linking the task's
own input directory. Startup, file delivery and public setup share the native
build timeout. After the actor stops, the original private tests are restored for
grading. `/tests/setup.sh` executes immediately before `/tests/test.sh`, with both
sharing the native verifier timeout. If private setup fails, checks are skipped.
A zero that setup explicitly writes to the native reward file is retained;
other setup failures abort grading. Verifier environment variables stay private
until grading.

This order is verified against the [original harness at
`c313b7b18f`](https://github.com/MLI-lab/OT-next-data/tree/c313b7b18f5245619a13da12e6943d703ea5b0fb).
The runtime verifier patch, verifier review and public task setup match their
recorded hashes in frozen contract
`e3179104ebff6933a2054c9ccdb692cfdc4d7626e186bbfaf06ee21881e0ad29`:
the runtime verifier patch `97be873552fe710949ad3fe1825a9d083a78523bc69d1b1dd036604a7271baf9`,
the verifier review `051a80e170b933230ad4fd0e1459f44bd9901e2b399e388a9959c2fc73358517`,
and public task setup `8de3f30f61f0167785b66f68a642854ae68ee7667450d2c2a99cdfdf0509338c`.
Converted tasks record the URI and SHA-256 digest of the protocol manifest, which
binds this contract and the historical harness files.
