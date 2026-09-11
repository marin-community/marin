# Research: `Artifact.from_id` + `ArtifactRegistry`

Background research for the design doc. Persisted so the 1-pager can stay short.

## 1. `Artifact` today

`lib/marin/src/marin/execution/artifact.py` (118 lines). Static-method-only loader/saver for step outputs — no instance state.

- `Artifact.from_path(base_path: str | StepSpec, artifact_type: type[T] | None = None) -> T | PathMetadata | dict[str, Any]` (artifact.py:32–61)
  - Loads `{base_path}/artifact.json` (legacy: `.artifact`) and deserializes via Pydantic `BaseModel.model_validate_json()` when `artifact_type` is provided; otherwise returns the raw dict.
  - Type check: `if not issubclass(artifact_type, BaseModel): raise TypeError(...)` (l.56–58). So `artifact_type` MUST be a Pydantic `BaseModel`.
  - Fallback: if the sidecar is missing but a sibling `.executor_status` reads `SUCCESS`, synthesize `PathMetadata(path=base_path)` (l.64–83).
- `Artifact.save(value, base_path)` (artifact.py:86–97): writes `artifact.json` for any Pydantic `BaseModel`, dataclass, or JSON-serializable.

**What an Artifact IS in this codebase**: the JSON metadata envelope around a step's output. Concrete payloads are Pydantic models like `PathMetadata` (single path), `PathsMetadata` (sharded), `NormalizedData`, `TokenizedAttrData`, `LevanterStoreData`. The artifact is metadata + pointer(s), not the bytes.

## 2. Call sites (representative)

All call sites are **lazy deserialization at consumption time** — a downstream step opens an upstream output by its computed path and asks for a typed view.

- `lib/marin/src/marin/processing/tokenize/attributes.py:39–58` — `Artifact.from_path(step, NormalizedData)` → produces `TokenizedAttrData`, `.save()` to emit.
- `lib/marin/src/marin/processing/classification/deduplication/fuzzy_minhash.py:38,125` — chained normalize → minhash → fuzzy-dups via `from_path`.
- `lib/marin/src/marin/processing/tokenize/store_builder.py:145` — collects multiple tokenized split artifacts.
- `lib/marin/src/marin/execution/step_runner.py:140,165` — `Artifact.save(result, output_path)` after a step function returns.

No call site constructs `Artifact` directly; everything goes through `from_path` / `save`.

## 3. Adjacent registry / id concepts

- **No existing artifact registry.** No module named `registry` in `lib/marin/src/marin/execution/`.
- `lib/marin/src/marin/datakit/ingestion_manifest.py:1–80` — Pydantic ingestion manifests (source policies, sampling caps, fingerprints). Governance sidecar, not a lookup table.
- `lib/marin/src/marin/execution/executor.py:30–77,848–875` — executor versioning: `VersionedValue` wrapper marks fields that participate in the step's hash. The **path itself is the version**; no id → path map.

## 4. Versioning conventions in Marin

Content-addressed path hashes:

- `lib/marin/src/marin/execution/step_spec.py:90–101` — `hash_id = sha256(json.dumps({name, hash_attrs, sorted(dep_names)}))[:8]` (l.96). Step output path: `{marin_prefix()}/{name}_{hash_id}` (l.116). Example: `gs://marin-us-central2/documents/fineweb-resiliparse-8c2f3a`.
- Schema-level versioning lives inside the artifact JSON itself — e.g. `NormalizedData.version = "v1"` (`lib/marin/src/marin/datakit/normalize.py:75`).

So Marin already has two version-like notions: the **step-hash** (8-char hex, content-addressed, machine-derived) and the **schema-version string** inside the payload (semver-ish, human-curated). The new `Artifact.from_id(id, version)` introduces a third axis — a **user-facing logical name + version** decoupled from both.

## 5. Filesystem abstraction

No unified wrapper; code switches between two layers:

- `rigging.filesystem` — `marin_prefix()` (root path), `open_url(...)` context manager for read/write across GCS + local. Imported in artifact.py:9 and used at l.49, l.53, l.88.
- `lib/marin/src/marin/utils.py` — fsspec helpers: `fsspec_exists`, `fsspec_glob`, `fsspec_mkdirs`. `url_to_fs(...)` is used elsewhere to obtain a filesystem object from a URL.

The registry should be backed by `rigging.filesystem` / `url_to_fs` so it works on both GCS and local paths with no hardcoded scheme.

## 6. The "optional model" parameter

User asked for `Artifact.from_id(<id>, <version>, <optional-model>)`. Strongly likely interpretation: **a Pydantic schema class to deserialize the payload into** — i.e. it parallels the existing `artifact_type` parameter on `from_path` (artifact.py:33). When `None`, you get the raw dict / `PathMetadata` fallback; when provided, you get a typed instance.

I'll confirm this with the user in Phase 3 — but treating "model" as "Pydantic schema" is consistent with every existing call site and with the artifact.py contract.

## 7. Prior design docs

Skimmed `.agents/projects/`. None on artifact identity / registries / GCS reorganization.

## 8. GitHub issues

`gh issue list --search "artifact registry"`, `gh issue list --search "artifact id"`:

- #5864 (OPEN) — `[marin] _discover_files treats artifact.json sidecar as JSONL data`. Tangential; confirms `artifact.json` is now a known sidecar convention.
- #5057 (OPEN) — unrelated (evals).

No prior issue dedicated to artifact-id or a registry.

---

## Prior art (web pass)

Quick read of well-known artifact registries to anchor reviewer expectations.

**MLflow Model Registry.** Flat string `name` + monotonically-increasing integer per name. Mutable aliases (`@champion`) and stage tags layered on top. Stores metadata + URI pointer; bytes live in the artifact store. Typed via "flavor" recorded in an `MLmodel` YAML. **SQL-backed** service.

**W&B Artifacts.** `entity/project/name` + `type` ("dataset", "model", ...). Version is monotonic `v0, v1, ...` per collection, **assigned by content checksum** (re-logging identical content doesn't bump). Mutable aliases. Server + content-addressed blob store.

**DVC.** ID = workspace-relative path of a `.dvc` pointer; version = **content hash (MD5)**. Registry is flat files on disk (`.dvc` + `dvc.lock`) tracked in git. Untyped blobs.

**Hugging Face Hub.** ID = `namespace/repo_name` + `repo_type`; version (`revision`) is a **git ref** (commit SHA, branch, or tag). Registry literally is git (LFS for big files).

### Convergent choices

- Universal: `(name, version)` pair; name is a string, often namespaced.
- Universal: registry stores **metadata + pointer**, not bytes.
- Universal: **mutable aliases** (`latest`, `best`, …) layered over immutable versions.
- Strong trend: content-hash-derived immutability (DVC, W&B, HF git SHA). MLflow's pure integer counter is the outlier.

### Divergent choices reviewers will probe

- **Version flavor**: monotonic int vs. content hash vs. user-supplied semver/tag.
- **Namespacing**: flat vs. `user/repo` vs. `entity/project/name`.
- **Typed vs. blob**: a cheap `type: str` field enables tooling later.
- **Registry transport**: SQL service vs. flat files in storage. A manifest-file design is firmly in the DVC / HF-as-git camp — say so explicitly.

### Pitfalls of small in-process registries

- **Non-atomic manifest writes** → corrupt registry on crash. Tempfile + rename locally; **GCS generation-preconditioned write** (`if-generation-match`) for object stores.
- **Concurrent writers** → lost updates. Either object-store CAS or document a single-writer / leased-writer model.
- **Mutable versions** → silent reproducibility loss. Versions must be immutable; new content ⇒ new version.
- **Alias drift** → record the *resolved* immutable version alongside any alias in run logs.
- **Pointer-only entries** → record a content hash even if version is human-supplied (DVC and W&B both do).
- **Absolute URIs in the manifest** → non-portable across buckets/regions. Store relative keys + a separately-configured root.
- **No GC / orphan story** → decide upfront whether the registry is append-only.
- **Unbounded metadata schemas** → pin a minimal typed schema (`id, version, type, uri, content_hash, created_at, metadata: dict`) from day one.

Sources (snapshot 2026-05-20):
- MLflow Model Registry docs
- W&B Artifacts (aliases, versioning)
- DVC internal files (`.dvc`, `dvc.lock`)
- Hugging Face Hub revisions
