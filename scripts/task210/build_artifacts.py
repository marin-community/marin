"""Task #210: stage, describe and (optionally) upload the two artifact dataset repos.

Usage: build_artifacts.py <cruxeval|two-expert> [--upload]
Runs on cw-us-east-02a. Copies each listed S3 object to /tmp/artifacts/<repo>/<path relative to marin/>, checks size, computes
sha256, packs each folder of more than 1,000 files into one tar (the Hub allows 10,000 files per folder) and re-hashes its
members, writes MANIFEST.tsv (path, bytes, sha256, S3 origin, issue, stored_in, purpose) and README.md. Without --upload it
prints README.md and the repo file count per folder, and uploads nothing. With --upload (needs HF_TOKEN in the env, never printed) it creates the
public dataset repo, uploads only the repo files that are missing or different (README.md and MANIFEST.tsv always), removes
stale ones, in commits of 1,000 operations (the Hub allows 128 commits per hour), then checks every file's size and sha256 (LFS) or git blob id (small files) on the Hub.
"""
import collections, concurrent.futures as cf, hashlib, os, pathlib, posixpath, sys, tarfile
import fsspec

B = "marin-us-east-02a/marin"
fs = fsspec.filesystem("s3")
I20 = "https://github.com/marin-community/marin/issues/9920"
I84 = "https://github.com/marin-community/marin/issues/9984"
T, K = "teacher-traces", "users/karan"
SETS = {
    "cruxeval": ("open-athena/Snowball-67B-A2B-CruxEval-Transfer-Artifacts-2026.10", 9920, I20,
                 "The GLM teacher responses come from [GLM 5.3](https://huggingface.co/zai-org/GLM-5.3). ",
                 "CruxEval transfer experiment: plain SFT, rejection-sampled SFT and on-policy distillation (OPD) from expert E into student B", [
        (f"{T}/cruxeval-test-exposed/task133-20261007-glm", "GLM teacher requests (`requests.jsonl`, also the prompt source for later sampling) and responses on the 1,600 CruxEval prompts; training data for expert E"),
        (f"{T}/cruxeval-test-exposed/task133-20261007-prepared-v2", "Expert E SFT training set built from the GLM responses (prepared rows, audits, scores)"),
        (f"{T}/cruxeval-test-exposed/step92-fdc3c3c5/task133-readiness.json", "Readiness record of the S3 copy of student B used as anchor/teacher base"),
        (f"{T}/cruxeval-test-exposed/step92-fdc3c3c5/.marinskyrl-model-manifest.json", "MarinSkyRL manifest of the S3 copy of student B"),
        (f"{T}/cruxeval-transfer/task145-e-pilot-20261008", "Expert E sampling pilot (small run before the k=8 pool)"),
        (f"{T}/cruxeval-transfer/task145-e-k8-20261008", "Expert E k=8 teacher-trace pool on the CruxEval prompts (source of the plain and RS SFT sets)"),
        (f"{T}/cruxeval-transfer/task145-plain-prepared-20261008", "Plain SFT training set (one E trace per prompt), prepared"),
        (f"{T}/cruxeval-transfer/task145-rs-prepared-20261009", "Rejection-sampled SFT training set (verified E traces), prepared"),
        (f"{K}/cruxeval-opd/data/cruxeval-1600-prompt-only.parquet", "OPD prompt set (1,600 CruxEval prompts, no responses)"),
        (f"{K}/skyrl/snowball-opd-cruxeval-e-2/", "OPD run launch config and terminal record (run snowball-opd-cruxeval-e-2)"),
        (f"{K}/skyrl/snowball-opd-cruxeval-e-2/exports/global_step_75/policy/.marinskyrl-model-manifest.json", "MarinSkyRL manifest of the OPD update-75 export"),
    ]),
    "two-expert": ("open-athena/Snowball-67B-A2B-Two-Expert-Transfer-Artifacts-2026.10", 9984, I84, "",
                   "Two-expert transfer experiment: experts E (CruxEval) and E_2 (MMLU-Pro) into student B by mixed SFT, mixed RS-SFT, MOPD and weight merges", [
        (f"{T}/mopd2/task162-e2-prepared-20261009-v1", "Expert E_2 SFT training set (MMLU-Pro), prepared"),
        (f"{T}/mopd2/task163-e2-step254-k8-20261010", "Expert E_2 step-254 k=8 teacher-trace pool on the MMLU-Pro prompts, with per-response grades"),
        (f"{T}/mopd2/task163-e2-step254-k8-audit-20261010", "Audit of the E_2 k=8 pool (first-correct selection, uncovered prompts)"),
        (f"{T}/mopd2/task165-mixed-plain-prepared-20261010", "Mixed plain SFT training set (E + E_2 traces), prepared"),
        (f"{T}/mopd2/task165-mmlu-plain-prepared-20261010", "MMLU-Pro-only plain SFT set built in the same task as the mixed set (see its manifest.json)"),
        (f"{T}/mopd2/task166-mixed-rs-prepared-20261010", "Mixed rejection-sampled SFT training set, prepared"),
        (f"{T}/mopd2/task166-mmlu-rs-prepared-20261010", "MMLU-Pro-only RS-SFT set built in the same task as the mixed set (see its manifest.json)"),
        (f"{K}/mopd2/data/mopd2-3206-prompt-only.parquet", "MOPD and mixed-SFT prompt universe: 3,206 prompts, teacher_route in {cruxeval, mmlupro}, no responses"),
        (f"{K}/skyrl/snowball-mopd2-cruxeval-mmlupro-2ep/", "MOPD run launch config and terminal record (run snowball-mopd2-cruxeval-mmlupro-2ep)"),
        (f"{K}/skyrl/snowball-mopd2-cruxeval-mmlupro-2ep/exports/global_step_100/policy/.marinskyrl-model-manifest.json", "MarinSkyRL manifest of the MOPD step-100 export"),
        (f"{K}/mopd2/merges/ta-a0.5-b0.5/r3/merge-recipe.json", "Recipe of merge TA a=b=0.5 (published model)"),
        (f"{K}/mopd2/merges/ta-a0.5-b0.5/r3/merge-manifest.json", "Output manifest of merge TA a=b=0.5"),
        (f"{K}/mopd2/merges/ties-d0.5-a1.0-b1.0/r3/merge-recipe.json", "Recipe of merge TIES d=0.5 (published model)"),
        (f"{K}/mopd2/merges/ties-d0.5-a1.0-b1.0/r3/merge-manifest.json", "Output manifest of merge TIES d=0.5"),
        (f"{K}/mopd2/merges/ta-a1.0-b1.0/r3/merge-recipe.json", "Recipe of merge TA a=b=1.0 (weights not published; rebuild from B, E, E_2 with this recipe)"),
        (f"{K}/mopd2/merges/ta-a1.0-b1.0/r3/merge-manifest.json", "Output manifest of merge TA a=b=1.0 (tensor list and sizes of the deleted weights)"),
    ]),
}


def expand(entry):
    """A trailing '/' means: only the files directly in that folder. A folder means: everything under it. Else one file."""
    if entry.endswith("/"):
        return [p for p in fs.ls(f"{B}/{entry}", detail=True) if p["type"] == "file"]
    if fs.isdir(f"{B}/{entry}"):
        return [dict(i, name=p) for p, i in fs.find(f"{B}/{entry}", detail=True).items()]
    return [fs.info(f"{B}/{entry}")]


def fetch(info, local):
    rel = info["name"][len(B) + 1:]
    dst = local / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or dst.stat().st_size != info["size"]:
        fs.get(info["name"], str(dst))
    data = dst.read_bytes()
    assert len(data) == info["size"], (rel, len(data), info["size"])
    return rel, info["size"], hashlib.sha256(data).hexdigest(), hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


key = sys.argv[1]
repo, issue, url, provider, title, entries = SETS[key]
local = pathlib.Path("/tmp/artifacts") / repo.replace("/", "__")
rows = []
with cf.ThreadPoolExecutor(32) as ex:
    for entry, purpose in entries:
        infos = expand(entry)
        assert infos, entry
        for rel, size, sha, gsha in ex.map(lambda i: fetch(i, local), infos):
            rows.append((rel, size, sha, gsha, entry, purpose))
rows.sort()

# The Hub allows at most 10,000 files per folder, so every folder with more than 1,000 files is stored as one uncompressed
# tar (<folder>.tar, members named by their full path, so extracting at the repo root restores the S3 layout).
counts = collections.Counter(posixpath.dirname(r[0]) for r in rows)
packed = sorted(d for d, n in counts.items() if n > 1000)
stored_in = {r[0]: (posixpath.dirname(r[0]) + ".tar" if posixpath.dirname(r[0]) in packed else r[0]) for r in rows}
tars = {}
for d in packed:
    members = [r for r in rows if posixpath.dirname(r[0]) == d]
    with tarfile.open(local / f"{d}.tar", "w", format=tarfile.PAX_FORMAT) as tf:
        for rel, size, *_ in members:
            ti = tf.gettarinfo(str(local / rel), arcname=rel)
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = 0, 0, 0, "", "", 0o644
            with open(local / rel, "rb") as f:
                tf.addfile(ti, f)
    want = {r[0]: (r[1], r[2]) for r in members}
    with tarfile.open(local / f"{d}.tar") as tf:
        got = {m.name: (m.size, hashlib.sha256(tf.extractfile(m).read()).hexdigest()) for m in tf.getmembers()}
    assert got == want, d
    data = (local / f"{d}.tar").read_bytes()
    tars[f"{d}.tar"] = (len(data), hashlib.sha256(data).hexdigest(), hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(), len(members))
    print(f"PACKED {d}.tar {len(members)} files {len(data)} bytes sha256 {tars[f'{d}.tar'][1]} (members re-hashed OK)", flush=True)

(local / "MANIFEST.tsv").write_text("path\tbytes\tsha256\ts3_origin\tissue\tstored_in\tpurpose\n" + "".join(
    f"{r[0]}\t{r[1]}\t{r[2]}\ts3://{B}/{r[0]}\t#{issue}\t{stored_in[r[0]]}\t{r[5]}\n" for r in rows))

groups = collections.OrderedDict()
for rel, size, sha, gsha, entry, purpose in rows:
    g = groups.setdefault(entry, [purpose, 0, 0, []])
    g[1] += 1; g[2] += size; g[3].append((rel, size, sha))
md = [f"---\nlicense: apache-2.0\npretty_name: {repo.split('/')[1]}\ntags:\n- marin\n- snowball\n- distillation\n---\n",
      f"# {repo.split('/')[1]}\n",
      f"Small artifacts from the {title}. Experiment report: {url}.\n",
      "These files were moved from CoreWeave S3 (`s3://marin-us-east-02a/marin/`) when the experiment's S3 data was cleaned up "
      "(marin task #210). Each file keeps its S3 path relative to `marin/`, so `s3_origin` = `s3://marin-us-east-02a/marin/<path>`. "
      "The S3 copies are deleted after this upload is verified.\n",
      f"Totals: {len(rows):,} files, {sum(r[1] for r in rows)/1e6:,.1f} MB. `MANIFEST.tsv` lists every file with bytes, sha256, "
      "S3 origin, issue, the repo file that holds it (`stored_in`) and purpose.\n",
      "The Hub allows at most 10,000 files per folder. So each folder with more than 1,000 files is stored as one uncompressed tar, "
      "`<folder>.tar`. Its members keep their full paths, so `tar -xf <folder>.tar` at the repo root restores the original layout.\n",
      "## Contents\n", "| S3 origin (under `s3://marin-us-east-02a/marin/`) | Purpose | Files | MB |", "|---|---|---:|---:|"]
for entry, (purpose, n, b, _) in groups.items():
    md.append(f"| `{entry}` | {purpose} | {n:,} | {b/1e6:,.2f} |")
md += ["", "## License", "", f"Apache-2.0. {provider}Teacher-model outputs remain subject to their providers' terms."]
if tars:
    md += ["", "## Packed folders", "", "| Repo file | Files inside | Bytes | sha256 |", "|---|---:|---:|---|"]
    md += [f"| `{t}` | {n:,} | {b:,} | `{sha}` |" for t, (b, sha, _, n) in tars.items()]
md += ["", "## Files (folders with more than 20 files are summarised above; see MANIFEST.tsv)", "", "| Path | Bytes | sha256 |", "|---|---:|---|"]
for entry, (purpose, n, b, files) in groups.items():
    if n <= 20:
        md += [f"| `{rel}` | {size:,} | `{sha}` |" for rel, size, sha in files]
(local / "README.md").write_text("\n".join(md) + "\n")

# What the repo must hold: loose files (path, bytes, sha256, git blob id) and the tars.
expect = {r[0]: (r[1], r[2], r[3]) for r in rows if stored_in[r[0]] == r[0]}
expect.update({t: v[:3] for t, v in tars.items()})
print(f"STAGED {repo} {len(rows)} files {sum(r[1] for r in rows)} bytes at {local}; repo files {len(expect)} ({len(tars)} tars)", flush=True)

if "--upload" not in sys.argv:
    print("README-BEGIN\n" + (local / "README.md").read_text() + "README-END", flush=True)
    for d, n in sorted(collections.Counter(posixpath.dirname(p) for p in expect).items()):
        print(f"REPO-FOLDER {d or '.'} {n} files", flush=True)
    sys.exit(0)

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi  # noqa: E402

api = HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)


def hub_files():
    return {s.rfilename: s for s in api.dataset_info(repo, files_metadata=True).siblings}


def mismatched(hub):
    return [p for p, (size, sha, gsha) in expect.items() if p not in hub or hub[p].size != size or
            (hub[p].lfs.sha256 if hub[p].lfs else hub[p].blob_id) != (sha if hub[p].lfs else gsha)]


hub = hub_files()
todo = mismatched(hub) + ["README.md", "MANIFEST.tsv"]
stale = sorted(p for p in hub if p not in expect and p not in ("README.md", "MANIFEST.tsv", ".gitattributes"))
stale_dirs = [d for d in packed if any(p.startswith(d + "/") for p in stale)]
stale = [p for p in stale if posixpath.dirname(p) not in stale_dirs]
print(f"TODO add {len(todo)} files, remove {len(stale)} stale files and {len(stale_dirs)} unpacked folders", flush=True)
ops = [CommitOperationDelete(path_in_repo=d + "/", is_folder=True) for d in stale_dirs] + [CommitOperationDelete(path_in_repo=p) for p in stale] + [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(local / p)) for p in todo]
for i in range(0, len(ops), 1000):
    part = ops[i:i + 1000]
    c = api.create_commit(repo, repo_type="dataset", commit_message=f"Task #210 artifacts: sync ({i + len(part)}/{len(ops)} operations)", operations=part)
    print(f"COMMIT {c.oid} {i + len(part)}/{len(ops)}", flush=True)
info = api.dataset_info(repo, files_metadata=True)
hub = {s.rfilename: s for s in info.siblings}
bad = mismatched(hub)
extra = sorted(p for p in hub if p not in expect and p not in ("README.md", "MANIFEST.tsv", ".gitattributes"))
print(f"UPLOADED {repo} sha {info.sha} files {len(hub)} mismatched {len(bad)} {bad[:10]} extra {len(extra)} {extra[:10]} "
      f"url https://huggingface.co/datasets/{repo}/tree/{info.sha}", flush=True)
