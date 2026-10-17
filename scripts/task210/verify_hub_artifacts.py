"""Task #210: download both artifact dataset repos at their pinned commits (no token) and hash every file in MANIFEST.tsv.

Usage: verify_hub_artifacts.py <download dir>. Loose files are hashed on disk; packed files are hashed inside their tar.
Prints one HUBCHECK line per repo: manifest rows, mismatches, and tar members not listed in the manifest.
"""
import hashlib, pathlib, sys, tarfile
from huggingface_hub import snapshot_download

REPOS = [("open-athena/Snowball-67B-A2B-CruxEval-Transfer-Artifacts-2026.10", "98c153c05ddd02068c5415144696e01e9a61c0a2"),
         ("open-athena/Snowball-67B-A2B-Two-Expert-Transfer-Artifacts-2026.10", "7a96dade2635b0bd76a1a67cb5381fad2d6b2be8")]
root = pathlib.Path(sys.argv[1])
for repo, rev in REPOS:
    d = root / repo.split("/")[1]
    snapshot_download(repo, repo_type="dataset", revision=rev, local_dir=d, token=False)
    rows = [l.split("\t") for l in (d / "MANIFEST.tsv").read_text().splitlines()[1:]]
    got = {}
    for tarname in sorted({r[5] for r in rows if r[5] != r[0]}):
        with tarfile.open(d / tarname) as tf:
            for m in tf.getmembers():
                got[m.name] = (m.size, hashlib.sha256(tf.extractfile(m).read()).hexdigest())
    bad = 0
    for path, size, sha, origin, issue, stored, purpose in rows:
        if stored == path:
            b = (d / path).read_bytes()
            g = (len(b), hashlib.sha256(b).hexdigest())
        else:
            g = got.pop(path, None)
        bad += g != (int(size), sha)
    print(f"HUBCHECK {repo}@{rev[:8]} manifest rows {len(rows)} mismatched {bad} unlisted tar members {len(got)}")
