"""Task #210: dump HF file manifests (name, size, sha256 for LFS files, git blob id otherwise) for the #9920/#9984 repos."""
import json, sys
from huggingface_hub import HfApi

REPOS = {
    "B": ("open-athena/Snowball-67B-A2B-10T-Mixed-RLVR-Sync-Step92", "fdc3c3c5fff489d3fb04dcbc2387c179b1bf3fab"),
    "E272": ("open-athena/Snowball-67B-A2B-CruxEval-Expert-E-Step272", "fb4f8051a344fec59f902bdd2ca876ac1b2f3296"),
    "E2_254": ("open-athena/Snowball-67B-A2B-MMLUPro-Expert-E2-Step254", "acb97183af633533df4f4f63d83adc1e0a7a9748"),
    "PLAIN399": ("open-athena/Snowball-67B-A2B-CruxEval-Plain-SFT-Step399", "3e34be3f884ff737f85354b060ea831ce0c90636"),
    "RS282": ("open-athena/Snowball-67B-A2B-CruxEval-RS-SFT-Step282", "073d128cb1e53652126cce37a7cc2c96625d3c80"),
    "OPD75": ("open-athena/Snowball-67B-A2B-CruxEval-OPD-Step75", "9d1893207226cf72f701187e1d5260d9fc3c931d"),
    "PLAIN803": ("open-athena/Snowball-67B-A2B-MixedPlain-SFT-Step803", "1a0bcf70fa4b599701d5e0c9f0864452d531a9dc"),
    "RS727": ("open-athena/Snowball-67B-A2B-MixedRS-SFT-Step727", "95dd3f4ab7487a67907b5a22748a2c2bc77770cc"),
    "MOPD100": ("open-athena/Snowball-67B-A2B-MOPD-Step100", "4bb972592c0094d51d29321f9af2a6defa2ccea1"),
    "TA05": ("open-athena/Snowball-67B-A2B-Merge-TA0.5", "07001c3333337bdea61647c74bcb227f60310fa6"),
    "TIES": ("open-athena/Snowball-67B-A2B-Merge-TIES-d0.5", "ffb5b68590fdf7c5fe01351a332e7915d574bc61"),
}
api = HfApi()
out = {}
for k, (repo, rev) in REPOS.items():
    info = api.model_info(repo, revision=rev, files_metadata=True)
    files = {s.rfilename: {"size": s.size, "sha256": s.lfs.sha256 if s.lfs else None, "blob": s.blob_id} for s in info.siblings}
    out[k] = {"repo": repo, "rev": rev, "sha": info.sha, "private": info.private, "files": files}
    print(k, repo, info.sha[:8], "private" if info.private else "public", len(files), f"{sum(f['size'] or 0 for f in files.values())/1e9:.2f} GB", file=sys.stderr)
json.dump(out, open(sys.argv[1], "w"), indent=1)
