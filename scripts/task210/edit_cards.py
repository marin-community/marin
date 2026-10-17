"""Task #210: add an 'S3 source deleted' line to the live model cards after the S3 exports are deleted.

Usage: edit_cards.py <hf-manifests json> <date YYYY-MM-DD> [--apply]
Without --apply: prints the unified diff per repo, writes nothing. With --apply: needs HF_TOKEN in the env (never printed);
commits only README.md. For each repo it reads the card at main (live text, not a stale copy), checks that every safetensors
file on main still has the sha256 verified at the pinned commit, and inserts the line right after the line holding
'Source export `s3://'. Repos without that line (e.g. B, whose card we did not write) are skipped and reported.
"""
import difflib, json, os, sys
from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

manifests, date, apply = json.load(open(sys.argv[1])), sys.argv[2], "--apply" in sys.argv
api = HfApi(token=os.environ["HF_TOKEN"] if apply else None)
ISSUE = {"E272": 9920, "PLAIN399": 9920, "RS282": 9920, "OPD75": 9920}
for key, m in manifests.items():
    repo, pinned = m["repo"], m["rev"]
    main = api.model_info(repo, files_metadata=True)
    now = {s.rfilename: s.lfs.sha256 for s in main.siblings if s.rfilename.endswith(".safetensors") and s.lfs}
    want = {f: v["sha256"] for f, v in m["files"].items() if f.endswith(".safetensors")}
    if now != want:
        print(f"SKIP {key} {repo}: weights on main {main.sha[:8]} differ from verified {pinned[:8]}")
        continue
    old = open(hf_hub_download(repo, "README.md", revision=main.sha)).read()
    lines = old.split("\n")
    idx = [i for i, l in enumerate(lines) if "Source export `s3://" in l]
    if len(idx) != 1:
        print(f"SKIP {key} {repo}: {len(idx)} 'Source export `s3://' lines; card not ours or unexpected")
        continue
    if "S3 source deleted on" in old:
        print(f"SKIP {key} {repo}: already edited")
        continue
    note = (f"S3 source deleted on {date}; weights verified identical to commit {pinned[:8]} by sha256 "
            f"(see https://github.com/marin-community/marin/issues/{ISSUE.get(key, 9984)}).")
    new = "\n".join(lines[: idx[0] + 1] + ["", note] + lines[idx[0] + 1:])
    print(f"== {key} {repo} main={main.sha[:8]}")
    sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), new.splitlines(True), "README.md@main", "README.md@new", n=1))
    if apply:
        c = api.create_commit(repo, [CommitOperationAdd("README.md", new.encode())], commit_message=f"Note S3 source deletion ({date})",
                              parent_commit=main.sha)
        print(f"EDITED {key} {repo} {c.oid}")
