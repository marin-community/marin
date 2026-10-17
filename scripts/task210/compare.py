"""Task #210: compare S3 export hashes (s3-hashes txt) with HF manifests (json). Prints one verdict line per repo."""
import collections, json, sys

hf = json.load(open(sys.argv[1]))
s3 = collections.defaultdict(dict)
for line in open(sys.argv[2]):
    p = line.split()
    if p[0] == "HASH":
        s3[p[1]][p[2]] = {"size": int(p[3]), "sha256": p[4], "git": p[5]}
LABEL = {"B": "b-copy", "E272": "e272", "E2_254": "e2-254", "PLAIN399": "plain399", "RS282": "rs282", "OPD75": "opd75",
         "PLAIN803": "plain803", "RS727": "rs727", "MOPD100": "mopd100", "TA05": "ta05", "TIES": "ties"}
for k, lab in LABEL.items():
    h, s = hf[k]["files"], s3[lab]
    weights = [f for f in h if f.endswith(".safetensors")]
    bad = [f for f in h if f not in ("README.md", ".gitattributes") and
           (f not in s or s[f]["size"] != h[f]["size"] or (h[f]["sha256"] or s[f]["git"]) != (s[f]["sha256"] if h[f]["sha256"] else h[f]["blob"]))]
    only_s3 = sorted(f for f in s if f not in h)
    print(f"{k:9} {hf[k]['repo']}@{hf[k]['sha'][:8]} hf_files={len(h)} safetensors={len(weights)} "
          f"weights_GB={sum(h[f]['size'] for f in weights)/1e9:.2f} mismatched={len(bad)} {bad[:5]} s3_only={only_s3}")
