# extracts mds from 1 processed stack exchange dump (just does some jq and head)
import argparse
import os
import json

import fsspec

argparser = argparse.ArgumentParser()
argparser.add_argument("-n", type=int, default=1000)
argparser.add_argument("-o", default=None)
argparser.add_argument("input")

args = argparser.parse_args()

if args.o is None:
    base = os.path.basename(args.input)
    output_dir = base.split(".")[0]
else:
    output_dir = args.o

os.makedirs(output_dir, exist_ok=True)
print(f"Writing to {output_dir}")

with fsspec.open(args.input, "r", compression="infer") as f:
    for i, line in enumerate(f):
        if i >= args.n:
            break
        data = json.loads(line)
        md = data["text"]
        with open(os.path.join(output_dir, f"{i}.md"), "w") as f:
            f.write(md)







