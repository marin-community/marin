# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json
import re


def load(path):
    out = []
    with open(path) as f:
        lines = f.readlines()
    for l in lines:
        if "RESULT " not in l:
            continue
        s = l.split("RESULT ", 1)[1].strip()
        try:
            out.append(json.loads(s, strict=False))
        except Exception:
            try:
                out.append(json.loads(re.sub(r'\\(?!["\\/bfnrtu])', r"\\\\", s), strict=False))
            except Exception:
                pass
    return out
