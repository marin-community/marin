"""Extract jit_train_step HLO text from an xplane.pb's /host:metadata plane."""
import sys
sys.path.insert(0, sys.argv[2])
from overlap import load
from google.protobuf.internal import decoder

def field_bytes(buf, want):
    pos = 0; n = len(buf)
    while pos < n:
        tag, pos = decoder._DecodeVarint(buf, pos)
        fnum, wt = tag >> 3, tag & 7
        if wt == 0:
            _, pos = decoder._DecodeVarint(buf, pos)
        elif wt == 2:
            ln, pos = decoder._DecodeVarint(buf, pos)
            if fnum == want:
                return buf[pos:pos+ln]
            pos += ln
        elif wt == 1: pos += 8
        elif wt == 5: pos += 4
        else: raise ValueError(wt)
    return None

xs = load(sys.argv[1])
meta = [p for p in xs.planes if p.name == "/host:metadata"][0]
names = {int(k): v.name for k, v in meta.stat_metadata.items()}
best = None
for k, m in meta.event_metadata.items():
    if m.name.startswith("jit_train_step"):
        for s in m.stats:
            if names.get(int(s.metadata_id)) == "Hlo Proto":
                if best is None or len(s.bytes_value) > len(best[1]):
                    best = (m.name, s.bytes_value)
print("module", best[0], len(best[1]))
mod = field_bytes(best[1], 1)
from jax._src.lib import xla_client
comp = xla_client.XlaComputation(mod)
txt = comp.as_hlo_text()
open(sys.argv[3], "w").write(txt)
print("hlo text chars", len(txt))
