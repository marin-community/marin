"""Map HLO instruction name -> (opcode, op_name) from the train-step HloProto in an xplane."""
import sys, pickle, collections
sys.path.insert(0, sys.argv[2])
from overlap import load
from google.protobuf.internal import decoder

def fields(buf):
    pos = 0; n = len(buf)
    while pos < n:
        tag, pos = decoder._DecodeVarint(buf, pos)
        fnum, wt = tag >> 3, tag & 7
        if wt == 0:
            v, pos = decoder._DecodeVarint(buf, pos); yield fnum, v
        elif wt == 2:
            ln, pos = decoder._DecodeVarint(buf, pos); yield fnum, buf[pos:pos+ln]; pos += ln
        elif wt == 1: yield fnum, buf[pos:pos+8]; pos += 8
        elif wt == 5: yield fnum, buf[pos:pos+4]; pos += 4
        else: raise ValueError(wt)

xs = load(sys.argv[1])
meta = [p for p in xs.planes if p.name == "/host:metadata"][0]
names = {int(k): v.name for k, v in meta.stat_metadata.items()}
best = None
for k, m in meta.event_metadata.items():
    if m.name.startswith("jit_train_step"):
        for s in m.stats:
            if names.get(int(s.metadata_id)) == "Hlo Proto" and (best is None or len(s.bytes_value) > len(best)):
                best = s.bytes_value
mod = dict((f, v) for f, v in fields(best) if f == 1)[1]
comps = [v for f, v in fields(mod) if f == 3]
instr = {}
comp_ops = {}
comp_id_name = {}
for c in comps:
    cname = None; cid = None; ins = []
    for f, v in fields(c):
        if f == 1: cname = v.decode()
        elif f == 5: cid = v
        elif f == 2: ins.append(v)
    comp_id_name[cid] = cname
    opnames = collections.Counter()
    for i in ins:
        name = opcode = op_name = None; called = []
        for f, v in fields(i):
            if f == 1: name = v.decode()
            elif f == 2: opcode = v.decode()
            elif f == 7:
                for ff, vv in fields(v):
                    if ff == 2: op_name = vv.decode()
            elif f == 38: called.append(v)
        instr[name] = [opcode, op_name, called, cname]
        if op_name: opnames[op_name] += 1
    comp_ops[cid] = opnames
# fill fusion op_names from called computation
filled = 0
for name, rec in instr.items():
    if not rec[1] and rec[2]:
        cnt = collections.Counter()
        for cid in rec[2]:
            cnt.update(comp_ops.get(cid, {}))
        if cnt:
            rec[1] = cnt.most_common(1)[0][0]; filled += 1
print("instructions", len(instr), "with op_name", sum(1 for r in instr.values() if r[1]), "filled", filled)
pickle.dump({k: (v[0], v[1], v[3]) for k, v in instr.items()}, open(sys.argv[3], "wb"))
