"""Approx HBM bytes per HLO instruction (operands + outputs) for non-GEMM kernels."""
import sys, re, pickle
txt = open(sys.argv[1]).read().splitlines()
BY = {"bf16": 2, "f32": 4, "f16": 2, "s32": 4, "u32": 4, "s8": 1, "u8": 1, "pred": 1, "s64": 8, "u64": 8, "f8e4m3fn": 1, "f8e5m2": 1, "s16": 2, "u16": 2}
shape_re = re.compile(r"(bf16|f32|f16|f8e4m3fn|f8e5m2|s32|u32|s8|u8|pred|s64|u64|s16|u16)\[([0-9,]*)\]")
def nbytes(sh):
    t = 0
    for dt, dd in sh:
        n = 1
        for x in dd.split(","):
            if x: n *= int(x)
        t += n * BY[dt]
    return t
outb = {}
parsed = []
comp = None
for line in txt:
    if line and not line.startswith(" ") and "{" in line:
        comp = line.split(" ")[0]
    m = re.match(r"\s*(ROOT )?%?([\w.\-]+) = (.*)", line)
    if not m: continue
    name, rest = m.group(2), m.group(3)
    if rest.startswith("("):
        head = rest[: rest.index(")") + 1]
    else:
        head = rest.split("(", 1)[0]
    outb[name] = nbytes(shape_re.findall(head))
    parsed.append((name, rest, comp))
res = {}
for name, rest, comp in parsed:
    if "(" not in rest: continue
    opcode_args = rest.split("(", 1)
    try:
        args = rest.split(" fusion(", 1)[1] if " fusion(" in rest else None
    except Exception:
        args = None
    if args is None: continue
    args = args.split(")", 1)[0]
    ins = sum(outb.get(a.strip().lstrip("%"), 0) for a in args.split(","))
    res[name] = ins + outb.get(name, 0)
print("fusions", len(res))
pickle.dump(res, open(sys.argv[2], "wb"))
