"""Parse HLO text for cuBLAS GEMM custom-calls -> flops; operand shapes via a name->shape map."""
import sys, re, pickle
txt = open(sys.argv[1]).read().splitlines()
shape_re = re.compile(r"(bf16|f32|f16|f8e4m3fn|f8e5m2|s32|u32|s8|pred|u8|s64)\[([0-9,]*)\]")
def dims(s): return [int(x) for x in s.split(",") if x] if s else []
shapes = {}
parsed = []
for line in txt:
    m = re.match(r"\s*(ROOT )?%?([\w.\-]+) = (.*)", line)
    if not m: continue
    name, rest = m.group(2), m.group(3)
    head = rest.split("(", 1)[0] if not rest.startswith("(") else rest[: rest.index(")") + 1]
    sh = shape_re.findall(head)
    if sh: shapes[name] = dims(sh[0][1])
    parsed.append((name, rest))
out = {}
for name, rest in parsed:
    if 'custom_call_target="__cublas' not in rest: continue
    args = rest.split("custom-call(", 1)[1].split(")", 1)[0]
    ops = [a.strip().lstrip("%") for a in args.split(",")]
    lhs = shapes.get(ops[0]); outd = shapes.get(name)
    if lhs is None or outd is None: continue
    mc = re.search(r'"lhs_contracting_dimensions":\[([^\]]*)\]', rest)
    cd = [int(x) for x in re.findall(r"\d+", mc.group(1))] if mc else [len(lhs) - 1]
    k = 1
    for c in cd: k *= lhs[c]
    o = 1
    for x in outd: o *= x
    out[name] = (2 * o * k, lhs, outd)
print("gemms", len(out))
pickle.dump(out, open(sys.argv[2], "wb"))
