import sys, collections, pickle
sys.path.insert(0, sys.argv[2])
from overlap import load, plane_events
xs = load(sys.argv[1])
planes = {p.name: p for p in xs.planes}
dev = planes["/device:GPU:0"]
rows = []
for lname, name, s, e, st in plane_events(dev):
    if not lname.startswith("Stream"): continue
    rows.append((lname, name, s, e, st.get("tf_op") or "", st.get("hlo_op") or "", st.get("hlo_module") or "", st.get("kernel_details") or ""))
host = planes["/host:CPU"]
launches = sorted(s for _, n, s, _e, _st in plane_events(host) if n.startswith("CommonPjRtLoadedExecutable::Execute (jit_train_step"))
pickle.dump({"rows": rows, "launches": launches}, open(sys.argv[3], "wb"))
print(len(rows), "rows;", len(launches), "launches")
