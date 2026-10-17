"""Task #210: build the keep/move/delete table (markdown) from inventory-raw NODE lines.

Each row is (issue, class, what, includes, excludes). Bytes = sum(include nodes) - sum(exclude nodes).
"""
import sys

node = {}
for line in open(sys.argv[1]):
    p = line.split()
    if p[0] == "NODE":
        node[p[1]] = (int(p[2]), int(p[3]), p[4])

X = "experiments"
K = "users/karan"
T = "teacher-traces"
c133 = f"{X}/task133-cruxeval/checkpoints/karan/cruxeval-step92-lr1e-05"
p145 = f"{X}/task145-cruxeval/checkpoints/karan/cruxeval-plain-e-from-b-lr1e-06/2026.10.08.1"
r145 = f"{X}/task145-cruxeval/checkpoints/karan/cruxeval-rs-e-from-b-lr1e-06/2026.10.08.1"
e147 = f"{X}/task147-cruxeval/checkpoints/karan/cruxeval-eplus-step272-lr1e-06/2026.10.08.1"
e157 = f"{X}/task157-e2/checkpoints/karan/e2-mmlupro-step92-lr1e-05/2026.10.09.1"
p165 = f"{X}/task165-mixed/checkpoints/karan/mixed-plain-e-from-b-lr1e-06"
r166 = f"{X}/task166-mixed/checkpoints/karan/mixed-rs-e-from-b-lr1e-06/2026.10.10.1"
opd = f"{K}/skyrl/snowball-opd-cruxeval-e-2"
mopd = f"{K}/skyrl/snowball-mopd2-cruxeval-mmlupro-2ep"
mg = f"{K}/mopd2/merges"
HF = "covered: HF repo verified by sha256"

def steps(base, kind, ss):
    return [f"{base}/{kind}/step-{s}" for s in ss]

ROWS = [
    # ---- #9920 ----
    ("9920", "DELETE", f"Expert E step-272 HF export ({HF}: CruxEval-Expert-E-Step272@fb4f8051)", [f"{c133}/2026.10.07.5/hf/step-272"], []),
    ("9920", "DELETE", "Expert E intermediate HF exports step-91, step-182", steps(f"{c133}/2026.10.07.5", "hf", [91, 182]), []),
    ("9920", "DELETE", "Expert E superseded first run 2026.10.07.2 (HF step-91)", [f"{c133}/2026.10.07.2"], []),
    ("9920", "DELETE", "task133 run metadata, executor dataset stubs, tokenized cache (rebuildable from the prepared set below)",
     [f"{X}/task133-cruxeval"], [f"{X}/task133-cruxeval/checkpoints/karan/cruxeval-step92-lr1e-05"]),
    ("9920", "DELETE", "Plain SFT native Levanter checkpoints step-100/200/300/399 (weights + optimizer state)", [f"{p145}/checkpoints"], []),
    ("9920", "DELETE", f"Plain SFT step-399 HF export ({HF}: CruxEval-Plain-SFT-Step399@3e34be3f)", [f"{p145}/hf/step-399"], []),
    ("9920", "DELETE", "Plain SFT intermediate HF exports step-100/200/300", steps(p145, "hf", [100, 200, 300]), []),
    ("9920", "DELETE", "RS-SFT native Levanter checkpoints step-94/188/282/375", [f"{r145}/checkpoints"], []),
    ("9920", "DELETE", f"RS-SFT step-282 HF export ({HF}: CruxEval-RS-SFT-Step282@073d128c)", [f"{r145}/hf/step-282"], []),
    ("9920", "DELETE", "RS-SFT intermediate HF exports step-94/188/375", steps(r145, "hf", [94, 188, 375]), []),
    ("9920", "DELETE", "task145 run metadata, executor dataset stubs, tokenized caches",
     [f"{X}/task145-cruxeval"], [p145, r145]),
    ("9920", "DELETE", "E+ control (E + 2 epochs, LR 1e-6) native checkpoints step-91/181", [f"{e147}/checkpoints"], []),
    ("9920", "OWNER", "E+ control HF exports step-91, step-181 (not on HF; default DELETE; alt MOVE step-181 -> Snowball-67B-A2B-CruxEval-EPlus-Step181, 134 GB upload)",
     steps(e147, "hf", [91, 181]), []),
    ("9920", "DELETE", "task147 metadata stubs, tokenized cache", [f"{X}/task147-cruxeval"], [e147]),
    ("9920", "DELETE", f"OPD update-75 export ({HF}: CruxEval-OPD-Step75@9d189320)", [f"{opd}/exports/global_step_75"], []),
    ("9920", "DELETE", "OPD intermediate/after exports global_step_25/50/100", [f"{opd}/exports/global_step_{s}" for s in (25, 50, 100)], []),
    ("9920", "MOVE-DS", "OPD run launch config/terminal record (top-level files of snowball-opd-cruxeval-e-2)", [opd], [f"{opd}/exports"]),
    ("9920", "DELETE", "OPD smoke run export global_step_2 (snowball-opd-cruxeval-e-smoke-2)", [f"{K}/skyrl/snowball-opd-cruxeval-e-smoke-2"], []),
    ("9920", "DELETE", "OPD failed/smoke launch stubs (snowball-opd-cruxeval-e, -e-smoke: resolved-launch.yaml only)",
     [f"{K}/skyrl/snowball-opd-cruxeval-e", f"{K}/skyrl/snowball-opd-cruxeval-e-smoke"], []),
    ("9920", "MOVE-DS", "OPD prompt set cruxeval-1600-prompt-only.parquet", [f"{K}/cruxeval-opd"], []),
    ("9920", "DELETE", f"Student B copy used as merge anchor/teacher ({HF}: 10T-Mixed-RLVR-Sync-Step92@fdc3c3c5; 2 extra json files -> MOVE-DS)",
     [f"{T}/cruxeval-test-exposed/step92-fdc3c3c5"], []),
    ("9920", "MOVE-DS", "GLM teacher responses for E (task133-20261007-glm, 1,603 files)", [f"{T}/cruxeval-test-exposed/task133-20261007-glm"], []),
    ("9920", "MOVE-DS", "E training set, prepared (task133-20261007-prepared-v2)", [f"{T}/cruxeval-test-exposed/task133-20261007-prepared-v2"], []),
    ("9920", "MOVE-DS", "E k=8 teacher-trace pool + pilot (task145-e-k8, task145-e-pilot)",
     [f"{T}/cruxeval-transfer/task145-e-k8-20261008", f"{T}/cruxeval-transfer/task145-e-pilot-20261008"], []),
    ("9920", "MOVE-DS", "Plain and RS SFT training sets, prepared (task145-plain/rs-prepared)",
     [f"{T}/cruxeval-transfer/task145-plain-prepared-20261008", f"{T}/cruxeval-transfer/task145-rs-prepared-20261009"], []),
    # ---- #9984 ----
    ("9984", "DELETE", "E_2 native Levanter checkpoints step-85/170/254", [f"{e157}/checkpoints"], []),
    ("9984", "DELETE", f"E_2 step-254 HF export ({HF}: MMLUPro-Expert-E2-Step254@acb97183)", [f"{e157}/hf/step-254"], []),
    ("9984", "DELETE", "E_2 intermediate HF exports step-85/170", steps(e157, "hf", [85, 170]), []),
    ("9984", "DELETE", "task157 metadata stubs, tokenized cache", [f"{X}/task157-e2"], [e157]),
    ("9984", "DELETE", "Mixed plain SFT native checkpoints step-201/402/603/803", [f"{p165}/2026.10.10.2/checkpoints"], []),
    ("9984", "DELETE", f"Mixed plain step-803 HF export ({HF}: MixedPlain-SFT-Step803@1a0bcf70)", [f"{p165}/2026.10.10.2/hf/step-803"], []),
    ("9984", "DELETE", "Mixed plain intermediate HF exports step-201/402/603", steps(f"{p165}/2026.10.10.2", "hf", [201, 402, 603]), []),
    ("9984", "DELETE", "task165 aborted run 2026.10.10.1, metadata stubs, tokenized cache", [f"{X}/task165-mixed"], [f"{p165}/2026.10.10.2"]),
    ("9984", "DELETE", "Mixed RS-SFT native checkpoints step-182/364/546/727", [f"{r166}/checkpoints"], []),
    ("9984", "DELETE", f"Mixed RS step-727 HF export ({HF}: MixedRS-SFT-Step727@95dd3f4a)", [f"{r166}/hf/step-727"], []),
    ("9984", "DELETE", "Mixed RS intermediate HF exports step-182/364/546", steps(r166, "hf", [182, 364, 546]), []),
    ("9984", "DELETE", "task166 metadata stubs, tokenized cache", [f"{X}/task166-mixed"], [r166]),
    ("9984", "DELETE", f"MOPD step-100 export ({HF}: MOPD-Step100@4bb97259)", [f"{mopd}/exports/global_step_100"], []),
    ("9984", "DELETE", "MOPD intermediate export global_step_50", [f"{mopd}/exports/global_step_50"], []),
    ("9984", "MOVE-DS", "MOPD run launch config/terminal record (top-level files of snowball-mopd2-cruxeval-mmlupro-2ep)", [mopd], [f"{mopd}/exports"]),
    ("9984", "DELETE", "MOPD smoke run export global_step_2", [f"{K}/skyrl/snowball-mopd2-cruxeval-mmlupro-smoke"], []),
    ("9984", "DELETE", f"Merge TA a=b=0.5 r3 ({HF}: Merge-TA0.5@07001c33; recipe/manifest json -> MOVE-DS)", [f"{mg}/ta-a0.5-b0.5"], []),
    ("9984", "DELETE", f"Merge TIES d=0.5 r3 ({HF}: Merge-TIES-d0.5@ffb5b685; recipe/manifest json -> MOVE-DS)", [f"{mg}/ties-d0.5-a1.0-b1.0"], []),
    ("9984", "OWNER", "Merge TA a=b=1.0 r3 (reported arm, withheld for runaway generation; not on HF; deterministic from B/E/E_2 + recipe; default DELETE, alt MOVE -> Snowball-67B-A2B-Merge-TA1.0, 134 GB)",
     [f"{mg}/ta-a1.0-b1.0/r3"], []),
    ("9984", "DELETE", "Merge TA a=b=1.0 partial earlier attempt (13 shards + recipe at the run root)", [f"{mg}/ta-a1.0-b1.0"], [f"{mg}/ta-a1.0-b1.0/r3"]),
    ("9984", "MOVE-DS", "MOPD prompt set mopd2-3206-prompt-only.parquet", [f"{K}/mopd2/data"], []),
    ("9984", "MOVE-DS", "E_2 training set, prepared (task162-e2-prepared-v1)", [f"{T}/mopd2/task162-e2-prepared-20261009-v1"], []),
    ("9984", "MOVE-DS", "E_2 k=8 teacher-trace pool with grades + audit (task163-*)",
     [f"{T}/mopd2/task163-e2-step254-k8-20261010", f"{T}/mopd2/task163-e2-step254-k8-audit-20261010"], []),
    ("9984", "MOVE-DS", "Mixed plain/RS and MMLU-only plain/RS training sets, prepared (task165/166-*-prepared)",
     [f"{T}/mopd2/task165-mixed-plain-prepared-20261010", f"{T}/mopd2/task165-mmlu-plain-prepared-20261010",
      f"{T}/mopd2/task166-mixed-rs-prepared-20261010", f"{T}/mopd2/task166-mmlu-rs-prepared-20261010"], []),
    # ---- not #9920/#9984 but under the listed users/karan prefixes ----
    ("other", "DELETE", "Older OPD/MOPD smoke data (snowball-opd-smoke, snowball-mopd-ultra; listed in the task)",
     [f"{K}/snowball-opd-smoke", f"{K}/snowball-mopd-ultra"], []),
    ("other", "OWNER", "Earlier MOPD-32k / PR-763 / tiny-wikitext leftovers, all tiny (snowball-mopd-32k data, pr-763-smoke*, checkpoints/tiny-wikitext, 26 skyrl smoke stubs); default DELETE",
     [f"{K}/snowball-mopd-32k", f"{K}/pr-763-smoke", f"{K}/pr-763-smoke-nojb", f"{K}/checkpoints"] +
     [k for k in node if k.startswith(f"{K}/skyrl/") and k.count("/") == 3 and any(s in k for s in ("pr-763", "mopd-32k", "mopd-async", "mopd-ultra", "opd-math"))], []),
]

def size(inc, exc):
    b = sum(node[i][0] for i in inc if i in node) - sum(node[e][0] for e in exc if e in node)
    n = sum(node[i][1] for i in inc if i in node) - sum(node[e][1] for e in exc if e in node)
    mt = max((node[i][2] for i in inc if i in node), default="")
    missing = [i for i in inc if i not in node]
    return b, n, mt, missing

tot = {}
out = ["| # | Issue | Class | What | S3 prefix(es) under s3://marin-us-east-02a/marin/ | GB | Objects | Newest |", "|---|---|---|---|---|---|---|---|"]
for i, (iss, cls, what, inc, exc) in enumerate(ROWS, 1):
    b, n, mt, missing = size(inc, exc)
    assert not missing, (what, missing)
    tot[cls] = tot.get(cls, 0) + b
    tot[(iss, cls)] = tot.get((iss, cls), 0) + b
    paths = "<br>".join(f"`{p}`" for p in inc[:6]) + (f"<br>… +{len(inc)-6} more" if len(inc) > 6 else "")
    if exc:
        paths += "<br>minus " + ", ".join(f"`{e.rsplit('/', 1)[1]}`" for e in exc)
    out.append(f"| {i} | {iss} | {cls} | {what} | {paths} | {b/1e9:,.2f} | {n:,} | {mt[:10]} |")
print("\n".join(out))
print()
for k, v in sorted(tot.items(), key=str):
    print(f"TOTAL {k if isinstance(k, str) else ' '.join(k)}: {v/1e9:,.2f} GB")
print(f"TOTAL all rows: {sum(v for k, v in tot.items() if isinstance(k, str))/1e9:,.2f} GB")
