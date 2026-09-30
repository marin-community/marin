"""Per-GPU power and SM clock (busy samples) from W&B system metrics. Usage: wandb_power.py <run> [...]"""
import wandb, sys, numpy as np
api = wandb.Api()
for rid in sys.argv[1:]:
    try:
        r = api.run(f"marin-community/marin_moe/{rid}")
    except Exception as e:
        print(rid, "ERR", e); continue
    h = r.history(stream="systemMetrics", samples=5000, pandas=True)
    rt = h["_runtime"].to_numpy()
    out = [rid, f"n={len(h)}"]
    for g in range(4):
        p = h.get(f"system.gpu.{g}.powerWatts"); c = h.get(f"system.gpu.{g}.smClock"); u = h.get(f"system.gpu.{g}.gpu")
        if p is None: continue
        busy = (u.to_numpy() > 90) & (p.to_numpy() > 600)
        pp = p.to_numpy()[busy]; cc = c.to_numpy()[busy]
        out.append(f"gpu{g}: P mean {pp.mean():.0f} p10/50/90 {np.percentile(pp,10):.0f}/{np.percentile(pp,50):.0f}/{np.percentile(pp,90):.0f} W clk mean {cc.mean():.0f} p10/50/90 {np.percentile(cc,10):.0f}/{np.percentile(cc,50):.0f}/{np.percentile(cc,90):.0f} (n={busy.sum()})")
    s = r.summary
    out.append(f"mfu={s.get('throughput/mfu')}")
    print("\n   ".join(out))
