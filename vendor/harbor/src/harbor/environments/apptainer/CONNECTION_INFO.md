# JSC Cluster Connection Info

## Jupiter (LOCAL machine - no SSH needed)
- We are ON Jupiter. All commands run locally.
- `squeue -u $USER` shows all jobs (including Booster partition).
- Booster is a PARTITION on Jupiter, not a separate cluster.
- Booster partition: `--partition=booster` (GPU nodes, e.g. jpbo-*)
- Job logs: `/e/scratch/jureap59/ot/users/nezhurina1/experiments/logs/`
- Scratch: `/e/scratch/jureap59/ot/`
- Project: `/e/project1/jureap59/marianna/ot/`

## JUWELS Cluster (SEPARATE cluster - SSH required)
- Host: juwels01.fz-juelich.de
- SSH key: <ssh_key>
- Username: nezhurina1
- Example: `ssh -i  <ssh_key> nezhurina1@juwels01.fz-juelich.de`
- Partition: `batch` (CPU nodes, e.g. jwc08n*)
- Worker logs: `/p/scratch/transfernetx/nezhurina1/apptainer_workers_*.out`
- SIF cache: `/p/scratch/transfernetx/nezhurina1/sif_cache/`
- Apptainer staging: `/tmp/apptainer_staging/`

## Bridge Architecture (server-on-Jupiter, per-node dispatcher)

```
Jupiter (bridge server, localhost:9920)
    ↑ reverse SSH tunnel
JUWELS login node (localhost:9920)
    ↑ direct access (same network)
JUWELS compute nodes:
    [1 dispatcher/node] → polls localhost:9920 → distributes to local queue
    [N workers/node]    → consume from local queue (NO HTTP polling)
    [workers]           → POST results directly to localhost:9920
```

- Bridge server runs ON Jupiter (fast for Harbor client, no tunnel overhead)
- Reverse SSH tunnel: JUWELS login:9920 → Jupiter:9920
- Per-node dispatcher reduces tunnel traffic: 32 pollers (not 512)
- Workers never touch HTTP for polling — only for posting results
- Harbor client accesses localhost:9920 directly (or 10.128.1.1:9920 from Booster)

### Setup
1. `bash start_bridge.sh --juwels` (from Jupiter — starts server + reverse tunnel)
2. `sbatch juwels_workers.sbatch` (on JUWELS — starts dispatchers + workers)
3. Set `APPTAINER_BRIDGE_URL=http://localhost:9920` in RL training config
