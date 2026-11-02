# Apptainer Bridge Workflow

## Architecture

```
Jupiter (Booster)                    JUWELS Cluster
┌─────────────────┐                 ┌──────────────────────────┐
│ Bridge Server    │◄── SSH tunnel ─┤ Login Node (jwlogin01i)  │
│ localhost:9920   │                │                          │
│                  │                │ Reverse tunnel:          │
│ Harbor training  │                │  jwlogin:9920 → Jup:9920│
│ job (BenSkyRL)   │                └──────────┬───────────────┘
│                  │                           │
│ Booster access:  │                ┌──────────▼───────────────┐
│ 10.128.1.1:9920  │                │ 32 Compute Nodes         │
└─────────────────┘                │ Each runs:               │
                                   │  1 dispatcher (polls)    │
                                   │  16 workers (exec jobs)  │
                                   │  1 SOCKS proxy tunnel    │
                                   │  N apptainer instances   │
                                   └──────────────────────────┘
```

## Components

### Bridge Server (Jupiter)
- **Location**: `/e/project1/jureap59/marianna/ot/harbor_patched/src/harbor/environments/apptainer_bridge/server.py`
- **Port**: 9920 (0.0.0.0)
- **Start**: `nohup python3 server.py --port 9920 --host 0.0.0.0 > /tmp/apptainer_bridge_server.log 2>&1 &`
- **Status**: `curl http://localhost:9920/status`

### Workers (JUWELS)
- **Sbatch**: `/p/scratch/transfernetx/nezhurina1/apptainer_bridge/juwels_workers.sbatch`
- **Worker script**: `/p/scratch/transfernetx/nezhurina1/apptainer_bridge/worker.py`
- **Partition**: batch, Account: projectnucleus, 32 nodes, 24h
- **Submit**: `ssh juwels01 'sbatch /p/scratch/transfernetx/nezhurina1/apptainer_bridge/juwels_workers.sbatch'`
- **Auto-chain**: Workers auto-submit a successor job (MAX_CHAIN=5)

### SSH Keys
- Jupiter → JUWELS: `~/.ssh/docker_jusuf`
- JUWELS compute → JUWELS login (proxy): `~/.ssh/docker` (must exist on JUWELS home)

## Start Procedure

### 1. Start bridge server on Jupiter
```bash
BRIDGE_DIR=/e/project1/jureap59/marianna/ot/harbor_patched/src/harbor/environments/apptainer_bridge
bash $BRIDGE_DIR/start_bridge.sh --juwels
```
Or manually:
```bash
pkill -f "server.py --port 9920" || true
nohup python3 $BRIDGE_DIR/server.py --port 9920 --host 0.0.0.0 > /tmp/apptainer_bridge_server.log 2>&1 &
```

### 2. Sync worker.py to JUWELS
```bash
scp -i ~/.ssh/docker_jusuf \
    $BRIDGE_DIR/worker.py \
    nezhurina1@juwels01.fz-juelich.de:/p/scratch/transfernetx/nezhurina1/apptainer_bridge/worker.py
```

### 3. Submit workers on JUWELS
```bash
ssh -i ~/.ssh/docker_jusuf nezhurina1@juwels01.fz-juelich.de \
    'sbatch /p/scratch/transfernetx/nezhurina1/apptainer_bridge/juwels_workers.sbatch'
```

### 4. Verify
```bash
curl http://localhost:9920/status  # workers_alive=true
```

## SIF Cache

### Location
- **Primary**: `/p/scratch/transfernetx/nezhurina1/sif_cache/` (used by workers)
- **Build cache**: `/p/scratch/projectnucleus/nezhurina1/sif_cache/` (symlinked into primary)

### R2EGym (1785 tasks)
- 10 unique Docker repos, 1785 unique images (each has `/testbed` with repo checkout)
- All SIFs pre-built with `tmux` via `Bootstrap: localimage` overlay
- Marker files: `*.tmux_done` indicate rebuild complete
- Cache key: `build_r2egym-NNNN-{dockerfile_hash}.sif`

### SweSmith (2500 tasks)
- 38 unique base images, 2500 unique Dockerfiles (differ by `git checkout` branch)
- Only 38 base SIFs pre-built: `swesmith_base_{base_image_hash}.sif`
- Worker does `git checkout` at runtime via `_extract_run_commands()`
- Cache key: `swesmith_base_{sha256(base_image)[:12]}.sif`

### NL2Bash (1570 tasks)
- All share same Dockerfile (ubuntu:22.04 + seeds)
- Single SIF, symlinked for all tasks
- COPY instructions handled by worker's `_apply_dockerfile_copies()`

## Proxy (Internet Access for Containers)

JUWELS compute nodes have no internet. SweSmith test scripts need internet
for `apt-get`, `uv add`, `curl`, etc.

### How it works
1. Worker launcher starts SOCKS proxy tunnel: `ssh -D PORT user@jwlogin22i`
2. Worker bind-mounts `libproxychains4.so` and config into container
3. Worker's `exec()` prefixes commands with `export LD_PRELOAD=... PROXYCHAINS_CONF_FILE=...;`
4. All network calls inside container go through SOCKS proxy

### Prerequisites
- SSH key `~/.ssh/docker` must exist on JUWELS (copy from Jupiter's `~/.ssh/docker_jusuf`)
- `~/.local/bin/proxychains4` and `~/.local/lib/libproxychains4.so` on JUWELS
- Apptainer home dirs symlinked to scratch to avoid quota:
  ```bash
  ln -sf /p/scratch/transfernetx/nezhurina1/apptainer_instance_logs ~/.apptainer/instances/logs
  ln -sf /p/scratch/transfernetx/nezhurina1/apptainer_home_cache ~/.apptainer/cache
  ```

### R2EGym does NOT need proxy
R2EGym test scripts only run pytest locally — no internet needed.

## Worker Features

### Base Image SIF Fallback (SweSmith)
When no task-specific SIF exists, worker checks `payload.base_image`:
1. Computes `sha256(base_image)[:12]`
2. Looks for `swesmith_base_{hash}.sif` in SIF cache
3. Starts instance from base SIF
4. Runs remaining Dockerfile RUN commands (e.g., `git checkout`) at runtime

### Dockerfile COPY Replication
Worker's `_apply_dockerfile_copies()` parses COPY instructions from the
Dockerfile and replicates file movements in the workspace. This is needed
because the SIF build ignores COPY instructions.

### Bridge Health Check (30 min timeout)
`harbor_env.py` monitors `workers_alive` status. If workers go down
(e.g., between 24h JUWELS job restarts), it waits up to 30 minutes for
the new worker job to start before failing. Configurable via
`BRIDGE_WORKERS_DEAD_TIMEOUT` env var.

## Pre-building SIFs

### R2EGym
```bash
# Generate build list on Jupiter
python3 -c "..." > /tmp/r2egym_build_list.txt
scp /tmp/r2egym_build_list.txt juwels:/p/scratch/transfernetx/nezhurina1/

# Pull images (Jusuf scraper or JUWELS devel, need internet)
sbatch prebuild_r2egym_sifs.sbatch

# Add tmux overlay (JUWELS batch with proxychains, or Jusuf scraper)
sbatch rebuild_r2egym_juwels_batch.sbatch

# Or sequentially from login node
ssh juwels01 'for sif in ...; do apptainer build --disable-cache ...; done'
```

### SweSmith
```bash
# Only 38 base images needed
sbatch prebuild_swesmith_bases.sbatch  # on Jusuf scraper (has internet)
```

### Symlink to worker cache
```bash
# Workers look in /p/scratch/transfernetx/nezhurina1/sif_cache/
for f in /p/scratch/projectnucleus/nezhurina1/sif_cache/*.sif; do
    ln -sf "$f" /p/scratch/transfernetx/nezhurina1/sif_cache/$(basename $f)
done
```

## Training Scripts

### R2EGym 8B SFT on Bridge
```bash
bash /e/project1/jureap59/marianna/ot/bash-scripts/run_rl_apptainer_bridge_r2egym_sft_8b.sh
```
- Model: `laion/r2egym-nl2bash-stack-bugsseq-fixthink-again`
- 12 nodes, TP=1, max_episodes=64, timeout=3600s
- SIF cache: `/p/scratch/projectnucleus/nezhurina1/sif_cache`

### SweSmith 8B SFT on Bridge
```bash
bash /e/project1/jureap59/marianna/ot/bash-scripts/run_rl_apptainer_bridge_swesmith_sft_8b.sh       # no rope
bash /e/project1/jureap59/marianna/ot/bash-scripts/run_rl_apptainer_bridge_swesmith_sft_8b_rope.sh  # with rope scaling
```

### NL2Bash 32B on Bridge
```bash
bash /e/project1/jureap59/marianna/ot/bash-scripts/run_rl_apptainer_bridge_nl2bash_resume_step31.sh
```

## Common Issues

| Problem | Cause | Fix |
|---------|-------|-----|
| `disk quota exceeded` in `~/.apptainer` | Home dir inode/space quota | Symlink `~/.apptainer/cache` and `~/.apptainer/instances/logs` to scratch |
| `Connection refused` in container | SOCKS proxy tunnel not running | Check `~/.ssh/docker` exists on JUWELS, verify SSH to jwlogin22i works |
| `name 'log' is not defined` in worker | Using `log()` instead of `print()` | Fix in worker.py |
| `SIF build failed` for SweSmith | `base_image` not forwarded by server | Ensure `server.py` passes `base_image` in JOB_START payload |
| `No SIF found` | SIFs in projectnucleus but workers look in transfernetx | Symlink SIFs to `/p/scratch/transfernetx/nezhurina1/sif_cache/` |
| Workers not connecting | Reverse SSH tunnel down | Run `start_bridge.sh --juwels` or manually restart tunnel |
| `LD_PRELOAD` stripped by Apptainer | Security feature | Set `LD_PRELOAD` inside `bash -c` command, not via `--env` |
| Zero rewards on SweSmith | test.sh needs internet but container has none | Enable proxychains in worker (bind-mount .so + set LD_PRELOAD in command) |
| R2EGym missing tmux | SIFs built with `apptainer pull` (no %post) | Rebuild with `Bootstrap: localimage` overlay adding tmux |

## Monitoring

```bash
# Bridge status
curl http://localhost:9920/status | python3 -m json.tool

# JUWELS workers
ssh juwels01 'squeue -u nezhurina1 | grep worker'

# Worker logs
ssh juwels01 'tail -20 /p/scratch/transfernetx/nezhurina1/apptainer_workers_JOBID.out'

# Training rewards
grep "avg_raw_reward" /e/scratch/reformo/nezhurina1/experiments/logs/LOGFILE.log
```
