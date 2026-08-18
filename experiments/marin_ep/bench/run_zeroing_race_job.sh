#!/bin/bash
# Iris-job driver for repro_fused2_zeroing_race.py on one GB200 tray.
# Sets up the GPU env, runs the 4-process repro with user-triggered core
# dumps armed, and on a wedge dumps the stuck rank and prints the transport
# warp's registers (spin: R6=current semaphore value, R7=expected).
set -u
cd /app
NIGHTLY=https://us-python.pkg.dev/ml-oss-artifacts-published/jax-public-nightly-artifacts-registry
uv sync --all-packages --extra=gpu > /tmp/uvsync.log 2>&1
uv pip install \
  "$NIGHTLY/jax/jax-0.11.1.dev20260809-py3-none-any.whl" \
  "$NIGHTLY/jaxlib/jaxlib-0.11.1.dev20260809-cp312-cp312-manylinux_2_27_aarch64.whl" \
  "$NIGHTLY/jax-cuda13-plugin/jax_cuda13_plugin-0.11.1.dev20260809-cp312-cp312-manylinux_2_27_aarch64.whl" \
  "$NIGHTLY/jax-cuda13-pjrt/jax_cuda13_pjrt-0.11.1.dev20260809-py3-none-manylinux_2_27_aarch64.whl" \
  >> /tmp/uvsync.log 2>&1
uv pip uninstall nvidia-nccl-cu12 >> /tmp/uvsync.log 2>&1 || true
uv pip install --no-deps --reinstall nvidia-nccl-cu13==2.30.7 >> /tmp/uvsync.log 2>&1
uv run --no-sync python -c "import jax; print('jax', jax.__version__)" || { cat /tmp/uvsync.log | tail -20; exit 1; }
echo SETUP_OK

DUMPDIR=/tmp/gpudumps; mkdir -p $DUMPDIR
pids=()
for i in 0 1 2 3; do
  MARIN_EP_COORD=127.0.0.1:9971 MARIN_EP_NUM_PROCS=4 MARIN_EP_PROC_ID=$i \
  CUDA_VISIBLE_DEVICES=$i \
  CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 \
  CUDA_COREDUMP_PIPE=$DUMPDIR/corepipe_%p \
  CUDA_COREDUMP_FILE=$DUMPDIR/core_%p.nvcudmp \
  CUDA_COREDUMP_GENERATION_FLAGS=skip_global_memory,skip_abort \
    timeout 2400 uv run --no-sync python experiments/marin_ep/bench/repro_fused2_zeroing_race.py 12 \
    > /tmp/repro_$i.log 2>&1 &
  pids+=($!)
done

# Watch: wedge = some ranks REPRO_COMPLETED while others stall >4 min after
# the last global progress line.
for t in $(seq 1 120); do
  sleep 20
  alive=0
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
  [ "$alive" -eq 0 ] && break
  done_n=$(grep -l REPRO_COMPLETED /tmp/repro_*.log 2>/dev/null | wc -l)
  if [ "$done_n" -gt 0 ] && [ "$alive" -gt 0 ]; then
    sleep 240
    still=0
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && still=$((still+1)); done
    if [ "$still" -gt 0 ]; then
      echo "WEDGE_SUSPECT: $done_n done, $still stuck"
      nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader
      P=$(ls $DUMPDIR/corepipe_* 2>/dev/null | head -1)
      for P in $(ls $DUMPDIR/corepipe_* 2>/dev/null); do
        timeout 10 bash -c "echo 1 > '$P'" && echo "triggered $P" || echo "trigger timeout $P (proc done)"
      done
      sleep 40
      TOOLDIR=/tmp/cudagdb
      if [ ! -x $TOOLDIR/bin/cuda-gdb ]; then
        mkdir -p $TOOLDIR; cd $TOOLDIR
        BASE=https://developer.download.nvidia.com/compute/cuda/redist
        for pkg in cuda_gdb cuda_cuobjdump cuda_nvdisasm; do
          curl -fsSL -o $pkg.tar.xz "$BASE/$pkg/linux-sbsa/$pkg-linux-sbsa-13.2.86-archive.tar.xz" && tar xJf $pkg.tar.xz
        done
        mkdir -p bin; cp -a cuda_*-archive/bin/* bin/ 2>/dev/null
        cd /app
      fi
      export PATH=$TOOLDIR/bin:$PATH
      for D in $(ls $DUMPDIR/core_*.nvcudmp 2>/dev/null); do
        echo "==== DUMP $D"
        cuda-gdb -q -batch -ex "target cudacore $D" \
          -ex "info cuda kernels" \
          -ex "cuda block (0,0,0) thread (260,0,0)" -ex "bt" -ex "info registers" \
          -ex "info cuda threads" 2>/dev/null | head -60
      done
      echo WEDGE_CONFIRMED
      pkill -f repro_fused2_zeroing_race || true
      break
    fi
  fi
done
for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done
echo "--- per-rank tails"
for i in 0 1 2 3; do echo "-- rank $i"; tail -4 /tmp/repro_$i.log; done
echo DRIVER_DONE
