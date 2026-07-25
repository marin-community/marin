ep25_pythonpath=$(find "$PWD/lib" -mindepth 2 -maxdepth 2 -type d -name src -print | paste -sd:)
env IRIS_USER=mwittmann PYTHONPATH="$ep25_pythonpath:$PWD" \
  .venv/bin/iris --cluster=marin job run --no-wait \
  --target-cluster cw-us-east-08a --priority interactive \
  --job-name ep25d2-mxfp8-numerics-20260725-v2 \
  --enable-extra-resources --gpu GB200x1 --cpu 16 --memory 96GB \
  --extra gpu \
  -- python experiments/grug/moe/standalone/check_mxfp8_expert_mlp.py \
  --out /tmp/ep25d2-mxfp8-numerics-20260725-v2.json
