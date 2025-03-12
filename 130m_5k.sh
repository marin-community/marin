for i in {1..3}; do
    python3 marin/run/ray_run.py -e WANDB_API_KEY 1c85c63399be786e59026e288175122f49a434b0 -- python optimizer_sweep/exp725_muonsweep_130M_5k.py  --force_run_failed True
done
