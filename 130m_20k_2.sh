for i in {1..10}; do
    python3 marin/run/ray_run.py -e WANDB_API_KEY 1c85c63399be786e59026e288175122f49a434b0 -- python optimizer_sweep/exp725_cautioussweep_130M_20k.py  --force_run_failed True
done


for i in {1..10}; do
    python3 marin/run/ray_run.py -e WANDB_API_KEY 1c85c63399be786e59026e288175122f49a434b0 -- python optimizer_sweep/exp725_lionsweep_130M_20k.py  --force_run_failed True
done

for i in {1..10}; do
    python3 marin/run/ray_run.py -e WANDB_API_KEY 1c85c63399be786e59026e288175122f49a434b0 -- python optimizer_sweep/exp725_soapsweep_130M_20k.py  --force_run_failed True
done

for i in {1..10}; do
    python3 marin/run/ray_run.py -e WANDB_API_KEY 1c85c63399be786e59026e288175122f49a434b0 -- python optimizer_sweep/exp725_scionsweep_130M_20k.py  --force_run_failed True
done