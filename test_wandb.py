import wandb
from optimizer_sweep.utils import sweeping

wandb_ids = ['sweep-725-130m-5k47ccedlr0.016-wd0.1-minlr0-warmup2000-b10.9-b20', 'sweep-725-130m-5kcfc602lr0.008-wd0.1-minlr0-warmup2000-b10.9-b20', 'sweep-725-130m-5kab65dflr0.032-wd0.1-minlr0-warmup2000-b10.9-b20', 'sweep-725-130m-5k9c496blr0.016-wd0-minlr0-warmup2000-b10.9-b20.9', 'sweep-725-130m-5kf0868blr0.016-wd0.2-minlr0-warmup2000-b10.9-b20', 'sweep-725-130m-5kf04578lr0.016-wd0.1-minlr0.05-warmup2000-b10.9-', 'sweep-725-130m-5kf9d772lr0.016-wd0.1-minlr0.1-warmup2000-b10.9-b', 'sweep-725-130m-5k9f1fc3lr0.016-wd0.1-minlr0-warmup500-b10.9-b20.', 'sweep-725-130m-5k120f91lr0.016-wd0.1-minlr0-warmup1000-b10.9-b20', 'sweep-725-130m-5k659702lr0.016-wd0.1-minlr0-warmup4000-b10.9-b20', 'sweep-725-130m-5k1378a4lr0.016-wd0.1-minlr0-warmup2000-b10.8-b20', 'sweep-725-130m-5k777dcflr0.016-wd0.1-minlr0-warmup2000-b10.95-b2', 'sweep-725-130m-5k991af9lr0.016-wd0.1-minlr0-warmup2000-b10.98-b2', 'sweep-725-130m-5ka77155lr0.016-wd0.1-minlr0-warmup2000-b10.9-b20', 'sweep-725-130m-5kd68f97lr0.016-wd0.1-minlr0-warmup2000-b10.9-b20', 'sweep-725-130m-5k3c25eblr0.016-wd0.1-minlr0-warmup2000-b10.9-b20', 'sweep-725-130m-5k644faelr0.016-wd0.1-minlr0-warmup2000-b10.9-b20', 'sweep-725-130m-5k662391lr0.016-wd0.1-minlr0-warmup2000-b10.9-b20', 'sweep-725-130m-5k221e1flr0.016-wd0.1-minlr0-warmup2000-b10.9-b20']
baseline_id = wandb_ids[0]
print(baseline_id)
status, best_id = sweeping(wandb_ids, baseline_id)
print('Status: ', status)
print('Best id: ', best_id)