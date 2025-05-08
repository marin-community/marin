from wandb import Api
import os

ENTITY    = "understanding-sam"
PROJECT   = "marin"
RUN_ID    = "norm_ablation_combined-d82ece"
TARGET_DIR = f"./wandb/{RUN_ID}"

api = Api()
run = api.run(f"{ENTITY}/{PROJECT}/{RUN_ID}")

os.makedirs(TARGET_DIR, exist_ok=True)
for file in run.files():                # list all uploaded files :contentReference[oaicite:0]{index=0}
    file.download(root=TARGET_DIR)      # download each into TARGET_DIR :contentReference[oaicite:1]{index=1}
