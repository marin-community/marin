"""One training recipe, three accelerators — to isolate why GPU loss trails TPU loss.

The exp153 (GPU) and exp146/exp166 (TPU) sweeps differ in a dozen incidental ways, so a
loss gap between them proves nothing. This runs a *single* code path on TPU, H100 and
GB200, changing only ``ResourceConfig``, and pins both platforms to token caches that were
verified byte-identical. Anything left is the thing we are hunting.

Deliberately simplified from exp166: no augmentation, no seeded-checkpoint machinery, no
sweep grid, no resource ladder. Short runs, loss logged every step.

``v6e-4`` and ``GB200x1`` are both **4 devices**, so they share a mesh shape, a data-parallel
width and a per-device batch. That pair is the controlled comparison; H100x8 varies the mesh
on purpose, to separate "different silicon" from "different mesh".

Every checkpoint goes to TTL'd temp storage on whichever cloud it runs on — never to a
permanent prefix.

    PLATFORM=tpu   uv run python scratch/parity_train.py --preview
    PLATFORM=h100  uv run python scratch/parity_train.py --preview
"""

import argparse
import hashlib
import os
from dataclasses import dataclass, replace

from fray.types import ANY_REGION, ResourceConfig
from levanter.data.text.datasets import BlockShuffleConfig
from levanter.layers.attention import AttentionBackend
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.config import AdamConfig
from marin.execution.lazy import ArtifactStep, lower
from marin.execution.step_runner import StepRunner
from marin.experiment.data import tokenized
from marin.experiment.train import train_lm
from marin.processing.tokenize.tokenize import TokenizedCache
from marin.rl.placement import marin_prefix_for_region
from marin.training.training import LevanterCheckpoint

# --- Recipe, copied verbatim from exp166 so the TPU side is the known-good one ----------

MODEL_CONFIG = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
    rope=Llama3RotaryEmbeddingsConfig(),
)
SEQ_LEN = 8192
DATA_SEED = 0
SHUFFLE = BlockShuffleConfig(io_block_size=256, window_blocks=512, perm_type="feistel")
WARMUP = 0.1
LR_SCHEDULE = "cosine"
# Overridable so the hyperparameter ablation reuses this harness byte-for-byte instead of a
# forked copy: exp153 ran lr 1e-3 / wd 0.8, exp146's winner ran lr 3.1623e-3 / wd 0.2.
LEARNING_RATE = float(os.environ.get("LR", 1e-3))
WEIGHT_DECAY = float(os.environ.get("WD", 0.2))
BATCH_SIZE = 32  # divides 4 (v6e-4, GB200x1) and 8 (H100x8) so dp is the full mesh everywhere

TOKENIZER = "timodonnell/contacts-v1-tokenizer@5d68a24a899f"
TEXT_KEY = "document"

# Caches verified byte-identical on 2026-07-29 (scratch/compare_caches.py). Pinned rather
# than resolved through marin_prefix(), so neither platform can silently tokenize its own.
# The TPU region is pinned because marin refuses a cross-region GCS read (correctly --
# that is the expensive direction). All of marin-dev's v6e regions hold a contacts-v1 cache;
# europe-west4 is the default because it is a v6e zone with good v6e-4 availability.
#
# Resolve the bucket through marin_prefix_for_region rather than f"marin-{region}": the
# mapping is not mechanical (europe-west4 lives in gs://marin-eu-west4), and guessing it
# cost a scheduling round.
TPU_REGION = os.environ.get("TPU_REGION", "europe-west4")
TPU_BUCKET = marin_prefix_for_region(TPU_REGION)

TRAIN_CACHE = {
    "gcs": f"{TPU_BUCKET}/tokenized/contacts-v1/2026.07.13.1",
    "s3": "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1/2026.07.25",
}
VAL_CACHE = {
    "gcs": f"{TPU_BUCKET}/tokenized/contacts-v1-val/2026.07.13.1",
    "s3": "s3://marin-us-east-02a/MarinFold/exp154_qwen_contacts_v1/tokenized/contacts-v1-val/2026.07.25",
}

CHECKPOINT_TTL_DAYS = 1
WANDB_GROUP = "exp153-tpu-gpu-parity"


@dataclass(frozen=True)
class Platform:
    """One accelerator target: its resources, its cloud, and its attention kernel."""

    name: str
    cloud: str  # "gcs" or "s3" -- selects the pinned cache and the temp bucket
    devices: int
    cluster: str  # the iris --target-cluster (TPU runs go to marin-dev directly)
    resources: ResourceConfig
    attn: AttentionBackend | None


PLATFORMS: dict[str, Platform] = {
    # 4 chips. The TPU reference; runs on marin-dev, not marin.
    "tpu": Platform(
        name="v6e-4",
        cloud="gcs",
        devices=4,
        cluster="marin-dev",
        resources=ResourceConfig.with_tpu("v6e-4", cpu=32, ram="128g", disk="50g", regions=[TPU_REGION]),
        attn=None,  # TPU default (splash); the known-good path
    ),
    # 8 GPUs -- different mesh from the TPU on purpose.
    "h100": Platform(
        name="H100x8",
        cloud="s3",
        devices=8,
        cluster="cw-us-east-02a",
        resources=ResourceConfig.with_gpu("H100", count=8, cpu=32, ram="256g", disk="256g", regions=[ANY_REGION]),
        attn=AttentionBackend.JAX_FLASH,
    ),
    # 4 GPUs -- same device count and mesh as v6e-4. The controlled comparison.
    "gb200": Platform(
        name="GB200x4",
        cloud="s3",
        devices=4,
        cluster="cw-us-east-08a",
        resources=ResourceConfig.with_gpu("GB200", count=4, cpu=32, ram="256g", disk="256g", regions=[ANY_REGION]),
        attn=AttentionBackend.JAX_FLASH,
    ),
}


def platform() -> Platform:
    key = os.environ.get("PLATFORM", "").strip().lower()
    if key not in PLATFORMS:
        raise SystemExit(f"set PLATFORM to one of {sorted(PLATFORMS)}; got {key!r}")
    return PLATFORMS[key]


def env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def attention_override() -> AttentionBackend | None:
    """``ATTN`` forces a backend, so the TPU kernel can be compared against the GPU one."""
    raw = os.environ.get("ATTN", "").strip().upper()
    return AttentionBackend[raw] if raw else None


def run_id(plat: Platform, steps: int, attn: AttentionBackend | None) -> str:
    """Identity carries everything that could change the curve, so runs never collide."""
    bits = f"{plat.name}-b{BATCH_SIZE}-s{steps}-lr{LEARNING_RATE:g}-wd{WEIGHT_DECAY:g}-{attn or 'default'}"
    digest = hashlib.sha256(bits.encode()).hexdigest()[:6]
    hp = f"-lr{LEARNING_RATE:g}-wd{WEIGHT_DECAY:g}".replace(".", "p")
    return f"parity-{plat.name.lower()}-b{BATCH_SIZE}-s{steps}{hp}-{digest}"


# Explicit temp roots per cloud. ``marin_temp_bucket`` resolves from ambient config, which
# on this dev box yields a local file:// path and inside a CoreWeave pod picks up the GCS
# cluster config -- neither is what we want written. Both prefixes below were verified to
# carry a real 1-day delete rule: S3 reports an x-amz-expiration date naming marin-ttl-1d,
# and gs://marin-us-east5 has "Delete age=1 matchesPrefix=[tmp/ttl=1d/]".
TEMP_ROOT = {
    "gcs": f"{TPU_BUCKET}/tmp/ttl={CHECKPOINT_TTL_DAYS}d/parity",
    "s3": f"s3://marin-us-east-02a/tmp/ttl={CHECKPOINT_TTL_DAYS}d/parity",
}


def output_path(plat: Platform, name: str) -> str:
    """TTL'd temp storage on this platform's own cloud. Never a permanent prefix."""
    return f"{TEMP_ROOT[plat.cloud]}/{name}"


def cache(plat: Platform, split: str) -> ArtifactStep[TokenizedCache]:
    pinned = (TRAIN_CACHE if split == "train" else VAL_CACHE)[plat.cloud]
    return tokenized(
        name=f"tokenized/contacts-v1{'' if split == 'train' else '-val'}",
        tokenizer=TOKENIZER,
        version="2026.07.13.1" if plat.cloud == "gcs" else "2026.07.25",
        paths=[pinned],
        text_key=TEXT_KEY,
        validation=split != "train",
        pin=pinned,
    )


def build(plat: Platform, steps: int, eval_every: int, eval_batches: int) -> ArtifactStep[LevanterCheckpoint]:
    attn = attention_override() or plat.attn
    model = replace(MODEL_CONFIG, attn_backend=attn) if attn is not None else MODEL_CONFIG
    name = run_id(plat, steps, attn)

    step = train_lm(
        name=f"parity/{name}",
        model=model,
        optimizer=AdamConfig(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup=WARMUP,
            lr_schedule=LR_SCHEDULE,
        ),
        datasets={cache(plat, "train"): 1.0},
        validation=[cache(plat, "val")],
        batch_size=BATCH_SIZE,
        seq_len=SEQ_LEN,
        num_train_steps=steps,
        z_loss_weight=None,
        evals=None,
        resources=plat.resources,
        version="2026.07.29",
        steps_per_eval=eval_every,
        wandb_project=os.environ.get("WANDB_PROJECT", "marin"),
        wandb_group=WANDB_GROUP,
        run_id=name,
        tags=[
            "parity",
            f"platform={plat.name}",
            f"devices={plat.devices}",
            f"attn={attn or 'default'}",
            f"batch={BATCH_SIZE}",
            f"steps={steps}",
        ],
        env_vars={"WANDB_ENTITY": os.environ["WANDB_ENTITY"]} if os.environ.get("WANDB_ENTITY") else None,
    )

    base_build = step.build_config

    def build_config(ctx):
        pod = base_build(ctx)
        trainer = replace(
            pod.train_config.trainer,
            max_eval_batches=eval_batches,  # cheap evals, run often
            log_xla_hlo=False,
            checkpointer=replace(pod.train_config.trainer.checkpointer, keep=None),
        )
        data = replace(
            pod.train_config.data,
            shuffle=SHUFFLE,
            components={k: replace(c, pack=True) for k, c in pod.train_config.data.components.items()},
        )
        train_config = replace(pod.train_config, trainer=trainer, data=data, data_seed=DATA_SEED)
        return replace(pod, train_config=train_config)

    return replace(step, build_config=build_config, override_path=output_path(plat, name))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="print the resolved plan, submit nothing")
    args = parser.parse_args()

    plat = platform()
    steps = env_int("STEPS", 60)
    eval_every = env_int("EVAL_EVERY", 10)
    eval_batches = env_int("EVAL_BATCHES", 4)
    attn = attention_override() or plat.attn
    name = run_id(plat, steps, attn)
    out = output_path(plat, name)

    if args.preview:
        print(
            f"PARITY [{plat.name}] cloud={plat.cloud} devices={plat.devices} cluster={plat.cluster}\n"
            f"  run_id={name}\n"
            f"  model=1.5B qwen3 seq={SEQ_LEN} batch={BATCH_SIZE} "
            f"per_device={BATCH_SIZE // plat.devices}\n"
            f"  lr={LEARNING_RATE:g} wd={WEIGHT_DECAY} warmup={WARMUP} sched={LR_SCHEDULE} seed={DATA_SEED}\n"
            f"  attn={attn or 'accelerator default'}\n"
            f"  steps={steps} eval_every={eval_every} eval_batches={eval_batches}\n"
            f"  train_cache={TRAIN_CACHE[plat.cloud]}\n"
            f"  output={out}",
            flush=True,
        )
        if not out.startswith(("gs://", "s3://")) or "/tmp/ttl=" not in out:
            raise SystemExit(f"REFUSING: output {out} is not TTL'd temp storage")
        return

    if "/tmp/ttl=" not in output_path(plat, name):
        raise SystemExit("REFUSING: checkpoints would not go to TTL'd temp storage")
    StepRunner().run([lower(build(plat, steps, eval_every, eval_batches))])


if __name__ == "__main__":
    main()
