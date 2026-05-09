# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
train_gpt_simple_adamh.py

This file descends from the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt).
It was prepared as a simplified version of the speedrun for use in neural net optimization research.

Differs from `train_gpt_simple.py` only in the init and the optimizer: the matrix-parameter
Muon update is replaced by AdamH (Adam preconditioning of the gradient followed by a
hyperball projection that preserves the Frobenius norm of every hidden 2D weight matrix at
every step). The residual-side projections / mlp.fc are initialised with per-module
multipliers on the default Kaiming init so AdamH has non-zero matrices to operate on from
step 0. AdamH hidden-matrix hyperparameters follow the AdamW baseline of this benchmark:
betas=(0.9, 0.95), eps=1e-8, warmup_steps=250, no weight decay (the hyperball constraint
already prevents norm growth).
"""

import os
import sys

with open(sys.argv[0]) as f:
    code = f.read()  # read the code of this file ASAP, for logging
import time
import uuid
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW

########################################
#              Dataloader              #
########################################


def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)  # header is 256 int32
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])  # number of tokens (claimed)
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())  # avoid bytes->array copy
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens


def distributed_data_generator(filename_pattern: str, batch_size: int, seq_len=1024):
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    files = sorted(Path.cwd().glob(filename_pattern))
    assert batch_size % world_size == 0
    local_batch_size = batch_size // world_size
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + batch_size + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos + rank * local_batch_size :][: local_batch_size + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += batch_size
        yield inputs.view(-1, seq_len), targets.view(-1, seq_len)


########################################
#             Architecture             #
########################################


def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return (norm(x.float()) * self.gains).type_as(x)


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncate RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        hdim = self.num_heads * self.head_dim
        self.q = Linear(dim, hdim)
        self.k = Linear(dim, hdim)
        self.v = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), scale=0.12, is_causal=True
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.fc = Linear(dim, hdim)
        self.proj = Linear(hdim, dim)

    def forward(self, x: Tensor):
        x = self.fc(x)
        x = x.relu().square()
        x = self.proj(x)
        return x


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim)
        self.mlp = MLP(dim)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


########################################
#              Optimizer               #
########################################


@torch.no_grad()
def scale_invariant_update_(param: Tensor, update: Tensor, lr: float, eps: float = 1e-10) -> None:
    """Hyperball-constrained step: take a preconditioned update of size lr * ||param||,
    then renormalise back onto the Frobenius sphere of the parameter's initial radius. Preserves
    ||param|| exactly across training; the invariant lets us drop weight decay on hidden
    matrices entirely (the constraint already prevents norm growth)."""
    p_norm = param.norm()
    u_norm = update.norm()
    new_param = param - lr * update * p_norm / torch.clamp(u_norm, min=eps)
    new_norm = torch.clamp(new_param.norm(), min=eps)
    param.copy_(new_param / new_norm * p_norm)


class AdamH(torch.optim.Optimizer):
    """AdamH: Adam-preconditioned gradient (m_t / sqrt(v_t)+eps after bias correction)
    applied via a Frobenius-norm-preserving hyperball projection. Used here for ALL hidden
    2D weight matrices -- q, k, v, mlp.fc, attn.proj, mlp.proj -- under non-zero (Kaiming-
    derived) init."""

    def __init__(self, params, lr=0.005, betas=(0.9, 0.95), eps=1e-8):
        assert isinstance(params, list) and len(params) >= 1 and isinstance(params[0], torch.nn.Parameter)
        defaults = dict(lr=lr, betas=betas, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
                bc1 = 1 - beta1**step
                bc2 = 1 - beta2**step
                update = (exp_avg / bc1) / ((exp_avg_sq / bc2).sqrt() + eps)
                scale_invariant_update_(p, update, group["lr"])


########################################
#                Setup                 #
########################################

# torchrun sets these env variables
device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
torch.cuda.set_device(device)
dist.init_process_group(backend="nccl", device_id=device)
dist.barrier()
# this code can be run equivalently with 1, 2, 4, or 8 gpus.
assert 8 % dist.get_world_size() == 0

# logging setup
if dist.get_rank() == 0:
    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{uuid.uuid4()}.txt"
    print(logfile)


def print0(s, console=False, log=True):
    if dist.get_rank() == 0:
        if console:
            print(s)
        if log:
            with open(logfile, "a") as f:
                print(s, file=f)


# we begin by logging this file itself
print0(code)
print0("=" * 100)
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0(f"Running on device_name={torch.cuda.get_device_name(device)} with world_size={dist.get_world_size()}")
print0("=" * 100)

val_tokens = 20 * 524288
batch_size = 8 * 64 * 1024
mbs = 64
train_loader = distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)
val_inputs, val_targets = next(distributed_data_generator("data/fineweb10B/fineweb_val_*.bin", val_tokens))

model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
model.compile(dynamic=False)


########################################
#       Init & Optim Hyperparams       #
########################################

# we want to minimize this while still reaching 3.28 val loss
train_steps = 4875
matrix_lr = 0.018
h_warmup_steps = 250

# initialize model parameters. Per-module multipliers on the default nn.Linear Kaiming-uniform
# init (std = 1/sqrt(3*fan_in), so ~0.0208 for fan_in=768 and ~0.0104 for fan_in=3072):
#   - attn.proj.weight (fan_in=768):  default x 1.25 -> std ≈ 0.026
#   - mlp.proj.weight  (fan_in=3072): default x 3.0  -> std ≈ 0.031
#   - mlp.fc.weight    (fan_in=768):  default x 1.5  -> std ≈ 0.031
# qkv weights keep their default init. The vocab head (proj.weight) and all "proj" biases are
# zeroed so initial logits are 0.
for name, p in model.named_parameters():
    if name.endswith(".attn.proj.weight"):
        p.data.mul_(1.25)
    elif name.endswith(".mlp.proj.weight"):
        p.data.mul_(3.0)
    elif name.endswith(".mlp.fc.weight"):
        p.data.mul_(1.5)
    elif name == "proj.weight":
        p.data.zero_()
    elif "proj" in name:
        p.data.zero_()

# create the optimizer(s)
optimizer1 = AdamW(
    [
        dict(params=[model.embed.weight], lr=0.3),
        dict(params=[model.proj.weight], lr=1 / 320),
        dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.01),
    ],
    betas=(0.8, 0.95),
    eps=1e-10,
    weight_decay=0,
    fused=True,
)
optimizer2 = AdamH([p for p in model.blocks.parameters() if p.ndim >= 2], lr=matrix_lr, betas=(0.9, 0.95), eps=1e-8)
optimizers = [optimizer1, optimizer2]
for group in optimizer1.param_groups:
    group["schedule_type"] = "aux"
for group in optimizer2.param_groups:
    group["schedule_type"] = "h"
assert set(p for opt in optimizers for group in opt.param_groups for p in group["params"]) == set(model.parameters())
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]


# learning rate schedule: linear warmup (h group only) then stable then linear decay. The h
# (AdamH) group warms up over h_warmup_steps and uses full linear cooldown over the entire
# run (cooldown_frac=1.0). The aux (AdamW) group uses no warmup and a shorter cooldown
# (cooldown_frac=0.4) to keep the embed/head learning longer before tapering.
def set_hparams(step):
    progress = step / train_steps
    assert 0 <= progress < 1
    for opt in optimizers:
        for group in opt.param_groups:
            if group["schedule_type"] == "h":
                cooldown_frac = 1.0
                if step < h_warmup_steps:
                    eta = step / h_warmup_steps
                elif progress < 1 - cooldown_frac:
                    eta = 1.0
                else:
                    eta = (1 - progress) / cooldown_frac
            else:
                cooldown_frac = 0.4
                if progress < 1 - cooldown_frac:
                    eta = 1.0
                else:
                    eta = (1 - progress) / cooldown_frac
            group["lr"] = group["initial_lr"] * eta


########################################
#        Training and Validation       #
########################################

for p in model.parameters():
    dist.broadcast(p.detach(), 0)
# start the clock
training_time = 0
dist.barrier()
t0 = time.perf_counter()
for step in range(train_steps + 1):

    # --------------- VALIDATION SECTION -----------------
    if step == train_steps or step % 125 == 0:
        # stop the clock
        dist.barrier()
        training_time += time.perf_counter() - t0
        model.eval()
        val_loss = 0
        with torch.no_grad():
            assert len(val_inputs) % mbs == 0
            for i in range(len(val_inputs) // mbs):
                val_loss += model(val_inputs[i * mbs : (i + 1) * mbs], val_targets[i * mbs : (i + 1) * mbs])
        dist.all_reduce(val_loss, op=dist.ReduceOp.SUM)
        val_loss /= val_tokens
        print0(
            f"step:{step}/{train_steps} val_loss:{val_loss:.5f} train_time:{training_time:.3f}s"
            + f" step_avg:{1000*training_time/max(step, 1):.2f}ms",
            console=True,
        )
        model.train()
        # start the clock again
        dist.barrier()
        t0 = time.perf_counter()

    if step == train_steps:
        break

    # --------------- TRAINING SECTION -----------------
    inputs, targets = next(train_loader)
    # accumulate across microbatches in case we are running with fewer than 8 gpus
    assert len(inputs) % mbs == 0
    for i in range(len(inputs) // mbs):
        model(inputs[i * mbs : (i + 1) * mbs], targets[i * mbs : (i + 1) * mbs]).backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    # set optimization hyperparameters and take a step
    set_hparams(step)
    for opt in optimizers:
        opt.step()
    model.zero_grad(set_to_none=True)
    approx_training_time = training_time + (time.perf_counter() - t0)
    print0(
        f"step:{step+1}/{train_steps} train_time:{approx_training_time:.3f}s"
        + f" step_avg:{1000*approx_training_time/(step + 1):.2f}ms",
        console=True,
        log=False,
    )

dist.destroy_process_group()
