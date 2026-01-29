## NanoGPT_Features_v0

This PR adds 3 speedruns for sizes 150m, 270m, and 460m param models with a subset of the modded-nanogpt features. The 270m mirrors the scale of NanoGPT.
* 150m: 1.271 bpb, 2.19e17 model flops, 1.12e18 hw flops
* 270m: 1.175 bpb, 8.7e17 model flops, 3.92e18 hw flops
* 460m: 1.113 bpb, 2.8e18 model flops, 1.19e19 hw flops

One objective here is to baseline the two repos to identify speedup opportunities. As a result, I am not ablating individual changes and instead want to add enough ML features such that the remaining speed gap can be isolated to non-ML components. For this draft, I am treating the single-file hackable transformer as an experimental messy scratchpad and leaving standardization of functions like partial rope, relu^2, and json logging integration out of scope. flops_per_token is an estimate, as lambdas are treated as rounding errors.

### Features included
*  Partial RoPE. Leave half of the head dimensions stationary. Also substantially increasing rotary frequency. 
* QK Norm. Apply RMSNorm to Q and K.
* Norm after embed
* RMS Norm instead of LayerNorm
* 2.5 TPP. Replacing the default 20x from Chinchilla (This seems drastically different? Maybe this metric needs a correction factor for embed/lm_head params?)
* Relu^2 MLP. Acts as a computationally efficient version of ReGLU with tied weights between gate and up projection.
* X0 Skip. 
* exponential decay of resid. Single lambda for each layer.
* backout lambda. Model learns single param to de-weight first 2/3 layers before lm_head projection.
* reduced head counts. Roughly cutting head count in half and doubling head_dim
* 0 init out projections. (May only be relevant for first 50 steps)
* boosted attn scale. Using 1.35/sqrt(head_dim)

### Some larger modeling differences with NanoGPT
* Uses the GPT2-tokenizer with 50,000 tokens, whereas the marin-tokenizer is defaulting to 128,256 vocab-size. This means that for small models there is a substantial amount of compute locked in the lm_head projection. In terms of total param count, the 150m model has 80% of its params in the embedding and lm_head. I don't know enough about this repo yet to test other tokenizers.
* Uses fp8 on the lm_head.
* Schedule based updates. Updates the momentum terms, attention window sizes, batch size, and rotary params throughout training.
* Parameter Group specific lr. In particular, the embed is set to 75x the lr of the lm_head.
* Attention Masking. Short/Short/Short/Short/Long attention window configuration 
* Data Sampling. I am not aware yet of how this run does data sampling, but I expect differences here.

There are ~20 other minor differences that could be interesting to explore in a more scientific manner at some point.
### FLOP Gap

For forward pass flops per token (lm_head, mlp, attn) NanoGPT is (77M, 104M, 79M) = 260M, whereas this 270M parameter run is (197M, 104M, 122M) = 422M FLOPS. This run was 22 throughput/mfu whereas NanoGPT is roughly around 45 throughput/mfu. Hence, 3x speed gap.

### Notes


When I tested https://wandb.ai/marin-community/marin/runs/hacktx_300m_stdattn_4096-77b709?nw=nwuserabhinavg4 on a single H100 I got 13 MFU instead of 21 MFU, which leads me to believe either the GPU I was allocated was poor, or there is a substantial aspect here of finding architectures that are well tuned to leverage the gpu/tpu specifics of the hardware. I got more reasonable MFU on H100 when I decreased the seq_len and replaced gated SILU with Relu^2.

A large number of parameters such as learning rates, seq_len, and batch size are left unmodified across scales, so I am not infering much from performance outside the 270m run. Checking different values was left out of scope. The throughput of the 130M run dropped by 10% for the last 50% of the run, unsure why.
