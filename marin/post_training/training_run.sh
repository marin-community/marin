#!/bin/bash

# Updated training_run.sh with checkpoint resumption capability

# Source the checkpoint detection functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/find_checkpoint.sh"

(
    source ~/miniconda3/bin/activate llama3_train
<<<<<<< HEAD
    export RUN_NAME="llama3_8b_math_test_experiment"
    export GCLOUD_TOKEN_PATH="$HOME/.config/gcloud/application_default_credentials.json"
    export GCLOUD_PROJECT="hai-gcp-models"
=======
    export RUN_NAME="5envs_restart"
    export GCLOUD_TOKEN_PATH="$HOME/.config/gcloud/application_default_credentials.json"
    export GCLOUD_PROJECT="hai-gcp-models"
    
    # Default model paths (base model)
    DEFAULT_PARAMS="gs://marin-us-central2/checkpoints/Llama-3.1-8B-Instruct-converted/params.msgpack"
    DEFAULT_TOKENIZER="meta-llama/Meta-Llama-3-8B-Instruct"
    DEFAULT_CONFIG="gs://marin-us-central2/checkpoints/Llama-3.1-8B-Instruct-converted/config.json"
    
    # Try to find the latest checkpoint
    echo "==========================================="
    echo "Checking for existing checkpoints..."
    echo "Run name: $RUN_NAME"
    echo "==========================================="
    
    if latest_checkpoint=$(find_latest_checkpoint "$RUN_NAME"); then
        echo "✓ Found existing checkpoint: $latest_checkpoint"
        echo "Will resume training from checkpoint"
        
        # Use checkpoint paths
        PARAMS_PATH="$latest_checkpoint/params.msgpack"
        CONFIG_PATH="$latest_checkpoint/config.json"
        # Note: Keep using HuggingFace tokenizer as it doesn't change
        TOKENIZER_PATH="$DEFAULT_TOKENIZER"
        
        echo "Using checkpoint paths:"
        echo "  Params: $PARAMS_PATH"
        echo "  Config: $CONFIG_PATH"
        echo "  Tokenizer: $TOKENIZER_PATH"
        
    else
        echo "✗ No existing checkpoints found"
        echo "Will start training from base model"
        
        # Use default paths
        PARAMS_PATH="$DEFAULT_PARAMS"
        CONFIG_PATH="$DEFAULT_CONFIG"
        TOKENIZER_PATH="$DEFAULT_TOKENIZER"
        
        echo "Using base model paths:"
        echo "  Params: $PARAMS_PATH"
        echo "  Config: $CONFIG_PATH"  
        echo "  Tokenizer: $TOKENIZER_PATH"
    fi
    
    echo "==========================================="
    echo "Starting training..."
    echo "==========================================="
    
>>>>>>> ffec06b9 (auto relaunch failed experiment with the lastest checkpoint)
    python -m post_training.train \
            --load_model="paths:{
                \"params\": \"$PARAMS_PATH\",
                \"tokenizer\": \"$TOKENIZER_PATH\",
                \"config\": \"$CONFIG_PATH\"
            }" \
            --output_dir="gs://marin-us-central2/post_training/experiments/$RUN_NAME" \
            --sharding="1,4,1,-1" \
            --num_train_steps=256 \
            --max_input_length=1024 \
            --max_output_length=1025 \
            --train_bsize=32 \
            --decode_bsize=128\
            --prefill_bsize=16 \
<<<<<<< HEAD
            --reference_logprobs_bsize=256 \
            --n_prompts_per_step=16 \
            --log_freq=8 \
            --num_eval_examples=1024 \
            --save_model_freq=0 \
            --environments_path="post_training/environments.json" \
            --wandb_project="math_rloo_math_test_experiments" \
=======
            --reference_logprobs_bsize=128 \
            --n_prompts_per_step=4 \
            --log_freq=1 \
            --num_eval_examples=32 \
            --save_model_freq=5 \
            --environments_path="post_training/environments.json" \
            --wandb_project="mlebench_tpu" \
>>>>>>> ffec06b9 (auto relaunch failed experiment with the lastest checkpoint)
            --inference_param_dtype="bf16" \
            --inference_activation_dtype="bf16" \
            --training_param_dtype="fp32" \
            --training_activation_dtype="bf16" \
            --optim_config="adamw:{
                \"init_lr\": 5e-7,
                \"end_lr\": 5e-7,
                \"lr\": 5e-7,
                \"lr_warmup_steps\": 0,
                \"lr_decay_steps\": 2048,
                \"b1\": 0.9,
                \"b2\": 0.95,
                \"clip_gradient\": 1.0,
                \"weight_decay\": 0.00,
                \"bf16_momentum\": false,
                \"multiply_by_parameter_scale\": false,
                \"weight_decay_exclusions\": [],
                \"schedule\": \"cos\",
                \"grad_accum_steps\": 16
            }" \
            --logger_config="{
                \"online\": true,
                \"prefix\": \"$RUN_NAME\",
                \"prefix_to_id\": true
            }" \
            --checkpointer_config="{
                \"save_optimizer_state\": false,
                \"save_float_dtype\": \"bf16\"
            }" \
            --generation_config="{
                \"max_output_length\": 1025,
                \"temperature\": 1.0,
                \"stop_tokens\": [[524, 9399], [694, 9399], [4005, 9399], [6199, 9399], [8217, 9399], [9169, 9399], [12817, 9399], [19203, 9399], [20264, 9399], [22246, 9399], [27147, 9399], [128001]],
                \"n_generations\": 64
            }" \
            --test_generation_config="{
                \"max_output_length\": 1025,
                \"temperature\": 0.0,
                \"stop_tokens\": [[524, 9399], [694, 9399], [4005, 9399], [6199, 9399], [8217, 9399], [9169, 9399], [12817, 9399], [19203, 9399], [20264, 9399], [22246, 9399], [27147, 9399], [128001]],
                \"n_generations\": 1
            }" \
            --model_config_override="{
                \"bos_token_id\": 128000,
                \"eos_token_id\": 128001,
                \"pad_token_id\": 128002,
                \"max_sequence_length\": 2048,
                \"remat_block\": \"nothing_saveable\",
                \"resid_pdrop\": 0.00,
                \"embd_pdrop\": 0.00,
                \"attn_pdrop\": 0.00
            }" \
            --train_attention_kernel_config="splash:{
                \"block_size\": 256
            }" \
            --prefill_attention_kernel_config="splash:{
                \"block_size\": 256
            }" \
            --generate_attention_kernel_config="paged:{
                \"page_size\": 256,
                \"pages_per_compute_block\": 1,
                \"inline_seq_dim\": true,
                \"use_int8\": false
            }" \
            --pad_token_id=128002 \
            --kl_coef=1e-3
)
