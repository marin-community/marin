(
    source ~/miniconda3/bin/activate llama3_train
    export RUN_NAME="llama3_8b_math_test_experiment"
    export GCLOUD_TOKEN_PATH="$HOME/.config/gcloud/application_default_credentials.json"
    export GCLOUD_PROJECT="hai-gcp-models"
    python -m post_training.train \
            --load_model="paths:{
                \"params\": \"gs://marin-us-central2/checkpoints/Llama-3.1-8B-Instruct-converted/params.msgpack\",
                \"tokenizer\": \"meta-llama/Meta-Llama-3-8B-Instruct\",
                \"config\": \"gs://marin-us-central2/checkpoints/Llama-3.1-8B-Instruct-converted/config.json\"
            }" \
            --output_dir="gs://marin-us-central2/experiments/math_rloo_test_experiments/" \
            --sharding="1,4,1,-1" \
            --num_train_steps=2048 \
            --max_input_length=256 \
            --max_output_length=1025 \
            --train_bsize=64 \
            --decode_bsize=1024 \
            --prefill_bsize=16 \
            --reference_logprobs_bsize=256 \
            --n_prompts_per_step=16 \
            --log_freq=8 \
            --num_eval_examples=1024 \
            --save_model_freq=0 \
            --environments_path="post_training/environments.json" \
            --wandb_project="math_rloo_math_test_experiments" \
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
