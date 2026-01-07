#!/usr/bin/env python3
"""
Restore original vocab sizes to fine-tuned models by padding embeddings.
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import torch
import os
import argparse
import tempfile
import shutil
import subprocess

# Mapping of fine-tuned models to their target vocab sizes
MODELS_TO_FIX = [
    # {
    #     "name": "qwen2.5-1.5b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209a1_sft_qwen2pt5_1pt5b_instruct_ot3_bsz512_lr8e_5-eb7076/hf/step-11718/",
    #     "target_vocab_size": 151936,  # Base Qwen2.5-1.5B-Instruct
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2209a1_sft_qwen2pt5_1pt5b_instruct_ot3_bsz512_lr8e_5-eb7076/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-3b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209b1_sft_qwen2pt5_3b_instruct_openthoughts3_bsz512_lr8e_5-c7d431/hf/step-11718/",
    #     "target_vocab_size": 151936,  # Base Qwen2.5-3B-Instruct
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2209b1_sft_qwen2pt5_3b_instruct_openthoughts3_bsz512_lr8e_5-c7d431/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2199b_sft_qwen2pt5_7b_instruct_openthoughts3_bsz512_lr8e_5-92772b/hf/step-11718/",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2199b_sft_qwen2pt5_7b_instruct_openthoughts3_bsz512_lr8e_5-92772b/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen3-8b-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2199c_sft_qwen3_8b_openthoughts3_bsz512_lr8e_5-accb91/hf/step-11718/",
    #     "target_vocab_size": 151936,  # Base Qwen3-8B
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2199c_sft_qwen3_8b_openthoughts3_bsz512_lr8e_5-accb91/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen3-8b-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209a2_sft_qwen2pt5_1pt5b_instruct_openthoughts4_1pt2m_qwen3_-0f8594/hf/step-11718/",
    #     "target_vocab_size": 151936,  # Base Qwen2.5-1.5B-Instruct
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2209a2_sft_qwen2pt5_1pt5b_instruct_openthoughts4_1pt2m_qwen3_-0f8594/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-3b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209b2_sft_qwen2pt5_3b_instruct_openthoughts4_1pt2m_qwen3_3b_-88f693/hf/step-11718/",
    #     "target_vocab_size": 151936,  # Base Qwen2.5-3B-Instruct
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2209b2_sft_qwen2pt5_3b_instruct_openthoughts4_1pt2m_qwen3_3b_-88f693/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2209c2_sft_qwen2pt5_7b_instruct_openthoughts4_1pt2m_qwen3_3b_-740b7d/hf/step-11718/",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2209c2_sft_qwen2pt5_7b_instruct_openthoughts4_1pt2m_qwen3_3b_-740b7d/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-32b-instruct-finetuned",
    #     "path": "gs://marin-us-central1/checkpoints/exp2209d2_sft_qwen2pt5_32b_instruct_ot4_1pt2m_qwen3_3b_bsz512_lr-86c9db/hf/step-11718/",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-32B-Instruct (different!)
    #     "output_path": "gs://marin-us-central1/checkpoints/exp2209d2_sft_qwen2pt5_32b_instruct_ot4_1pt2m_qwen3_3b_bsz512_lr-86c9db/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned",
    #     "path": "gs://marin-us-central2/checkpoints/exp2199b_redo_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-2d659d/hf/step-11718",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
    #     "output_path": "gs://marin-us-central2/checkpoints/exp2199b_redo_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-2d659d/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned",
    #     "path": "gs://marin-us-central2/checkpoints/exp2199b_redo_pt2_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-1a1aff/hf/step-10500/",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
    #     "output_path": "gs://marin-us-central2/checkpoints/exp2199b_redo_pt2_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-1a1aff/hf/step-10500-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262a_ot4_math30k_qwen3_32b_bsz128_lr4e_5-ad01cb/hf/step-1170/",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2262a_ot4_math30k_qwen3_32b_bsz128_lr4e_5-ad01cb/hf/step-1170-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262b_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-41ff16/hf/step-1170/",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2262b_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-41ff16/hf/step-1170-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned",
    #     "path": "gs://marin-us-central2/checkpoints/exp2199b_redo3_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-c05011/hf/step-11718/",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
    #     "output_path": "gs://marin-us-central2/checkpoints/exp2199b_redo3_sft_qwen2pt5_7b_instruct_ot3_bsz512_lr8e_5-c05011/hf/step-11718-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262c_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-b39be3/hf/step-3000/",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2262c_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-b39be3/hf/step-3000-padded-vocab/",
    # },
    # {
    #     "name": "qwen2.5-7b-instruct-finetuned",
    #     "path": "gs://marin-us-east5/checkpoints/exp2262e_ot4_math30k_qwen3_32b_bsz128_lr4e_5-51aefe/hf/step-2340/",
    #     "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
    #     "output_path": "gs://marin-us-east5/checkpoints/exp2262e_ot4_math30k_qwen3_32b_bsz128_lr4e_5-51aefe/hf/step-2340-padded-vocab/",
    # },
    {
        "name": "qwen2.5-7b-instruct-finetuned",
        "path": "gs://marin-us-east5/checkpoints/exp2262f_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-cfac80/hf/step-2340/",
        "target_vocab_size": 152064,  # Base Qwen2.5-7B-Instruct (different!)
        "output_path": "gs://marin-us-east5/checkpoints/exp2262f_ot4_math30k_qwen3_235b_a22b_bsz128_lr4e_5-cfac80/hf/step-2340-padded-vocab/",
    },
]

def check_gcs_path_exists(gcs_path):
    """Check if a GCS path exists."""
    result = subprocess.run(
        ["gsutil", "ls", gcs_path],
        capture_output=True,
        text=True
    )
    return result.returncode == 0

def check_remote_vocab_size(gcs_path):
    """Check vocab_size of model already in GCS without downloading."""
    config_path = os.path.join(gcs_path.rstrip('/'), 'config.json')
    
    try:
        result = subprocess.run(
            ["gsutil", "cat", config_path],
            capture_output=True,
            text=True,
            check=True
        )
        import json
        config = json.loads(result.stdout)
        return config.get('vocab_size')
    except Exception:
        return None
def download_from_gcs(gcs_path, local_path):
    """Download model from GCS to local path."""
    print(f"📥 Downloading from GCS: {gcs_path}")
    print(f"   to local: {local_path}")
    
    os.makedirs(local_path, exist_ok=True)
    ret = os.system(f"gsutil -m cp -r {gcs_path}* {local_path}/")
    
    if ret != 0:
        raise RuntimeError(f"Failed to download from GCS (exit code {ret})")
    
    print("✓ Download complete")
    
    # Check if files are in a subdirectory (happens with trailing / in GCS path)
    items = os.listdir(local_path)
    if len(items) == 1 and os.path.isdir(os.path.join(local_path, items[0])):
        # Files are in a subdirectory, move them up
        subdir = os.path.join(local_path, items[0])
        print(f"📁 Files downloaded to subdirectory, moving up from: {items[0]}")
        
        # Move all files from subdirectory to parent
        for item in os.listdir(subdir):
            src = os.path.join(subdir, item)
            dst = os.path.join(local_path, item)
            shutil.move(src, dst)
        
        # Remove empty subdirectory
        os.rmdir(subdir)
        print("✓ Files moved to correct location")
    
    # DEBUG: List what was downloaded
    print(f"\n🔍 Downloaded files:")
    for item in os.listdir(local_path):
        print(f"  - {item}")
    
    # Check if config.json exists and is valid
    config_path = os.path.join(local_path, 'config.json')
    if os.path.exists(config_path):
        import json
        with open(config_path) as f:
            config = json.load(f)
        print(f"✓ config.json exists, model_type: {config.get('model_type', 'MISSING')}")
    else:
        print(f"✗ config.json NOT FOUND at {config_path}")
        raise RuntimeError("config.json not found after download")
    
    return local_path

def pad_model_embeddings(model_path, target_vocab_size, output_path, debug=False, force=False):
    """
    Pad model embeddings to target vocab size without changing vocabulary.
    
    Args:
        model_path: Path to the fine-tuned model (can be GCS path)
        target_vocab_size: Target vocabulary size (from base model)
        output_path: Where to save the padded model (can be GCS path)
        debug: If True, save locally only and don't upload to GCS
        force: If True, reprocess even if output already exists
    """
    print(f"\n{'='*80}")
    print(f"Processing: {model_path}")
    print(f"Target vocab size: {target_vocab_size}")
    if debug:
        print("🐛 DEBUG MODE: Will save locally only, no GCS upload")
    print(f"{'='*80}\n")
    
    # Check if output already exists and has correct vocab size
    if not force and output_path.startswith("gs://"):
        if check_gcs_path_exists(output_path):
            print(f"🔍 Checking if output already exists at: {output_path}")
            remote_vocab_size = check_remote_vocab_size(output_path)
            
            if remote_vocab_size == target_vocab_size:
                print(f"✅ Output already exists with correct vocab_size={target_vocab_size}")
                print(f"✅ Skipping (use --force to reprocess)")
                return
            elif remote_vocab_size is not None:
                print(f"⚠️  Output exists but has vocab_size={remote_vocab_size}, expected {target_vocab_size}")
                print(f"   Will reprocess...")
            else:
                print(f"⚠️  Output exists but couldn't read config. Will reprocess...")
    
    # Check local output for debug mode
    if not force and debug:
        local_output = f"/tmp/padded_models/{os.path.basename(output_path.rstrip('/'))}"
        if os.path.exists(local_output):
            config_path = os.path.join(local_output, 'config.json')
            if os.path.exists(config_path):
                print(f"🔍 Checking if local output already exists at: {local_output}")
                try:
                    import json
                    with open(config_path) as f:
                        config = json.load(f)
                    local_vocab_size = config.get('vocab_size')
                    
                    if local_vocab_size == target_vocab_size:
                        print(f"✅ Local output already exists with correct vocab_size={target_vocab_size}")
                        print(f"✅ Skipping (use --force to reprocess)")
                        print(f"🐛 Model saved at: {local_output}")
                        return
                except Exception:
                    pass
    
    # Download from GCS if needed
    local_model_path = model_path
    temp_download_dir = None
    
    if model_path.startswith("gs://"):
        temp_download_dir = tempfile.mkdtemp(prefix="model_download_")
        try:
            local_model_path = download_from_gcs(model_path, temp_download_dir)
        except Exception as e:
            print(f"✗ Failed to download model: {e}")
            shutil.rmtree(temp_download_dir, ignore_errors=True)
            raise
    
    try:
        # Load model and tokenizer
        print("\nLoading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(local_model_path, trust_remote_code=True)
        config = AutoConfig.from_pretrained(local_model_path)
        
        current_vocab_size = config.vocab_size
        print(f"Current vocab size: {current_vocab_size}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Current embedding shape: {model.get_input_embeddings().weight.shape}")
        
        if current_vocab_size == target_vocab_size:
            print("✓ Vocab size already matches target. No padding needed.")
            return
        
        if current_vocab_size > target_vocab_size:
            print(f"✗ ERROR: Current vocab size ({current_vocab_size}) > target ({target_vocab_size})")
            return
        
        padding_size = target_vocab_size - current_vocab_size
        print(f"\n📝 Adding {padding_size} padding tokens...")
        
        # Resize token embeddings (adds padding with random initialization)
        model.resize_token_embeddings(target_vocab_size)
        
        # Update config
        model.config.vocab_size = target_vocab_size
        
        # Verify
        new_embedding_shape = model.get_input_embeddings().weight.shape
        print(f"\n✓ Model embedding size after resize: {new_embedding_shape}")
        print(f"✓ New vocab size in config: {model.config.vocab_size}")
        print(f"✓ Tokenizer vocab size (unchanged): {len(tokenizer)}")
        
        # Verify divisibility by 4
        assert target_vocab_size % 4 == 0, f"Target vocab size {target_vocab_size} not divisible by 4!"
        print(f"✓ Verified: {target_vocab_size} % 4 = {target_vocab_size % 4} (divisible by 4)")
        
        # Save to local path
        if debug:
            local_output = f"/tmp/padded_models/{os.path.basename(output_path.rstrip('/'))}"
        else:
            local_output = tempfile.mkdtemp(prefix="model_output_")
        
        print(f"\n💾 Saving to local path: {local_output}")
        os.makedirs(local_output, exist_ok=True)
        model.save_pretrained(local_output)
        tokenizer.save_pretrained(local_output)
        print("✓ Saved locally")
        
        # Verify saved files
        print("\n✓ Saved files:")
        for file in sorted(os.listdir(local_output)):
            file_path = os.path.join(local_output, file)
            if os.path.isfile(file_path):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  - {file} ({size_mb:.2f} MB)")
        
        # Load and verify
        print("\n🔍 Verifying saved model...")
        verify_config = AutoConfig.from_pretrained(local_output)
        print(f"  - Config vocab_size: {verify_config.vocab_size}")
        verify_model = AutoModelForCausalLM.from_pretrained(local_output, torch_dtype=torch.bfloat16)
        print(f"  - Embedding shape: {verify_model.get_input_embeddings().weight.shape}")
        print("✓ Verification passed!")
        
        if debug:
            print(f"\n🐛 DEBUG MODE: Skipping GCS upload")
            print(f"🐛 Model saved locally at: {local_output}")
            print(f"🐛 To upload manually, run:")
            print(f"🐛   gsutil -m cp -r {local_output}/* {output_path}")
        else:
            # Upload to GCS
            if output_path.startswith("gs://"):
                print(f"\n☁️  Uploading to GCS: {output_path}")
                ret = os.system(f"gsutil -m cp -r {local_output}/* {output_path}")
                if ret == 0:
                    print("✓ Upload complete")
                else:
                    print(f"✗ Upload failed with exit code {ret}")
                    return
                
                # Cleanup local output
                print("🧹 Cleaning up local files...")
                shutil.rmtree(local_output, ignore_errors=True)
                print("✓ Cleanup complete")
            else:
                # Just move if local path
                print(f"\n📦 Moving to output path: {output_path}")
                os.makedirs(output_path, exist_ok=True)
                os.system(f"cp -r {local_output}/* {output_path}/")
                shutil.rmtree(local_output, ignore_errors=True)
        
        print(f"\n✓ Successfully padded model to vocab_size={target_vocab_size}")
        if debug:
            print(f"✓ Saved locally to: {local_output}")
        else:
            print(f"✓ Saved to: {output_path}")
        print()
        
    finally:
        # Cleanup downloaded model
        if temp_download_dir and os.path.exists(temp_download_dir):
            print(f"🧹 Cleaning up temporary download directory...")
            shutil.rmtree(temp_download_dir, ignore_errors=True)

def main():
    parser = argparse.ArgumentParser(description="Pad model embeddings to restore original vocab sizes")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: save locally only, don't upload to GCS"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Process only this specific model (e.g., 'qwen2.5-1.5b-instruct-finetuned')"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output already exists"
    )
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("VOCAB SIZE PADDING SCRIPT")
    if args.debug:
        print("🐛 DEBUG MODE ENABLED")
    if args.force:
        print("⚡ FORCE MODE: Will reprocess existing outputs")
    print("="*80)
    print("\nThis script will pad fine-tuned model embeddings back to base vocab sizes")
    print("without changing the actual vocabulary or tokenizer.\n")
    
    models_to_process = MODELS_TO_FIX
    if args.model:
        models_to_process = [m for m in MODELS_TO_FIX if m["name"] == args.model]
        if not models_to_process:
            print(f"✗ ERROR: Model '{args.model}' not found in MODELS_TO_FIX")
            print(f"Available models: {[m['name'] for m in MODELS_TO_FIX]}")
            return
        print(f"Processing only: {args.model}\n")
    
    skipped = []
    processed = []
    failed = []
    
    for model_info in models_to_process:
        try:
            # Check if already done before processing
            output_exists = False
            if not args.force and model_info["output_path"].startswith("gs://"):
                if check_gcs_path_exists(model_info["output_path"]):
                    vocab_size = check_remote_vocab_size(model_info["output_path"])
                    if vocab_size == model_info["target_vocab_size"]:
                        output_exists = True
            
            if output_exists and not args.force:
                print(f"\n✅ {model_info['name']}: Already exists with correct vocab_size, skipping")
                skipped.append(model_info['name'])
                continue
            
            pad_model_embeddings(
                model_path=model_info["path"],
                target_vocab_size=model_info["target_vocab_size"],
                output_path=model_info["output_path"],
                debug=args.debug,
                force=args.force
            )
            processed.append(model_info['name'])
        except Exception as e:
            print(f"\n✗ ERROR processing {model_info['name']}: {e}\n")
            import traceback
            traceback.print_exc()
            failed.append(model_info['name'])
            continue
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if processed:
        print(f"\n✅ Processed ({len(processed)}):")
        for name in processed:
            print(f"  - {name}")
    
    if skipped:
        print(f"\n⏭️  Skipped ({len(skipped)}) - already exist with correct vocab_size:")
        for name in skipped:
            print(f"  - {name}")
    
    if failed:
        print(f"\n❌ Failed ({len(failed)}):")
        for name in failed:
            print(f"  - {name}")
    
    if args.debug and processed:
        print("\n🐛 DEBUG MODE: Models saved locally to /tmp/padded_models/")
        print("\nTo upload to GCS after verification:")
        for model_info in models_to_process:
            if model_info['name'] in processed:
                local_path = f"/tmp/padded_models/{os.path.basename(model_info['output_path'].rstrip('/'))}"
                print(f"\n  # {model_info['name']}")
                print(f"  gsutil -m cp -r {local_path}/* {model_info['output_path']}")
    elif processed:
        print("\n✅ Padded models uploaded to GCS")
    
    print("\nUpdate your model paths to use the -padded versions:")
    print("  OLD: .../step-11718/")
    print("  NEW: .../step-11718-padded-vocab/")
    print()

if __name__ == "__main__":
    main()