"""Diagnostic script to debug tokenizer loading on the Ray cluster."""

import json
import os
import shutil
import tempfile

import fsspec


UNIFIED_TOKENIZER_PATH = "gs://marin-us-west4/tokenizers/llama3-unified-144k"

FILES = ["tokenizer_config.json", "tokenizer.json", "special_tokens_map.json"]


def check_disk_space():
    """Check disk space on relevant partitions."""
    print("=== Disk Space ===")
    for path in ["/tmp", os.path.expanduser("~"), "/"]:
        try:
            usage = shutil.disk_usage(path)
            print(f"  {path}: total={usage.total // (1024**2)}MB, "
                  f"used={usage.used // (1024**2)}MB, "
                  f"free={usage.free // (1024**2)}MB "
                  f"({usage.free / usage.total * 100:.1f}% free)")
        except Exception as e:
            print(f"  {path}: ERROR - {e}")
    print()


def check_gcs_files():
    """Check the GCS files directly."""
    print("=== GCS File Info ===")
    fs = fsspec.filesystem("gs")
    for fname in FILES:
        remote = f"marin-us-west4/tokenizers/llama3-unified-144k/{fname}"
        try:
            info = fs.info(remote)
            print(f"  {fname}: size={info['size']} bytes")
        except Exception as e:
            print(f"  {fname}: ERROR - {e}")
    print()


def test_download():
    """Test downloading each file and verify integrity."""
    print("=== Download Test ===")
    fs, path = fsspec.core.url_to_fs(UNIFIED_TOKENIZER_PATH)

    with tempfile.TemporaryDirectory() as tmpdir:
        for fname in FILES:
            remote_path = os.path.join(path, fname)
            local_path = os.path.join(tmpdir, fname)

            try:
                # Get remote size
                remote_info = fs.info(remote_path)
                remote_size = remote_info["size"]

                # Download
                fs.get(remote_path, local_path)

                # Check local size
                local_size = os.path.getsize(local_path)
                match = "OK" if local_size == remote_size else f"MISMATCH (remote={remote_size})"
                print(f"  {fname}: downloaded {local_size} bytes - {match}")

                # Validate JSON
                if fname.endswith(".json"):
                    with open(local_path) as f:
                        data = f.read()
                    json.loads(data)
                    print(f"    -> Valid JSON ({len(data)} chars)")
            except Exception as e:
                print(f"  {fname}: ERROR - {type(e).__name__}: {e}")

                # If download happened but JSON failed, show tail of file
                if os.path.exists(local_path):
                    local_size = os.path.getsize(local_path)
                    print(f"    -> Local file exists: {local_size} bytes (expected {remote_size})")
                    with open(local_path) as f:
                        content = f.read()
                    print(f"    -> Last 200 chars: {repr(content[-200:])}")
    print()


def test_load_tokenizer():
    """Test loading the tokenizer using the same code path as default_train."""
    print("=== Load Tokenizer Test ===")
    try:
        from levanter.compat.hf_checkpoints import load_tokenizer
        tok = load_tokenizer(UNIFIED_TOKENIZER_PATH)
        print(f"  SUCCESS: vocab_size={len(tok)}, type={type(tok).__name__}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")
    print()


if __name__ == "__main__":
    print(f"Python: {os.sys.version}")
    print(f"CWD: {os.getcwd()}")
    print(f"HOME: {os.path.expanduser('~')}")
    print()

    check_disk_space()
    check_gcs_files()
    test_download()
    test_load_tokenizer()
