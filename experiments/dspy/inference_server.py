import subprocess
import sys
import time

import click
import requests
from click import command, option


def is_package_installed(package_name: str) -> bool:
    """Check if a package is installed."""
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError:
        return False


def ensure_vllm_installed(device_type: str) -> None:
    """Ensure the correct vLLM package is installed."""
    required_package = "vllm-tpu" if device_type == "tpu" else "vllm"
    other_package = "vllm" if device_type == "tpu" else "vllm-tpu"
    
    # Check if the correct package is installed
    if is_package_installed(required_package):
        print(f"{required_package} is already installed")
        return
    
    # If wrong package is installed, uninstall it first
    if is_package_installed(other_package):
        print(f"Uninstalling {other_package}...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", other_package], check=True)
    
    # Install the correct package
    print(f"Installing {required_package}...")
    subprocess.run([sys.executable, "-m", "pip", "install", required_package], check=True)
    print(f"{required_package} installed successfully")


def start_vllm_server(model_path: str, port: int, device_type: str) -> None:
    """Start a vLLM server as a background process."""
    ensure_vllm_installed(device_type)
    command_str = f"vllm serve {model_path} --trust-remote-code --port {port}"
    print(f"Using {device_type.upper()}, starting vLLM server: {command_str}")
    
    process = subprocess.Popen(
        command_str,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    server_url = f"http://127.0.0.1:{port}/v1"
    
    # Wait for server to be ready
    while True:
        try:
            response = requests.get(f"{server_url}/models", timeout=5)
            if response.status_code == 200:
                print(f"vLLM server ready at {server_url} (PID: {process.pid})")
                break
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2)


@command(
    "host-model",
    help="Host a model on a local inference server",
)
@option("--model-path", type=str, help="Path to the model to host", required=True)
@option("--port", type=int, help="Port to host the model on", default=8000)
@option("--device-type", type=click.Choice(["gpu", "tpu"]), help="Type of device to host the model on", required=True)
def start_inference_server(model_path: str, port: int, device_type: str) -> None:
    """
    Start a vLLM inference server.
    """
    start_vllm_server(model_path, port=port, device_type=device_type)