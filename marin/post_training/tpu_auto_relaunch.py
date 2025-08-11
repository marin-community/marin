#!/usr/bin/env python3
"""
TPU Auto-Relaunch Script

This script monitors TPU instances for preemptions and automatically relaunches
failed jobs on new TPU instances.
"""

import os
import re
import sys
import json
import time
import subprocess
import logging
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tpu_auto_relaunch.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TPUInstance:
    name: str
    zone: str
    accelerator_type: str
    cores: int
    state: str

class TPUManager:
    def __init__(self, launcher_script_path: str = "launcher.py", training_script: str = "training_run.sh"):
        self.launcher_script_path = launcher_script_path
        self.training_script = training_script
        self.current_tpu = None
        self.job_running = False
        
    def run_command(self, command: str, capture_output: bool = True) -> Tuple[int, str, str]:
        """Run a shell command and return (returncode, stdout, stderr)"""
        logger.debug(f"Running command: {command}")
        try:
            result = subprocess.run(
                command, 
                shell=True, 
                capture_output=capture_output,
                text=True,
                timeout=300  # 5 minute timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out: {command}")
            return -1, "", "Command timed out"
        except Exception as e:
            logger.error(f"Error running command: {e}")
            return -1, "", str(e)

    def list_available_tpus(self, zones: List[str] = None) -> List[TPUInstance]:
        """List all available TPUs with v5p accelerator type"""
        if zones is None:
            zones = ["us-central1-a", "us-central1-b", "us-central1-c", "us-east5-a"]
        
        all_tpus = []
        for zone in zones:
            command = f'gcloud compute tpus tpu-vm list --zone={zone} --filter="acceleratorType:v5p" --format="json"'
            returncode, stdout, stderr = self.run_command(command)
            
            if returncode == 0 and stdout.strip():
                try:
                    tpus_data = json.loads(stdout)
                    for tpu_data in tpus_data:
                        # Extract core count from accelerator type (e.g., "v5p-8" -> 8)
                        accelerator_type = tpu_data.get('acceleratorType', '')
                        cores_match = re.search(r'v5p-(\d+)', accelerator_type)
                        cores = int(cores_match.group(1)) if cores_match else 0
                        
                        # Extract just the TPU name (remove any path prefixes)
                        full_name = tpu_data['name']
                        tpu_name = full_name.split('/')[-1]  # Get last part after any slashes
                        
                        tpu = TPUInstance(
                            name=tpu_name,
                            zone=zone,
                            accelerator_type=accelerator_type,
                            cores=cores,
                            state=tpu_data.get('state', 'UNKNOWN')
                        )
                        if tpu.state == 'READY':
                            all_tpus.append(tpu)
                            logger.debug(f"Found TPU: {tpu_name} in {zone} with {cores} cores")
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse TPU list JSON for zone {zone}")
            else:
                logger.warning(f"Failed to list TPUs in zone {zone}: {stderr}")
        
        # Sort by cores (descending) for priority selection
        all_tpus.sort(key=lambda x: x.cores, reverse=True)
        logger.info(f"Found {len(all_tpus)} available TPUs")
        return all_tpus

    def is_tpu_accessible(self, tpu_name: str, zone: str) -> bool:
        """Check if TPU is accessible via SSH"""
        command = f'gcloud compute tpus tpu-vm ssh {tpu_name} --zone {zone} --command "echo tpu_accessible" --ssh-flag="-o ConnectTimeout=10"'
        returncode, stdout, stderr = self.run_command(command)
        
        if returncode == 0:
            logger.info(f"TPU {tpu_name} is accessible")
            return True
        
        # Check for specific preemption indicators
        preemption_indicators = [
            "NOT_FOUND",
            "does not exist",
            "Connection refused",
            "Connection timed out",
            "No route to host"
        ]
        
        error_text = stderr + stdout
        for indicator in preemption_indicators:
            if indicator in error_text:
                logger.warning(f"TPU {tpu_name} appears to be preempted: {indicator}")
                return False
        
        logger.warning(f"TPU {tpu_name} SSH failed but unclear if preempted: {error_text}")
        return False

    def update_launcher_script(self, new_tpu: TPUInstance):
        """Update the launcher.py script with new TPU information"""
        try:
            with open(self.launcher_script_path, 'r') as f:
                content = f.read()
            
            # Find the available_tpus list and update it
            pattern = r'available_tpus = \[(.*?)\]'
            match = re.search(pattern, content, re.DOTALL)
            
            if match:
                # Add the new TPU to the list, removing any duplicates first
                new_tpu_entry = f'("{new_tpu.name}", "{new_tpu.zone}")'
                
                # Extract existing entries
                existing_entries = re.findall(r'\("([^"]+)", "([^"]+)"\)', match.group(1))
                
                # Remove duplicate if exists
                filtered_entries = [(name, zone) for name, zone in existing_entries 
                                  if name != new_tpu.name]
                
                # Add new TPU at the beginning (highest priority)
                all_entries = [(new_tpu.name, new_tpu.zone)] + filtered_entries
                
                new_tpu_list = ',\n        '.join([f'("{name}", "{zone}")' for name, zone in all_entries])
                new_available_tpus = f'available_tpus = [\n        {new_tpu_list},\n    ]'
                
                updated_content = re.sub(pattern, new_available_tpus, content, flags=re.DOTALL)
                
                # Backup original file
                backup_path = f"{self.launcher_script_path}.backup"
                with open(backup_path, 'w') as f:
                    f.write(content)
                
                # Write updated content
                with open(self.launcher_script_path, 'w') as f:
                    f.write(updated_content)
                
                logger.info(f"Updated launcher script with TPU {new_tpu.name}")
                return True
            else:
                logger.error("Could not find available_tpus list in launcher script")
                return False
                
        except Exception as e:
            logger.error(f"Error updating launcher script: {e}")
            return False

    def setup_tpu(self, tpu_name: str) -> bool:
        """Run setup on the specified TPU"""
        logger.info(f"Setting up TPU {tpu_name}")
        
        # Ensure checkpoint detection script is executable
        try:
            subprocess.run(["chmod", "+x", "find_checkpoint.sh"], check=False)
        except:
            pass
        
        # First update the launcher script with the new TPU
        if self.current_tpu:
            if not self.update_launcher_script(self.current_tpu):
                logger.error("Failed to update launcher script before setup")
                return False
        command = f"python {self.launcher_script_path} setup --project={tpu_name}"
        returncode, stdout, stderr = self.run_command(command, capture_output=False)
        
        if returncode == 0:
            logger.info(f"Successfully set up TPU {tpu_name}")
            
            # Verify checkpoint detection script is available on TPU
            zone = self.current_tpu.zone if self.current_tpu else "us-central1-a"
            check_cmd = f'gcloud compute tpus tpu-vm ssh {tpu_name} --zone {zone} --command "test -f ~/llama3_train/post_training/find_checkpoint.sh && echo checkpoint_script_found"'
            ret, out, _ = self.run_command(check_cmd)
            if ret == 0 and "checkpoint_script_found" in out:
                logger.info("Checkpoint detection script verified on TPU")
            else:
                logger.warning("Checkpoint detection script may not be available on TPU")
            
            return True
        else:
            logger.error(f"Failed to set up TPU {tpu_name}: {stderr}")
            return False

    def check_for_checkpoints(self, tpu_name: str, run_name: str = None) -> Dict[str, str]:
        """Check for existing checkpoints via SSH to TPU"""
        if not run_name:
            # Try to extract RUN_NAME from training script
            try:
                with open(self.training_script, 'r') as f:
                    content = f.read()
                    import re
                    match = re.search(r'RUN_NAME="([^"]+)"', content)
                    if match:
                        run_name = match.group(1)
            except:
                logger.warning("Could not extract RUN_NAME from training script")
        
        if not run_name:
            logger.warning("No RUN_NAME provided, skipping checkpoint check")
            return {"found": False}
        
        logger.info(f"Checking for existing checkpoints for run: {run_name}")
        
        # Run checkpoint detection on the TPU
        checkpoint_cmd = f'source ~/miniconda3/bin/activate llama3_train && cd ~/llama3_train/post_training && bash find_checkpoint.sh info "{run_name}"'
        returncode, stdout, stderr = self.run_command(
            f'gcloud compute tpus tpu-vm ssh {tpu_name} --zone {self.current_tpu.zone if self.current_tpu else "us-central1-a"} --command "{checkpoint_cmd}"'
        )
        
        if returncode == 0 and "CHECKPOINT_FOUND=true" in stdout:
            # Parse the checkpoint info
            info = {}
            for line in stdout.strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    info[key.lower()] = value
            
            logger.info(f"Found existing checkpoint: {info.get('checkpoint_path', 'unknown')}")
            logger.info(f"Step number: {info.get('step_number', 'unknown')}")
            return info
        else:
            logger.info("No existing checkpoints found")
            return {"checkpoint_found": "false"}

    def launch_job(self, tpu_name: str) -> bool:
        """Launch training job on the specified TPU"""
        logger.info(f"Launching job on TPU {tpu_name}")
        
        # Ensure launcher script is updated with current TPU before launching
        if self.current_tpu:
            if not self.update_launcher_script(self.current_tpu):
                logger.error("Failed to update launcher script before launch")
                return False
        
        # Check for existing checkpoints before launching
        checkpoint_info = self.check_for_checkpoints(tpu_name)
        if checkpoint_info.get("checkpoint_found") == "true":
            logger.info("Training will resume from existing checkpoint")
        else:
            logger.info("Training will start from base model")
        
        command = f"python {self.launcher_script_path} launch {self.training_script} --project={tpu_name}"
        returncode, stdout, stderr = self.run_command(command, capture_output=False)
        
        if returncode == 0:
            logger.info(f"Successfully launched job on TPU {tpu_name}")
            self.job_running = True
            return True
        else:
            logger.error(f"Failed to launch job on TPU {tpu_name}: {stderr}")
            return False

    def find_best_replacement_tpu(self, failed_tpu_name: str) -> Optional[TPUInstance]:
        """Find the best replacement TPU (highest core count, different from failed one)"""
        available_tpus = self.list_available_tpus()
        
        # Filter out the failed TPU
        replacement_candidates = [tpu for tpu in available_tpus if tpu.name != failed_tpu_name]
        
        if not replacement_candidates:
            logger.error("No replacement TPUs available")
            return None
        
        # Return the TPU with the most cores
        best_tpu = replacement_candidates[0]  # Already sorted by cores descending
        logger.info(f"Selected replacement TPU: {best_tpu.name} ({best_tpu.cores} cores)")
        return best_tpu

    def handle_tpu_preemption(self) -> bool:
        """Handle TPU preemption by finding replacement and relaunching"""
        logger.info("Handling TPU preemption...")
        
        replacement_tpu = self.find_best_replacement_tpu(self.current_tpu.name if self.current_tpu else "")
        if not replacement_tpu:
            return False
        
        # Update current TPU reference BEFORE setup/launch
        self.current_tpu = replacement_tpu
        logger.info(f"Selected replacement TPU: {replacement_tpu.name} in zone {replacement_tpu.zone}")
        
        # Update launcher script with new TPU
        if not self.update_launcher_script(replacement_tpu):
            return False
        
        # Setup new TPU (this will also update launcher script again to be sure)
        if not self.setup_tpu(replacement_tpu.name):
            return False
        
        # Launch job on new TPU
        if not self.launch_job(replacement_tpu.name):
            return False
        
        return True

    def monitor_and_relaunch(self, initial_tpu_name: str = None, check_interval: int = 60):
        """Main monitoring loop"""
        logger.info("Starting TPU monitoring and auto-relaunch service")
        
        # Initialize with current TPU or find one
        if initial_tpu_name:
            # Find the zone for the initial TPU
            available_tpus = self.list_available_tpus()
            current_tpu_info = next((tpu for tpu in available_tpus if tpu.name == initial_tpu_name), None)
            if current_tpu_info:
                self.current_tpu = current_tpu_info
                logger.info(f"Using specified TPU: {self.current_tpu.name}")
                # Ensure the initial TPU is in the launcher script
                if not self.update_launcher_script(self.current_tpu):
                    logger.error("Failed to update launcher script with initial TPU")
                    return
            else:
                logger.error(f"Initial TPU {initial_tpu_name} not found in available TPUs")
                return
        else:
            # Find the best available TPU
            available_tpus = self.list_available_tpus()
            if not available_tpus:
                logger.error("No TPUs available to start with")
                return
            self.current_tpu = available_tpus[0]
            logger.info(f"Starting with TPU: {self.current_tpu.name}")
            # Ensure the selected TPU is in the launcher script
            if not self.update_launcher_script(self.current_tpu):
                logger.error("Failed to update launcher script with selected TPU")
                return

        # Run setup on the initial TPU BEFORE the monitoring loop
        logger.info("Running initial setup on the selected TPU...")
        if not self.setup_tpu(self.current_tpu.name):
            logger.error("Initial TPU setup failed. Please check logs. Aborting.")
            return
        logger.info("Initial setup complete.")            
        
        consecutive_failures = 0
        max_consecutive_failures = 3
        
        while True:
            try:
                if self.current_tpu and self.is_tpu_accessible(self.current_tpu.name, self.current_tpu.zone):
                    logger.info(f"TPU {self.current_tpu.name} is healthy")
                    consecutive_failures = 0
                    
                    # If no job is running, try to launch one
                    if not self.job_running:
                        logger.info("No job running, attempting to launch...")
                        self.launch_job(self.current_tpu.name)
                    
                else:
                    logger.warning(f"TPU {self.current_tpu.name} is not accessible - attempting recovery")
                    consecutive_failures += 1
                    
                    if consecutive_failures >= max_consecutive_failures:
                        logger.error(f"Maximum consecutive failures ({max_consecutive_failures}) reached. Stopping.")
                        break
                    
                    self.job_running = False
                    if not self.handle_tpu_preemption():
                        logger.error("Failed to handle TPU preemption")
                        time.sleep(check_interval * 2)  # Wait longer on failure
                        continue
                    
                    consecutive_failures = 0  # Reset on successful recovery
                
                logger.info(f"Waiting {check_interval} seconds before next check...")
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal. Shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in monitoring loop: {e}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.error("Too many consecutive errors. Stopping.")
                    break
                time.sleep(check_interval)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="TPU Auto-Relaunch Script")
    parser.add_argument("--tpu-name", help="Initial TPU name to monitor")
    parser.add_argument("--launcher-script", default="launcher.py", help="Path to launcher script")
    parser.add_argument("--training-script", default="training_run.sh", help="Training script to launch")
    parser.add_argument("--check-interval", type=int, default=60, help="Check interval in seconds")
    parser.add_argument("--list-tpus", action="store_true", help="List available TPUs and exit")
    
    args = parser.parse_args()
    
    manager = TPUManager(args.launcher_script, args.training_script)
    
    if args.list_tpus:
        tpus = manager.list_available_tpus()
        print("\nAvailable TPUs:")
        for tpu in tpus:
            print(f"  {tpu.name} ({tpu.zone}) - {tpu.accelerator_type} - {tpu.cores} cores - {tpu.state}")
        return
    
    manager.monitor_and_relaunch(args.tpu_name, args.check_interval)

if __name__ == "__main__":
    main()
