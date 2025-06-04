import json
import glob
import uuid
import time
import openai
import subprocess
import datasets
from marin_env import MarinEnv, EnvStep

class MLEBenchEnv(MarinEnv):

    def __init__(self, endpoint: str, **kwargs):
        # Initialize endpoint
        self.endpoint = endpoint
        self.client = openai.OpenAI()
        self.model_name = kwargs.get("model_name", None)

    def step(self, **kwargs):
        run_name = f"{uuid.uuid4().hex}"
        commands = ["python", "third_party/mlebench/run_agent.py", "--agent-id", "aide/gpt-4-turbo", "--competition-set", "third_party/mlebench/experiments/splits/spaceship-titanic.txt", "--run-name", f"{run_name}"]
        process = subprocess.Popen(commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        stdout, stderr = process.communicate()

        # Decode the byte strings to print them
        if stdout:
            print("--- Subprocess Output ---")
            print(stdout.decode())
        
        if stderr:
            print("--- Subprocess Error ---")
            print(stderr.decode())

        # You can now use the output or error as needed
        result = {
            "stdout": stdout.decode() if stdout else "",
            "stderr": stderr.decode() if stderr else ""
        }

        submission_file = glob.glob(f"third_party/mlebench/{run_name}/*/spaceship-titanic*/submission/submission.csv")
        print(submission_file)

        cmd = f"mlebench grade-sample {submission_file} spaceship-titanic"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("result", result.stderr)
        try:
            result = '{' + result.stderr.split('{')[1]
            result = json.loads(result)
            reward = result["score"]
        except Exception as e:
            print(e)
            reward = None
        print("reward", reward)
        return EnvStep(llm_in=llm_in, llm_out=llm_out, reward=reward)

def main():
    env = MLEBenchEnv("", model_name="gpt-4-turbo")
    for i in range(1):
        print(f"step={i}", env.step(temperature=0.7, max_tokens=512))


if __name__ == "__main__":
    main()
