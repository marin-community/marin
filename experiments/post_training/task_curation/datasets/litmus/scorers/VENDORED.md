# Litmus answer scorer

The adapter ships the pure answer extraction and reward functions from NVIDIA’s [Apache-2.0 Litmus verifier](https://github.com/NVIDIA-NeMo/Gym/blob/e87f88065db6105548194e1f12feea720c129cca/resources_servers/litmus_agent/app.py). Both release splits contain direct-answer requests. The packaged grader reads the final text response and the original row contract and uses the server’s default numeric tolerances. Metrics, transport types and the optional sandbox service stay in NeMo Gym.
