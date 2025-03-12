import random
with open('optimizer_sweep/utils.py', 'a') as f:
    f.write(f"\nprint({random.randint(0, 100000)})")
