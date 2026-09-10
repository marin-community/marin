#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
import numpy as np
def g(df):
    time = df.time.tolist()
    car = df.car.tolist()
    nearest_neighbour = []
    euclidean_distance = []
    for i in range(len(df)):
        n = 0
        d = np.inf
        for j in range(len(df)):
            if df.loc[i, 'time'] == df.loc[j, 'time'] and df.loc[i, 'car'] != df.loc[j, 'car']:
                t = np.sqrt(((df.loc[i, 'x'] - df.loc[j, 'x'])**2) + ((df.loc[i, 'y'] - df.loc[j, 'y'])**2))
                if t < d:
                    d = t
                    n = df.loc[j, 'car']
        nearest_neighbour.append(n)
        euclidean_distance.append(d)
    return pd.DataFrame({'time': time, 'car': car, 'nearest_neighbour': nearest_neighbour, 'euclidean_distance': euclidean_distance})

df = g(df.copy())

PY
