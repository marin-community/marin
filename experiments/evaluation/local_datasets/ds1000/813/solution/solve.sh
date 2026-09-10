#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
p_guess = (pmin + pmax)/2
bounds = np.c_[pmin, pmax]
fp   = lambda p, x: p[0]*x[0]+p[1]*x[1]
e    = lambda p, x, y: ((fp(p,x)-y)**2).sum()
sol = sciopt.minimize(e, p_guess, bounds=bounds, args=(x,y))
result = sol.x


PY
