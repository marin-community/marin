#!/bin/sh
set -eu

# Write reference solution into the container solution file.
cat <<'PY' > /solution/solution.py
# set title
# plt.title(myTitle, loc='center', wrap=True)
from textwrap import wrap

ax = plt.gca()
ax.set_title("\n".join(wrap(myTitle, 60)), loc="center", wrap=True)
# axes.set_title("\n".join(wrap(myTitle, 60)), loc='center', wrap=True)
PY
