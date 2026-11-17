const bayesian = `from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

n = 100
failures = 2
threshold = 0.95
confidence = 0.95
prior_a, prior_b = 1, 1
assert 0 <= failures <= n and n > 0
assert 0 < threshold < 1 and 0 < confidence < 1
assert prior_a > 0 and prior_b > 0
posterior = beta(prior_a + n - failures, prior_b + failures)
probability = posterior.sf(threshold)
low, high = posterior.ppf([0.025, 0.975])
acceptable = [f for f in range(n + 1)
              if beta.sf(threshold, prior_a + n - f, prior_b + f) >= confidence]
tolerated = str(max(acceptable)) if acceptable else "None at this trial count"

out = Path("output")
x = np.linspace(max(0, posterior.ppf(0.0001)), 1, 1000)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(x, posterior.pdf(x), color="#256b60", linewidth=2)
ax.fill_between(x, posterior.pdf(x), where=x >= threshold, color="#9ac6b6", alpha=0.6)
ax.axvline(threshold, color="#bf7245", linestyle="--", label=f"Target: {threshold:.1%}")
ax.set(xlabel="True success probability", ylabel="Posterior density", title="What do the trials tell us?")
ax.legend()
fig.tight_layout()
fig.savefig(out / "posterior.png", dpi=150)
plt.close(fig)

rows = []
for f in range(min(n, max(failures + 3, 10)) + 1):
    p = beta.sf(threshold, prior_a + n - f, prior_b + f)
    rows.append(f"| {f} | {p:.2%} | {'Yes' if p >= confidence else 'No'} |")
report = f"""# Eval confidence report

With **{failures} failures in {n} trials**, the posterior probability that the true
success rate exceeds **{threshold:.1%}** is **{probability:.2%}**.

| Measure | Result |
| --- | --- |
| Posterior mean | {posterior.mean():.2%} |
| 95% equal-tail credible interval | {low:.2%} – {high:.2%} |
| Required posterior confidence | {confidence:.1%} |
| Maximum tolerated failures | {tolerated} |

![Posterior success probability](posterior.png)

## Failure budget

| Failures | P(success rate > target) | Meets confidence? |
| --- | --- | --- |
""" + "\\n".join(rows) + f"""

## Assumptions

Independent Bernoulli trials with a shared, fixed success probability and a
Beta({prior_a}, {prior_b}) prior. This is a Bayesian posterior probability;
it depends on the prior and does not correct for correlated trials or dataset shift.
"""
(out / "index.md").write_text(report)
print(f"P(p > {threshold}) = {probability:.6f}")
print(f"Maximum tolerated failures: {tolerated}")
`;

const seaborn = `from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
sns.set_theme(style="whitegrid", palette="Set2")
out = Path("output")
n = 240
frame = pd.DataFrame({
    "model": np.repeat(["Baseline", "Candidate A", "Candidate B"], n // 3),
    "score": np.concatenate([rng.beta(a, b, n // 3) for a, b in [(8, 3), (12, 3), (16, 3)]]),
    "latency_ms": np.concatenate([rng.lognormal(mu, .25, n // 3) for mu in [4, 4.2, 4.1]]),
})
frame.to_csv(out / "observations.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 4))
sns.violinplot(data=frame, x="model", y="score", hue="model", legend=False, ax=ax)
ax.set_title("Score distribution by model")
fig.tight_layout()
fig.savefig(out / "scores.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 4))
sns.scatterplot(data=frame, x="latency_ms", y="score", hue="model", alpha=.7, ax=ax)
ax.set_title("Latency and score")
fig.tight_layout()
fig.savefig(out / "latency.png", dpi=150)
plt.close(fig)

means = frame.groupby("model")[["score", "latency_ms"]].mean()
rows = [f"| {model} | {row.score:.3f} | {row.latency_ms:.1f} |" for model, row in means.iterrows()]
(out / "index.md").write_text("""# Model comparison

A reproducible report from 240 synthetic observations (seed 42).
Replace the generated data with your uploaded CSV to explore real results.

## Average performance

| Model | Mean score | Mean latency (ms) |
| --- | --- | --- |
""" + "\\n".join(rows) + """

## Score distributions

![Score distributions](scores.png)

## Latency tradeoffs

![Latency versus score](latency.png)

[Download the observations](observations.csv)
""")
print(frame.groupby("model").agg({"score": ["mean", "std"], "latency_ms": "mean"}))
`;

const storage = `from pathlib import Path
import os
from importlib.metadata import version
from rigging.fsutil.parquet import parquet_lines

# marina's installed marin-rigging and PyArrow are available by default.
# filesystem_for() inside parquet_lines() routes gs:// and s3:// URLs and
# reads inherited credentials. Keep keys in the environment.
url = "s3://marin-us-east-02a/marin/data/rl/data/rl/gretel_text_to_sql-56330d3f5621db60/2026.10.07.1/final/part-00000.parquet"
rows = 5
assert 1 <= rows <= 20
preview = "\\n".join(parquet_lines(url, rows=rows))

pairs = {
    "CoreWeave": ("CW_KEY_ID", "CW_KEY_SECRET"),
    "R2": ("R2_KEY_ID", "R2_KEY_SECRET"),
    "AWS": ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"),
}
credentials = [f"| {backend} | {'Available' if all(os.environ.get(key) for key in keys) else 'Not configured'} |"
               for backend, keys in pairs.items()]
report = f"""# Object storage preview

Source: {url}

Read with **marin-rigging {version("marin-rigging")}** and **PyArrow {version("pyarrow")}**.
The preview reads the Parquet footer and up to {rows} rows from its first row group.
The reader skips rows when the first row group exceeds its preview budget.

## Inherited credential sources

Only availability is shown; secret values remain in the process environment.
GCS URLs use the host's application default credentials.

| Backend | Key pair |
| --- | --- |
""" + "\\n".join(credentials) + """

## Schema and rows

\x60\x60\x60text
""" + preview + """
\x60\x60\x60

[Download the text preview](preview.txt)
"""
Path("output/index.md").write_text(report)
Path("output/preview.txt").write_text(preview)
print(f"Previewed {url}")
`;

export const examples = {
  bayesian: {
    title: "Bayesian eval",
    entrypoint: "main.py",
    packages: "numpy\nscipy\nmatplotlib",
    files: [{ path: "main.py", content: bayesian, encoding: "utf-8" }],
  },
  seaborn: {
    title: "Seaborn report",
    entrypoint: "main.py",
    packages: "numpy\npandas\nseaborn\nmatplotlib",
    files: [{ path: "main.py", content: seaborn, encoding: "utf-8" }],
  },
  storage: {
    title: "Object storage preview",
    entrypoint: "main.py",
    packages: "",
    files: [{ path: "main.py", content: storage, encoding: "utf-8" }],
  },
  blank: {
    title: "Untitled project",
    entrypoint: "main.py",
    packages: "",
    files: [
      {
        path: "main.py",
        content:
          'from pathlib import Path\n\nPath("output/index.md").write_text("# Hello, Python\\n\\nYour report starts here.")\nprint("Report written to output/index.md")\n',
        encoding: "utf-8",
      },
    ],
  },
};
