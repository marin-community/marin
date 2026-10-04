# Snowball model family tree

[Open the Snowball model family tree](https://public.applets.marina.oa.dev/a/d3b5517a-c119-4fd9-999d-362183422bf3/)
to explore 118 model and run nodes across the 2.7T, 5.7T and 10T cooldown families
without signing in. This is a curated snapshot of the evidence available on
October 4, 2026. The stable URL opens the current applet revision.

The lineage map opens with the whole tree visible and its shared pretrain centered.
Click a node to expand its summary, algorithms, compute estimate and evidence.
Wheel or trackpad input zooms; dragging pans. The side panel has zoom controls,
ancestry, direct children and **Focus this branch**. **Read all** presents the
summaries as cards. Search includes model repository IDs and GitHub issue numbers.

**Timeline** defaults to the 10T family and a September 7 opening date. The cooldown
stays on the left while descendants appear in chronological columns. Switch to
2.7T or 5.7T with the timeline's cooldown selector. Space starts or pauses playback
outside editable controls. Normal playback takes 12 seconds; Fast takes 6 and Slow
24. The date scrubber and **Show all** allow direct navigation. Click a timeline
node for its date basis and full evidence.

Organization color records experiment ownership. The complete 2.7T branch belongs
to open-athena, including exports hosted under other Hub namespaces. Dashed links
identify inferred ancestry. Timeline dates can identify a checkpoint label, run
name, source report or Hub repository creation; they do not all identify training
completion. Undated nodes appear at the end. Column spacing follows chronology
without encoding elapsed duration.

The graph includes [the broad-domain RLVR campaign](https://github.com/marin-community/marin/issues/9735),
including the Antidoom step-20 checkpoint trained with a frozen-reference KL
penalty from public Antidoom step 12. The step-45 export's exact earlier weight ancestry remains qualified. Compute labels
identify the estimate's scope; unknown training and rollout totals remain unknown.

## Maintain the applet

The source package is `infra/marina/applets/snowball_family_tree/`.
`graph-data.json` contains the reviewed nodes, parents, source links, compute
assumptions and per-node timeline dates. Update that file when adding a checkpoint.
Adding a checkpoint normally requires only a graph update. The extracts
`evidence/evidence-glm53async48-telemetry.json` and
`evidence/evidence-antidoom12-telemetry.json` support scoped RL compute estimates
for their respective runs. Update or add an extract when new token accounting
changes an estimate, and link it from the node's evidence. Run records link to the
public campaign archive.

Build and validate from the repository root:

```bash
uv run marina validate infra/marina/applets/snowball_family_tree
uv run marina publish infra/marina/applets/snowball_family_tree --local --mode public
```

The local command starts a disposable Marina/Postgres instance and prints its
revision URL. Ctrl-C removes it. Before updating the public applet, inspect its
current revision:

```bash
uv run marina applets versions d3b5517a-c119-4fd9-999d-362183422bf3 --json
uv run marina publish infra/marina/applets/snowball_family_tree \
  --update d3b5517a-c119-4fd9-999d-362183422bf3 \
  --base-version CURRENT_VERSION --mode public
```

Replace `CURRENT_VERSION` with the reported current version. The manifest's build
command produces `dist/` from the HTML, JavaScript, CSS, graph and token-count
extracts. Publication uses that generated package. Generated files are ignored by
Git. Asset URLs are relative, and executable scripts use packaged files to satisfy
Marina's content security policy.
