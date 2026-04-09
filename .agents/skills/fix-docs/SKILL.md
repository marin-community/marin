Your task is to fix the markdown docs within `lib/iris`, `lib/zephyr` and `lib/fray` so that they maximally comply with the principles below. Do NOT fix docs outside of the aforementioned directories.

Your output: You will dispatch sub-agents that will (1) thoroughly parse the code and the docs and (2) make all the documentation changes that are deemed appropriate, locally. You will commit the changes locally into a single commit, inform the user of the commit, and summarize the changes you made. Under no circumstances should you push any commit to the repo without explicit approval from the user.

Principles:
- Agents are the primary consumers of documentation
    - Why: Coding agents are expected to write most of (all?) the code in Marin
    - How:
        - Humans typically use agents as "documentation brokers"
        - Documentation is clearly segregated by intended audience (e.g., Humans, Agents):
            - Human docs should live in `docs/`. Human-specific documentation is typically high-level and "getting started". It is designed to be concise and maximally clear.
            - Agent docs should live in the directory structure, close to the code they pertain to (agents are good at navigating codebases). Agent docs are designed to be highly token-efficient.
- Agentic docs follow standards:
    - Why: So agents know where to look
    - How: The entry points for agent documentation are:
        - `**/*/AGENTS.md` (spread out throughout the codebase). These files contain:
            - Targeted guidance for agents, for their respective code subtrees
            - A recursive ToC / Index of the content inside the subtree
        - `**/*/OPS.md` (also spread out throughout the codebase):
            - An extensive field guide for agents on how to gather telemetry information about the systems in the subtree, run commands, debug, ...
        - `.agents/skills/*/SKILL.md`: Prompts for common agentic workflows
- Avoid rot:
    - Why: Rot confuses agents and wastes context
    - How: When you identify a document that is out of date with the current codebase, either update it (if it's recent and the change is small) or archive it to `.agents/project/YYYYMMDD_filename.md` (if it's for historical reading only). The code is the source of truth — agents are good at reading code when it's relevant to the task at hand.
- Agent context quality and quantity are paramount:
    - Why: The context window is all the model sees about our world. Model performance is known to degrade once the context window gets too full
    - How: Docs for agents should focus on:
        - "Negative guidance": Assume agents are highly smart / capable. Docs should focus on avoiding sharp edges based on _observed_ suboptimal agentic behaviors.
        - Markdown files are used primarily as indexes / ToC: to allow agents to be more **context-efficient** (i.e., help the agent "if you want to learn about X, read file Y")

Common markdown doc issues:
- Some of the design docs were used to guide agents during initial system design; most of them are stale at this point. Move these documents to `.agents/project/YYYYMMDD_filename.md` based on the date of the first commit (use `git`).
